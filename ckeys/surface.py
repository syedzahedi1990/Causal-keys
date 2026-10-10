"""Emitted-form scoring (stage 8, preregistration J): the log-probability of a candidate answer word summed over the
surface forms a model emits it in, computed exactly in one forward pass per batch row.

Forms of a word w (W = w.capitalize()) are frame + w and frame + W for each frame:
  FRAMES_SIGMA   " ", "", " The ", " the ", "The ", "the "                                    (12 forms; reviewer-named)
  FRAMES_E_FIXED FRAMES_SIGMA + locatives " In the ", " in the ", " On the ", " on the ", " At the ", " at the ",
                 " Inside the ", " inside the " + markdown " **", "**"                       (32 forms; primary)
plus, per model, the frames discovered on calibration generations (ckeys.generate.discover_frames), which may name
the item's agents/object/other location as {a}, {b}, {o}, {d} and are instantiated per item.

A form's token sequence is its continuation of the prompt: ids(prefill + form) minus ids(prefill) when that boundary is
stable, else the form's standalone ids (a model that has read the prefill's last token can only emit standalone
pieces). Forms with identical sequences are counted once, and no sequence may be a proper prefix of another, so the
events summed are disjoint. log P(form) = sum_t log p(f_t | prompt, f_<t) is read from a token trie appended after the
prompt: every proper prefix of every form is one node, attending to the whole prompt and to its own ancestors (4D
tree mask: boolean for sdpa, 0 / finfo.min for eager, as ckeys.knockout), at position T - 1 + depth. The prompt's last
position scores each form's first token; the node holding f_1..f_{t-1} scores f_t. Prompt positions see only the
prompt (causal), so clamps on prompt positions act exactly as in a plain pass.
"""
from __future__ import annotations

import torch

FRAMES_SIGMA = (" ", "", " The ", " the ", "The ", "the ")
FRAMES_E_FIXED = FRAMES_SIGMA + (" In the ", " in the ", " On the ", " on the ", " At the ", " at the ",
                                 " Inside the ", " inside the ", " **", "**")


def form_ids(tok, form: str, prefill: str = "Answer:") -> tuple[int, ...]:
    b = tok(prefill, add_special_tokens=False).input_ids
    full = tok(prefill + form, add_special_tokens=False).input_ids
    if full[:len(b)] == b and len(full) > len(b):
        return tuple(full[len(b):])
    return tuple(tok(form, add_special_tokens=False).input_ids)


def instantiate(frame: str, names: dict | None) -> str | None:
    """A discovered frame with {a}/{b}/{o}/{d} filled from ``names``; None if it names a slot the item lacks."""
    if "{" not in frame:
        return frame
    try:
        return frame.format(**(names or {}))
    except (KeyError, IndexError):
        return None


class FormSet:
    """Token sequences of every form of every candidate word, and the trie of their proper prefixes.

    ``frames``: a sequence of frames, or a dict {"sigma": frames, "E": frames} naming nested form sets; every set's
    log-sum is reported by ``score``. Each set must contain the bare forms " w" (so the lowercase L score is a field)."""

    def __init__(self, tok, words, frames=None, names: dict | None = None, prefill: str = "Answer:"):
        if frames is None:
            frames = {"sigma": FRAMES_SIGMA, "E": FRAMES_E_FIXED}
        if not isinstance(frames, dict):
            frames = {"E": tuple(frames)}
        self.words, self.sets = list(words), {}
        self.tok, self.prefill = tok, prefill
        allseq = set()
        for name, fr in frames.items():
            per = {}
            for w in self.words:
                seqs = {}
                for f in fr:
                    f = instantiate(f, names)
                    if f is None:
                        continue
                    for x in (w, w.capitalize()):
                        s = form_ids(tok, f + x, prefill)
                        seqs.setdefault(s, f + x)
                per[w] = seqs
                allseq |= set(seqs)
            self.sets[name] = per
        allseq = sorted(allseq)
        for i, s in enumerate(allseq):          # disjoint events: no sequence is a proper prefix of another
            for t in allseq[i + 1:]:
                if len(t) > len(s) and t[:len(s)] == s:
                    raise ValueError(f"form {tok.decode(list(s))!r} is a proper prefix of {tok.decode(list(t))!r}")
                if t[:len(s)] != s:
                    break
        self.seqs = allseq
        self.nodes = sorted({s[:k] for s in allseq for k in range(1, len(s))}, key=lambda n: (len(n), n))
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.lower = {w: form_ids(tok, " " + w, prefill) for w in self.words}
        assert all(self.lower[w] in self.sets[n][w] for n in self.sets for w in self.words), "every set needs ' w'"

    def __len__(self):
        return len(self.nodes)

    def check_decode(self):
        """Every form's sequence decodes back to its string (up to leading-space normalisation of a standalone
        continuation); returns the list of mismatches."""
        bad = []
        for per in self.sets.values():
            for seqs in per.values():
                for s, f in seqs.items():
                    d = self.tok.decode(list(s))
                    if d != f and d.strip() != f.strip():
                        bad.append((f, d))
        return bad


def trie_inputs(ids: torch.Tensor, fs: FormSet, impl: str, dtype=torch.float32):
    """ids [R, T] -> (ids2 [R, T+N], position ids [R, T+N], mask [R, 1, T+N, T+N])."""
    R, T = ids.shape
    N = len(fs)
    ext = torch.tensor([n[-1] for n in fs.nodes], dtype=ids.dtype)
    ids2 = torch.cat([ids, ext[None].expand(R, N)], 1)
    pos = torch.cat([torch.arange(T), torch.tensor([T + len(n) - 1 for n in fs.nodes], dtype=torch.long)])
    m = torch.zeros(T + N, T + N, dtype=torch.bool)
    m[:T, :T] = torch.ones(T, T, dtype=torch.bool).tril()
    m[T:, :T] = True
    for i, n in enumerate(fs.nodes):
        for k in range(1, len(n) + 1):
            m[T + i, T + fs.index[n[:k]]] = True
    m = m[None, None].expand(R, 1, T + N, T + N)
    if impl == "eager":
        m = torch.where(m, torch.tensor(0.0, dtype=dtype), torch.tensor(torch.finfo(dtype).min, dtype=dtype))
    return ids2, pos[None].expand(R, -1), m


def _chain(lp, s, fs):
    v = lp[:, 0, s[0]]
    for t in range(1, len(s)):
        v = v + lp[:, 1 + fs.index[s[:t]], s[t]]
    return v


def _fields(lp, fs, topk):
    """lp [R, 1+N, V] float32 log-softmax (row 0: the answer position; 1+i: trie node i)."""
    res = {"L": {}, "forms": {}, "first": lp[:, 0]}
    seqlp = {s: _chain(lp, s, fs) for s in fs.seqs}
    for name, per in fs.sets.items():
        res[name] = {}
        for w in fs.words:
            F = torch.stack([seqlp[s] for s in per[w]], 1)
            res[name][w] = torch.logsumexp(F, 1)
            res["forms"].setdefault(name, {})[w] = {f: seqlp[s] for s, f in per[w].items()}
    for w in fs.words:
        res["L"][w] = lp[:, 0, fs.lower[w][0]] if len(fs.lower[w]) == 1 else _chain(lp, fs.lower[w], fs)
    p0 = lp[:, 0].exp()
    starts = sorted({s[0] for s in fs.seqs})
    res["gap"] = 1.0 - p0[:, starts].sum(1)                      # first-token mass that begins no form
    v, i = p0.topk(topk, 1)
    res["top"] = (i, v)
    return res


@torch.no_grad()
def score(model, ids: torch.Tensor, fs: FormSet, topk: int = 10, mask_extra=None):
    """One forward over prompt + trie. Returns {"L": {w: [R]}, <set name>: {w: [R]} (log-sum over that set's forms),
    "forms": {set: {w: {form: [R]}}}, "first": [R, V] log-softmax at the answer position, "gap": [R],
    "top": (ids [R, k], probs [R, k])}. Clamps (ckeys.clamp) act at prompt positions as usual: they index positions
    < T, which the trie columns never are."""
    impl = model.config._attn_implementation
    dtype = next(model.parameters()).dtype
    dev = next(model.parameters()).device
    if len(fs) == 0:
        out = model(ids.to(dev), use_cache=False, logits_to_keep=1)
        return _fields(torch.log_softmax(out.logits.float(), -1), fs, topk)
    ids2, pos, mask = trie_inputs(ids.cpu(), fs, impl, dtype)
    out = model(ids2.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev), use_cache=False,
                logits_to_keep=len(fs) + 1)
    return _fields(torch.log_softmax(out.logits.float(), -1), fs, topk)


@torch.no_grad()
def score_reference(model, ids: torch.Tensor, fs: FormSet, topk: int = 10):
    """The same fields by one plain causal forward per form sequence (the exactness reference, and the fallback if the
    tree pass fails its BF16 floor)."""
    dev = next(model.parameters()).device
    T, N, V = ids.shape[1], len(fs), None
    rows = {}
    first = None
    for s in fs.seqs + [l for l in fs.lower.values()]:
        x = torch.cat([ids, torch.tensor([s], dtype=ids.dtype).expand(ids.shape[0], -1)], 1).to(dev)
        lp = torch.log_softmax(model(x, use_cache=False).logits.float(), -1)
        if first is None:
            first = lp[:, T - 1]
            V = lp.shape[-1]
        for t in range(1, len(s)):
            rows[s[:t]] = lp[:, T - 1 + t]
    lp = torch.zeros(ids.shape[0], 1 + N, V)
    lp[:, 0] = first.cpu()
    for n, i in fs.index.items():
        lp[:, 1 + i] = rows[n].cpu()
    return _fields(lp, fs, topk)
