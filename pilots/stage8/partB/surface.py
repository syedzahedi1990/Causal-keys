"""Prototype of ckeys/surface.py (Part B of stage 8): score each candidate as the logsumexp over its surface forms,
exactly, in ONE forward pass per batch row, by appending a token trie after the prompt with a tree-attention 4D mask
and explicit position ids.

Forms of a candidate word w (W = w.capitalize()):
  SIGMA (primary):  " w", " W", "w", "W", and A + " w", A + " W" for A in (" The", " the", "The", "the")   -> 12 forms
  SIGMA_PLUS (robustness): SIGMA + {" In the", " in the", " On the", " on the", " At the", " at the"} x {w, W}
                           + {" **w", " **W", "**w", "**W"}                                              -> 28 forms
A form's token sequence is its continuation of the prompt (ids(prefill + form) minus ids(prefill) when the boundary is
stable, else the form's standalone ids: a model that has read the prompt's last token can only emit standalone pieces).
Distinct forms with identical token sequences are counted once. log S(w) = logsumexp_f log P(f), with
log P(f) = sum_t log p(f_t | prompt, f_<t) read from the trie: the prompt's last position scores f_1; the trie node
holding f_1..f_{t-1} scores f_t.
"""
from __future__ import annotations

import torch

ARTICLES = (" The", " the", "The", "the")
LOCATIVES = (" In the", " in the", " On the", " on the", " At the", " at the", " Inside the", " inside the")


def surface_forms(word: str, plus: bool = False) -> list[str]:
    W = word.capitalize()
    f = [" " + word, " " + W, word, W] + [a + " " + x for a in ARTICLES for x in (word, W)]
    if plus:
        f += [l + " " + x for l in LOCATIVES for x in (word, W)] + [" **" + word, " **" + W, "**" + word, "**" + W]
    return f


def form_ids(tok, form: str, prefill: str = "Answer:") -> tuple[int, ...]:
    b = tok(prefill, add_special_tokens=False).input_ids
    full = tok(prefill + form, add_special_tokens=False).input_ids
    if full[:len(b)] == b and len(full) > len(b):
        return tuple(full[len(b):])
    return tuple(tok(form, add_special_tokens=False).input_ids)


class FormSet:
    """Token sequences of every form of every candidate, and the trie of their proper prefixes."""

    def __init__(self, tok, words, plus=False, prefill="Answer:"):
        self.words = list(words)
        self.seqs = {w: sorted({form_ids(tok, f, prefill) for f in surface_forms(w, plus)}) for w in self.words}
        nodes = sorted({s[:k] for w in self.words for s in self.seqs[w] for k in range(1, len(s))}, key=lambda n: (len(n), n))
        self.nodes = nodes                       # trie nodes, parents before children
        self.index = {n: i for i, n in enumerate(nodes)}
        self.lower = {w: tok.encode(" " + w, add_special_tokens=False) for w in self.words}

    def __len__(self):
        return len(self.nodes)


def trie_inputs(ids: torch.Tensor, fs: FormSet, impl: str, dtype=torch.float32):
    """ids [R, T] -> (ids2 [R, T+N], pos [R, T+N], mask [R, 1, T+N, T+N]) with tree attention (bool for sdpa,
    additive for eager)."""
    R, T = ids.shape
    N = len(fs)
    tok_ext = torch.tensor([n[-1] for n in fs.nodes], dtype=ids.dtype)
    ids2 = torch.cat([ids, tok_ext[None].expand(R, N)], 1)
    pos = torch.cat([torch.arange(T), torch.tensor([T + len(n) - 1 for n in fs.nodes], dtype=torch.long)])
    m = torch.zeros(T + N, T + N, dtype=torch.bool)
    m[:T, :T] = torch.ones(T, T, dtype=torch.bool).tril()
    m[T:, :T] = True
    for i, n in enumerate(fs.nodes):
        for k in range(1, len(n) + 1):
            m[T + i, T + fs.index[n[:k]]] = True       # ancestors and itself
    m = m[None, None].expand(R, 1, T + N, T + N).clone()
    if impl == "eager":
        m = torch.where(m, torch.tensor(0.0, dtype=dtype), torch.tensor(torch.finfo(dtype).min, dtype=dtype))
    return ids2, pos[None].expand(R, -1).clone(), m


@torch.no_grad()
def score(model, ids: torch.Tensor, fs: FormSet):
    """Forward with the trie. Returns dict: 'sigma' {w: [R]} logsumexp over forms, 'lower' {w: [R]} the plain
    lowercase ' w' log-prob, 'forms' {w: [R, n_forms]}, 'first' [R, V] log-softmax at the answer position."""
    impl = model.config._attn_implementation
    dtype = next(model.parameters()).dtype
    ids2, pos, mask = trie_inputs(ids.cpu(), fs, impl, dtype)
    dev = next(model.parameters()).device
    T = ids.shape[1]
    N = len(fs)
    out = model(ids2.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev), use_cache=False,
                logits_to_keep=N + 1)
    lp = torch.log_softmax(out.logits.float(), -1)     # [R, N+1, V]: index 0 = answer position T-1, 1+i = node i
    res = {"sigma": {}, "lower": {}, "forms": {}, "first": lp[:, 0]}
    for w in fs.words:
        cols = []
        for s in fs.seqs[w]:
            v = lp[:, 0, s[0]]
            for t in range(1, len(s)):
                v = v + lp[:, 1 + fs.index[s[:t]], s[t]]
            cols.append(v)
        F = torch.stack(cols, 1)
        res["forms"][w] = F
        res["sigma"][w] = torch.logsumexp(F, 1)
        res["lower"][w] = lp[:, 0, fs.lower[w][0]]
    return res


@torch.no_grad()
def score_reference(model, ids: torch.Tensor, fs: FormSet):
    """The same quantities by separate plain forward passes (one per form): the exactness reference."""
    dev = next(model.parameters()).device
    res = {"sigma": {}, "forms": {}}
    for w in fs.words:
        cols = []
        for s in fs.seqs[w]:
            x = torch.cat([ids, torch.tensor([s], dtype=ids.dtype).expand(ids.shape[0], -1)], 1).to(dev)
            lp = torch.log_softmax(model(x, use_cache=False).logits.float(), -1)
            T = ids.shape[1]
            v = lp[:, T - 1, s[0]]
            for t in range(1, len(s)):
                v = v + lp[:, T - 1 + t, s[t]]
            cols.append(v)
        F = torch.stack(cols, 1)
        res["forms"][w] = F
        res["sigma"][w] = torch.logsumexp(F, 1)
    return res
