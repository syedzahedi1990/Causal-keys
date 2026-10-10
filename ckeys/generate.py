"""Greedy generation under cache interventions (stage 8, preregistration J), answer parsing and frame discovery.

``greedy`` runs the prompt once with the KV cache on, then one token per step. Batch rows share the prompt ids (the
rows differ only by per-row clamp tables), so there is no padding. Clamps on prompt positions (ckeys.clamp) act in the
prompt pass, where the cached K/V at those positions are written, and pass through on the one-token decode steps; this
equals clamping at every step of a cache-free decode (tests/test_generate.py). Rows stop at EOS, at a newline, or when
``stop`` says so; finished rows keep being fed but their tokens are not recorded.
"""
from __future__ import annotations

import collections
import re

import torch

from .story import LOCATIONS

CAND_RE = re.compile(r"(?i)\b(" + "|".join(LOCATIONS) + r")")


END_OF_TURN = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>", "</s>", "<eos>")


def _eos_ids(model, tok):
    """The ids that end an answer: generation_config's eos ids, the tokenizer's eos, and the chat templates' end-of-turn
    tokens that exist in this vocabulary (generation_config lists only <eos> for Gemma-2 and Yi-1.5, whose turns end with
    <end_of_turn> and <|im_end|>)."""
    e = model.generation_config.eos_token_id if getattr(model, "generation_config", None) else None
    ids = set(e if isinstance(e, (list, tuple)) else [e]) | {getattr(tok, "eos_token_id", None)}
    conv = getattr(tok, "convert_tokens_to_ids", None)
    unk = getattr(tok, "unk_token_id", None)
    for t in END_OF_TURN:
        i = conv(t) if conv else None
        if isinstance(i, int) and i != unk:
            ids.add(i)
    return ids - {None}


@torch.no_grad()
def greedy(model, tok, ids: torch.Tensor, max_new: int = 16, stop=None, newline: bool = True):
    """ids [R, T] -> list of R lists of generated ids (EOS excluded; a newline token ends the row and is excluded)."""
    dev = next(model.parameters()).device
    R = ids.shape[0]
    eos = _eos_ids(model, tok)
    out = model(ids.to(dev), use_cache=True, logits_to_keep=1)
    past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
    res, done = [[] for _ in range(R)], [False] * R
    for step in range(max_new):
        for r in range(R):
            if done[r]:
                continue
            t = int(nxt[r])
            if t in eos or (newline and "\n" in tok.decode([t])):
                done[r] = True
                continue
            res[r].append(t)
            if stop is not None and stop(r, res[r]):
                done[r] = True
        if all(done) or step == max_new - 1:
            break
        out = model(nxt[:, None].to(dev), past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
    return res


@torch.no_grad()
def greedy_reference(model, tok, ids: torch.Tensor, max_new: int = 16, stop=None, newline: bool = True):
    """The same by cache-free full forwards at every step (the exactness reference)."""
    dev = next(model.parameters()).device
    eos = _eos_ids(model, tok)
    res = []
    for r in range(ids.shape[0]):
        x, gen = ids[r:r + 1].to(dev), []
        for _ in range(max_new):
            t = int(model(x, use_cache=False).logits[0, -1].argmax())
            if t in eos or (newline and "\n" in tok.decode([t])):
                break
            gen.append(t)
            if stop is not None and stop(r, gen):
                break
            x = torch.cat([x, torch.tensor([[t]], device=dev)], 1)
        res.append(gen)
    return res


@torch.no_grad()
def argmax_chain(model, ids: torch.Tensor, cont: list) -> torch.Tensor:
    """Teacher-forced argmax chain: bool [R] whether, with ``cont`` appended, each next token of ``cont`` is the argmax
    at its position (equivalently, greedy generation reproduces ``cont``; used under hooks where generation would leave
    a row-restricted hook's frame)."""
    dev = next(model.parameters()).device
    T = ids.shape[1]
    x = torch.cat([ids, torch.tensor([cont], dtype=ids.dtype).expand(ids.shape[0], -1)], 1).to(dev)
    lg = model(x, use_cache=False, logits_to_keep=len(cont) + 1).logits
    pred = lg[:, :len(cont)].argmax(-1).cpu()
    return (pred == torch.tensor(cont)[None]).all(1)


def candidate_stop(tok):
    """stop(r, ids): True once the decoded text contains a candidate location word followed by a non-letter (or once
    a candidate word is complete at the end of a token)."""
    def f(r, ids):
        s = tok.decode(ids)
        m = CAND_RE.search(s)
        return bool(m) and (m.end() < len(s) or s.endswith(m.group(0)))
    return f


def parse_answer(text: str) -> str:
    """The first candidate location named in ``text`` (case-insensitive, plural allowed), else 'other'."""
    m = CAND_RE.search(text)
    return m.group(1).lower() if m else "other"


def extract_frame(text: str, names: dict) -> str | None:
    """The text before the first candidate word, with the item's names replaced by {a}, {b}, {o}, {d}; None if no
    candidate, or the frame has a newline or is longer than 60 characters."""
    m = CAND_RE.search(text)
    if not m:
        return None
    fr = text[:m.start()]
    if "\n" in fr or len(fr) > 60:
        return None
    for k in sorted(names, key=lambda k: -len(names[k])):
        if names[k]:
            fr = re.sub(r"\b" + re.escape(names[k]) + r"\b", "{" + k + "}", fr)
    return fr


def discover_frames(records, fixed, min_share=0.02, min_count=2, max_frames=16):
    """records: iterable of (arm, frame or None). A frame not in ``fixed`` is admitted if it occurs in at least
    ``min_share`` (and at least ``min_count``) of some arm's generations; at most ``max_frames``, most frequent first
    (ties by string)."""
    per_arm, total = collections.defaultdict(collections.Counter), collections.Counter()
    n_arm = collections.Counter()
    for arm, fr in records:
        n_arm[arm] += 1
        if fr is None or fr in fixed:
            continue
        per_arm[arm][fr] += 1
        total[fr] += 1
    ok = {fr for arm, c in per_arm.items() for fr, k in c.items() if k >= min_count and k >= min_share * n_arm[arm]}
    return sorted(ok, key=lambda fr: (-total[fr], fr))[:max_frames]
