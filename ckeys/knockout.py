"""Attention knockout by a 4D mask: chosen (row, column) pairs get attention weight exactly 0 in every layer and
head (the row is renormalised over its remaining columns), plus the row/column groups of the belief-story prompts.

transformers passes a 4D ``attention_mask`` through as is: sdpa takes a boolean mask (True = attend), eager adds
a float mask (0 / finfo.min), so the mask must match ``model.config._attn_implementation``. Never fully mask a
row (NaN under sdpa); column 0 and the diagonal are always kept. All forbidden rows must lie after the clamped
position p so that positions <= p compute identically under every mask.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .encoding import candidate_ids
from .story import LOCATIONS, pick_x


def knockout_mask(T: int, pairs: Sequence[tuple[int, int]], B: int = 1, dtype: torch.dtype | None = None,
                  impl: str = "sdpa") -> torch.Tensor:
    """``[B, 1, T, T]`` causal mask with ``pairs`` forbidden: boolean for sdpa, additive (0 / finfo.min) for eager."""
    assert impl in ("sdpa", "eager"), impl
    m = torch.ones(T, T, dtype=torch.bool).tril()
    for r, c in pairs:
        assert r != c and c != 0, (r, c)
        m[r, c] = False
    m = m.view(1, 1, T, T).expand(B, 1, T, T).clone()
    if impl == "eager":
        dtype = dtype or torch.float32
        return torch.where(m, torch.tensor(0.0, dtype=dtype), torch.finfo(dtype).min)
    return m


def knockout_masks(T: int, pairs_per_row: Sequence[Sequence[tuple[int, int]]], dtype=None, impl="sdpa"):
    """One mask per batch row, concatenated to ``[R, 1, T, T]``."""
    return torch.cat([knockout_mask(T, p, 1, dtype, impl) for p in pairs_per_row])


def mask_for_model(model, T: int, pairs, B: int = 1) -> torch.Tensor:
    impl = model.config._attn_implementation
    assert impl in ("sdpa", "eager"), impl
    return knockout_mask(T, pairs, B, next(model.parameters()).dtype, impl)


def masks_for_model(model, T: int, pairs_per_row) -> torch.Tensor:
    impl = model.config._attn_implementation
    assert impl in ("sdpa", "eager"), impl
    return knockout_masks(T, pairs_per_row, next(model.parameters()).dtype, impl)


def rows_in(offsets, a: int, b: int) -> list[int]:
    return [i for i, (s, e) in enumerate(offsets) if e > a and s < b]


def groups(tok, text: str, ids: torch.Tensor, offsets, p: int, core: dict, arm: str, X: str | None = None,
           query: str | None = None) -> dict:
    """Row/column groups of one belief-story prompt (``text`` = the full chat text, ``ids`` = [1, T], ``offsets`` =
    character offsets per token): p, a = T-1, c_prev, R_cand (candidate-word rows after p; 6, or 0 under NONE),
    R_trk (B, S, X rows of R_cand), R_oth, R_q (the question line), R_post (rows after the question/mention
    block except a and R_cand), R_tail (rows > p not in R_cand), R_all (rows > p), C_init (pre-p rows holding
    the initial-location word)."""
    T = ids.shape[1]
    cid = candidate_ids(tok, "P1")
    X = X or pick_x(core)
    toks = ids[0].tolist()
    R_cand = [i for i in range(p + 1, T) if toks[i] in cid]
    assert len(R_cand) in (6, 0), (arm, R_cand)
    trk = {cid[LOCATIONS.index(w)] for w in (core["base"], core["source"], X)}
    R_trk = [i for i in R_cand if toks[i] in trk]
    R_oth = [i for i in R_cand if i not in R_trk]
    if query is None:
        q0 = text.index("\nQuestion: ")
        q1 = text.index("\n", q0 + 1)
    else:
        q0 = text.index("\nQuestion: " + query)
        q1 = q0 + len("\nQuestion: " + query)
    R_q = rows_in(offsets, q0 + 1, q1)
    assert R_q and min(R_q) > p, (arm, R_q)
    block_end = max(R_q + R_cand)
    R_post = [i for i in range(block_end + 1, T - 1) if i not in R_cand]
    R_tail = [i for i in range(p + 1, T) if i not in R_cand]
    init_id = cid[LOCATIONS.index(core["initial"])]
    C_init = [i for i in range(p) if toks[i] == init_id]
    own = f"the {core['object']} is in the {core['initial']}."
    o0 = text.index(own)
    assert any(i in rows_in(offsets, o0, o0 + len(own)) for i in C_init), (arm, C_init)
    assert 1 <= len(C_init) <= 2 and toks[p - 1] == tok.encode(" the", add_special_tokens=False)[-1], (arm, C_init, toks[p - 1])
    return dict(p=p, a=T - 1, c_prev=p - 1, R_cand=R_cand, R_trk=R_trk, R_oth=R_oth, R_q=R_q, R_post=R_post,
                R_tail=R_tail, R_all=list(range(p + 1, T)), C_init=C_init)


def target_ids(tok) -> dict:
    """Per location: the lower-case id, the id of the capitalised form when single-token (else its first piece,
    flagged ``cap_single=False``). The six first pieces are asserted distinct. ``loc_mass`` lists the ids whose
    probability counts as location mass (lower-case plus single-token capitalised forms)."""
    out, first = {}, []
    for w in LOCATIONS:
        lo = tok.encode(" " + w, add_special_tokens=False)
        cap = tok.encode(" " + w.capitalize(), add_special_tokens=False)
        assert len(lo) == 1, w
        out[w] = {"lower": lo[0], "cap": cap[0], "cap_single": len(cap) == 1}
        first.append(cap[0])
    assert len(set(first)) == 6 and not set(first) & {v["lower"] for v in out.values()}, first
    out["loc_mass"] = [v["lower"] for w, v in out.items() if w != "loc_mass"] + \
                      [v["cap"] for w, v in out.items() if w != "loc_mass" and v["cap_single"]]
    return out


def on_target(top1: int, w: str, tid: dict) -> bool:
    return top1 in (tid[w]["lower"], tid[w]["cap"])
