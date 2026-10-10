"""Rows, tables and per-item passes of stage 8 part A (preregistration J): counterfactual SQuAD items, span clamps.

An item in a format is encoded with each of its entities in the passage (B = the answer, S, X, Z): the four prompts have
one length and differ only inside B's entity span P (ckeys.natural_formats.check_item). A row of a batch on the B prompt
names a donor run for the key and for the value at every position of P in every layer from 0 (``Row``); its tables
are the captured K / V of the donor runs (ckeys.clamp.stack_rows layout, [R, |P|, D]) and are written by
ckeys.clamp.clamp_kv. ID = (B, B) is the in-batch self-clamp (the identity); KV_S = (S, S) reproduces the S run.
Exploratory rows change K or V only from an onset layer (B's own tables below it, which is the identity) or only at
the first or the remaining span positions.

Row names: ID, K_S, V_S, KV_S, K_X, V_X, KV_X (every format); K_Z, V_Z, KV_Z and the cue-conflict, flag-only and
copy-fallback rows KS_VX, KX_VS, KS_VZ, KZ_VS (NOM, OPTA, LETA); exploratory K_S@on, V_S@on (onset), K_S^1, K_S^r,
V_S^1, V_S^r (first span position, the rest).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .clamp import capture_kv, clamp_kv
from .generate import greedy
from .natural_formats import (LETTER_FORMATS, answer_of, check_item, decision_ids, encode_item, norm,
                              passage_range, span_positions)

ENT_KEYS = (("B", "answer"), ("S", "S"), ("X", "X"), ("Z", "Z"))


@dataclass(frozen=True)
class Row:
    k: str                 # donor run of the key ("B", "S", "X", "Z")
    v: str                 # donor run of the value
    onset: int = 0         # layers below the onset take B's own tables
    piece: str = "all"     # "all", "first" (only P[0] takes the donors) or "rest" (only P[1:])

    def donor(self, l: int, ch: str, i: int) -> str:
        if l < self.onset or (self.piece == "first" and i != 0) or (self.piece == "rest" and i == 0):
            return "B"
        return self.k if ch == "k" else self.v


BASE = {"ID": Row("B", "B"), "K_S": Row("S", "B"), "V_S": Row("B", "S"), "KV_S": Row("S", "S"),
        "K_X": Row("X", "B"), "V_X": Row("B", "X"), "KV_X": Row("X", "X")}
ZROWS = {"K_Z": Row("Z", "B"), "V_Z": Row("B", "Z"), "KV_Z": Row("Z", "Z")}
CUE = {"KS_VX": Row("S", "X"), "KX_VS": Row("X", "S"), "KS_VZ": Row("S", "Z"), "KZ_VS": Row("Z", "S")}
FULL_FORMATS = ("NOM", "OPTA", "LETA")          # the formats with the Z, cue-conflict, flag-only and copy-fallback rows
LETA_REDUCED = ("ID", "KV_S", "KV_X", "KV_Z", "KS_VX", "KX_VS")   # LETA under the deadline: accuracy and cue conflict
COMPETENCE = ("ID", "KV_S", "KV_X")


def core_rows(fmt: str, reduced: bool = False) -> dict:
    rows = dict(BASE) | (ZROWS | CUE if fmt in FULL_FORMATS else {})
    if reduced:
        assert fmt == "LETA"
        rows = {n: r for n, r in rows.items() if n in LETA_REDUCED}
    return rows


def explore_rows(fmt: str, nL: int) -> dict:
    """E1 onset rows (from round(0.3 L)) in NOM and OPTA; E2 piece rows: K_S at the first / remaining span positions in
    OPTA, V_S in NOM; ID as the in-batch reference."""
    on = round(0.3 * nL)
    rows = {"ID": BASE["ID"], "K_S": BASE["K_S"], "V_S": BASE["V_S"],
            "K_S@on": Row("S", "B", onset=on), "V_S@on": Row("B", "S", onset=on)}
    if fmt == "OPTA":
        rows |= {"K_S^1": Row("S", "B", piece="first"), "K_S^r": Row("S", "B", piece="rest")}
    if fmt == "NOM":
        rows |= {"V_S^1": Row("B", "S", piece="first"), "V_S^r": Row("B", "S", piece="rest")}
    return rows


def tables(kv: dict, rows, nL: int, nP: int, which: str = "kv") -> dict:
    """{(l, ch): [R, nP, D]}: row r's entry at span position i is ``kv[rows[r].donor(l, ch, i)][(l, ch)][i]``."""
    return {(l, ch): torch.stack([torch.stack([kv[r.donor(l, ch, i)][(l, ch)][i] for i in range(nP)]) for r in rows])
            for l in range(nL) for ch in which}


def gen_text(tok, ids, gen) -> str:
    """The text a generation adds to the prompt, decoded together with the prompt (sentencepiece decoders drop the
    leading space of a sequence's first token)."""
    pre, full = tok.decode(list(ids)), tok.decode(list(ids) + list(gen))
    return full[len(pre):] if full.startswith(pre) else tok.decode(list(gen))


def end_ids(tok) -> set:
    """Ids that end an answer: the tokenizer's special tokens (all_special_ids and the added tokens marked special). Some
    generation configs list only <eos> and not the end of the chat turn (Gemma-2-9b-it: <end_of_turn>; Yi-1.5-9B-Chat:
    <|im_end|>), so greedy decoding goes on past it; its text would otherwise be parsed as part of the answer."""
    added = getattr(tok, "added_tokens_decoder", None) or {}
    return set(tok.all_special_ids) | {i for i, t in added.items() if getattr(t, "special", False)}


def cut(gen, ends) -> list:
    """A generation up to its first special token (excluded)."""
    return next((list(gen[:n]) for n, t in enumerate(gen) if t in ends), list(gen))


class Invalid(Exception):
    """An item that is not valid for a tokenizer and frame in a format (skipped, with its reason)."""


def prep(tok, it, fmt, frame):
    """Encodings of the B, S, X, Z prompts, the span P and the frame-aware decision ids; raises Invalid(reason)."""
    enc = {Y: encode_item(tok, it, fmt, it[k]) for Y, k in ENT_KEYS}
    tb, ib, ob, (c0, c1) = enc["B"]
    P = span_positions(ob, c0, c1)
    for Y in "SXZ":
        iy = enc[Y][1]
        if len(iy) != len(ib):
            raise Invalid(f"{fmt}: length {Y}")
        diff = [i for i, (x, y) in enumerate(zip(ib, iy)) if x != y]
        if not diff or not set(diff) <= set(P):
            raise Invalid(f"{fmt}: diff outside span {Y}")
    try:
        dd = decision_ids(tok, tb, ib, it, fmt, frame, strict=it["stratum"] != "SP")
    except AssertionError as ex:
        raise Invalid(f"{fmt}: {ex}") from None
    return dict(fmt=fmt, text=tb, offsets=ob, ids={Y: list(enc[Y][1]) for Y in "BSXZ"}, texts={Y: enc[Y][0] for Y in "BSXZ"},
                P=P, T=len(ib), **dd)


def valid_all(tok, it, formats, frame):
    """None when the item is valid in every format for this tokenizer and frame, else the first reason."""
    bad = check_item(tok, it, formats)
    if bad:
        return bad
    try:
        for f in formats:
            prep(tok, it, f, frame)
    except Invalid as ex:
        return str(ex)
    return None


@torch.no_grad()
def capture(model, d, layers, dev):
    """One batch of the B, S, X, Z prompts + w: K/V at P in every layer ({run: {(l, ch): [|P|, D]}}) and each run's
    log-probabilities of the decision tokens at the decision position ({run: {Y: lp}})."""
    x = torch.tensor([d["ids"][Y] + d["w"] for Y in "BSXZ"])
    with capture_kv(model, d["P"], layers) as C:
        lg = model(x.to(dev), use_cache=False, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
    kv = {Y: {k: v[i] for k, v in C.items()} for i, Y in enumerate("BSXZ")}
    runs = {Y: {c: float(lg[i, t]) for c, t in d["dec"].items()} for i, Y in enumerate("BSXZ")}
    return kv, runs


@torch.no_grad()
def score_rows(model, d, kv, rows: dict, nL, dev, target="S"):
    """The rows on B prompt + c_target, teacher-forced: per row the decision log-probabilities {Y: lp}, the option mass
    (sum of the four options' decision probabilities), the decision argmax, and the log-probabilities of the tokens of
    c_target after the decision token. Returns {row: {...}}."""
    names = list(rows)
    c, j = d["c"][target], d["j"]
    tabs = tables(kv, [rows[n] for n in names], nL, len(d["P"]))
    x = torch.tensor([d["ids"]["B"] + c]).expand(len(names), -1)
    keep = len(c) - j + 1
    with clamp_kv(model, d["P"], tabs, range(nL)):
        lg = model(x.to(dev), use_cache=False, logits_to_keep=keep).logits.float().log_softmax(-1)
    opts = [Y for Y in ("B", "S", "X", "D") if Y in d["dec"]]
    out = {}
    for r, n in enumerate(names):
        l0 = lg[r, 0]
        out[n] = {"lp": {Y: float(l0[t]) for Y, t in d["dec"].items()},
                  "mass": float(l0[[d["dec"][Y] for Y in opts]].exp().sum()), "argmax": int(l0.argmax()),
                  "cont": [float(lg[r, t - j, c[t]]) for t in range(j + 1, len(c))]}
    return out


def decided_stop(tok, it, fmt):
    """stop(r, ids) for ckeys.generate.greedy: True once the generation's answer class can no longer change. Entity
    formats: the first line has >= W + 2 normalized words, W the most words of a normalized candidate (appending text
    changes at most the last normalized word, so the first W + 1 are final, and ``matches`` against every candidate is
    fixed). Letter formats: two characters after the skipped leading markdown (letter_of reads at most two), neither of
    them the replacement character of an incomplete multi-byte character (which a later token completes). Every format:
    a special token (end_ids) ends the answer, which is then read up to it (cut)."""
    ends = end_ids(tok)
    if fmt in LETTER_FORMATS:
        def first2(ids):
            s = tok.decode(ids).lstrip().lstrip("*_`([\"' ")
            return len(s) >= 2 and "\ufffd" not in s[:2]
        return lambda r, ids: ids[-1] in ends or first2(ids)
    W = max(len(norm(it[k]).split()) for k in ("answer", "S", "X", "Z", "D"))
    return lambda r, ids: ids[-1] in ends or len(norm(tok.decode(ids).split("\n")[0]).split()) >= W + 2


@torch.no_grad()
def generate_rows(model, tok, it, d, kv, rows: dict, nL, max_new, extra=None, stop=True):
    """Greedy generation (ckeys.generate.greedy: prompt pass with the clamps, decode steps read the clamped cache) of
    the rows on the B prompt, plus ``extra`` = {name: run} unclamped rows on another run's prompt (e.g. {"S_run": "S"}).
    ``stop``: end a row once its answer class is decided (decided_stop). Per row: ids (up to the first special token,
    cut), text, the entity it names (answer_of) and g1 (the token after w when the generation starts with w)."""
    extra = extra or {}
    names = list(rows) + list(extra)
    tabs = tables(kv, [rows[n] for n in rows] + [rows.get("ID", next(iter(rows.values())))] * len(extra), nL, len(d["P"]))
    ids = torch.tensor([d["ids"]["B"]] * len(rows) + [d["ids"][Y] for Y in extra.values()])
    keep = torch.tensor([True] * len(rows) + [False] * len(extra))
    with clamp_kv(model, d["P"], tabs, range(nL), per_row=keep):
        gens = greedy(model, tok, ids, max_new=max_new, stop=decided_stop(tok, it, d["fmt"]) if stop else None)
    w, ends = d["w"], end_ids(tok)
    out = {}
    for r, n in enumerate(names):
        g = cut(gens[r], ends)
        txt = gen_text(tok, ids[r].tolist(), g)
        out[n] = {"ids": g, "text": txt, "who": answer_of(txt, it, d["fmt"]),
                  "g1": g[len(w)] if len(g) > len(w) and g[:len(w)] == w else None}
    return out


def passage_positions(tok, d, it, run="S"):
    """Token positions of the passage in the prompt of ``run`` (the KIVI positions of J-A7)."""
    ent = {"B": it["answer"], "S": it["S"], "X": it["X"], "Z": it["Z"]}[run]
    p0, p1 = passage_range(d["texts"][run], it, d["fmt"], ent)
    off = tok(d["texts"][run], add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    return span_positions(off, p0, p1)


def letters(fmt):
    return fmt in LETTER_FORMATS
