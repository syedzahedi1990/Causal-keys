"""Stage 9, edit and dose arm (docs/stage9/SPEC.md sections 5.4, 6, 8, 9.1-9.2, 9.7, 11.3, 11.5; predictions
docs/STAGE9_PREDICTIONS.md P5a-c, B1-B3, B-F1, B-F2): edits written at the writing token p of the 80 new
canonical-lexicon cores E9 (ckeys.stage9.edit_cores, cond_E), read in four formats (NONE, AFTER, LETTER, POST).

Per core (prep9: the B, S, X, pi(S), pi(X) prompts of every format, one length per format, a single differing position p,
the same p and an identical prefix through p in every format, so the tables below are the same tensors in all four):
  natural pass   experiments.stage8_edits.natural_pass: the block outputs h_v,l(p) at l in EDIT_BLOCKS = (3, 7, 11, 15) and
                 the K/V at p from block 4 for v in (B, S, X, pi(S), pi(X)); targets t in (S, X)
  edit vectors   (FP32, written at (p, l) by experiments.stage8_edits.edit_pass, forwards of the natural pass's 5-row shape)
                   E1|t       h_B + mu1[l][t] - mu1[l][B]           mu1: results/gpu_stage8c/calib/<key>.pt (pinned sha256)
                   E1x{c}|t   h_B + c (mu1[l][t] - mu1[l][B])       c in (0.5, 0.75, 1.5), dose blocks (3, 7) only
                   E4b|t      fixed_subspace_interchange(U)(h_B, h_pi(t))   U = the stored seed-102 basis (qwen7, mistral7)
                   T|S        h_S,l(p)                               pipeline check only: its tables must equal the
                                                                     natural S tables (max |diff| in T_check, 0 expected)
  table rows     clamp K/V at p in layers >= l + 1 (below, p's K/V are B's own), in batch order (SPEC 8.1):
                   self                       B, B
                   nat|t                      t, t
                   E1|t, E4b|t                the edit's tables
                   E1x{c}|t                   the edit's tables                                     (blocks 3, 7)
                   J{a}|t                     B + a(t - B), B + a(t - B); a in 0.25 .. 1.25          (blocks 3, 7)
                   K1|t, V1|t                 t, B / B, t                                           (blocks 3, 7)
                   K{a}|t, V{a}|t             B + a(t - B), B / B, B + a(t - B); a in (0.5, 0.8)    (blocks 3, 7)
                   K{a}V1|t, K1V{a}|t         B + a(t - B), t / t, B + a(t - B); a in (0.5, 0.8)    (blocks 3, 7)
                   self2                      B, B (an in-batch determinism check, not a BF16 floor: two rows
                                              with the same input and tables in one forward come out bitwise
                                              equal, also in BF16; the cross-batch floor is self against the
                                              CAA batch's self row of the same cell, B in another batch shape
                                              with p's own K/V instead of the prefix pass's tables)
                 Interpolated tables are computed in FP32 (ckeys.edits.interp on FP32 copies; a = 1 is t's table itself)
                 and cast to the model's dtype by the clamp. Every cell's rows are ONE scoring forward led by self (at
                 most --chunk rows), so each row and its self row share a batch.
  CAA rows       all-position steering, not a table row: a forward hook on blocks(model)[l] adds mu1[l][t] - mu1[l][B]
                 (per batch row; zero for the leading self row) to the block output at every sequence index >= p of a
                 forward longer than p (prompt pass, trie pass, cache-free forwards) and at every index of a shorter one
                 (a cached one-token decode step, whose absolute position is past p). At index p alone this is the
                 single-site E1. Batch: self, CAA|S, CAA|X, scored and generated separately from the table rows.
  per cell       (format x block) ckeys.surface.score over prompt + trie under the clamps (or the hook): E = the
                 log-sum over the 32 fixed E forms (+ the stage-8c frames: none discovered), L = log p(" w"), six-vectors
                 in LOCATIONS order (LETTER: letter i names location i); greedy generation (ckeys.generate.greedy, 16
                 tokens, the KV cache on): a row ends at EOS / end of turn, a newline, 16 tokens or the stop rule (word
                 formats: ckeys.fresh.stop_fn over LOCATIONS; LETTER: ckeys.stage9.letter_stop); answers by
                 ckeys.fresh.parse (word formats) or ckeys.stage9.parse_letter (LETTER). Under LETTER every row
                 also stores "word" = ckeys.fresh.parse(text, LOCATIONS), as the stories arm does, so that a letter
                 answer contradicted by a word of the same text ("A drawer" read as option A, the box) can be
                 counted; the answer itself is SPEC 9.2's either way.
  stats          per table row Z and target t at block l: ckeys.edits.kv_stats(Z, t, B, layers >= l + 1) (proj_k = the
                 delivered key ratio r_K, proj_v, cos_k, cos_v, ratio_k, ratio_v, nu) and, for the residual rows (nat,
                 E1, E1x{c}, E4b), cos_resid and ratio_resid of the displacement at (p, l) against h_t - h_B (null for the
                 interpolated rows). CAA's K/V at p are E1's, so its r_K is E1's (no separate record).

Stages (one process each; BF16, sdpa, use_cache=False in every scoring pass; scripts/gpu_stage9.sh):
  check  tokenizer and config only, no weights: the E9 pin; prep9 of every E9 core in the four formats; fresh.raw_prompt
         of every E9 item equal to ckeys.encoding.raw_prompt; the word form sets (with the stage-8c frames) and the
         letter set build and decode back; the calib (and, at qwen7 and mistral7, the das) files on disk equal their
         pins, the calib json records the same sha256, mu1 and the bases have the model's width and the bases
         orthonormal rows -> OUT/check/<tag>_edits.json; exit 1 when a check fails
  eval   every E9 core -> OUT/edits/<tag>.json (SPEC 11.5), resumed by core (an existing file with the same commit,
         revision, calib and das files, population and row layout keeps its cores), rewritten every 5 cores; exit 3
         when the deadline (STAGE8_DEADLINE minus --reserve-min) stops it before the last core
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct (or the --model given), FP32, CPU; n = 2 E9 cores; mu1 =
ckeys.edits.class_means over the first 3 cores of H_fit (ckeys.edits.populations()["H"][:150]); the E4b basis (qwen7,
mistral7) = the transpose of the Q factor of qr(randn(hidden, 16), Generator seed 102); calib_file, das_file "TEST";
outputs tagged TEST_<key>, by default under results/gpu_stage9_test (--out overrides; the real default is
results/gpu_stage9).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from ckeys import edits, fresh, stage9
from ckeys.clamp import clamp_kv
from ckeys.encoding import LETTERS, WRAPPER_USED, candidate_ids, raw_prompt
from ckeys.generate import greedy
from ckeys.interventions import _out_tensor, _with_tensor, blocks, fixed_subspace_interchange, hooks
from ckeys.story import LOCATIONS, PAIR_SWAP, pick_x, record
from ckeys.surface import FormSet, score
from experiments.format_factorial import provenance as ff_provenance
from experiments.ioi_factorial import device_name
from experiments.stage6_heads import write_atomic
from experiments.stage8_edits import NAT_ROWS, edit_pass, natural_pass, split_tables, verified

TINY = "Qwen/Qwen2.5-0.5B-Instruct"
KEYS = ("mistral7", "llama8", "qwen7")
E4B_KEYS = ("qwen7", "mistral7")          # the models with stored E4b bases (stage 8c dasfit)
DAS_SEED = 102                            # E4b: the random-init fits; seed 101 (PCA init, E4a) is not used
RANK = 16
TARGETS = ("S", "X")
MULTS = (0.5, 0.75, 1.5)                  # E1 multipliers (dose blocks)
ALPHAS_J = (0.25, 0.5, 0.65, 0.8, 0.9, 1.25)
ALPHAS_KV = (0.5, 0.8)
MAX_NEW = 16
CHUNK = 64
STAGE8C = "results/gpu_stage8c"
H_FIT_TEST = 3                            # TEST_MODE: mu1 from the first 3 cores of H_fit


def log(s):
    print(s, flush=True)


def r5(x):
    if isinstance(x, (list, tuple)):
        return [r5(v) for v in x]
    return round(float(x), 5)


# --------------------------------------------------------------------------- prompts (SPEC 6)
def prep9(tok, core: dict, values=None, formats=stage9.EDIT_FORMATS) -> dict:
    """ckeys.edits.prep for the stage-9 formats: the ids of every format for each value written at p (default B, S, X,
    pi(S), pi(X)); asserts one length per format, a single differing position p, the same p in every format and an
    identical prefix through p in every format (so an edit's tables are the same tensors in all of them)."""
    assert "NONE" in formats, "the natural pass reads the NONE prefix"
    X = pick_x(core)
    vals = values or {"B": core["base"], "S": core["source"], "X": X, "piS": PAIR_SWAP[core["source"]],
                      "piX": PAIR_SWAP[X]}
    ids = {f: {k: edits.prompt_ids(tok, core, f, v) for k, v in vals.items()} for f in formats}
    p = None
    for f in formats:
        shapes = {v.shape for v in ids[f].values()}
        assert len(shapes) == 1, (f, edits.key(core), "prompts of different lengths", shapes)
        B = ids[f]["B"][0]
        for k, v in ids[f].items():
            if vals[k] == vals["B"]:
                continue
            dpos = (v[0] != B).nonzero().flatten().tolist()
            assert len(dpos) == 1, (f, k, edits.key(core), "differing positions", dpos)
            p = dpos[0] if p is None else p
            assert dpos[0] == p, (f, k, edits.key(core), "another p", dpos, p)
    assert p is not None, "no value differs from B"
    for f in formats:
        for k in vals:
            assert torch.equal(ids[f][k][:, :p + 1], ids["NONE"][k][:, :p + 1]), (f, k, "prefix through p differs")
    return {"core": core, "X": X, "vals": vals, "ids": ids, "p": p, "prefix": ids["NONE"]["B"][:, :p + 1],
            "iB": LOCATIONS.index(vals["B"]), "iS": LOCATIONS.index(core["source"]), "iX": LOCATIONS.index(X),
            "iPiS": LOCATIONS.index(vals["piS"]) if "piS" in vals else None,
            "iPiX": LOCATIONS.index(vals["piX"]) if "piX" in vals else None}


def letter_form_set(tok) -> FormSet:
    return FormSet(tok, LETTERS, {"E": edits.LETTER_FRAMES})


def form_set9(tok, fmt: str, core: dict, frames=()) -> FormSet:
    """SPEC 9.1: the letter forms " X", "X", " **X", "**X" under LETTER, else the 32 fixed E frames plus the frames
    (instantiated with the core's names) over LOCATIONS."""
    return letter_form_set(tok) if fmt == "LETTER" else edits.form_set(tok, fmt, core, frames)


def answer9(text: str, it, fmt: str) -> tuple[str, str]:
    """(location or "other", how in {"word", "letter", "none"})."""
    if fmt == "LETTER":
        return stage9.parse_letter(text, it)
    w = fresh.parse(text, LOCATIONS)
    return w, ("word" if w != "other" else "none")


def gen_fields(text: str, it, fmt: str) -> dict:
    """The answer fields of a generated row: ans and how (SPEC 9.2); under LETTER also "word" = ckeys.fresh.parse(text,
    LOCATIONS) (the first location the text names, or "other"), so that a letter answer contradicted by a word in the
    same text ("A drawer", read as option A = the box) can be counted, as in the stories arm. The answer is SPEC 9.2's
    either way."""
    a, how = answer9(text, it, fmt)
    out = {"ans": a, "how": how}
    if fmt == "LETTER":
        out["word"] = fresh.parse(text, LOCATIONS)
    return out


def stop9(tok, it, fmt: str):
    return stage9.letter_stop(tok, it) if fmt == "LETTER" else fresh.stop_fn(tok, LOCATIONS)


# --------------------------------------------------------------------------- rows (SPEC 8.1)
def row_specs(l: int, e4b: bool) -> list[tuple]:
    """The table rows of a cell at block l in batch order: (name, target or None, K source, V source); a source is
    ("B",), ("T",) (the target's natural table), ("I", a) (B + a(t - B)) or ("E", vector name)."""
    B, T = ("B",), ("T",)
    rows = [("self", None, B, B)]
    kinds = [("nat", T, T), ("E1", "E", "E")] + ([("E4b", "E", "E")] if e4b else [])
    if l in stage9.DOSE_BLOCKS:
        kinds += [(f"E1x{c}", "E", "E") for c in MULTS]
        kinds += [(f"J{a}", ("I", a), ("I", a)) for a in ALPHAS_J]
        kinds += [("K1", T, B), ("V1", B, T)]
        kinds += [(f"K{a}", ("I", a), B) for a in ALPHAS_KV] + [(f"V{a}", B, ("I", a)) for a in ALPHAS_KV]
        kinds += [(f"K{a}V1", ("I", a), T) for a in ALPHAS_KV] + [(f"K1V{a}", T, ("I", a)) for a in ALPHAS_KV]
    for kind, ks, vs in kinds:
        for t in TARGETS:
            name = f"{kind}|{t}"
            rows.append((name, t, ("E", name) if ks == "E" else ks, ("E", name) if vs == "E" else vs))
    rows.append(("self2", None, B, B))
    return rows


def row_names(l: int, e4b: bool) -> list[str]:
    return [r[0] for r in row_specs(l, e4b)]


RESID_KINDS = ("nat", "E1", "E4b") + tuple(f"E1x{c}" for c in MULTS)    # rows written as residual vectors


def vectors(l: int, h: dict, mu1: dict, U, d: dict) -> dict:
    """The edit vectors at block l (FP32 [D] each): E1|t, E1x{c}|t (dose blocks), E4b|t (U given) and T|S; ``h``
    {v: {l: [D]}} the natural residuals of NAT_ROWS, ``U`` {l: [RANK, D]} or None."""
    hB = h["B"][l]
    out = {}
    for t in TARGETS:
        it = d["iS"] if t == "S" else d["iX"]
        delta = mu1[l][it] - mu1[l][d["iB"]]
        out[f"E1|{t}"] = hB + delta
        if l in stage9.DOSE_BLOCKS:
            for c in MULTS:
                out[f"E1x{c}|{t}"] = hB + c * delta
        if U is not None:
            out[f"E4b|{t}"] = fixed_subspace_interchange(U[l])(hB[None], h["pi" + t][l][None])[0]
    out["T|S"] = h["S"][l].clone()
    return out


def chan(src, ch: str, tabB: dict, tabt: dict | None, tabE: dict, layers) -> dict:
    """{q: [D]} FP32 table of one channel from a row's source (SPEC 8.1)."""
    if src[0] == "B":
        return {q: tabB[(q, ch)].float() for q in layers}
    if src[0] == "T":
        return {q: tabt[(q, ch)].float() for q in layers}
    if src[0] == "E":
        return {q: tabE[src[1]][(q, ch)].float() for q in layers}
    a = float(src[1])
    if a == 1.0:
        return {q: tabt[(q, ch)].float() for q in layers}
    fB = {(q, ch): tabB[(q, ch)].float() for q in layers}
    ft = {(q, ch): tabt[(q, ch)].float() for q in layers}
    return edits.interp(fB, ft, a, layers, ch)


def build_rows(specs, tabB: dict, tabT: dict, tabE: dict, layers) -> list[tuple]:
    """[(name, K {q: [D]}, V {q: [D]})] in batch order."""
    out = []
    for name, t, ks, vs in specs:
        tt = tabT[t] if t is not None else None
        out.append((name, chan(ks, "k", tabB, tt, tabE, layers), chan(vs, "v", tabB, tt, tabE, layers)))
    return out


def as_tab(K: dict, V: dict) -> dict:
    return {(q, "k"): K[q] for q in K} | {(q, "v"): V[q] for q in V}


def row_stats(rows, specs, tabB: dict, tabT: dict, layers, vecs: dict, h: dict, l: int) -> dict:
    """SPEC 8.2: kv_stats of every table row with a target against that target's natural tables, plus the residual
    cosine and norm ratio for the rows written as residual vectors (null for the interpolated rows)."""
    fB = {k: v.float() for k, v in tabB.items()}
    out = {}
    for (name, K, V), (_, t, _, _) in zip(rows, specs):
        if t is None:
            continue
        ft = {k: v.float() for k, v in tabT[t].items()}
        s = edits.kv_stats(as_tab(K, V), ft, fB, layers)
        kind = name.split("|")[0]
        if kind in RESID_KINDS:
            hB = h["B"][l]
            dn = h[t][l] - hB
            dz = (h[t][l] if kind == "nat" else vecs[name]) - hB
            s |= {"cos_resid": edits.cos(dz, dn), "ratio_resid": float(dz.norm() / dn.norm().clamp_min(1e-30))}
        else:
            s |= {"cos_resid": None, "ratio_resid": None}
        out[name] = {k: (None if v is None else r5(v)) for k, v in s.items()}
    return out


# --------------------------------------------------------------------------- the all-position CAA hook (SPEC 8.1)
@contextlib.contextmanager
def caa_hook(model, layer: int, p: int, vecs: torch.Tensor, only_p: bool = False):
    """Add ``vecs[r]`` (FP32 [R, D], one per batch row) to the output of block ``layer``: at every sequence index >= p
    of a forward longer than p, and at every index of a shorter one (a cached one-token decode step, whose absolute
    position is past p). The sum is formed in FP32 and cast once to the stream's dtype, so at index p it is the
    single-site edit h + v. ``only_p``: index p only, and only in forwards longer than p (the single-site check)."""
    def fn(_m, _i, out):
        h = _out_tensor(out)
        assert h.shape[0] == vecs.shape[0], (h.shape, vecs.shape)
        if h.shape[1] > p:
            sl = slice(p, p + 1) if only_p else slice(p, None)
        elif only_p:
            return out
        else:
            sl = slice(0, None)
        v = vecs.to(h.device, torch.float32)[:, None, :]
        new = h.clone()
        new[:, sl] = (h[:, sl].float() + v).to(h.dtype)
        return _with_tensor(out, new)
    with hooks([blocks(model)[layer].register_forward_hook(fn)]):
        yield


# --------------------------------------------------------------------------- one batch: scores and generation
def six(res: dict, name: str, fs: FormSet, i: int) -> list[float]:
    """Six scores of batch row ``i`` in LOCATIONS order (LETTER: letter j names location j)."""
    return [float(res[name][w][i]) for w in fs.words]


@torch.no_grad()
def run_batch(model, tok, ids: torch.Tensor, fs: FormSet, names: list, it, fmt: str, ctx, gen: bool = True,
              max_new: int = MAX_NEW) -> dict:
    """Score (one forward over prompt + trie) and generate the batch rows ``names`` on the prompt ``ids`` [1, T]
    inside the context ``ctx()`` (the clamp or the CAA hook). Returns {name: {"E", "L", "gen", "ans", "how", "n_tok"}}
    (LETTER: also "word", gen_fields)."""
    dev = next(model.parameters()).device
    R = len(names)
    x = ids.expand(R, -1).to(dev)
    with ctx():
        res = score(model, x, fs)
    out = {n: {"E": r5(six(res, "E", fs, i)), "L": r5(six(res, "L", fs, i))} for i, n in enumerate(names)}
    if gen:
        with ctx():
            g = greedy(model, tok, x, max_new=max_new, stop=stop9(tok, it, fmt))
        prompt = ids[0].tolist()
        for i, n in enumerate(names):
            text = fresh.gen_text(tok, prompt, g[i])
            out[n].update(gen=text, **gen_fields(text, it, fmt), n_tok=len(g[i]))
    return out


def table_ctx(model, p: int, rows: list, layers):
    tabs = {}
    for q in layers:
        tabs[(q, "k")] = torch.stack([r[1][q] for r in rows])[:, None]
        tabs[(q, "v")] = torch.stack([r[2][q] for r in rows])[:, None]
    return lambda: clamp_kv(model, [p], tabs, layers)


# --------------------------------------------------------------------------- one core
@torch.no_grad()
def eval_core(model, tok, index: int, core: dict, it, mu1: dict, U, frames, nL: int, blocks_=stage9.EDIT_BLOCKS,
              formats=stage9.EDIT_FORMATS, gen: bool = True, max_new: int = MAX_NEW, chunk: int = CHUNK,
              d: dict | None = None, keep=None) -> dict:
    """The record of SPEC 11.5 for one E9 core. ``keep`` (tests): a dict that receives the per-block tables and vectors."""
    d = d or prep9(tok, core)
    p = d["p"]
    res, kv = natural_pass(model, d, list(blocks_), nL)
    h = {k: {l: res[l][i] for l in blocks_} for i, k in enumerate(NAT_ROWS)}
    fs = {f: form_set9(tok, f, core, frames) for f in formats}
    rec = {"index": index, "core": core, "X": d["X"], "p": p, "iB": d["iB"], "iS": d["iS"], "iX": d["iX"],
           "iPiS": d["iPiS"], "iPiX": d["iPiX"], "cluster": [core["base"], core["source"]], "T_check": {},
           "stats": {}, "cells": {}}
    for l in blocks_:
        layers = list(range(l + 1, nL))
        vecs = vectors(l, h, mu1, U, d)
        vn = list(vecs)
        kvE = edit_pass(model, d, l, torch.stack([vecs[n] for n in vn]), layers)
        tabE = {n: split_tables(kvE, i, layers) for i, n in enumerate(vn)}
        tabB = split_tables(kv, 0, layers)
        tabT = {"S": split_tables(kv, 1, layers), "X": split_tables(kv, 2, layers)}
        rec["T_check"][str(l)] = max(float((tabE["T|S"][k].float() - tabT["S"][k].float()).abs().max()) for k in tabT["S"])
        specs = row_specs(l, U is not None)
        rows = build_rows(specs, tabB, tabT, tabE, layers)
        assert len(rows) <= chunk, f"{len(rows)} table rows at block {l} exceed the chunk {chunk}"
        rec["stats"][str(l)] = row_stats(rows, specs, tabB, tabT, layers, vecs, h, l)
        names = [r[0] for r in rows]
        delta = torch.zeros(3, mu1[l].shape[-1])
        for j, t in enumerate(TARGETS):
            delta[1 + j] = mu1[l][d["iS"] if t == "S" else d["iX"]] - mu1[l][d["iB"]]
        tctx = table_ctx(model, p, rows, layers)
        cctx = lambda: caa_hook(model, l, p, delta)  # noqa: E731
        for f in formats:
            ids = d["ids"][f]["B"]
            cell = run_batch(model, tok, ids, fs[f], names, it, f, tctx, gen, max_new)
            cell["caa"] = run_batch(model, tok, ids, fs[f], ["self", "CAA|S", "CAA|X"], it, f, cctx, gen, max_new)
            rec["cells"][f"{f}@{l}"] = cell
        if keep is not None:
            keep[l] = {"vecs": vecs, "tabE": tabE, "tabB": tabB, "tabT": tabT, "rows": rows, "delta": delta, "h": h}
    return rec


# --------------------------------------------------------------------------- stored calibration means and bases
def resolve(a, rel: str) -> Path:
    """The file of a committed stage-8c artefact under --stage8c (default the repository's results/gpu_stage8c)."""
    base = Path(a.stage8c)
    base = base if base.is_absolute() else stage9.ROOT / base
    return base / rel


def load_calib(a, hidden: int | None = None) -> dict:
    """mu1 {l: [6, D] FP32} at EDIT_BLOCKS from results/gpu_stage8c/calib/<key>.pt (its sha256 asserted equal to the
    pin before the load, and equal to the one calib/<key>.json records) and the json's discovered frames."""
    rel = f"{STAGE8C}/calib/{a.key}.pt"
    pin = stage9.ARTEFACT_SHA256[rel]
    f = resolve(a, f"calib/{a.key}.pt")
    h = stage9.file_sha(f)
    assert h == pin, f"{f}: sha256 {h} differs from the pin {pin}"
    J = json.loads(f.with_suffix(".json").read_text())
    assert J["calib_pt_sha256"] == pin, f"{f.with_suffix('.json')}: calib_pt_sha256 {J['calib_pt_sha256']} != the pin"
    T = torch.load(f, weights_only=False)
    mu1 = {l: T["mu1"][l].float() for l in stage9.EDIT_BLOCKS}
    for l, m in mu1.items():
        assert m.shape[0] == 6 and (hidden is None or m.shape == (6, hidden)), (l, tuple(m.shape), hidden)
    return {"mu1": mu1, "frames": list(J["frames"]), "calib_file": rel, "calib_sha256": h}


def load_das(path, pin: str, blocks_=stage9.EDIT_BLOCKS, hidden: int | None = None) -> dict:
    """The stored E4b bases {l: U [RANK, D] FP32} (seed DAS_SEED) of a stage-8c das file: the file's sha256 is
    asserted equal to ``pin`` before it is loaded; every U has RANK orthonormal rows (and width ``hidden``)."""
    h = stage9.file_sha(path)
    assert h == pin, f"{path}: sha256 {h} differs from the pin {pin}"
    D = torch.load(path, weights_only=False)
    out = {}
    for l in blocks_:
        U = D[(l, DAS_SEED)].float()
        assert U.dim() == 2 and U.shape[0] == RANK and (hidden is None or U.shape[1] == hidden), (l, tuple(U.shape), hidden)
        assert float((U @ U.T - torch.eye(RANK)).abs().max()) < 1e-4, f"basis ({l}, {DAS_SEED}) rows not orthonormal"
        out[l] = U
    return out


def das_of(a, hidden: int | None = None) -> tuple[dict | None, dict]:
    """(bases or None, provenance fields) of the key (outside TEST_MODE)."""
    if a.key not in E4B_KEYS:
        return None, {"das_file": None, "das_sha256": None, "das_seed": DAS_SEED}
    rel = f"{STAGE8C}/das/{a.key}.pt"
    pin = stage9.ARTEFACT_SHA256[rel]
    f = resolve(a, f"das/{a.key}.pt")
    U = load_das(f, pin, hidden=hidden)
    j = f.with_suffix(".json")
    if j.exists():
        rec = json.loads(j.read_text()).get("das_pt_sha256")
        assert rec == pin, f"{j}: das_pt_sha256 {rec} != the pin"
    return U, {"das_file": rel, "das_sha256": pin, "das_seed": DAS_SEED}


def test_basis(hidden: int) -> torch.Tensor:
    """TEST_MODE's E4b stand-in: U [RANK, hidden] with orthonormal rows (Q of a seeded Gaussian, transposed)."""
    g = torch.Generator().manual_seed(DAS_SEED)
    return torch.linalg.qr(torch.randn(hidden, RANK, generator=g))[0].T.contiguous()


def test_means(model, tok) -> dict:
    """TEST_MODE's mu1 stand-in: ckeys.edits.class_means over the first H_FIT_TEST cores of H_fit."""
    H = edits.populations()["H"][:edits.N_HFIT]
    return edits.class_means(model, tok, H[:H_FIT_TEST], stage9.EDIT_BLOCKS)


def stand_ins(a, model, tok) -> tuple[dict, dict | None, list, dict]:
    """(mu1, bases or None, frames, provenance fields) for the run: the pinned stage-8c files, or the TEST stand-ins."""
    hidden = model.config.hidden_size
    if a.test:
        mu1 = test_means(model, tok)
        U = {l: test_basis(hidden) for l in stage9.EDIT_BLOCKS} if a.key in E4B_KEYS else None
        return mu1, U, [], {"calib_file": "TEST", "calib_sha256": None, "das_file": "TEST" if U else None,
                            "das_sha256": None, "das_seed": DAS_SEED, "frames": []}
    c = load_calib(a, hidden)
    U, dp = das_of(a, hidden)
    return c["mu1"], U, c["frames"], {"calib_file": c["calib_file"], "calib_sha256": c["calib_sha256"], **dp,
                                     "frames": c["frames"]}


# --------------------------------------------------------------------------- model, provenance, files
def load_model(a, need_model=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if not need_model:
        return AutoConfig.from_pretrained(a.model, revision=a.revision), tok, dev
    kw = dict(dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation=a.attn)
    if dev == "cuda":
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    assert model.config._attn_implementation == a.attn
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model, tok, dev


def prov(a, model=None, dev=None, extra=None) -> dict:
    ver = verified(a.model) if a.model and Path(a.model).is_dir() else None
    p = ff_provenance(a) | stage9.run_meta(a.test) | {
        "stage9": True, "arm": "edits", "stage": a.stage, "model": a.model, "model_key": a.key,
        "revision": a.revision or (ver or {}).get("revision"), "verified": ver, "test_mode": a.test,
        "formats": list(a.formats), "formats_not_run": [f for f in stage9.EDIT_FORMATS if f not in a.formats],
        "blocks": list(a.blocks), "dose_blocks": [l for l in stage9.DOSE_BLOCKS if l in a.blocks],
        "onsets": [l + 1 for l in a.blocks], "kv_only_onsets": [l + 1 for l in stage9.DOSE_BLOCKS if l in a.blocks],
        "max_new": a.max_new, "reserve_min": a.reserve_min, "chunk": a.chunk, "timings": {},
        "population": "E9", "population_sha256": stage9.POP_SHA256["E9"], "skipped_items": [],
        "multipliers": list(MULTS), "alphas_joint": list(ALPHAS_J), "alphas_kv": list(ALPHAS_KV)}
    if model is not None:
        cfg = model.config
        p |= {"dtype": str(next(model.parameters()).dtype), "attn_implementation": cfg._attn_implementation,
              "device": device_name(dev), "n_layers": len(blocks(model)), "hidden": cfg.hidden_size,
              "n_heads": cfg.num_attention_heads, "n_kv_heads": cfg.num_key_value_heads}
    p["wrapper"] = dict(WRAPPER_USED)
    return p | (extra or {})


def cores_of(a) -> list[tuple[dict, object]]:
    """[(core, item)] of E9 (pin asserted), the first --n."""
    cores, items = stage9.edit_cores(), stage9.edit_items()
    pairs = list(zip(cores, items))
    return pairs[:a.n] if a.n else pairs


def time_left(a) -> float:
    """Seconds before STAGE8_DEADLINE minus the reserve of the later steps (inf without a deadline)."""
    dl = os.environ.get("STAGE8_DEADLINE")
    return math.inf if not dl else float(dl) - time.time() - 60.0 * a.reserve_min


def out_file(a, kind) -> Path:
    o = Path(a.out)
    return {"check": o / "check" / f"{a.tag}_edits.json", "eval": o / "edits" / f"{a.tag}.json"}[kind]


# --------------------------------------------------------------------------- stage check (tokenizer and config only)
def stage_check(a):
    cfg, tok, _ = load_model(a, need_model=False)
    t0, fails, skipped, info = time.time(), [], {}, {}
    hidden, nL = cfg.hidden_size, cfg.num_hidden_layers
    if max(a.blocks) + 1 >= nL:
        fails.append(f"block {max(a.blocks)} leaves no layer above it in {nL} blocks")
    files = {"calib_file": "TEST" if a.test else None, "calib_sha256": None, "das_file": None, "das_sha256": None,
             "das_seed": DAS_SEED, "frames": []}
    if a.test:
        files["das_file"] = "TEST" if a.key in E4B_KEYS else None
    else:
        try:
            c = load_calib(a, hidden)
            files |= {"calib_file": c["calib_file"], "calib_sha256": c["calib_sha256"], "frames": c["frames"]}
        except (AssertionError, OSError, KeyError, ValueError) as ex:
            fails.append(f"calib: {ex}")
        try:
            _, dp = das_of(a, hidden)
            files |= dp
        except (AssertionError, OSError, KeyError, ValueError) as ex:
            fails.append(f"das: {ex}")
    try:
        pairs = cores_of(a)
    except AssertionError as ex:
        pairs = []
        fails.append(f"population E9: {ex}")
    try:
        candidate_ids(tok, "AFTER")
    except AssertionError as ex:
        fails.append(f"locations: {ex}")
    ps, nodes, bad_decode = [], {}, []
    try:
        lfs = letter_form_set(tok)
        bad_decode += [("LETTER", x) for x in lfs.check_decode()]
        nodes["LETTER"] = [len(lfs), len(lfs)]
    except ValueError as ex:
        fails.append(f"letter FormSet: {ex}")
    raw_bad = []
    for core, it in pairs:
        try:
            d = prep9(tok, core)
            ps.append(d["p"])
        except AssertionError as ex:
            skipped.setdefault("E9/prep9", []).append([it.key, repr(ex)[:300]])
            continue
        for f in stage9.EDIT_FORMATS:
            for v in d["vals"].values():
                rec = record(core, "direct", v)
                if fresh.raw_prompt(it, f, v) != raw_prompt(f, rec["story"], rec["query"]):
                    raw_bad.append((it.key, f, v))
            if f == "LETTER":
                continue
            try:
                fs = form_set9(tok, f, core, files["frames"])
                n = nodes.setdefault(f, [len(fs), len(fs)])
                nodes[f] = [min(n[0], len(fs)), max(n[1], len(fs))]
                bad_decode += [(it.key, f, x) for x in fs.check_decode()]
            except ValueError as ex:
                skipped.setdefault(f"E9/{f}", []).append([it.key, f"form set: {ex}"])
    if raw_bad:
        fails.append(f"{len(raw_bad)} E9 prompts differ from ckeys.encoding.raw_prompt: {raw_bad[:3]}")
    if bad_decode:
        fails.append(f"{len(bad_decode)} forms do not decode back: {bad_decode[:5]}")
    if skipped:
        fails.append(f"skipped items: { {k: len(v) for k, v in skipped.items()} }")
    info |= {"p_positions": sorted(set(ps)), "trie_nodes": nodes, "n_cores": len(pairs),
             "rows": {str(l): row_names(l, files["das_file"] is not None) for l in a.blocks}}
    P = prov(a, None, None, {**files, "n_items": len(pairs), "hidden": hidden, "n_layers": nL,
                             "skipped_items": [[k, *x] for k, v in skipped.items() for x in v],
                             "seconds": round(time.time() - t0, 1)})
    rep = {"provenance": P, "pass": not fails, "fails": fails, "skipped": skipped, **info}
    write_atomic(rep, out_file(a, "check"))
    log(f"check {'PASS' if not fails else 'FAIL'}: {len(fails)} failures {fails[:3]}; p {info['p_positions']}; "
        f"trie nodes {nodes}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------- stage eval
@torch.no_grad()
def stage_eval(a):
    model, tok, dev = load_model(a)
    nL = len(blocks(model))
    assert max(a.blocks) + 1 < nL, f"block {max(a.blocks)} leaves no layer above it in {nL} blocks"
    t0 = time.time()
    mu1, U, frames, files = stand_ins(a, model, tok)
    pairs = cores_of(a)
    rows = {str(l): row_names(l, U is not None) for l in a.blocks}
    P = prov(a, model, dev, {**files, "n_items": len(pairs), "e4b": U is not None, "rows": rows})
    P["timings"]["setup"] = round(time.time() - t0, 1)
    log(f"calib {files['calib_file']} ({files['calib_sha256']}), das {files['das_file']} ({files['das_sha256']}), "
        f"frames {frames}; {len(pairs)} cores, blocks {a.blocks}, formats {a.formats}")
    path = out_file(a, "eval")
    recs = []
    if path.exists():
        old = json.loads(path.read_text())
        op = old["provenance"]
        same = ("git_commit", "revision", "verified", "hidden", "calib_sha256", "das_sha256", "population_sha256",
                "n_items", "n_layers", "formats", "blocks", "max_new", "test_mode") + (("model",) if a.test else ())
        if all(op.get(k) == P.get(k) for k in same) and old.get("rows") == rows:
            recs = old["cores"]
            P["skipped_items"] = op.get("skipped_items", [])
            P["timings"]["eval_before"] = op.get("timings", {}).get("eval")
            log(f"resuming {path}: {len(recs)} cores kept")
        else:
            log(f"{path} exists from another commit, revision, calibration, population or layout: rerun from the start")
    done = {r["index"] for r in recs} | {s["index"] for s in P["skipped_items"]}

    def write():
        recs.sort(key=lambda r: r["index"])
        P["n_done"] = len(recs)
        P["cores_not_run"] = [i for i, _ in enumerate(pairs) if i not in {r["index"] for r in recs}
                              and i not in {s["index"] for s in P["skipped_items"]}]
        P["finished_utc"] = stage9.run_meta(a.test)["finished_utc"]
        write_atomic({"provenance": P, "population": "E9", "rows": rows, "cores": recs}, path)

    per, status, t1, new = None, 0, time.time(), 0
    for i, (core, it) in enumerate(pairs):
        if i in done:
            continue
        left = time_left(a)
        if left <= 0 or (per is not None and left < 1.2 * per):
            status = 3
            log(f"deadline (minus {a.reserve_min} min reserve): {len(recs)} of {len(pairs)} cores done; stopping (exit 3)")
            break
        ts = time.time()
        try:
            d = prep9(tok, core)
        except AssertionError as ex:
            P["skipped_items"].append({"index": i, "key": it.key, "reason": repr(ex)[:300]})
            continue
        recs.append(eval_core(model, tok, i, core, it, mu1, U, frames, nL, a.blocks, a.formats, True, a.max_new,
                              a.chunk, d=d))
        dt = time.time() - ts
        per = dt if per is None else 0.8 * per + 0.2 * dt
        new += 1
        P["timings"]["eval"] = round(time.time() - t1, 1)
        if new % 5 == 0:
            write()
        log(f"core {i} ({it.key}): {dt:.1f}s; T_check {recs[-1]['T_check']} ({len(recs)}/{len(pairs)})")
    P["timings"]["eval"] = round(time.time() - t1, 1)
    write()
    log(f"wrote {path}: {len(recs)} cores; not run {P['cores_not_run']}")
    return status


# --------------------------------------------------------------------------- main
def args_of(argv=None) -> argparse.Namespace:
    """The parsed and normalised arguments (TEST_MODE from --test or the environment; --out defaults to
    results/gpu_stage9, in TEST_MODE to results/gpu_stage9_test, so a manual TEST run never writes into the real results)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "eval"])
    ap.add_argument("--model", default=None, help="a local verified directory (s8_fetch) or a Hub id")
    ap.add_argument("--key", required=True, choices=KEYS, help="the model key of scripts/stage8_models.json")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--n", type=int, default=0, help="E9 cores (0 = all 80)")
    ap.add_argument("--formats", default=",".join(stage9.EDIT_FORMATS))
    ap.add_argument("--blocks", default=",".join(map(str, stage9.EDIT_BLOCKS)))
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--chunk", type=int, default=CHUNK, help="the largest table-row batch allowed (one forward per cell)")
    ap.add_argument("--stage8c", default=STAGE8C, help="the committed stage-8c results (calib and das files)")
    ap.add_argument("--reserve-min", type=float, default=0.0, help="minutes the pipeline needs after this step")
    ap.add_argument("--out", default=None, help="default results/gpu_stage9 (TEST_MODE: results/gpu_stage9_test)")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, n = 2")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    a.out = a.out or str(stage9.ROOT / ("results/gpu_stage9_test" if a.test else "results/gpu_stage9"))
    if a.test:   # the pipeline passes the 0.5B hub id (s8_fetch in TEST_MODE); the tests pass a tiny random model
        a.model, a.revision, a.dtype = a.model or TINY, None, "float32"
        a.n = a.n or 2
    assert a.model, "--model is required"
    a.formats = [f for f in a.formats.split(",") if f]
    assert a.formats and all(f in stage9.EDIT_FORMATS for f in a.formats), a.formats
    a.formats = [f for f in stage9.EDIT_FORMATS if f in a.formats]
    a.blocks = [int(x) for x in a.blocks.split(",") if x]
    assert a.blocks and all(l in stage9.EDIT_BLOCKS for l in a.blocks), a.blocks
    a.blocks = [l for l in stage9.EDIT_BLOCKS if l in a.blocks]
    a.tag = ("TEST_" if a.test else "") + a.key
    return a


def main(argv=None):
    a = args_of(argv)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    return {"check": stage_check, "eval": stage_eval}[a.stage](a) or 0


if __name__ == "__main__":
    sys.exit(main())   # 3: complete up to the deadline, cores not run; 1: the check failed
