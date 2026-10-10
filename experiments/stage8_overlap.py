"""The overlap screen of P-2026-10-10-J part C (J-C-SCREEN, J-C-WIN; docs/PREREGISTRATION.md) on Prakash et al.'s (2026)
material, at Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Reuses experiments/prakash_swap.py and ckeys/causaltom.py by
import (their stories, raw wrapper, seed-10 pool, LM filter, BIND sweep, exchange rows and natural clamp).

Steps of one process (one model):
  1. preflight (prakash_swap.preflight: pool hash, single-token entities, lengths and positions in every format), their LM
     filter on the 320 pool pairs; the population is the first 150 passing pairs (every passing pair if fewer).
  2. the BIND sweep under NO-MENTION over every block l (prakash_swap.sweep): IIA(l), Phi(l) = mean[m(M_l) - m(B)].
  3. W_B = {l : Phi(l) >= 3 nats and IIA(l) >= 0.5}. The natural clamp under OPTIONS-AFTER at the onsets of the fixed grid
     round(x nL / 28) for x in {0, 3, ..., 27} and at l + 1 for every l in W_B (prakash_swap.run_pair, clamp rows only):
     s_ID(OPTIONS-AFTER, l0) = mean ID_K / (mean ID_K + mean ID_V) (H's definition; point estimate here, the scorer
     recomputes it with intervals).
  4. The window W = {l in W_B : s_ID(OPTIONS-AFTER, l + 1) >= 0.4}; l_w = min W. If W is not empty, H's exchange for BIND
     at l_w and the natural clamp at l_w + 1 in NO-MENTION, QNAMES and OPTIONS-AFTER (J-C-WIN) -> window.json.
Outputs under {out}/overlap/<tag>/: screen.json, window.json (only with a window). TEST_MODE: Qwen2.5-0.5B FP32 on the CPU,
no LM filter, n = 2, sweep over blocks 0, 4, ..., 20, 23; NO-MENTION and OPTIONS-AFTER only for the window.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys import causaltom as ct
from ckeys.interventions import blocks
from experiments import prakash_swap as ps
from experiments.stage6_heads import write_atomic
from experiments.stage8_edits import verified

GRID28 = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
WIN_PHI, WIN_IIA, WIN_SID = 3.0, 0.5, 0.4
FORMATS = ("NO-MENTION", "QNAMES", "OPTIONS-AFTER")


def log(s):
    print(s, flush=True)


def s_id_point(cells, l0):
    """H's s_ID from run_pair clamp cells at onset l0 (point estimate)."""
    def D(c, row, t):
        return c["lp"][row][t] - c["lp"]["ID"][t]

    def idx(c, ch):
        return 0.5 * ((D(c, f"{ch}_S", "S") - D(c, f"{ch}_X", "S")) + (D(c, f"{ch}_X", "X") - D(c, f"{ch}_S", "X")))
    cs = [c for c in cells if c["l0"] == l0]
    k = np.mean([idx(c, "K") for c in cs])
    v = np.mean([idx(c, "V") for c in cs])
    return float(k / (k + v)) if k + v > 0 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="the verified local directory (s8_fetch) or a Hub id")
    ap.add_argument("--key", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prakash-repo", default=None)
    ap.add_argument("--out", default="results/gpu_stage8c")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    formats = FORMATS
    if a.test:
        a.model, a.dtype, a.n = "Qwen/Qwen2.5-0.5B-Instruct", "float32", min(a.n, 2)
        formats = ("NO-MENTION", "OPTIONS-AFTER")
    tag = ("TEST_" if a.test else "") + a.key
    out = Path(a.out) / "overlap" / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    rel = ct.load(a.prakash_repo)
    pairs = ct.pool(rel)
    tok = AutoTokenizer.from_pretrained(a.model)
    pf = ps.preflight(tok, rel, pairs, sorted(set(ct.FORMATS) | set(ct.EXTRA_FORMATS)))
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    kw = dict(dtype=getattr(torch, a.dtype), attn_implementation="sdpa")
    if dev == "cuda":
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    nL = len(blocks(model))
    prov = ps.provenance(a, model) | {"test_mode": a.test, "model_key": a.key, "preflight": pf, "verified": verified(a.model)}
    t0 = time.time()
    if a.test:
        fl, pop = [], list(range(a.n))
    else:
        fl = ps.lm_filter(model, tok, pairs)
        pop = [f["i"] for f in fl if f["ok"]][:a.n]
    log(f"population: {len(pop)} pairs ({time.time() - t0:.0f}s)")
    layers = [0, 4, 8, 12, 16, 20, nL - 1] if a.test else list(range(nL))
    items = [ps.item(tok, rel, pairs[i], "NO-MENTION", i) for i in pop]
    rows, summ = ps.sweep(model, tok, items, "BIND", layers, a.batch)
    WB = [l for l in layers if summ[l]["Phi"] >= WIN_PHI and summ[l]["IIA"] >= WIN_IIA]
    grid = sorted({min(nL - 1, round(x * nL / 28)) for x in GRID28})
    onsets = sorted(set(grid) | {l + 1 for l in WB if l + 1 < nL})
    cells, logs = [], []
    logf = lambda s: (logs.append(s), None)  # noqa: E731
    for i in pop:
        L = ps.item(tok, rel, pairs[i], "OPTIONS-AFTER", i)
        _, cl = ps.run_pair(model, tok, L, [], onsets, False, True, nL, a.test, logf)
        cells += [c | {"format": "OPTIONS-AFTER"} for c in cl]
    sid = {l0: s_id_point(cells, l0) for l0 in onsets}
    W = [l for l in WB if l + 1 < nL and sid.get(l + 1, float("nan")) >= WIN_SID]
    lw = min(W) if W else None
    body = {"population": pop, "filter": fl, "n_pass": None if a.test else sum(f["ok"] for f in fl), "layers": layers,
            "sweep_summary": summ, "sweep_rows": rows, "W_B": WB, "grid": grid, "onsets": onsets, "clamp_cells": cells,
            "s_id_point": sid, "window": W, "l_w": lw, "rule": {"phi": WIN_PHI, "iia": WIN_IIA, "s_id": WIN_SID}}
    prov["timings"] = {"screen": round(time.time() - t0, 1)}
    write_atomic({"provenance": prov} | body, out / "screen.json")
    log(f"BIND W_B {WB}; s_ID(OPTIONS-AFTER) {({k: round(v, 3) for k, v in sid.items()})}; window {W}, l_w {lw}")
    if lw is None and a.test:   # TEST_MODE: exercise the window path at the block with the largest Phi (labelled, never scored)
        lw = max(layers[:-1], key=lambda l: summ[l]["Phi"])
        log(f"TEST_MODE: no window; the window exchange runs at block {lw} to exercise the code")
    if lw is None:
        return
    ex, cl = [], []
    for f in formats:
        for i in pop:
            L = ps.item(tok, rel, pairs[i], f, i)
            e, c = ps.run_pair(model, tok, L, [("BIND", lw)], [lw + 1], True, True, nL, a.test, logf)
            ex += [x | {"format": f} for x in e]
            cl += [x | {"format": f} for x in c]
        log(f"  window exchange {f}: {len(pop)} pairs ({time.time() - t0:.0f}s)")
    prov["timings"]["window"] = round(time.time() - t0, 1)
    write_atomic({"provenance": prov, "population": pop, "l_w": lw, "formats": list(formats), "rows": list(ps.ROWS),
                  "cells": ex, "clamp_cells": cl, "b0_log": logs, "test_forced": lw not in W}, out / "window.json")


if __name__ == "__main__":
    main()
