"""Exploratory attention probe of part (e): name-mover and duplicate-token heads in GPT-2 small, PLAIN vs INLINE.

A separate eager-attention FP32 instance (sdpa returns no weights). Per head, mean over cores of the attention
weight: name movers (Wang et al. 9.9, 9.6, 10.0) from END (the answer position) to the IO mention p and, in INLINE,
to the listed IO_B token; duplicate-token heads (0.1, 3.0, 0.10) from the listed IO_B token to p, from the listed
IO_S token to p, and from the second subject mention S2 to the first S1 (PLAIN); and under C_K(S) (the source run's
key at p clamped in every layer with the value at p held at the base run's, ckeys.clamp: the factorial's K_S@0 row)
the same weights from the listed IO_B and IO_S tokens to p and from END to p. Every [layer, head] matrix is saved so
any model can be probed; the labels are GPT-2 small's. Cores whose runs fail ckeys.ioi.encode_runs are skipped and
counted (provenance skipped_items).
Output: <out>/<model>.json ({"provenance", "n", "heads", "mats": {quantity: [L][H]}, "summary"}) and <model>_summary.txt.
TEST_MODE (--test or TEST_MODE=1): gpt2, FP32, CPU, n = 2 (--model and --n are honoured under TEST_MODE=1).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv
from ckeys.interventions import blocks
from ckeys.ioi import check_occurrences, encode_runs, make_cores, name_ids
from experiments.ioi_factorial import provenance

HEADS = {"name_mover": [(9, 9), (9, 6), (10, 0)], "duplicate": [(0, 1), (3, 0), (0, 10)]}
QUANT = ("PLAIN END->p", "PLAIN S2->S1", "INLINE END->p", "INLINE END->IO_B(list)", "INLINE IO_B(list)->p", "INLINE IO_S(list)->p",
         "INLINE K_S: END->p", "INLINE K_S: IO_B(list)->p", "INLINE K_S: IO_S(list)->p")


def where(ids, tid, k):
    return (ids[0] == tid).nonzero().flatten().tolist()[k]


@torch.no_grad()
def run(model, tok, cores, log=print):
    nL = len(blocks(model))
    nH = model.config.num_attention_heads if hasattr(model.config, "num_attention_heads") else model.config.n_head
    acc, n, skipped, t0 = {q: np.zeros((nL, nH)) for q in QUANT}, 0, 0, time.time()
    for core in cores:
        runs = {arm: encode_runs(tok, core, arm, False) for arm in ("PLAIN", "INLINE")}
        if any(v is None for v in runs.values()):
            skipped += 1
            log(f"  skipped core {core} ({[a for a, v in runs.items() if v is None]}: the runs differ in length or at more than one position)")
            continue
        for arm, ids in runs.items():
            check_occurrences(tok, core, arm, ids["B"])
        tid = name_ids(tok, core)
        att = {}
        for arm, ids in runs.items():
            p = (ids["B"][0] != ids["S"][0]).nonzero().item()
            out = model(ids["B"], use_cache=False, output_attentions=True)
            A = torch.stack([a[0] for a in out.attentions])  # [L, H, T, T]
            T = ids["B"].shape[1]
            if arm == "PLAIN":
                s1, s2 = where(ids["B"], tid["Subj"], 0), where(ids["B"], tid["Subj"], 1)
                att["PLAIN END->p"], att["PLAIN S2->S1"] = A[:, :, T - 1, p], A[:, :, s2, s1]
            else:
                ib, is_ = where(ids["B"], tid["B"], 1), where(ids["B"], tid["S"], 0)
                assert ib > p and is_ > p
                att["INLINE END->p"], att["INLINE END->IO_B(list)"] = A[:, :, T - 1, p], A[:, :, T - 1, ib]
                att["INLINE IO_B(list)->p"], att["INLINE IO_S(list)->p"] = A[:, :, ib, p], A[:, :, is_, p]
                with capture_kv(model, [p], range(nL)) as KS:
                    model(ids["S"], use_cache=False)
                with capture_kv(model, [p], range(nL)) as KB:
                    model(ids["B"], use_cache=False)
                tabs = {(l, "k"): KS[l, "k"][0] for l in range(nL)} | {(l, "v"): KB[l, "v"][0] for l in range(nL)}
                with clamp_kv(model, [p], tabs, range(nL)):
                    A = torch.stack([a[0] for a in model(ids["B"], use_cache=False, output_attentions=True).attentions])
                att["INLINE K_S: END->p"], att["INLINE K_S: IO_B(list)->p"], att["INLINE K_S: IO_S(list)->p"] = A[:, :, T - 1, p], A[:, :, ib, p], A[:, :, is_, p]
        for q in QUANT:
            acc[q] += att[q].numpy()
        n += 1
    log(f"  probe done n={n} skipped={skipped} ({time.time() - t0:.0f}s)")
    return {q: (v / max(n, 1)).tolist() for q, v in acc.items()}, n, skipped


def summarize(mats, n, heads=HEADS):
    lines = [f"n={n}; mean attention weight per head (labels are Wang et al.'s GPT-2 small heads)"]
    for q, M in mats.items():
        M = np.array(M)
        top = sorted(((M[l, h], l, h) for l in range(M.shape[0]) for h in range(M.shape[1])), reverse=True)[:5]
        lab = "  ".join(f"{g[:4]} {l}.{h} {M[l, h]:.3f}" for g, hs in heads.items() for l, h in hs if l < M.shape[0] and h < M.shape[1])
        lines.append(f"  {q:28s} {lab}   top-5: " + " ".join(f"{l}.{h}={v:.3f}" for v, l, h in top))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="results/gpu_stage5/ioi_attention")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--test", action="store_true", help="TEST_MODE: gpt2, FP32, CPU, n = 2")
    a = ap.parse_args(argv)
    test = a.test or bool(os.environ.get("TEST_MODE"))
    if test:
        a.dtype, a.n = "float32", 2 if a.test else a.n
        a.out = a.out if a.out != ap.get_default("out") else tempfile.mkdtemp(prefix="stage5_ioi_attention_test_")
    device = "cuda" if torch.cuda.is_available() and not test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation="eager").eval().to(device)
    cores = [dict(c) for c in make_cores(a.n, random.Random(a.seed))]
    mats, n, skipped = run(model, tok, cores, lambda s: print(s, flush=True))
    s = summarize(mats, n)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = a.model.split("/")[-1]
    label = "exploratory attention probe" + (", TEST_MODE" if test else "") + ("" if a.model == "gpt2" else "; the head labels are GPT-2 small's")
    json.dump({"provenance": provenance(a, label) | {"attn_implementation": model.config._attn_implementation, "skipped_items": skipped, "n_items": n}, "n": n, "heads": HEADS,
               "mats": mats, "summary": s}, open(f"{a.out}/{tag}.json", "w"))
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
