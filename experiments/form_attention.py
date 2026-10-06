"""Attention probe of preregistration G, part (d): do a variant's re-mention tokens attend to the writing token p
under its source key? Eager attention, n = 60 of the seed-0 cores, arms POST and the sentence variants.

Per core and arm, ``remention_attention.probe`` runs clean B and the base prompt with K_S / K_X clamped at p and the
value at p held at B's (the factorial's K_S@0 / K_X@0 rows, ckeys.clamp hooks); for every location's form span (its full token
sequence, located after p as an exact subsequence) the head-averaged attention from the span tokens to p is summed
over the span, per layer. A_v = mean over layers of 1/2 [att_S(K_S) - att_S(K_X) + att_X(K_X) - att_X(K_S)] and
a_v = A_v / A_POST are computed by analysis/stage5_parts/variants.py. Also stored: last-token-only attention and the
per-head maximum of the span sum. Output: <out>/<model>.json. TEST_MODE (--test, or TEST_MODE=1 keeping --n):
Qwen2.5-0.5B-Instruct, FP32, CPU, n = 3.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from ckeys.encoding import candidate_ids, import_arm_modules
from ckeys.story import LOCATIONS, make_cores
from ckeys.variants import form_spans
from experiments.format_factorial import provenance
from experiments.remention_attention import encode_item, load_model, probe

PROBE_ARMS = ("POST", "POST_THE", "POST_MODIF", "POST_TITLE", "POST_UPPER", "POST_PLURAL", "POST_SYN", "POST_FRMIX",
              "POST_DEMIX", "POST_FR", "POST_DE")
CONDS = ("B", "K_S", "K_X")
TEST_MODEL, TEST_N = "Qwen/Qwen2.5-0.5B-Instruct", 3


def locate_spans(seq: list[int], p: int, spans: list[list[int]]) -> list[list[int]] | None:
    """Rows of each form span after p, in order; None if one is not found."""
    out, pos = [], p + 1
    for sp in spans:
        hit = next((i for i in range(pos, len(seq) - len(sp) + 1) if seq[i:i + len(sp)] == sp), None)
        if hit is None:
            return None
        out.append(list(range(hit, hit + len(sp))))
        pos = hit + len(sp)
    return out


@torch.no_grad()
def item(model, tok, core, arm, cid, conds=CONDS):
    enc = encode_item(tok, core, arm)
    if enc is None:
        return None
    seq, p = enc["ids"]["B"][0].tolist(), enc["p"]
    spans = locate_spans(seq, p, form_spans(tok, arm))
    if spans is None:
        return None
    rows = [r for sp in spans for r in sp]
    out = probe(model, enc["ids"], p, rows, [p], conds)
    att = {}
    for c in conds:
        A = out["att"][c][:, :, :, 0]                       # [L, H, rows]
        per = {"span": [], "last": [], "head_max": []}
        k = 0
        for sp in spans:
            a = A[:, :, k:k + len(sp)]
            per["span"].append(a.mean(1).sum(-1).tolist())        # head-averaged, summed over the span: [L]
            per["last"].append(a[:, :, -1].mean(1).tolist())
            per["head_max"].append(a.sum(-1).max(1).values.tolist())
            k += len(sp)
        att[c] = per
    lp = {c: {w: out["lp"][c][i].item() for w, i in zip(LOCATIONS, cid)} for c in conds}
    return {"core": core, "X": enc["X"], "arm": arm, "p": p, "T": enc["T"], "spans": spans,
            "b": LOCATIONS.index(core["base"]), "s": LOCATIONS.index(core["source"]), "x": LOCATIONS.index(enc["X"]),
            "att": att, "lp": lp}


def run(model, tok, cores, arms, log=print):
    items, skipped, t0, cid = [], 0, time.time(), candidate_ids(tok, "POST")
    for arm in arms:
        for i, core in enumerate(cores):
            it = item(model, tok, core, arm, cid)
            if it is None:
                skipped += 1
                continue
            it["i"] = i
            items.append(it)
        log(f"  {arm} done ({time.time() - t0:.0f}s)")
    return items, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--arms", default=",".join(PROBE_ARMS))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gpu_stage5/form_attention")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--arm-modules", default="ckeys.variants")
    ap.add_argument("--test", action="store_true", help=f"TEST_MODE: {TEST_MODEL}, FP32, n = {TEST_N}")
    a = ap.parse_args()
    if a.test or os.environ.get("TEST_MODE"):
        a.model, a.dtype, a.device_map = TEST_MODEL, "float32", None
        a.n = TEST_N if a.test else a.n
    import_arm_modules(a.arm_modules)
    torch.set_grad_enabled(False)
    model, tok = load_model(a.model, a.dtype, a.revision, a.device_map, cpu=bool(a.test or os.environ.get("TEST_MODE")))
    items, skipped = run(model, tok, make_cores(a.n, random.Random(a.seed)), a.arms.split(","), lambda s: print(s, flush=True))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = f"{a.out}/{a.model.split('/')[-1]}.json"
    prov = provenance(a) | {"skipped_items": skipped, "attn_implementation": model.config._attn_implementation, "conds": CONDS}
    json.dump({"provenance": prov, "items": items}, open(f, "w"))
    print(f"wrote {f} ({len(items)} items, {skipped} skipped)")


if __name__ == "__main__":
    main()
