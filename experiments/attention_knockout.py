"""Attention knockout of the lookup route (part (b) of P-2026-10-05-G): chosen (row, column) attention weights are
set to exactly 0 in every layer and head (the row renormalised) while the key/value at the writing token p is
clamped as in format_factorial. Every effect is measured against the self-clamp row under the same mask.

Masks (ckeys.knockout.groups): M0 none; M1 R_cand x {p}; M2 R_cand x C_init (control column); M3 {a} x {p};
M4 (R_cand u {a}) x {p}; M8 R_all x {p} (sanity gate); exploratory M5 R_trk x {p}, M6 R_oth x {p}, M7 R_tail x {p},
M2b R_cand x {c_prev}, Mq R_q x {p}, Mpost R_post x {p}. Under NONE only M0, M3, M8, Mq, Mpost.
Per (arm, core): three clean forwards (B, S, X) capture K/V at p; one batched forward of the base prompt carries
8 clamp rows per mask (ID self-clamp, ID2 duplicate = within-batch noise floor, K_S, V_S, KV_S, K_X, V_X, KV_X)
with a per-row 4D mask (bool for sdpa, additive for eager). Stored per row: log p of S, B, X, init at a = T-1,
m = logp(S) - logp(B), restricted argmax and candidate mass, location mass, full-vocab top-5, on-target flags.
NONE additionally runs the exploratory emission-position pass (greedy teacher-forced continuation of the clean B
run to the first on-target top-1, then {a..a_emit} x {p}).
TEST_MODE (--test, or TEST_MODE=1 which keeps --n): Qwen2.5-0.5B-Instruct, FP32, CPU, n = 2, all arms and masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv, stack_rows
from ckeys.encoding import WRAPPER_USED, build_prompt, candidate_ids, chat_text
from ckeys.interventions import blocks
from ckeys.knockout import groups, masks_for_model, on_target, target_ids
from ckeys.story import LOCATIONS, make_cores, pick_x, record

ROWS = [("B", "B", "ID"), ("B", "B", "ID2"), ("S", "B", "K_S"), ("B", "S", "V_S"), ("S", "S", "KV_S"),
        ("X", "B", "K_X"), ("B", "X", "V_X"), ("X", "X", "KV_X")]  # (key donor, value donor, label)
MASKS = ["M0", "M1", "M2", "M2b", "M3", "M4", "M5", "M6", "M7", "M8", "Mq", "Mpost"]
NONE_MASKS = ["M0", "M3", "M8", "Mq", "Mpost"]  # M1/M2/M2b/M4-M7 are undefined (R_cand empty) or equal M3/M8
ARMS = ("AFTER", "P1", "POST", "NONE")
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def masks_of(arm):
    return NONE_MASKS if arm == "NONE" else MASKS


def mask_pairs(g):
    p, a = g["p"], g["a"]
    col = lambda rows, cols=(p,): [(r, c) for r in rows for c in cols]
    return {"M0": [], "M1": col(g["R_cand"]), "M2": col(g["R_cand"], g["C_init"]), "M2b": col(g["R_cand"], (g["c_prev"],)),
            "M3": col([a]), "M4": col(g["R_cand"] + [a]), "M5": col(g["R_trk"]), "M6": col(g["R_oth"]),
            "M7": col(g["R_tail"]), "M8": col(g["R_all"]), "Mq": col(g["R_q"]), "Mpost": col(g["R_post"])}


def logprobs_at(model, ids, mask=None):
    out = model(ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    return torch.log_softmax(out.logits[:, -1].float(), -1)


def readout(lp, tid, cid, tidx, core):
    top = lp.topk(5)
    t1 = int(top.indices[0])
    return {"lp": {k: lp[i].item() for k, i in tid.items()}, "m": (lp[tid["S"]] - lp[tid["B"]]).item(),
            "mass": lp[cid].exp().sum().item(), "argmax_cand": LOCATIONS[int(lp[cid].argmax())],
            "loc_mass": lp[tidx["loc_mass"]].exp().sum().item(),
            "top5": [top.indices.tolist(), [round(x, 4) for x in top.values.tolist()]],
            "on": {"B": on_target(t1, core["base"], tidx), "S": on_target(t1, core["source"], tidx)}}


def clamp_rows(model, ids, p, tabs, nL, mask, n_masks, rd):
    """Batched forward of ``ids`` expanded to 8 x n_masks rows under ``mask`` with the 8 clamp rows per mask."""
    R = n_masks * len(ROWS)
    tabs_R = {k: v.repeat(n_masks, 1, 1) for k, v in tabs.items()}
    with clamp_kv(model, [p], tabs_R, range(nL)):
        lp = logprobs_at(model, ids.expand(R, -1), mask)
    return [{lab: rd(lp[i * len(ROWS) + j]) for j, (_, _, lab) in enumerate(ROWS)} for i in range(n_masks)]


@torch.no_grad()
def emission_pass(model, ids, p, tabs, nL, tidx, core, steps, rd):
    """NONE, exploratory: continue the clean B run greedily (teacher-forced) until the top-1 is on target, then
    knock out {a..a_emit} x {p} and score at a_emit (both M0 and M3' rows at a_emit)."""
    seq, a = ids, ids.shape[1] - 1
    for s in range(steps + 1):
        t1 = int(logprobs_at(model, seq)[0].argmax())
        if on_target(t1, core["base"], tidx):
            break
        if s == steps:
            return {"a_emit": None, "tokens": seq[0, a + 1:].tolist()}
        seq = torch.cat([seq, torch.tensor([[t1]], device=seq.device)], 1)
    T2 = seq.shape[1]
    pairs = [(r, p) for r in range(a, T2)]
    mask = masks_for_model(model, T2, [[]] * len(ROWS) + [pairs] * len(ROWS)).to(seq.device)
    m0, m3 = clamp_rows(model, seq, p, tabs, nL, mask, 2, rd)
    return {"a_emit": T2 - 1, "tokens": seq[0, a + 1:].tolist(), "m": {"M0": m0, "M3e": m3}}


@torch.no_grad()
def run_item(model, tok, core, arm, device, tidx, emit_steps=12, chunk=0):
    X = pick_x(core)
    locs = {"B": core["base"], "S": core["source"], "X": X}
    ids, text = {}, {}
    for name, loc in locs.items():
        r = record(core, "direct", loc)
        text[name] = chat_text(tok, build_prompt(arm, r["story"], r["query"], core, X))
        ids[name] = tok(text[name], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    if len(diff) != 1 or (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() != diff:
        return None
    p, T = diff[0], ids["B"].shape[1]
    off = tok(text["B"], add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    g = groups(tok, text["B"], ids["B"].cpu(), off, p, core, arm, X, record(core, "direct", core["base"])["query"])
    nL, cid = len(blocks(model)), candidate_ids(tok, arm)
    tid = {k: cid[LOCATIONS.index(v)] for k, v in (locs | {"init": core["initial"]}).items()}
    rd = lambda lp: readout(lp, tid, cid, tidx, core)
    kv, clean = {}, {}
    for name in ("B", "S", "X"):
        with capture_kv(model, [p], range(nL)) as t:
            lp = logprobs_at(model, ids[name])[0]
        kv[name] = {k: v[0] for k, v in t.items()}
        clean[name] = rd(lp)
    tabs = stack_rows(kv, [lambda l, ch, r=r: r[0] if ch == "k" else r[1] for r in ROWS], range(nL))
    masks, pairs, out = masks_of(arm), mask_pairs(g), {}
    for m0 in range(0, len(masks), chunk or len(masks)):
        ms = masks[m0:m0 + (chunk or len(masks))]
        mask = masks_for_model(model, T, [pairs[m] for m in ms for _ in ROWS]).to(device)
        out |= dict(zip(ms, clamp_rows(model, ids["B"], p, tabs, nL, mask, len(ms), rd)))
    item = {"core": core, "X": X, "arm": arm, "view": "direct", "pos": p, "len": T, "n_layers": nL, "groups": g,
            "chat_sha256": hashlib.sha256(text["B"].encode()).hexdigest(), "clean": clean, "m": out}
    if arm == "NONE" and emit_steps:
        item["emit"] = emission_pass(model, ids["B"], p, tabs, nL, tidx, core, emit_steps, rd)
    return item


def per_core(results, arm, mask):
    """Per-core derived quantities of one arm x mask: ID_K, ID_V, d_K/d_V/d_KV (span), restricted accuracy and
    full-vocab on-target flags (acc/on_B from the self-clamp row, acc/on_S from the KV_S row), location and candidate
    mass of the self-clamp row, the within-batch and batch-vs-clean noise floors, |C_init|, competence."""
    out = {}
    for r in results:
        if r["arm"] != arm or mask not in r["m"]:
            continue
        m, c = r["m"][mask], r["core"]
        idr = m["ID"]
        dl = {k: {t: m[k]["lp"][t] - idr["lp"][t] for t in ("S", "B", "X")} for k in m}
        d = {k: m[k]["m"] - idr["m"] for k in m}
        out[json.dumps(c, sort_keys=True)] = dict(
            idK=0.5 * ((dl["K_S"]["S"] - dl["K_X"]["S"]) + (dl["K_X"]["X"] - dl["K_S"]["X"])),
            idV=0.5 * ((dl["V_S"]["S"] - dl["V_X"]["S"]) + (dl["V_X"]["X"] - dl["V_S"]["X"])),
            dK=d["K_S"], dV=d["V_S"], dKV=d["KV_S"], accB=idr["argmax_cand"] == c["base"],
            accS=m["KV_S"]["argmax_cand"] == c["source"], onB=idr["on"]["B"], onS=m["KV_S"]["on"]["S"],
            loc=idr["loc_mass"], mass=idr["mass"], floor=abs(d["ID2"]), floor_clean=abs(idr["m"] - r["clean"]["B"]["m"]),
            n_init=len(r["groups"]["C_init"]), top1={k: m[k]["top5"][0][0] for k in m}, L=r["n_layers"],
            competent=r["clean"]["B"]["argmax_cand"] == c["base"] and r["clean"]["S"]["argmax_cand"] == c["source"])
    return out


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    bs = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(t):
    return f"{t[0]:+7.2f} [{t[1]:+6.2f},{t[2]:+6.2f}]"


def summarize(res, arms=ARMS):
    lines = []
    for arm in arms:
        for mask in masks_of(arm):
            P = per_core(res, arm, mask)
            if not P:
                continue
            v = lambda k: np.array([x[k] for x in P.values()], float)
            P0 = per_core(res, arm, "M0")
            rK = v("idK").mean() / np.mean([x["idK"] for x in P0.values()])
            lines.append(f"{arm:5s} {mask:5s} n={len(P):3d} ID_K {fmt(boot(v('idK')))} r_K {rK:+5.2f}  ID_V {fmt(boot(v('idV')))}  "
                         f"span {v('dKV').mean():+6.2f} q_V {v('idV').mean() / v('dKV').mean():+5.2f}  acc_B {v('accB').mean():.2f} "
                         f"acc_S {v('accS').mean():.2f} on_B {v('onB').mean():.2f} on_S {v('onS').mean():.2f} loc {v('loc').mean():.2f} "
                         f"floor {v('floor').mean():.3f}/{v('floor_clean').mean():.3f}")
    return "\n".join(lines)


def provenance(a, model, tok, res, skipped):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    first = {arm: next((r for r in res if r["arm"] == arm), None) for arm in a.arms.split(",")}
    seen = sorted({i for r in res for m in r["m"].values() for row in m.values() for i in row["top5"][0]} |
                  {i for r in res for row in r["clean"].values() for i in row["top5"][0]})
    return {"args": vars(a), "git_commit": commit, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "dtype": str(next(model.parameters()).dtype),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "attn_implementation": model.config._attn_implementation, "wrapper": dict(WRAPPER_USED),
            "target_ids": target_ids(tok), "rows": ROWS, "masks": MASKS, "none_masks": NONE_MASKS, "skipped_items": skipped,
            "chat_sha256_core0": {arm: r["chat_sha256"] for arm, r in first.items() if r}, "len_core0": {arm: r["len"] for arm, r in first.items() if r},
            "c_init_counts": {str(k): int(v) for k, v in zip(*np.unique([len(r["groups"]["C_init"]) for r in res], return_counts=True))},
            "vocab": {i: tok.decode([i]) for i in seen}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gpu_stage5/knockout")
    ap.add_argument("--attn", default="sdpa", help="attention implementation: sdpa (bool mask) or eager (additive mask)")
    ap.add_argument("--emit-steps", type=int, default=12, help="emission-position pass under NONE (0 disables)")
    ap.add_argument("--chunk", type=int, default=0, help="masks per batched forward (0 = all masks in one batch)")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B, FP32, CPU, n = 2")
    a = ap.parse_args()
    if a.test or os.environ.get("TEST_MODE"):  # --test also fixes n = 2; TEST_MODE=1 keeps the caller's n
        a.n = min(a.n, 2) if a.test else a.n
        a.test, a.model, a.dtype = True, TEST_MODEL, "float32"
        print(f"TEST_MODE: {a.model}, float32, CPU, n = {a.n}", flush=True)
        a.out = a.out if a.out != ap.get_default("out") else tempfile.mkdtemp(prefix="stage5_knockout_test_")
    device = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = {"dtype": getattr(torch, a.dtype), "revision": a.revision, "attn_implementation": a.attn}
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map:
        model = model.to(device)
    device = next(model.parameters()).device
    tidx = target_ids(tok)
    cores = make_cores(a.n, random.Random(a.seed))
    res, skipped, t0 = [], 0, time.time()
    for arm in a.arms.split(","):
        for core in cores:
            r = run_item(model, tok, core, arm, device, tidx, a.emit_steps, a.chunk)
            if r is None:
                skipped += 1
            else:
                res.append(r)
        print(f"  {arm} done ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_s{a.seed}"
    prov = provenance(a, model, tok, res, skipped) | {"seconds": round(time.time() - t0)}
    json.dump({"provenance": prov, "results": res}, open(f"{a.out}/{tag}.json", "w"))
    s = json.dumps({k: v for k, v in prov.items() if k != "vocab"}) + "\n" + summarize(res, a.arms.split(","))
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)
    print(f"wrote {a.out}/{tag}.json ({time.time() - t0:.0f}s)")
    if not res:
        raise SystemExit("no valid items")


if __name__ == "__main__":
    main()
