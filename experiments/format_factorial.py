"""Gate G1: does a post-state re-mention open a key channel? Log-odds K/V factorial across format arms.

For each arm and core, clamp the critical token's cached key and/or value (read by later tokens) to the
source run's, for all layers >= l0, in the unpatched base story. Metric: m = logp(S) - logp(B) over the
full vocabulary at the answer position (letters for LETTER). d_C = m(C) - m(clean B).
Reports dK, dV, dKV, interaction, key share dK/(dK+dV), recovery dKV/(m_S - m_B), with core bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import ARMS, alphabet, candidate_ids, encode, raw_prompt
from ckeys.interventions import blocks, capture, hooks
from ckeys.story import LOCATIONS, make_cores, record

# rows: (key source, value source, l0) applied at the critical token for layers >= l0
ROWS = [("S", "B", 0), ("B", "S", 0), ("S", "S", 0), ("S", "B", 8), ("B", "S", 8),
        ("X", "B", 0), ("B", "X", 0), ("X", "X", 0)]
LABEL = {("S", "B"): "K_S", ("B", "S"): "V_S", ("S", "S"): "KV_S", ("X", "B"): "K_X", ("B", "X"): "V_X", ("X", "X"): "KV_X"}


def pick_x(core):
    used = {core["base"], core["source"], core["initial"], core["distractor_location"]}
    return next(l for l in LOCATIONS if l not in used)


@torch.no_grad()
def run_item(model, tok, core, arm, view, rows, device):
    X = pick_x(core)
    locs = {"B": core["base"], "S": core["source"], "X": X}
    ids = {}
    for name, loc in locs.items():
        r = record(core, view, loc)
        ids[name] = encode(tok, raw_prompt(arm, r["story"], r["query"])).to(device)
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    assert len(diff) == 1 and (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() == diff, diff
    pos = diff[0]
    ib = ids["B"]
    nL = len(blocks(model))
    alph = alphabet(arm)
    cid = candidate_ids(tok, arm)
    track = {"S": core["source"], "B": core["base"], "X": X, "init": core["initial"]}
    tid = {k: cid[LOCATIONS.index(v)] for k, v in track.items()}
    kv, clean = {}, {}
    for name in ("B", "S", "X"):
        with capture(model, range(nL), "k") as K, capture(model, range(nL), "v") as V:
            lp = torch.log_softmax(model(ids[name]).logits[0, -1].float(), -1)
        kv[name] = ({l: K[l][0, pos] for l in range(nL)}, {l: V[l][0, pos] for l in range(nL)})
        clean[name] = {"lp": {k: lp[i].item() for k, i in tid.items()}, "m": (lp[tid["S"]] - lp[tid["B"]]).item(),
                       "mass": lp[cid].exp().sum().item(), "argmax_cand": LOCATIONS[int(lp[cid].argmax())]}
    rows = [r for r in rows if r[2] < nL]

    def who(row, l, ch):
        ks, vs, l0 = row
        return (ks if ch == "K" else vs) if l >= l0 else "B"

    tabs = {(l, ch): torch.stack([kv[who(r, l, ch)][0 if ch == "K" else 1][l] for r in rows])
            for l in range(nL) for ch in "KV"}
    hs = []
    for l in range(nL):
        at = blocks(model)[l].self_attn
        for mod, ch in ((at.k_proj, "K"), (at.v_proj, "V")):
            def hk(_m, _i, out, t=tabs[(l, ch)]):
                out = out.clone()
                out[:, pos] = t.to(out.dtype)
                return out
            hs.append(mod.register_forward_hook(hk))
    with hooks(hs):
        lp = torch.log_softmax(model(ib.expand(len(rows), -1)).logits[:, -1].float(), -1)
    out = {f"{LABEL[(k, v)]}@{l0}": {"m": (lp[i, tid["S"]] - lp[i, tid["B"]]).item(),
                                      "lp": {t: lp[i, j].item() for t, j in tid.items()}}
           for i, (k, v, l0) in enumerate(rows)}
    return {"core": core, "X": X, "arm": arm, "view": view, "pos": pos, "len": ib.shape[1], "clean": clean, "m": out}


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), (np.percentile(bs, 2.5), np.percentile(bs, 97.5))


def summarize(res):
    lines = []
    for arm in ARMS:
        for view in ("direct", "other_agent", "world"):
            R = [r for r in res if r["arm"] == arm and r["view"] == view]
            if not R:
                continue
            comp = [r for r in R if r["clean"]["B"]["argmax_cand"] == (r["core"]["base"] if view != "other_agent" else r["core"]["initial"])
                    and r["clean"]["S"]["argmax_cand"] == (r["core"]["source"] if view != "other_agent" else r["core"]["initial"])]
            mB = np.array([r["clean"]["B"]["m"] for r in R]); mS = np.array([r["clean"]["S"]["m"] for r in R])
            d = {k: np.array([r["m"][k]["m"] for r in R]) - mB for k in R[0]["m"]}
            dl = {k: {t: np.array([r["m"][k]["lp"][t] - r["clean"]["B"]["lp"][t] for r in R]) for t in ("S", "B", "X")}
                  for k in R[0]["m"]}
            span = mS - mB
            dK, dV, dKV = d["K_S@0"], d["V_S@0"], d["KV_S@0"]
            share = dK.mean() / (dK.mean() + dV.mean()) if abs(dK.mean() + dV.mean()) > 1e-6 else float("nan")
            mass = np.mean([r["clean"]["B"]["mass"] for r in R])
            lines.append(f"{arm:7s} {view:11s} n={len(R):3d} competent={len(comp):3d} mass={mass:.2f} span={span.mean():+6.2f}")
            for k in d:
                mu, (lo, hi) = boot(d[k])
                parts = "  ".join(f"dlogp({t})={dl[k][t].mean():+6.2f}" for t in ("S", "B", "X"))
                lines.append(f"    d[{k:7s}] {mu:+7.2f} [{lo:+6.2f},{hi:+6.2f}]   {parts}")
            mu, (lo, hi) = boot(dKV - dK - dV)
            lines.append(f"    interaction {mu:+7.2f} [{lo:+6.2f},{hi:+6.2f}]   key share {share:.3f}   recovery dKV/span {dKV.mean() / span.mean():.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--views", default="direct")
    ap.add_argument("--specificity-arms", default="P1,NONE", help="arms also run on the other_agent view")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/format_factorial")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(device).eval()
    cores = make_cores(a.n, random.Random(a.seed))
    jobs = [(arm, v) for arm in a.arms.split(",") for v in a.views.split(",")]
    jobs += [(arm, "other_agent") for arm in a.specificity_arms.split(",") if arm and arm in a.arms.split(",")]
    res, t0 = [], time.time()
    for arm, view in jobs:
        rows = ROWS if view != "other_agent" else ROWS[:3]
        for core in cores:
            r = run_item(model, tok, core, arm, view, rows, device)
            if r is not None:
                res.append(r)
        print(f"  {arm}/{view} done ({time.time() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.model.split('/')[-1]}_s{a.seed}"
    json.dump({"args": vars(a), "results": res}, open(f"{a.out}/{tag}.json", "w"))
    s = summarize(res)
    open(f"{a.out}/{tag}_summary.txt", "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
