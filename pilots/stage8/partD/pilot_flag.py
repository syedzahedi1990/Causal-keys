"""Pilot D-I at a small model (CPU, FP32): a3 ranking -> H*, flag fit on R, injection battery on E.
Usage: python pilot_flag.py MODEL N_RANK N_EVAL ARM_FIT ARMS_EVAL"""
import json
import math
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partD")
from flagkit import Inject, OCap, head_out, ks_of, lp_last, prep  # noqa: E402
from ckeys.clamp import clamp_kv  # noqa: E402
from ckeys.interventions import blocks, edits  # noqa: E402
from ckeys.story import make_cores  # noqa: E402
from experiments.stage6_heads import rank_of  # noqa: E402

torch.set_num_threads(int(__import__('os').environ.get('NT','3')))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
NR = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NE = int(sys.argv[3]) if len(sys.argv) > 3 else 20
ARMFIT = sys.argv[4] if len(sys.argv) > 4 else "P1"
ARMS_EVAL = (sys.argv[5] if len(sys.argv) > 5 else "P1").split(",")
OUT = sys.argv[6] if len(sys.argv) > 6 else None
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").eval()
cfg = model.config
nL, H = cfg.num_hidden_layers, cfg.num_attention_heads
hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
D = cfg.hidden_size
kstar = math.ceil(0.05 * nL * H)
oc, inj = OCap(model), Inject(model)
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)  # noqa: E731


@torch.no_grad()
def a3_and_flag_runs(d):
    """clean B and K_S-clamped runs with attentions; o_proj inputs at rows (rowS, rowB)."""
    ks = ks_of(model, d["ids"]["S"], d["p"])
    p = d["p"]
    oc.rows, oc.active = [d["rowS"], d["rowB"]], True
    try:
        ob = model(d["ids"]["B"], use_cache=False, logits_to_keep=1, output_attentions=True)
        zb = {l: oc.store[l][0] for l in range(nL)}
        with edits(model, [(l, "k", [p], (lambda h, l=l: ks[l].expand_as(h))) for l in range(nL)]):
            of = model(d["ids"]["B"], use_cache=False, logits_to_keep=1, output_attentions=True)
        zf = {l: oc.store[l][0] for l in range(nL)}
    finally:
        oc.active = False
    Ab = torch.stack([a[0].float() for a in ob.attentions])
    Af = torch.stack([a[0].float() for a in of.attentions])
    rS, rB = d["rowS"], d["rowB"]
    a3 = 0.5 * ((Af[:, :, rS, p] - Ab[:, :, rS, p]) + (Ab[:, :, rB, p] - Af[:, :, rB, p]))
    return a3.numpy(), zb, zf, ks


R = make_cores(NR, random.Random(0))
E = make_cores(NE, random.Random(1))
recs = []
for i, core in enumerate(R):
    d = prep(tok, core, ARMFIT)
    a3, zb, zf, _ = a3_and_flag_runs(d)
    recs.append((a3, zb, zf))
    if i % 5 == 0:
        log(f"rank story {i} T={d['T']}")
A3 = np.mean([r[0] for r in recs], 0)
Hs = [tuple(c) for c in rank_of(A3)[:kstar]]
log(f"ranked on {NR} ({ARMFIT}); k*={kstar}; top10 {Hs[:10]}")
byL = {}
for l, h in Hs:
    byL.setdefault(l, []).append(h)
# per-story flag per layer: 0.5 * [(out_KS - out_B)[rowS] + (out_B - out_KS)[rowB]]
flags = {l: [] for l in byL}
for _, zb, zf in recs:
    for l, hh in byL.items():
        dS = head_out(model, l, zf[l][0] - zb[l][0], hh, hd)
        dB = head_out(model, l, zb[l][1] - zf[l][1], hh, hd)
        flags[l].append(0.5 * (dS + dB))
Delta = {l: torch.stack(v).mean(0) for l, v in flags.items()}
# consistency: mean cosine of per-story flags to the mean, per layer; norm
for l in sorted(Delta):
    F = torch.stack(flags[l])
    cos = torch.nn.functional.cosine_similarity(F, Delta[l][None], dim=-1)
    print(f"  layer {l:2d} heads {byL[l]} |Delta| {Delta[l].norm():.3f} mean|flag| {F.norm(dim=-1).mean():.3f} cos(story,mean) {cos.mean():.2f}")
# global direction check: cos between layer flags
Ls = sorted(Delta)
U = torch.stack([Delta[l] / Delta[l].norm() for l in Ls])
C = U @ U.T
print("  mean off-diagonal cos between layer flags:", float((C.sum() - len(Ls)) / (len(Ls) ** 2 - len(Ls))))
if OUT:
    json.dump({"model": MODEL, "Hs": Hs}, open(OUT, "w"))
    torch.save({l: Delta[l] for l in Ls}, OUT.replace(".json", "_delta.pt"))
    torch.save(flags, OUT.replace(".json", "_flags.pt"))
g = torch.Generator().manual_seed(0)
RAND = [{l: (lambda v: v / v.norm() * Delta[l].norm())(torch.randn(D, generator=g)) for l in Ls} for _ in range(3)]


def addmap(B, T, spec):
    """spec: list per batch row of list of (row, vecdict, coef)."""
    out = {l: torch.zeros(B, T, D) for l in Ls}
    for b, items in enumerate(spec):
        for row, vec, c in items:
            for l in Ls:
                out[l][b, row] += c * vec[l]
    return out


@torch.no_grad()
def eval_story(d, arm):
    names = ["none", "addX", "move", "rand0", "rand1", "rand2", "addFirst", "addQ", "subB", "KX", "add2X", "addHalfX", "KS"]
    sp = {"none": [], "addX": [(d["rowX"], Delta, 1.0)], "move": [(d["rowX"], Delta, 1.0), (d["rowB"], Delta, -1.0)],
          "rand0": [(d["rowX"], RAND[0], 1.0)], "rand1": [(d["rowX"], RAND[1], 1.0)], "rand2": [(d["rowX"], RAND[2], 1.0)],
          "addFirst": [(d["first"], Delta, 1.0)], "addQ": [(d["qrow"], Delta, 1.0)], "subB": [(d["rowB"], Delta, -1.0)],
          "KX": [], "add2X": [(d["rowX"], Delta, 2.0)], "addHalfX": [(d["rowX"], Delta, 0.5)], "KS": []}
    B, T = len(names), d["T"]
    inj.add, inj.proj, inj.active = addmap(B, T, [sp[n] for n in names]), {}, True
    kx = ks_of(model, d["ids"]["X"], d["p"]) if True else None
    ksx = ks_of(model, d["ids"]["S"], d["p"])
    tab = {(l, "k"): torch.stack([kx[l] if n == "KX" else ksx[l] for n in names])[:, None] for l in range(nL)}
    pr = torch.tensor([n in ("KX", "KS") for n in names])
    try:
        with clamp_kv(model, [d["p"]], tab, range(nL), "k", per_row=pr):
            lp = lp_last(model, d["ids"]["B"].expand(B, -1))
    finally:
        inj.active, inj.add = False, {}
    c = lp[:, d["cid"]]
    return {n: c[i].tolist() for i, n in enumerate(names)}


res = {}
for arm in ARMS_EVAL:
    rows = []
    for core in E:
        d = prep(tok, core, arm)
        r = eval_story(d, arm)
        rows.append(dict(iB=d["iB"], iS=d["iS"], iX=d["iX"], iI=d["iI"], iD=d["iD"], r=r))
        if len(rows) % 5 == 0:
            log(f"eval {arm} story {len(rows)}")
    res[arm] = rows
    log(f"eval {arm} on {NE}")
    names = list(rows[0]["r"])
    mX = {n: np.array([x["r"][n][x["iX"]] - x["r"][n][x["iB"]] for x in rows]) for n in names}
    lX = {n: np.array([x["r"][n][x["iX"]] for x in rows]) for n in names}
    am = {n: np.array([int(np.argmax(x["r"][n])) for x in rows]) for n in names}
    iX = np.array([x["iX"] for x in rows]); iB = np.array([x["iB"] for x in rows]); iI = np.array([x["iI"] for x in rows])
    ref = mX["KX"] - mX["none"]
    print(f"  {arm}: clean argmax=B {np.mean(am['none'] == iB):.2f}; m_X(none) {mX['none'].mean():+.2f}; K_X effect on m_X {ref.mean():+.2f}")
    for n in names:
        dm = mX[n] - mX["none"]
        print(f"   {n:9s} dm_X {dm.mean():+7.2f} (sd {dm.std():5.2f})  iota {dm.mean() / ref.mean():+.3f}  dlpX {np.mean(lX[n] - lX['none']):+6.2f}  "
              f"argmax: X {np.mean(am[n] == iX):.2f} B {np.mean(am[n] == iB):.2f} init {np.mean(am[n] == iI):.2f}")
if OUT:
    json.dump({"model": MODEL, "Hs": Hs, "res": res, "Delta_norm": {l: float(Delta[l].norm()) for l in Ls}}, open(OUT, "w"))
log("done")
