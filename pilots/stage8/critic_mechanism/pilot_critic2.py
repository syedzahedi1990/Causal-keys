"""Critic pilot for part D (Qwen2.5-0.5B, CPU FP32). Checks three things the design does not test:
(1) identity-freeness of the flag: per-story flag under the key-only clamp K_S (p's value still B) against the flag
    under the full source run (key and value of S): cos and norm ratio per layer; share of flag variance explained
    by B's identity; cosine of Delta with the unembedding rows of the six locations;
(2) the identity-contrast currency: ID_inj = 1/2[(lS-lX)(move B->S) - (lS-lX)(move B->X)] against
    ID_K = 1/2[(lS-lX)(K_S) - (lS-lX)(K_X)] (six-way renormalised), the injection analogue of the paper's measure;
(3) the orthogonal-complement control: the story's own write delta(s) and its component orthogonal to Delta,
    injected as a move B->X.
Uses H* from the part-D pilot (pilot05.json, ranked on 16 stories of Random(0)). Fit on Random(0) stories 16..(16+NF)
(fresh to the H* ranking), evaluated on Random(7) (fresh)."""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PD = "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partD"
sys.path.insert(0, PD)
from flagkit import Inject, OCap, head_out, ks_of, lp_last, prep  # noqa: E402
from ckeys.clamp import clamp_kv  # noqa: E402
from ckeys.interventions import blocks, edits  # noqa: E402
from ckeys.story import LOCATIONS, make_cores  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "2")))
NF, NE = int(sys.argv[1]), int(sys.argv[2])
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
t0 = time.time()
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)  # noqa: E731
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
nL, H, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
hd = getattr(cfg, "head_dim", None) or D // H
Hs = [tuple(x) for x in json.load(open(os.path.join(PD, "pilot05.json")))["Hs"]]
byL = {}
for l, h in Hs:
    byL.setdefault(l, []).append(h)
Ls = sorted(byL)
oc, inj = OCap(model), Inject(model)


@torch.no_grad()
def writes(d):
    """o_proj inputs at (rowS, rowB) in: clean B, key-only K_S, full source run S."""
    p = d["p"]
    ks = ks_of(model, d["ids"]["S"], p)
    oc.rows, oc.active = [d["rowS"], d["rowB"]], True
    try:
        model(d["ids"]["B"], use_cache=False, logits_to_keep=1)
        zb = {l: oc.store[l][0] for l in Ls}
        with edits(model, [(l, "k", [p], (lambda h, l=l: ks[l].expand_as(h))) for l in range(nL)]):
            model(d["ids"]["B"], use_cache=False, logits_to_keep=1)
        zk = {l: oc.store[l][0] for l in Ls}
        model(d["ids"]["S"], use_cache=False, logits_to_keep=1)
        zs = {l: oc.store[l][0] for l in Ls}
    finally:
        oc.active = False
    fk, fs = {}, {}
    for l in Ls:
        hh = byL[l]
        fk[l] = 0.5 * (head_out(model, l, zk[l][0] - zb[l][0], hh, hd) + head_out(model, l, zb[l][1] - zk[l][1], hh, hd))
        fs[l] = 0.5 * (head_out(model, l, zs[l][0] - zb[l][0], hh, hd) + head_out(model, l, zb[l][1] - zs[l][1], hh, hd))
    return fk, fs


# ---- (1) fit + identity-freeness
FK, FS, BID, SID = {l: [] for l in Ls}, {l: [] for l in Ls}, [], []
R = make_cores(16 + NF, random.Random(0))[16:]
for i, core in enumerate(R):
    d = prep(tok, core, "P1")
    fk, fs = writes(d)
    for l in Ls:
        FK[l].append(fk[l])
        FS[l].append(fs[l])
    BID.append(LOCATIONS.index(core["base"])); SID.append(LOCATIONS.index(core["source"]))
log(f"fit on {NF} stories")
Delta = {l: torch.stack(FK[l]).mean(0) for l in Ls}
DeltaS = {l: torch.stack(FS[l]).mean(0) for l in Ls}
WU = model.lm_head.weight.float()
loc_ids = [tok.encode(" " + w)[0] for w in LOCATIONS]
BID = np.array(BID); SID = np.array(SID)
FKs = {l: torch.stack(FK[l]) for l in Ls}
def loo(w):
    keep = torch.tensor((BID != w) & (SID != w))
    return {l: FKs[l][keep].mean(0) for l in Ls}, int(keep.sum())
LOO = {w: loo(w) for w in range(6)}
print("leave-one-word-out fit sizes", {LOCATIONS[w]: LOO[w][1] for w in range(6)})
print("layer |Delta_K| |Delta_KV| cos(Delta_K,Delta_KV) mean_s cos(dK(s),dKV(s)) B-identity R^2 of dK  max|cos(Delta,W_U[loc])| mean|cos(Delta,W_U[rand])|")
g = torch.Generator().manual_seed(3)
rand_ids = torch.randint(0, WU.shape[0], (500,), generator=g)
for l in Ls:
    FKl, FSl = torch.stack(FK[l]), torch.stack(FS[l])
    cs = torch.nn.functional.cosine_similarity(FKl, FSl, dim=-1).mean()
    # share of variance of per-story flag explained by B identity (between-group SS / total SS)
    X = FKl.numpy()
    tot = ((X - X.mean(0)) ** 2).sum()
    between = sum(((X[BID == b].mean(0) - X.mean(0)) ** 2).sum() * (BID == b).sum() for b in np.unique(BID))
    u = Delta[l] / Delta[l].norm()
    cl = torch.nn.functional.cosine_similarity(WU[loc_ids], u[None], dim=-1).abs().max()
    cr = torch.nn.functional.cosine_similarity(WU[rand_ids], u[None], dim=-1).abs().mean()
    print(f"  {l:2d}  {Delta[l].norm():.3f}  {DeltaS[l].norm():.3f}  {float(torch.nn.functional.cosine_similarity(Delta[l], DeltaS[l], dim=0)):+.3f}"
          f"  {float(cs):+.3f}  {between / tot:.3f}  {float(cl):.3f}  {float(cr):.3f}")
print(f"  (B-identity R^2 under a null of {len(np.unique(BID))} groups on n={NF}: expected ~{(len(np.unique(BID)) - 1) / (NF - 1):.2f})")


def addmap(B, T, spec):
    out = {l: torch.zeros(B, T, D) for l in Ls}
    for b, items in enumerate(spec):
        for row, vec, c in items:
            for l in Ls:
                out[l][b, row] += c * vec[l]
    return out


# ---- (2), (3) evaluation
E = make_cores(NE, random.Random(7))
rows = []
for j, core in enumerate(E):
    d = prep(tok, core, "P1")
    own, _ = writes(d)                       # the story's own key-only write (a K_S pass)
    perp = {l: own[l] - (own[l] @ (Delta[l] / Delta[l].norm())) * (Delta[l] / Delta[l].norm()) for l in Ls}
    perp_n = {l: perp[l] / perp[l].norm() * Delta[l].norm() for l in Ls}
    rB, rS, rX = d["rowB"], d["rowS"], d["rowX"]
    DX, DS = LOO[d["iX"]][0], LOO[d["iS"]][0]
    names = ["none", "moveS", "moveX", "addS", "addX", "moveX_own", "moveX_perp", "moveX_perpN", "moveX_KV", "moveS_loo", "moveX_loo", "addX_loo", "KS", "KX"]
    sp = {"none": [], "moveS": [(rS, Delta, 1), (rB, Delta, -1)], "moveX": [(rX, Delta, 1), (rB, Delta, -1)],
          "addS": [(rS, Delta, 1)], "addX": [(rX, Delta, 1)],
          "moveX_own": [(rX, own, 1), (rB, own, -1)], "moveX_perp": [(rX, perp, 1), (rB, perp, -1)],
          "moveX_perpN": [(rX, perp_n, 1), (rB, perp_n, -1)], "moveX_KV": [(rX, DeltaS, 1), (rB, DeltaS, -1)],
          "moveS_loo": [(rS, DS, 1), (rB, DS, -1)], "moveX_loo": [(rX, DX, 1), (rB, DX, -1)], "addX_loo": [(rX, DX, 1)],
          "KS": [], "KX": []}
    B, T = len(names), d["T"]
    ksS, ksX = ks_of(model, d["ids"]["S"], d["p"]), ks_of(model, d["ids"]["X"], d["p"])
    tab = {(l, "k"): torch.stack([ksX[l] if n == "KX" else ksS[l] for n in names])[:, None] for l in range(nL)}
    pr = torch.tensor([n in ("KS", "KX") for n in names])
    inj.add, inj.proj, inj.active = addmap(B, T, [sp[n] for n in names]), {}, True
    try:
        with torch.no_grad(), clamp_kv(model, [d["p"]], tab, range(nL), "k", per_row=pr):
            lp = lp_last(model, d["ids"]["B"].expand(B, -1))
    finally:
        inj.active, inj.add = False, {}
    c = lp[:, d["cid"]]
    c = c - torch.logsumexp(c, -1, keepdim=True)    # six-way renormalised
    rows.append(dict(iB=d["iB"], iS=d["iS"], iX=d["iX"], r={n: c[i].tolist() for i, n in enumerate(names)}))
    if (j + 1) % 4 == 0:
        log(f"eval {j + 1}")
nm = list(rows[0]["r"])
lS = {n: np.array([x["r"][n][x["iS"]] for x in rows]) for n in nm}
lX = {n: np.array([x["r"][n][x["iX"]] for x in rows]) for n in nm}
lB = {n: np.array([x["r"][n][x["iB"]] for x in rows]) for n in nm}
cSX = {n: lS[n] - lX[n] for n in nm}
IDK = 0.5 * ((cSX["KS"] - cSX["none"]) - (cSX["KX"] - cSX["none"]))
IDinj = 0.5 * ((cSX["moveS"] - cSX["none"]) - (cSX["moveX"] - cSX["none"]))
IDloo = 0.5 * ((cSX["moveS_loo"] - cSX["none"]) - (cSX["moveX_loo"] - cSX["none"]))
IDadd = 0.5 * ((cSX["addS"] - cSX["none"]) - (cSX["addX"] - cSX["none"]))
NX = lX["KX"] - lX["none"]
print(f"n={len(rows)}  ID_K (6-way renorm) {IDK.mean():+.3f} (sd {IDK.std():.3f})")
print(f"  ID_inj(move) {IDinj.mean():+.3f}  ratio {IDinj.mean() / IDK.mean():+.3f};  ID_inj(add) {IDadd.mean():+.3f} ratio {IDadd.mean() / IDK.mean():+.3f}")
print(f"  ID_inj(move, leave-one-word-out flag) {IDloo.mean():+.3f} ratio {IDloo.mean() / IDK.mean():+.3f}")
print(f"  N_X = dl_X(K_X) {NX.mean():+.3f};  dl_B(K_S) {(lB['KS'] - lB['none']).mean():+.3f};  dl_X(K_S) {(lX['KS'] - lX['none']).mean():+.3f} (non-specific gain of X when only B loses its flag)")
for n in ["moveX_loo", "addX_loo", "moveX", "addX", "moveX_own", "moveX_perp", "moveX_perpN", "moveX_KV", "moveS", "addS"]:
    dX = lX[n] - lX["none"]
    print(f"  {n:12s} dl_X {dX.mean():+.3f}  iota {dX.mean() / NX.mean():+.3f}   dl_B {(lB[n] - lB['none']).mean():+.3f}")
json.dump({"rows": rows, "IDK": IDK.tolist(), "IDinj": IDinj.tolist()}, open(os.path.join(os.path.dirname(__file__), "pilot_critic2.json"), "w"))
log("done")
