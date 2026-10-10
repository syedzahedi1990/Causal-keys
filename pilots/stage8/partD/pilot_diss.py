"""Pilot of the dissociation measures at Qwen2.5-0.5B (CPU FP32; 0.5B already showed H_diss in the disclosed G pilots).
Uses H* and the P1 flag from pilot_flag.py (pilot05.json, pilot05_delta.pt). For POST: flag written along the P1 flag
(W ratio, cosines), and the read test: inject Delta_P1 (and Delta_POST) at the sentence row of X; outcome on
case-marginalised candidate log-probs. Same for P1 for the ratio."""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flagkit import Inject, OCap, head_out, ks_of, lp_last, prep  # noqa: E402
from ckeys.clamp import clamp_kv  # noqa: E402
from ckeys.interventions import blocks, edits  # noqa: E402
from ckeys.story import LOCATIONS, make_cores  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "3")))
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
NF, NE = int(sys.argv[1]), int(sys.argv[2])
t0 = time.time()
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)  # noqa: E731
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
nL, H, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
hd = cfg.hidden_size // H
J = json.load(open("pilot05.json"))
Hs = [tuple(c) for c in J["Hs"]]
DP1 = torch.load("pilot05_delta.pt")
Ls = sorted(DP1)
byL = {}
for l, h in Hs:
    byL.setdefault(l, []).append(h)
oc, inj = OCap(model), Inject(model)


def forms(word):
    out = []
    for f in (" " + word, " " + word.capitalize(), word, word.capitalize()):
        t = tok.encode(f, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return out


FORMS = [forms(w) for w in LOCATIONS]
print("forms per word:", [len(f) for f in FORMS])


def cand_lp(lp):
    """[B, 6] case-marginalised log-probs (logsumexp over single-token surface forms)."""
    return torch.stack([torch.logsumexp(lp[:, f], -1) for f in FORMS], -1)


@torch.no_grad()
def flag_story(d):
    ks = ks_of(model, d["ids"]["S"], d["p"])
    oc.rows, oc.active = [d["rowS"], d["rowB"]], True
    try:
        model(d["ids"]["B"], use_cache=False, logits_to_keep=1)
        zb = {l: oc.store[l][0] for l in Ls}
        with edits(model, [(l, "k", [d["p"]], (lambda h, l=l: ks[l].expand_as(h))) for l in range(nL)]):
            model(d["ids"]["B"], use_cache=False, logits_to_keep=1)
        zf = {l: oc.store[l][0] for l in Ls}
    finally:
        oc.active = False
    return {l: 0.5 * (head_out(model, l, zf[l][0] - zb[l][0], byL[l], hd) + head_out(model, l, zb[l][1] - zf[l][1], byL[l], hd)) for l in Ls}


F = make_cores(NF, random.Random(0))
E = make_cores(NE, random.Random(1))
flags = {"P1": [], "POST": []}
for arm in flags:
    for core in F:
        flags[arm].append(flag_story(prep(tok, core, arm)))
    log(f"flags {arm}")
DPOST = {l: torch.stack([f[l] for f in flags["POST"]]).mean(0) for l in Ls}
DP1f = {l: torch.stack([f[l] for f in flags["P1"]]).mean(0) for l in Ls}
for l in Ls:
    u = DP1[l] / DP1[l].norm()
    wP1 = np.mean([float(f[l] @ u) for f in flags["P1"]])
    wPO = np.mean([float(f[l] @ u) for f in flags["POST"]])
    cos = float(torch.nn.functional.cosine_similarity(DPOST[l], DP1[l], dim=0))
    print(f"  layer {l:2d} proj P1 {wP1:+.3f} POST {wPO:+.3f} ratio {wPO / wP1:+.2f}  |D_POST| {DPOST[l].norm():.3f} |D_P1| {DP1[l].norm():.3f} cos {cos:+.2f}")
WP1 = sum(np.mean([float(f[l] @ (DP1[l] / DP1[l].norm())) for f in flags["P1"]]) for l in Ls)
WPO = sum(np.mean([float(f[l] @ (DP1[l] / DP1[l].norm())) for f in flags["POST"]]) for l in Ls)
print(f"  omega = W(POST)/W(P1) summed over layers = {WPO / WP1:.3f}")


def addmap(B, T, spec):
    out = {l: torch.zeros(B, T, D) for l in Ls}
    for b, items in enumerate(spec):
        for row, vec, c in items:
            for l in Ls:
                out[l][b, row] += c * vec[l]
    return out


g = torch.Generator().manual_seed(0)
RAND = {l: (lambda v: v / v.norm() * DP1[l].norm())(torch.randn(D, generator=g)) for l in Ls}


@torch.no_grad()
def eval_story(d):
    names = ["none", "addX_P1", "addX_POST", "move_P1", "rand", "KX", "KS"]
    sp = {"none": [], "addX_P1": [(d["rowX"], DP1, 1.0)], "addX_POST": [(d["rowX"], DPOST, 1.0)],
          "move_P1": [(d["rowX"], DP1, 1.0), (d["rowB"], DP1, -1.0)], "rand": [(d["rowX"], RAND, 1.0)], "KX": [], "KS": []}
    B, T = len(names), d["T"]
    kx, ksx = ks_of(model, d["ids"]["X"], d["p"]), ks_of(model, d["ids"]["S"], d["p"])
    tab = {(l, "k"): torch.stack([kx[l] if n == "KX" else ksx[l] for n in names])[:, None] for l in range(nL)}
    pr = torch.tensor([n in ("KX", "KS") for n in names])
    inj.add, inj.active = addmap(B, T, [sp[n] for n in names]), True
    try:
        with clamp_kv(model, [d["p"]], tab, range(nL), "k", per_row=pr):
            lp = lp_last(model, d["ids"]["B"].expand(B, -1))
    finally:
        inj.active, inj.add = False, {}
    c = cand_lp(lp)
    return {n: c[i].tolist() for i, n in enumerate(names)}, float(c[0].exp().sum())


for arm in ("POST", "P1"):
    rows = []
    for core in E:
        d = prep(tok, core, arm)
        r, mass = eval_story(d)
        rows.append(dict(iB=d["iB"], iS=d["iS"], iX=d["iX"], r=r, mass=mass))
    names = list(rows[0]["r"])
    lX = lambda n: np.array([x["r"][n][x["iX"]] - np.log(np.exp(x["r"][n]).sum()) for x in rows])  # noqa: E731
    mX = lambda n: np.array([x["r"][n][x["iX"]] - x["r"][n][x["iB"]] for x in rows])  # noqa: E731
    am = lambda n: np.array([int(np.argmax(x["r"][n])) for x in rows])  # noqa: E731
    iB = np.array([x["iB"] for x in rows]); iX = np.array([x["iX"] for x in rows]); iS = np.array([x["iS"] for x in rows])
    cS = lambda n: np.array([x["r"][n][x["iS"]] - x["r"][n][x["iX"]] for x in rows])  # noqa: E731
    idK = 0.5 * ((cS("KS") - cS("none")) - (cS("KX") - cS("none")))
    print(f"{arm}: case-marg cand mass {np.mean([x['mass'] for x in rows]):.2f}; acc B {np.mean(am('none') == iB):.2f}; ID_K(case-marg) {idK.mean():+.2f} (sd {idK.std():.2f})")
    for n in names:
        print(f"   {n:10s} dm_X {np.mean(mX(n) - mX('none')):+6.2f}  dl_X(renorm) {np.mean(lX(n) - lX('none')):+6.2f}  argmax X {np.mean(am(n) == iX):.2f} B {np.mean(am(n) == iB):.2f}")
log("done")
