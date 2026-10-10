"""Pilot (0.5B, CPU FP32): IOI INLINE flag (fit with the belief H* at the listed IO_S / IO_B rows) and the belief P1 flag,
injected at the listed IO_X row; outcome lp(IO_X) renormalised over the four names, against K_X at p and a random
direction of equal norm."""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flagkit import Inject, OCap, head_out  # noqa: E402
from ckeys import ioi  # noqa: E402
from ckeys.clamp import clamp_kv  # noqa: E402
from ckeys.interventions import capture, edits  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "4")))
MODEL, NF, NE = "Qwen/Qwen2.5-0.5B-Instruct", int(sys.argv[1]), int(sys.argv[2])
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
cfg = model.config
nL, H, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
hd = D // H
Hs = [tuple(c) for c in json.load(open("pilot05.json"))["Hs"]]
DB = torch.load("pilot05_delta.pt")
Ls = sorted(DB)
byL = {}
for l, h in Hs:
    byL.setdefault(l, []).append(h)
oc, inj = OCap(model), Inject(model)


def keys(ids, p):
    with capture(model, range(nL), "k") as K:
        model(ids, use_cache=False, logits_to_keep=1)
    return {l: K[l][0, p].clone() for l in range(nL)}


def prep(core):
    ids = ioi.encode_runs(tok, core, "INLINE", chat=False)
    if ids is None:
        return None
    ib = ids["B"]
    p = (ib[0] != ids["S"][0]).nonzero().flatten().tolist()[0]
    tid = ioi.name_ids(tok, core)
    text = ioi.raw_prompt("INLINE", core, core["io_b"], False)
    par = ioi.parenthetical(core)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    c0 = text.index(par)
    G = {k: i for i, (s, e) in enumerate(enc.offset_mapping) if e > c0 and s < c0 + len(par)
         for k, v in tid.items() if ib[0, i].item() == v}
    return dict(ids=ids, p=p, tid=tid, G=G, T=ib.shape[1])


flags = []
for core in ioi.make_cores(NF, random.Random(6)):
    d = prep(core)
    if d is None:
        continue
    ks = keys(d["ids"]["S"], d["p"])
    oc.rows, oc.active = [d["G"]["S"], d["G"]["B"]], True
    with torch.no_grad():
        model(d["ids"]["B"], use_cache=False, logits_to_keep=1)
        zb = {l: oc.store[l][0] for l in Ls}
        with edits(model, [(l, "k", [d["p"]], (lambda h, l=l: ks[l].expand_as(h))) for l in range(nL)]):
            model(d["ids"]["B"], use_cache=False, logits_to_keep=1)
        zf = {l: oc.store[l][0] for l in Ls}
    oc.active = False
    flags.append({l: 0.5 * (head_out(model, l, zf[l][0] - zb[l][0], byL[l], hd) + head_out(model, l, zb[l][1] - zf[l][1], byL[l], hd)) for l in Ls})
DI = {l: torch.stack([f[l] for f in flags]).mean(0) for l in Ls}
for l in Ls:
    print(f"  layer {l:2d} |D_ioi| {DI[l].norm():.3f} |D_belief| {DB[l].norm():.3f} cos {float(torch.nn.functional.cosine_similarity(DI[l], DB[l], dim=0)):+.2f}")
g = torch.Generator().manual_seed(0)
RAND = {l: (lambda v: v / v.norm() * DI[l].norm())(torch.randn(D, generator=g)) for l in Ls}
names = ["none", "addX_ioi", "addX_belief", "rand", "KX"]
res = []
for core in ioi.make_cores(NE, random.Random(5)):
    d = prep(core)
    if d is None:
        continue
    kx = keys(d["ids"]["X"], d["p"])
    B, T, rx = len(names), d["T"], d["G"]["X"]
    add = {l: torch.zeros(B, T, D) for l in Ls}
    for l in Ls:
        add[l][1, rx], add[l][2, rx], add[l][3, rx] = DI[l], DB[l], RAND[l]
    inj.add, inj.active = add, True
    tab = {(l, "k"): kx[l][None, None].expand(B, 1, -1) for l in range(nL)}
    try:
        with torch.no_grad(), clamp_kv(model, [d["p"]], tab, range(nL), "k", per_row=torch.tensor([n == "KX" for n in names])):
            lp = torch.log_softmax(model(d["ids"]["B"].expand(B, -1), use_cache=False, logits_to_keep=1).logits[:, -1].float(), -1)
    finally:
        inj.active, inj.add = False, {}
    four = lp[:, [d["tid"][k] for k in ("B", "S", "X", "Subj")]]
    lX = (four[:, 2] - torch.logsumexp(four, -1)).numpy()
    res.append(lX - lX[0])
R = np.array(res)
print(f"[{time.time() - t0:.0f}s] n={len(R)}")
for i, n in enumerate(names):
    print(f"   {n:12s} dl(IO_X, 4-way) {R[:, i].mean():+.3f} (se {R[:, i].std() / np.sqrt(len(R)):.3f})")
