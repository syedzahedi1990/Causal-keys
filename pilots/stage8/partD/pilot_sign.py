"""Competence and the sign of ID_K under the IN / OUT questions (and P1) at a small model; no injection.
Rows per story: self (none), K_S, K_X at p from layer 0. Lowercase candidate ids (list formats)."""
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flagkit import ks_of, lp_last, prep  # noqa: E402
from ckeys.clamp import clamp_kv  # noqa: E402
from ckeys.story import make_cores  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "2")))
MODEL, N, ARMS = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
nL = model.config.num_hidden_layers
E = make_cores(N, random.Random(1))
for arm in ARMS:
    out = []
    for core in E:
        d = prep(tok, core, arm)
        ks, kx = ks_of(model, d["ids"]["S"], d["p"]), ks_of(model, d["ids"]["X"], d["p"])
        tab = {(l, "k"): torch.stack([ks[l], ks[l], kx[l]])[:, None] for l in range(nL)}
        with torch.no_grad(), clamp_kv(model, [d["p"]], tab, range(nL), "k", per_row=torch.tensor([False, True, True])):
            lp = lp_last(model, d["ids"]["B"].expand(3, -1))[:, d["cid"]]
        out.append((d, lp))
    c = lambda i: np.array([float(lp[i, d["iS"]] - lp[i, d["iX"]]) for d, lp in out])  # noqa: E731
    idK = 0.5 * ((c(1) - c(0)) - (c(2) - c(0)))
    am = np.array([int(lp[0].argmax()) for _, lp in out])
    ment = np.mean([a in (d["iB"], d["iI"], d["iD"]) for a, (d, _) in zip(am, out)])
    mass = np.mean([float(lp[0].exp().sum()) for _, lp in out])
    dB = np.mean([float(lp[1, d["iB"]] - lp[0, d["iB"]]) for d, lp in out])
    amKS_B = np.mean([int(lp[1].argmax()) == d["iB"] for d, lp in out])
    amKS_S = np.mean([int(lp[1].argmax()) == d["iS"] for d, lp in out])
    print(f"[{time.time() - t0:5.0f}s] {MODEL.split('/')[-1]} {arm}: n={N} mass {mass:.2f} clean argmax mentioned {ment:.2f}; "
          f"ID_K {idK.mean():+.2f} [se {idK.std() / np.sqrt(N):.2f}], frac<0 {np.mean(idK < 0):.2f}; dlp(B) under K_S {dB:+.2f}; "
          f"argmax under K_S: B {amKS_B:.2f} S {amKS_S:.2f}", flush=True)
