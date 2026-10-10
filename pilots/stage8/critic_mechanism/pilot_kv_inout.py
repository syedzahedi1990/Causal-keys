"""Critic pilot (CPU FP32): ID_K and ID_V under Q_IN / Q_OUT (and P1) at a small model, with a non-candidate key K_N.
Rows: self, K_S, K_X, V_S, V_X, K_N (key of the story whose event word is a non-candidate, 'garage' or the first
single-token alternative), all from layer 0, six-way renormalised and raw lowercase log-probs.
Question: does the value channel keep a positive (content) sign when the question asks 'not mentioned', while the key
channel flips? And does K_N act like removing B's flag (no specific beneficiary)?"""
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PD = "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partD"
sys.path.insert(0, PD)
from flagkit import lp_last, prep, raw  # noqa: E402
from ckeys.clamp import capture_kv, clamp_kv  # noqa: E402
from ckeys.story import make_cores, record  # noqa: E402
from experiments.row_restricted_keys import encode_with_offsets  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "2")))
MODEL, N, ARMS = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(",")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
nL = model.config.num_hidden_layers
NONCAND = [w for w in ("garage", "kitchen", "attic", "table", "bucket", "pocket") if len(tok.encode(" " + w)) == 1][0]
print("non-candidate word:", NONCAND)


def kv_of(ids, p):
    with torch.no_grad(), capture_kv(model, [p], range(nL), "kv") as st:
        model(ids, use_cache=False, logits_to_keep=1)
    return {k: v[0, 0].clone() for k, v in st.items()}


E = make_cores(N, random.Random(9))
for arm in ARMS:
    res = []
    for core in E:
        d = prep(tok, core, arm)
        p = d["p"]
        r = record(core, "direct", NONCAND)
        _, idsN, _ = encode_with_offsets(tok, raw(arm, r["story"], r["query"]))
        assert idsN.shape == d["ids"]["B"].shape and int((idsN[0] != d["ids"]["B"][0]).sum()) == 1
        S, X, Nn = kv_of(d["ids"]["S"], p), kv_of(d["ids"]["X"], p), kv_of(idsN, p)
        names = ["self", "K_S", "K_X", "V_S", "V_X", "K_N"]
        src = {"self": S, "K_S": S, "K_X": X, "V_S": S, "V_X": X, "K_N": Nn}
        tab = {(l, ch): torch.stack([src[n][(l, ch)] for n in names])[:, None] for l in range(nL) for ch in "kv"}
        prk = torch.tensor([n.startswith("K") for n in names])
        prv = torch.tensor([n.startswith("V") for n in names])
        with torch.no_grad(), clamp_kv(model, [p], {k: v for k, v in tab.items() if k[1] == "k"}, range(nL), "k", per_row=prk), \
                clamp_kv(model, [p], {k: v for k, v in tab.items() if k[1] == "v"}, range(nL), "v", per_row=prv):
            lp = lp_last(model, d["ids"]["B"].expand(len(names), -1))[:, d["cid"]]
        ln = lp - torch.logsumexp(lp, -1, keepdim=True)
        res.append((d, {n: ln[i].numpy() for i, n in enumerate(names)}, float(lp[0].exp().sum())))
    c = lambda n: np.array([x[1][n][x[0]["iS"]] - x[1][n][x[0]["iX"]] for x in res])  # noqa: E731
    idK = 0.5 * ((c("K_S") - c("self")) - (c("K_X") - c("self")))
    idV = 0.5 * ((c("V_S") - c("self")) - (c("V_X") - c("self")))
    se = lambda a: a.std() / np.sqrt(len(a))  # noqa: E731
    dB_N = np.array([x[1]["K_N"][x[0]["iB"]] - x[1]["self"][x[0]["iB"]] for x in res])
    dS_N = np.array([x[1]["K_N"][x[0]["iS"]] - x[1]["self"][x[0]["iS"]] for x in res])
    dX_N = np.array([x[1]["K_N"][x[0]["iX"]] - x[1]["self"][x[0]["iX"]] for x in res])
    dB_S = np.array([x[1]["K_S"][x[0]["iB"]] - x[1]["self"][x[0]["iB"]] for x in res])
    mass = np.mean([x[2] for x in res])
    print(f"[{time.time() - t0:5.0f}s] {MODEL.split('/')[-1]} {arm} n={N} mass {mass:.2f}: ID_K {idK.mean():+.2f} (se {se(idK):.2f}, frac>0 {np.mean(idK > 0):.2f}); "
          f"ID_V {idV.mean():+.2f} (se {se(idV):.2f}, frac>0 {np.mean(idV > 0):.2f})", flush=True)
    print(f"      K_N: dl_B {dB_N.mean():+.2f} (K_S: {dB_S.mean():+.2f}); dl_S {dS_N.mean():+.2f}; dl_X {dX_N.mean():+.2f}; "
          f"|dl_S - dl_X| {np.abs(dS_N - dX_N).mean():.2f}", flush=True)
