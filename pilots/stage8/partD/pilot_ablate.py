"""Pilot (0.5B, CPU FP32) of the directional ablation: at the layers of H*, at the six option rows, the attention output's
component along the unit flag direction is set to its mean over unflagged option rows (words absent from the story) on
the fit stories; random unit directions as controls. ID_K / ID_V from format_factorial.run_item (13 rows + 3 clean)."""
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flagkit import Inject, prep  # noqa: E402
import experiments.format_factorial as ff  # noqa: E402
from ckeys.interventions import blocks  # noqa: E402
from ckeys.story import LOCATIONS, make_cores  # noqa: E402

torch.set_num_threads(int(os.environ.get("NT", "4")))
MODEL, NF, NE = "Qwen/Qwen2.5-0.5B-Instruct", int(sys.argv[1]), int(sys.argv[2])
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
D = model.config.hidden_size
DP1 = torch.load("pilot05_delta.pt")
Ls = sorted(DP1)
U = {l: DP1[l] / DP1[l].norm() for l in Ls}
g = torch.Generator().manual_seed(1)
RU = [{l: (lambda v: v / v.norm())(torch.randn(D, generator=g)) for l in Ls} for _ in range(2)]
inj = Inject(model)
store = {}
hk = [blocks(model)[l].self_attn.o_proj.register_forward_hook(lambda _m, _i, o, l=l: store.__setitem__(l, o.detach())) for l in Ls]
dirs = {"flag": U, "rand0": RU[0], "rand1": RU[1]}
MU = {k: {l: [] for l in Ls} for k in dirs}
for core in make_cores(NF, random.Random(0)):
    d = prep(tok, core, "P1")
    absent = [d["G"][i] for i, w in enumerate(LOCATIONS) if w not in (core["base"], core["initial"], core["distractor_location"])]
    with torch.no_grad():
        model(d["ids"]["B"], use_cache=False, logits_to_keep=1)
    for k, u in dirs.items():
        for l in Ls:
            MU[k][l] += (store[l][0, absent] @ u[l]).tolist()
for h in hk:
    h.remove()
MU = {k: {l: float(np.mean(v)) for l, v in m.items()} for k, m in MU.items()}
print(f"[{time.time() - t0:.0f}s] mu fitted")
res = {k: [] for k in ["none"] + list(dirs)}
for core in make_cores(NE, random.Random(1)):
    d = prep(tok, core, "P1")
    for k in res:
        if k != "none":
            rm = torch.zeros(1, d["T"], dtype=torch.bool)
            rm[0, d["G"]] = True
            inj.proj, inj.add, inj.active = {l: (dirs[k][l], MU[k][l], rm) for l in Ls}, {}, True
        try:
            r = ff.run_item(model, tok, core, "P1", "direct", "cpu")
        finally:
            inj.active, inj.proj = False, {}
        m, idr = r["m"], r["m"]["ID@0"]
        dl = {kk: {t: m[kk]["lp"][t] - idr["lp"][t] for t in ("S", "B", "X")} for kk in m}
        idK = 0.5 * ((dl["K_S@0"]["S"] - dl["K_X@0"]["S"]) + (dl["K_X@0"]["X"] - dl["K_S@0"]["X"]))
        idV = 0.5 * ((dl["V_S@0"]["S"] - dl["V_X@0"]["S"]) + (dl["V_X@0"]["X"] - dl["V_S@0"]["X"]))
        res[k].append((idK, idV, r["clean"]["B"]["argmax_cand"] == core["base"]))
print(f"[{time.time() - t0:.0f}s] n={NE}")
base = np.array([x[0] for x in res["none"]])
for k, v in res.items():
    a = np.array(v, float)
    print(f"   {k:6s} ID_K {a[:, 0].mean():+.2f}  rho_K {a[:, 0].mean() / base.mean():+.2f}  ID_V {a[:, 1].mean():+.2f}  base argmax {a[:, 2].mean():.2f}")
