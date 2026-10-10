"""Mini-pilot (disclosed): NO-MENTION, multi-token e_S. Rows ID, K_S, V_S, KV_S teacher-forced on ' ' + e_S.
d_C(dec) and d_C(cont) = change vs ID in the decision-token log-prob and in the summed log-prob of the tokens after it;
interaction share = (d_KV - d_K - d_V) / d_KV."""
import json
import random
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA")
from ckeys.clamp import capture_kv, clamp_kv, stack_rows  # noqa: E402
from ckeys.interventions import blocks  # noqa: E402
from pilot_fast import window  # noqa: E402
from pilot_natural import cont_ids, encode  # noqa: E402

fmt = sys.argv[2] if len(sys.argv) > 2 else "NOM"
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32).eval()
nL = len(blocks(model))
valid = set(json.load(open("valid_ft.json")))
items = [window(i) for i in json.load(open("items_ft.json")) if i["id"] in valid and i["sub"] in ("PERSON", "ORG", "PLACE")]
items = [i for i in items if len(i["context"]) <= 300]
random.Random(5).shuffle(items)
ROWS = [("B", "B"), ("S", "B"), ("B", "S"), ("S", "S")]
out = []
with torch.no_grad():
    for it in items:
        if len(out) >= int(sys.argv[1]):
            break
        enc = {Y: encode(tok, it, fmt, e) for Y, e in (("B", it["answer"]), ("S", it["S"]))}
        tb, ib, ob, (c0, c1) = enc["B"]
        P = [i for i, (s, e) in enumerate(ob) if e > c0 and s < c1]
        if len(enc["S"][1]) != len(ib):
            continue
        c, k = cont_ids(tok, tb, ib, it["S"])
        if len(c) - k < 2:
            continue
        kv = {}
        for Y in ("B", "S"):
            with capture_kv(model, P, range(nL)) as t:
                model(torch.tensor([enc[Y][1]]), use_cache=False)
            kv[Y] = {kk: v[0] for kk, v in t.items()}
        tabs = stack_rows(kv, [lambda l, ch, r=r: r[0] if ch == "k" else r[1] for r in ROWS], range(nL))
        x = torch.tensor([ib + c]).expand(len(ROWS), -1)
        with clamp_kv(model, P, tabs, range(nL)):
            lg = model(x, use_cache=False).logits[:, len(ib) - 1:-1].float().log_softmax(-1)
        tl = lg[:, torch.arange(len(c)), torch.tensor(c)]
        dec, cont = tl[:, k], tl[:, k + 1:].sum(1)
        out.append(dict(id=it["id"], S=it["S"], n_cont=len(c) - k - 1, dec=(dec - dec[0]).tolist(), cont=(cont - cont[0]).tolist()))
        print(it["S"], "dec dK dV dKV", [round(v, 2) for v in out[-1]["dec"][1:]], "cont", [round(v, 2) for v in out[-1]["cont"][1:]], flush=True)
D = np.array([o["dec"] for o in out])
C = np.array([o["cont"] for o in out])
for name, M in (("decision", D), ("continuation", C)):
    dK, dV, dKV = M[:, 1].mean(), M[:, 2].mean(), M[:, 3].mean()
    print(f"{fmt} {name}: n={len(M)} dK={dK:+.2f} dV={dV:+.2f} dKV={dKV:+.2f} interaction share={(dKV - dK - dV) / dKV:+.2f}")
json.dump(out, open(f"pilot_cont_{fmt}.json", "w"))
