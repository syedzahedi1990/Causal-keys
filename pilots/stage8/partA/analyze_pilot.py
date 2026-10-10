"""Summarise a pilot_fast.py output: competence, emitted-form mass, ID_K / ID_V / s_ID (decision token), flip rates."""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA")
from pilot_natural import matches  # noqa: E402

res = json.load(open(sys.argv[1]))
by = collections.defaultdict(list)
for r in res:
    by[r["fmt"]].append(r)
print("exactness (max |KV_S row - S run|, first 2 items):", [round(r["exact"], 6) for r in res if "exact" in r])


def ids(r):
    lp = r["lp"]
    d = lambda row, Y: lp[row][Y] - lp["ID"][Y]  # noqa: E731
    idk = 0.5 * ((d("K_S", "S") - d("K_X", "S")) + (d("K_X", "X") - d("K_S", "X")))
    idv = 0.5 * ((d("V_S", "S") - d("V_X", "S")) + (d("V_X", "X") - d("V_S", "X")))
    return idk, idv


for f, R in by.items():
    comp = [r for r in R if r["ok_B"] and r["ok_S"]]
    pB = [np.exp(r["lp"]["ID"]["B"]) for r in R]
    print(f"\n{f}: n={len(R)} competent(B&S)={len(comp)}  median p(dec_B | ID)={np.median(pB):.3f}  "
          f"mean cand-mass={np.mean([r['mass']['ID'] for r in R]):.3f}")
    for name, S in (("all", R), ("competent", comp)):
        if not S:
            continue
        k, v = np.array([ids(r) for r in S]).T
        sid = k.mean() / (k.mean() + v.mean()) if (k.mean() + v.mean()) > 0 else float("nan")
        fk = np.mean([matches(r["gen"]["K_S"], r["S"]) for r in S])
        fv = np.mean([matches(r["gen"]["V_S"], r["S"]) for r in S])
        dk = np.mean([r["lp"]["K_S"]["S"] > max(r["lp"]["K_S"][y] for y in "BXD") for r in S])
        dv = np.mean([r["lp"]["V_S"]["S"] > max(r["lp"]["V_S"][y] for y in "BXD") for r in S])
        print(f"   [{name:9s}] n={len(S):2d} ID_K={k.mean():+6.2f} (sd {k.std():.2f})  ID_V={v.mean():+6.2f} (sd {v.std():.2f})  "
              f"s_ID={sid:+.2f}   flip-gen K_S={fk:.2f} V_S={fv:.2f}   flip-dec K_S={dk:.2f} V_S={dv:.2f}")
for r in res[:8]:
    print(r["fmt"], r["ans"], "|", r["S"], "| gens:", {k: v[:25] for k, v in r["gen"].items()})
