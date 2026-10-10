import json
import sys

import numpy as np

L = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]
d = json.load(open(sys.argv[1]))
items = d["items"]
depths = [int(x) for x in d["args"]["depths"].split(",")]
arms = d["args"]["arms"].split(",")
print("n items", len(items))
for sc in ("marg", "lower"):
    print(f"\n===== scorer {sc}")
    for arm in arms:
        mass = np.mean([it["arms"][arm]["mass_" + ("lower" if sc == "lower" else "marg")] for it in items])
        print(f"--- {arm}  clean-B candidate mass {mass:.2f}")
        for l in depths:
            line = f"l={l:2d} "
            idT = None
            for fam in ("T", "CI", "CO"):
                ids, flips = [], []
                for it in items:
                    r = it["arms"][arm][sc]
                    iS, iX = L.index(it["core"]["source"]), L.index(it["X"])
                    eS, eX = r[f"{fam}_S@{l}"], r[f"{fam}_X@{l}"]
                    ids.append(0.5 * (eS[iS] - eX[iS] + eX[iX] - eS[iX]))
                    flips += [int(np.argmax(eS) == iS), int(np.argmax(eX) == iX)]
                m = np.mean(ids)
                if fam == "T":
                    idT = m
                line += f"| {fam} ID_KV {m:6.2f} (x{m / idT:4.2f}) flip {np.mean(flips):.2f} "
            # random and binding
            dR, fR, phiB, fB = [], [], [], []
            for it in items:
                r = it["arms"][arm][sc]
                c = it["core"]
                iS, iB, iI = L.index(c["source"]), L.index(c["base"]), L.index(c["initial"])
                s0, eR, eB = r["self"], r[f"R_S@{l}"], r[f"BIND@{l}"]
                dR.append((eR[iS] - eR[iB]) - (s0[iS] - s0[iB]))
                fR.append(int(np.argmax(eR) == iS))
                phiB.append((eB[iI] - eB[iB]) - (s0[iI] - s0[iB]))
                fB.append(int(np.argmax(eB) == iI))
            line += f"| R dm_SB {np.mean(dR):+5.2f} flip {np.mean(fR):.2f} | BIND Phi {np.mean(phiB):+5.2f} flip->init {np.mean(fB):.2f}"
            print(line)
print("\n===== non-equivalence (NONE arm; identical prefix in every arm)")
arm = arms[0]
for l in depths:
    out = []
    for fam in ("CI", "CO", "T"):
        for key in ("resid", "cos", "normratio", "kv"):
            if fam == "T" and key != "kv":
                continue
            v = [it["arms"][arm]["nu"][f"{fam}_{t}@{l}_{key}"] for it in items for t in ("S", "X")]
            out.append(f"{fam}.{key} {np.median(v):.3f}")
    print(f"l={l:2d} " + "  ".join(out))
