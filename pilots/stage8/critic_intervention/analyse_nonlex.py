import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
FAMS = ["T", "CO", "SYN", "FR", "CI", "CIperp"]
ARMS = ["NONE", "P1", "LETTER"]
depths = sorted({int(n.split("@")[1]) for n in d[0]["arms"]["NONE"]["names"][1:]})
print(f"n stories {len(d)}")
ID = {}   # (arm, fam, l) -> list per story
FLIP = {}
for it in d:
    for arm in ARMS:
        a = it["arms"][arm]
        sc = np.array(a["scores"])
        nm = a["names"]
        iS, iX = a["iS"], a["iX"]
        self_ = sc[0]
        for l in depths:
            for fam in FAMS:
                rS = sc[nm.index(f"{fam}_S@{l}")] - self_
                rX = sc[nm.index(f"{fam}_X@{l}")] - self_
                idv = 0.5 * (rS[iS] - rX[iS] + rX[iX] - rS[iX])
                ID.setdefault((arm, fam, l), []).append(idv)
                fS = float(np.argmax(sc[nm.index(f"{fam}_S@{l}")]) == iS)
                fX = float(np.argmax(sc[nm.index(f"{fam}_X@{l}")]) == iX)
                FLIP.setdefault((arm, fam, l), []).extend([fS, fX])
for l in depths:
    print(f"\n=== block {l}")
    cos = {fam: np.mean([it["cos"][f"{fam}_{t}@{l}"][0] for it in d for t in "SX"]) for fam in FAMS}
    nr = {fam: np.mean([it["cos"][f"{fam}_{t}@{l}"][1] for it in d for t in "SX"]) for fam in FAMS}
    for arm in ARMS:
        nat = np.mean(ID[(arm, "T", l)])
        s = f"{arm:6s} nat ID {nat:6.2f} |"
        for fam in FAMS[1:]:
            phi = np.mean(ID[(arm, fam, l)]) / nat
            s += f" {fam} phi {phi:5.2f} fl {np.mean(FLIP[(arm, fam, l)]):.2f} |"
        print(s + f" T flip {np.mean(FLIP[(arm, 'T', l)]):.2f}")
    s = "transfer phi(LETTER)/phi(NONE), phi(P1)/phi(NONE): "
    for fam in FAMS[1:]:
        pn = np.mean(ID[("NONE", fam, l)]) / np.mean(ID[("NONE", "T", l)])
        pl = np.mean(ID[("LETTER", fam, l)]) / np.mean(ID[("LETTER", "T", l)])
        pp = np.mean(ID[("P1", fam, l)]) / np.mean(ID[("P1", "T", l)])
        s += f"{fam} {pl / pn:5.2f},{pp / pn:5.2f} | "
    print(s)
    print("resid cos with natural: " + " ".join(f"{f} {cos[f]:.2f} (norm x{nr[f]:.2f})" for f in FAMS[1:]))
