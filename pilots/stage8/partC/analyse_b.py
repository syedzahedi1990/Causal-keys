import json
import sys

import numpy as np

L = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]
d = json.load(open(sys.argv[1]))
items = d["items"]
depths = [int(x) for x in d["args"]["depths"].split(",")]
arms = d["args"]["arms"].split(",")
print("n items", len(items), "max exact-check", max(it["arms"][a]["exact_check"] for it in items for a in arms))


def idc(r, fam, l, mode, iS, iX):
    sfx = "" if mode == "KV" else "|" + mode
    eS, eX = r[f"{fam}_S@{l}{sfx}"], r[f"{fam}_X@{l}{sfx}"]
    return 0.5 * (eS[iS] - eX[iS] + eX[iX] - eS[iX])


for sc in ("marg", "lower"):
    print(f"\n===== scorer {sc}")
    for arm in arms:
        print(f"--- {arm}")
        for l in depths:
            line = f"l={l:2d} (onset {l + 1:2d}) "
            for fam in ("T", "CI", "CO"):
                K, V, KV = [], [], []
                for it in items:
                    r = it["arms"][arm][sc]
                    iS, iX = L.index(it["core"]["source"]), L.index(it["X"])
                    K.append(idc(r, fam, l, "K", iS, iX))
                    V.append(idc(r, fam, l, "V", iS, iX))
                    KV.append(idc(r, fam, l, "KV", iS, iX))
                k, v, kv = np.mean(K), np.mean(V), np.mean(KV)
                line += f"| {fam} kID {k / (k + v):+.2f} (K {k:5.2f} V {v:5.2f} KV {kv:5.2f}) "
            pk, pv, ph = [], [], []
            for it in items:
                r = it["arms"][arm][sc]
                c = it["core"]
                iB, iI = L.index(c["base"]), L.index(c["initial"])
                s0 = r["self"]
                m = lambda x: x[iI] - x[iB]
                ph.append(m(r[f"BIND@{l}"]) - m(s0))
                pk.append(m(r[f"BIND@{l}|K"]) - m(s0))
                pv.append(m(r[f"BIND@{l}|V"]) - m(s0))
            P = np.mean(ph)
            line += f"| BIND Phi {P:+5.2f} psiK {np.mean(pk) / P:+.2f} psiV {np.mean(pv) / P:+.2f}"
            print(line)

print("\n===== story-level discrimination D = mean|ID_K^E - ID_K^T| / mean ID_KV^T (scorer marg)")
for arm in arms:
    for l in depths:
        out = []
        for fam in ("CI", "CO"):
            dd, base = [], []
            for it in items:
                r = it["arms"][arm]["marg"]
                iS, iX = L.index(it["core"]["source"]), L.index(it["X"])
                dd.append(abs(idc(r, fam, l, "K", iS, iX) - idc(r, "T", l, "K", iS, iX)))
                base.append(idc(r, "T", l, "KV", iS, iX))
            out.append(f"{fam} D {np.mean(dd) / np.mean(base):.3f}")
        print(f"{arm:6s} l={l:2d} " + "  ".join(out))
