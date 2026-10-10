import math
# identity contrasts (ID_K, ID_V) at l0=0 from committed stage-1 summaries (seed-0, n=150)
nat = {"Qwen2.5-7B": {"LETTER": (20.38, 0.22), "P1": (21.16, 5.76), "POST": (5.52, 11.91), "NONE": (1.04, 16.60)},
       "Mistral-7B": {"LETTER": (15.18, 0.97), "P1": (14.91, 2.30), "POST": None, "NONE": (0.55, 13.65)}}
import sys
post_m = tuple(float(x) for x in sys.argv[1:3]) if len(sys.argv) > 2 else None
if post_m: nat["Mistral-7B"]["POST"] = post_m
for m, d in nat.items():
    print("==", m)
    for r in (0.33, 0.5, 0.67, 1.5, 2.0, 3.0):
        gaps = {}
        for f, kv in d.items():
            if kv is None: continue
            k, v = kv
            s = k / (k + v); e = r * k / (r * k + v)
            gaps[f] = e - s
        mad = sum(abs(g) for g in gaps.values()) / len(gaps)
        mx = max(abs(g) for g in gaps.values())
        cross = (r*d["LETTER"][0]/(r*d["LETTER"][0]+d["LETTER"][1])) - (r*d["NONE"][0]/(r*d["NONE"][0]+d["NONE"][1]))
        print(f" c_K/c_V={r:4.2f} gaps " + " ".join(f"{f}:{g:+.3f}" for f, g in gaps.items()) +
              f" | MAD {mad:.3f} max {mx:.3f} -> JC3 {'PASS' if mad <= 0.12 and mx <= 0.25 else 'fail'} | JC2 crossover {cross:.2f}")
