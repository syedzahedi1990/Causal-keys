"""Log-odds (S vs T) decomposition of Paper 1's fixed-value study; key share of the MC answer.

margin = logp(S) - logp(T) over the six candidate log scores, direct view, distinct B/S/T primary cores.
Key share s_K = (m[K_S+V_T] - m[K_T+V_T]) / (m[endpoint S] - m[endpoint T]).
"""
import gzip, json, sys, statistics as st
from pathlib import Path

ROOT = Path(sys.argv[1])
LOCS = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]
PS = [1, 0, 3, 2, 5, 4]
for study in ["fixed_value_qwen", "fixed_value_mistral"]:
    meta = json.load(open(ROOT / f"data/mechanism/{study}.json"))
    F = meta["row_fields"]
    rows = [dict(zip(F, json.loads(l))) for l in gzip.open(ROOT / f"data/mechanism/{study}.jsonl.gz")]
    full = {c["id"]: c for c in json.load(open(ROOT / "gpu/component_data/fixed_value_72.json"))["stories"]}
    cores = [full[c["id"]] for c in meta["cores"]]
    vals = {}
    for r in rows:
        if r["view"] != 0 or r["scores"] is None:
            continue
        c = cores[r["core"]]
        S, B = c["source"], c["base"]
        T = LOCS[PS[LOCS.index(S)]]
        if len({S, B, T}) < 3:
            continue
        m = r["scores"][LOCS.index(S)] - r["scores"][LOCS.index(T)]
        key = r["endpoint"] if r["endpoint"] else f"{r['frame']}:{r['condition']}"
        vals.setdefault(key, []).append(m)
    mean = {k: st.mean(v) for k, v in vals.items()}
    print(f"\n{study}: mean logp(S)-logp(T), n per cell shown")
    for k in ["S", "T", "B", "P", "M"] + [f"{f}:{c}" for f in "PM" for c in ["K_P__V_T", "K_M__V_T", "K_S__V_T", "K_T__V_T", "K_NULL__V_T"]]:
        if k in mean:
            print(f"  {k:16s} {mean[k]:+7.2f}  (n={len(vals[k])})")
    span = mean["S"] - mean["T"]
    for f in "PM":
        sK = (mean[f"{f}:K_S__V_T"] - mean[f"{f}:K_T__V_T"]) / span
        kM = (mean[f"{f}:K_P__V_T"] - mean[f"{f}:K_M__V_T"]) / span
        print(f"  frame {f}: key share s_K = {sK:.3f};  (K_P - K_M)/(S - T) = {kM:.3f}")
