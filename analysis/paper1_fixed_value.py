"""Do clamped critical-token KEYS determine content when VALUES are fixed at natural V_T?

Paper 1 fixed-value study: at the critical token, keys and values in blocks 6-40 (Mistral) /
6-80 (Qwen) are replaced, values fixed to the natural target run's V_T, and the recipient's own
attention output at that token is restored. Rows: frame (P or M recipient) x key condition.
"""
import gzip, json, sys, collections
from pathlib import Path

ROOT = Path(sys.argv[1])
LOCS = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]
PS = [1, 0, 3, 2, 5, 4]
CONDS = ["K_P__V_T", "K_M__V_T", "K_S__V_T", "K_T__V_T", "K_NULL__V_T", "K_B__V_B"]

for study in ["fixed_value_qwen", "fixed_value_mistral"]:
    meta = json.load(open(ROOT / f"data/mechanism/{study}.json"))
    F = meta["row_fields"]
    tok2loc = {v: k for k, v in meta["alphabets"]["locations"].items()}
    rows = [dict(zip(F, json.loads(l))) for l in gzip.open(ROOT / f"data/mechanism/{study}.jsonl.gz")]
    full = {c["id"]: c for c in json.load(open(ROOT / "gpu/component_data/fixed_value_72.json"))["stories"]}
    cores = [full[c["id"]] for c in meta["cores"]]
    print(f"\n######## {study}  (direct view; distinct B,S,T cores; 3 fits)")
    for avail in ["T in context (initial/distractor)", "T absent"]:
        tab = collections.defaultdict(collections.Counter)
        for r in rows:
            if r["view"] != 0:
                continue
            c = cores[r["core"]]
            S, B = c["source"], c["base"]
            T = LOCS[PS[LOCS.index(S)]]
            if len({S, B, T}) < 3:
                continue
            a = "T in context (initial/distractor)" if T in (c["initial"], c["distractor_location"]) else "T absent"
            if a != avail:
                continue
            loc = tok2loc.get(r["global_token_id"], "INVALID")
            lab = "S" if loc == S else "T" if loc == T else "B" if loc == B else "init" if loc == c["initial"] else "other"
            key = f"{r['frame']}:{r['condition']}" if r["endpoint"] is None else f"endpoint {r['endpoint']}"
            tab[key][lab] += 1
        print(f"  -- {avail}")
        order = [f"endpoint {e}" for e in "BSTPM"] + [f"{f}:{c}" for f in "PM" for c in CONDS]
        for k in order:
            if k not in tab:
                continue
            n = sum(tab[k].values())
            print(f"    {k:18s} n={n:4d}  " + "  ".join(f"{l}={tab[k][l]/n:5.1%}" for l in ["S", "T", "B", "init", "other"]))
