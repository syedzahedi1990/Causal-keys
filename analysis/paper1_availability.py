"""Retrospective H3 test on Paper 1's released per-story outputs (no GPU).

Question: does the learned remap M (and key addition K_M -> P) succeed only when the remapped
target T = pi(S) is already mentioned elsewhere in the base story (pointer / re-addressing account)?
Base story tokens mention exactly three locations: the object's initial location, the distractor
object's location, and the critical (base) location B.
"""
import gzip, json, sys, collections
from pathlib import Path

ROOT = Path(sys.argv[1])
LOCS = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]
MAPS = {"m1": [1, 2, 3, 4, 5, 0], "m3": [1, 0, 3, 2, 5, 4]}  # six-cycle; pair-swap (paper's m2)


def avail_class(core, T):
    if T == core["base"]:
        return "T=B (at critical token)"
    if T == core["initial"]:
        return "T=initial (elsewhere)"
    if T == core["distractor_location"]:
        return "T=distractor loc (elsewhere)"
    return "T absent from context"


def load(study):
    meta = json.load(open(ROOT / f"data/mechanism/{study}.json"))
    F = meta["row_fields"]
    tok2loc = {v: k for k, v in meta["alphabets"]["locations"].items()}
    rows = [dict(zip(F, json.loads(l))) for l in gzip.open(ROOT / f"data/mechanism/{study}.jsonl.gz")]
    return meta, rows, tok2loc


def analyse(study):
    meta, rows, tok2loc = load(study)
    cores = meta["cores"]
    end = {}  # (core, mapping, seed, endpoint, view) -> loc
    exch = {}
    for r in rows:
        loc = tok2loc.get(r["global_token_id"], "INVALID")
        if r["endpoint"] is not None:
            end[(r["core"], r["mapping"], r["seed"], r["endpoint"], r["view"])] = loc
        elif r["mask"] == "full":
            exch[(r["core"], r["mapping"], r["seed"], r["frame"], r["condition"], r["view"])] = loc
    print(f"\n######## {study}")
    for mp in sorted({r["mapping"] for r in rows}):
        stats = collections.defaultdict(lambda: collections.Counter())
        for ci, c in enumerate(cores):
            S = c["source"]; B = c["base"]
            if S == B:
                continue  # B=S collisions: no transfer opportunity
            T = LOCS[MAPS[mp][LOCS.index(S)]]
            cls = avail_class(c, T)
            for seed in (101, 102, 103):
                m = end.get((ci, mp, seed, "M", 0)); p = end.get((ci, mp, seed, "P", 0))
                if m is None:
                    continue
                st = stats[cls]
                st["n"] += 1
                st["M=T"] += m == T
                st["M=S"] += m == S
                st["M=B"] += m == B
                st["M=initial"] += m == c["initial"]
                st["P=S"] += p == S
                add = exch.get((ci, mp, seed, "P", "other", 0)); nul = exch.get((ci, mp, seed, "P", "null", 0))
                rem = exch.get((ci, mp, seed, "M", "other", 0))
                if add is not None:
                    st["add_n"] += 1
                    st["add=M"] += add == m
                    st["add=T"] += add == T
                    st["null=M"] += nul == m
                    st["rem=P"] += rem == p
        print(f"  mapping {mp}: direct view, B!=S cores x 3 fits")
        hdr = ["class", "n", "M=T", "M=S", "M=B", "M=init", "P=S", "K_M->P: =M", "=T", "null=M", "K_P->M: =P"]
        print("   " + " | ".join(hdr))
        for cls in ["T=B (at critical token)", "T=initial (elsewhere)", "T=distractor loc (elsewhere)", "T absent from context"]:
            st = stats.get(cls)
            if not st:
                continue
            n, a = st["n"], max(st["add_n"], 1)
            print(f"   {cls:30s} {n:4d} | {st['M=T']/n:5.1%} | {st['M=S']/n:5.1%} | {st['M=B']/n:5.1%} | {st['M=initial']/n:5.1%} | {st['P=S']/n:5.1%}"
                  f" | {st['add=M']/a:5.1%} | {st['add=T']/a:5.1%} | {st['null=M']/a:5.1%} | {st['rem=P']/a:5.1%}")


for s in ["native_qwen", "native_mistral", "mapping_qwen", "mapping_mistral"]:
    analyse(s)
