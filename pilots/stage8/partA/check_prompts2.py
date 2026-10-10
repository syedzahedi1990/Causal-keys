"""Tokenizer-only check (no model): for every item, format and tokenizer, the chat-wrapped B/S/X prompts have equal length
and differ only inside the entity span P; the continuation of ' ' + e_Y after the 'Answer:' prefill is prefix-stable;
decision tokens (first content token) of B, S, X, D are pairwise distinct; prompt length statistics."""
import collections
import json
import statistics
import sys

from transformers import AutoTokenizer

sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA")
from pilot_natural import cont_ids, encode  # noqa: E402

TOK = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
       "llama": "unsloth/Meta-Llama-3.1-8B-Instruct", "gemma": "unsloth/gemma-2-9b-it"}
FMTS = ["NOM", "OPTA", "OPTB", "MENA", "MENB", "LETA"]
items = json.load(open(sys.argv[1]))
toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOK.items()}
bad = collections.Counter()
lens = collections.defaultdict(list)
spanlen = collections.defaultdict(list)
ok_items = collections.defaultdict(set)
for it in items:
    for k, tok in toks.items():
        for f in FMTS:
            try:
                enc = {Y: encode(tok, it, f, e) for Y, e in (("B", it["answer"]), ("S", it["S"]), ("X", it["X"]), ("Y", it["Y"]))}
            except Exception as e:  # noqa: BLE001
                bad[(k, f, "encode:" + type(e).__name__)] += 1
                continue
            tb, ib, ob, (c0, c1) = enc["B"]
            P = [i for i, (s, e) in enumerate(ob) if e > c0 and s < c1]
            good = True
            for Y in ("S", "X", "Y"):
                iy = enc[Y][1]
                if len(iy) != len(ib):
                    good = False
                    bad[(k, f, "len")] += 1
                    break
                diff = [i for i, (x, y) in enumerate(zip(ib, iy)) if x != y]
                if not diff or not set(diff) <= set(P):
                    good = False
                    bad[(k, f, "diff")] += 1
                    break
            if good and f != "LETA":
                try:
                    decs = [cont_ids(tok, tb, ib, e) for e in (it["answer"], it["S"], it["X"], it["Y"], it["D"])]
                    pre = {tuple(c[:j]) for c, j in decs}
                    ds = [c[j] for c, j in decs]
                    if len(set(ds)) != 5 or len(pre) != 1:
                        good = False
                        bad[(k, f, "dec")] += 1
                except AssertionError:
                    good = False
                    bad[(k, f, "prefix")] += 1
            if good:
                ok_items[(k, f)].add(it["id"])
                lens[(k, f)].append(len(ib))
                spanlen[k].append(len(P))
for key, v in sorted(bad.items()):
    print("bad", key, v)
allok = None
for (k, f), s in sorted(ok_items.items()):
    print(f"{k:8s} {f:5s} ok {len(s):4d}  prompt tokens median {statistics.median(lens[(k, f)]):.0f} max {max(lens[(k, f)])}")
    allok = s if allok is None else allok & s
print("items valid in every tokenizer and format:", len(allok))
for k in toks:
    c = collections.Counter(spanlen[k])
    print(k, "span lengths", sorted(c.items())[:8])
by = collections.Counter(i["sub"] for i in items if i["id"] in allok)
print("by subtype", by, "articles", len({i['art'] for i in items if i['id'] in allok}))
json.dump(sorted(allok), open(sys.argv[2], "w"))
