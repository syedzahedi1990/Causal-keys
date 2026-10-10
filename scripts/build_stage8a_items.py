"""Build the stage-8 part-A items (preregistration J) from SQuAD v1.1 dev: ckeys.squad_items rules, validity in every
study tokenizer and format (ckeys.natural_formats.check_item), the seeded article split and caps. Writes
data/stage8a_items.json (the committed item file the GPU preflight must reproduce) and prints the counts.

    python scripts/build_stage8a_items.py <dev-v1.1.json> [--out data/stage8a_items.json] [--check]
--check rebuilds and compares with the committed file instead of writing it."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from ckeys.natural_formats import check_item  # noqa: E402
from ckeys.squad_items import build_items, cap, split_articles  # noqa: E402

TOKENIZERS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
              "llama": "unsloth/Meta-Llama-3.1-8B-Instruct", "gemma": "unsloth/gemma-2-9b-it"}
KEEP = ("id", "title", "art", "par", "question", "answer", "start", "type", "sub", "S", "X", "Z", "D", "D_in", "options",
        "stratum")


def build(path, tokenizers=TOKENIZERS):
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in tokenizers.items()}
    items, cnt = build_items(path, toks)
    valid, why = [], collections.Counter()
    for it in items:
        bad = next((f"{k}: {r}" for k, t in toks.items() if (r := check_item(t, it))), None)
        if bad:
            why[bad.split(":")[0] + ":" + bad.split(":")[1]] += 1
        else:
            valid.append(it)
    ft = [i for i in valid if i["stratum"] == "FT"]
    R_titles, E_titles = split_articles(ft)
    R = cap([i for i in ft if i["title"] in R_titles], per_article=8, seed=1)
    E = cap([i for i in ft if i["title"] in E_titles], per_article=10, seed=2)
    Y = cap([i for i in valid if i["stratum"] == "SP" and i["title"] in E_titles], per_article=10, seed=3)[:80]
    split = {i["id"]: "R" for i in R} | {i["id"]: "E" for i in E} | {i["id"]: "YEAR" for i in Y}
    out = [dict({k: i[k] for k in KEEP}, split=split[i["id"]], rank=n)
           for grp in (R, E, Y) for n, i in enumerate(grp)]
    return out, cnt, why, len(ft), sorted(R_titles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("squad")
    ap.add_argument("--out", default="data/stage8a_items.json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out, cnt, why, nft, R_titles = build(a.squad)
    for k in sorted(cnt):
        print(f"{k:24s} {cnt[k]}")
    for k, v in why.most_common():
        print(f"invalid {k:40s} {v}")
    by = collections.Counter((i["split"], i["sub"]) for i in out)
    print("valid FT items", nft, "| R articles", R_titles)
    print("split x subtype", dict(sorted(by.items())))
    print("E articles", len({i["title"] for i in out if i["split"] == "E"}), "E D_in", sum(i["D_in"] for i in out if i["split"] == "E"))
    blob = json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    print("sha256", hashlib.sha256(blob.encode()).hexdigest())
    if a.check:
        old = Path(a.out).read_text()
        assert old == blob, f"rebuild differs from {a.out}"
        print("rebuild identical to", a.out)
    else:
        Path(a.out).write_text(blob)


if __name__ == "__main__":
    main()
