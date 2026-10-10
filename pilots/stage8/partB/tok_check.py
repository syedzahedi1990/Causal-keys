"""Part B pilot 1 (tokenizer only, no model): for each candidate model, check that
 (a) the six lowercase ' w' candidates are single tokens (format_factorial requirement),
 (b) B/S/X prompts of every 2x2 arm differ at exactly one position (fresh seed cores),
 (c) the surface-form set (12 forms per candidate) and the number of distinct proper prefixes (extra passes),
 (d) whether the chat template accepts a system turn.
"""
import random, sys, json
sys.path.insert(0, "/home/user/Causal-keys")
from transformers import AutoTokenizer
from ckeys.encoding import encode, build_prompt, WRAPPER_USED, candidate_ids
from ckeys.story import LOCATIONS, make_cores, pick_x, record

MODELS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
          "allenai/OLMo-2-1124-7B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/gemma-2-9b-it",
          "microsoft/phi-4", "ibm-granite/granite-3.1-8b-instruct", "tiiuae/Falcon3-7B-Instruct"]
ARTICLES = [" The", " the", "The", "the"]


def forms(w):
    W = w.capitalize()
    direct = [" " + w, " " + W, w, W]
    art = [a + " " + x for a in ARTICLES for x in (w, W)]
    return direct + art


def main():
    cores = make_cores(150, random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
    out = {}
    for m in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as e:
            print(m, "LOAD FAIL", e); continue
        WRAPPER_USED["system_merged"] = False
        rep = {}
        try:
            cid = candidate_ids(tok, "AFTER"); rep["lower_single"] = True
        except AssertionError as e:
            rep["lower_single"] = str(e)
        # surface forms: token sequences appended after "Answer:" (encode the full text to respect merges)
        base = "Answer:"
        b_ids = tok(base, add_special_tokens=False).input_ids
        seqs, multi, prefixes, merged = {}, [], set(), []
        for w in LOCATIONS:
            for f in forms(w):
                ids = tok(base + f, add_special_tokens=False).input_ids
                if ids[:len(b_ids)] == b_ids:
                    s = tuple(ids[len(b_ids):])
                else:  # the form merges with the prompt's last token: use its standalone tokenisation
                    s = tuple(tok(f, add_special_tokens=False).input_ids)
                    merged.append(f)
                seqs[(w, f)] = s
                if len(s) > 1:
                    multi.append((f, [tok.decode([t]) for t in s]))
                for k in range(1, len(s)):
                    prefixes.add(s[:k])
        rep["boundary_merged_forms"] = sorted(set(f.strip() for f in merged))[:6] + [f"... {len(merged)} total"]
        rep["n_forms"] = len(seqs)
        rep["n_distinct_seqs"] = len(set(seqs.values()))
        rep["n_prefixes"] = len(prefixes)
        rep["prefixes"] = sorted(["|".join(tok.decode([t]) for t in p) for p in prefixes])
        rep["cap_space_single"] = all(len(seqs[(w, " " + w.capitalize())]) == 1 for w in LOCATIONS)
        rep["cap_space_multi"] = [w for w in LOCATIONS if len(seqs[(w, " " + w.capitalize())]) > 1]
        # first tokens shared across candidates (ambiguity of the first piece)
        first = {}
        for (w, f), s in seqs.items():
            first.setdefault(s[0], set()).add(w)
        rep["shared_first_tokens"] = {tok.decode([t]): sorted(v) for t, v in first.items() if len(v) > 1}
        # prompt alignment on fresh cores for the 2x2 arms
        bad = {}
        lens = {}
        for arm in ("AFTER", "BEFORE", "POST", "PRE", "NONE", "P1", "LETTER"):
            nb = 0
            for core in cores:
                X = pick_x(core)
                ids = {}
                for nm, loc in (("B", core["base"]), ("S", core["source"]), ("X", X)):
                    r = record(core, "direct", loc)
                    ids[nm] = encode(tok, build_prompt(arm, r["story"], r["query"], core, X))[0]
                if len({len(v) for v in ids.values()}) > 1:
                    nb += 1; continue
                d = (ids["B"] != ids["S"]).nonzero().flatten().tolist()
                dx = (ids["B"] != ids["X"]).nonzero().flatten().tolist()
                if len(d) != 1 or dx != d:
                    nb += 1
                lens[arm] = len(ids["B"])
            bad[arm] = nb
        rep["skipped_items_per_arm"] = bad
        rep["prompt_len"] = lens
        rep["system_merged"] = WRAPPER_USED["system_merged"]
        out[m] = rep
        print("==", m)
        for k, v in rep.items():
            print("  ", k, ":", v)
    json.dump(out, open("/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB/tok_check.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
