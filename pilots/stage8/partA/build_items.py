"""Part A pilot: build counterfactual SQuAD items (B = original passage, S/X = same-subtype substitutes of the answer
entity, D = distractor option) valid for a set of tokenizers. CPU only.

Rules (the design fixes these):
  * entity e_B = majority answer (>= 2 of 3 annotators), typed YEAR / NUMBER / NAME (sub-typed by the first wh-word of
    the question: who/whom/whose -> PERSON, where -> PLACE, else NAME_OTHER), occurring exactly once in the passage
    (word-bounded, case-sensitive) and not in the question (case-insensitive).
  * substitutes: NAME: SQuAD dev majority answers of the same subtype and word count from OTHER articles; NUMBER: the
    leading digit replaced (seeded); YEAR: same century, |dy| in [3, 60] (seeded).
    Absent from passage and question (case-insensitive), no word shared with each other or with e_B / e_D.
  * per tokenizer: the S and X passages tokenize to the same length as B and are identical outside the entity's token
    span; first content tokens (after any whitespace-only shared token) pairwise distinct among B, S, X, D  ("FT").
    YEAR items are kept in a separate stratum where FT is not required ("SP").
  * D: another majority answer of the same paragraph, same subtype, occurring in the passage, not in the question,
    FT-distinct (D_in = True); else an absent pool entity (D_in = False).
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys

from transformers import AutoTokenizer

YEAR = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])$")
NUM = re.compile(r"^(\d{1,3}(,\d{3})+|\d+)$")
CONNECT = {"of", "de", "the", "and", "von", "van", "da", "du", "la", "le", "del", "di", "y", "al", "for", "on", "in"}
CAPW = re.compile(r"^[A-Z][A-Za-z\.\-'&]*$")
STOP = {"The", "A", "An", "It", "He", "She", "They", "This", "That", "These", "Those", "His", "Her", "Its", "Their",
        "In", "On", "At", "I", "We", "You"}
WH = re.compile(r"\b(who|whom|whose|where|when|what|which|how|why)\b", re.I)


def etype(a):
    if YEAR.match(a):
        return "YEAR"
    if NUM.match(a):
        return "NUMBER"
    w = a.split()
    if 1 <= len(w) <= 4 and CAPW.match(w[0]) and w[0] not in STOP and not any(c.isdigit() for c in a) \
            and all(CAPW.match(x) or x in CONNECT for x in w) and CAPW.match(w[-1]):
        return "NAME"
    return None


def subtype(t, q):
    if t != "NAME":
        return t
    m = WH.search(q)
    wh = m.group(1).lower() if m else ""
    return {"who": "PERSON", "whom": "PERSON", "whose": "PERSON", "where": "PLACE"}.get(wh, "NAME_OTHER")


def occ(text, a, flags=0):
    return [m.start() for m in re.finditer(r"(?<![\w])" + re.escape(a) + r"(?![\w])", text, flags)]


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def majority(qa):
    texts = [x["text"] for x in qa["answers"]]
    a, n = collections.Counter(texts).most_common(1)[0]
    return a, n


def load(path):
    D = json.load(open(path))["data"]
    base, pool = [], collections.defaultdict(list)
    for ai, art in enumerate(D):
        for pi, par in enumerate(art["paragraphs"]):
            ctx = par["context"]
            par_ans = []
            for qa in par["qas"]:
                a, n = majority(qa)
                t = etype(a)
                if t is None:
                    continue
                st = subtype(t, qa["question"])
                par_ans.append((a, st, qa["question"]))
                if t == "NAME" and n >= 2:
                    pool[(st, len(a.split()))].append((ai, a))
            for qa in par["qas"]:
                a, n = majority(qa)
                t = etype(a)
                if t is None or n < 2:
                    continue
                o = occ(ctx, a)
                if len(o) != 1 or a.lower() in qa["question"].lower():
                    continue
                st = subtype(t, qa["question"])
                base.append(dict(id=qa["id"], art=ai, par=pi, title=art["title"], context=ctx, question=qa["question"],
                                 answer=a, start=o[0], type=t, sub=st, par_answers=par_ans))
    for k in pool:  # dedupe, keep first article
        seen, out = set(), []
        for ai, a in pool[k]:
            if a not in seen:
                seen.add(a)
                out.append((ai, a))
        pool[k] = out
    return base, pool


def words(s):
    return {w.lower() for w in re.findall(r"\w+", s)}


def first_content(tok, s):
    """First token id of ' ' + s (after 'Answer:') that is not whitespace-only, and its index."""
    pre = tok("Answer:", add_special_tokens=False).input_ids
    ids = tok("Answer: " + s, add_special_tokens=False).input_ids
    assert ids[:len(pre)] == pre, (s, ids, pre)
    cont = ids[len(pre):]
    for i, t in enumerate(cont):
        if tok.decode([t]).strip():
            return t, i, cont
    return None, None, cont


def span_ok(tok, ctx, start, a, b):
    """ctx with a at start replaced by b: same token count, identical outside the entity span (offset-based)."""
    ctx2 = ctx[:start] + b + ctx[start + len(a):]
    e1 = tok(ctx, add_special_tokens=False, return_offsets_mapping=True)
    e2 = tok(ctx2, add_special_tokens=False, return_offsets_mapping=True)
    if len(e1.input_ids) != len(e2.input_ids):
        return False, None
    P = [i for i, (s, e) in enumerate(e1.offset_mapping) if e > start and s < start + len(a)]
    diff = [i for i, (x, y) in enumerate(zip(e1.input_ids, e2.input_ids)) if x != y]
    if not set(diff) <= set(P) or not diff:
        return False, None
    return True, P


def build(path, tok_names, n_cap=None, verbose=False):
    toks = {k: AutoTokenizer.from_pretrained(v) for k, v in tok_names.items()}
    base, pool = load(path)
    cnt = collections.Counter()
    items = []
    for it in base:
        a, t, st, ctx, q = it["answer"], it["type"], it["sub"], it["context"], it["question"]
        rng = random.Random(seed_of(it["id"]))
        cnt[f"cand_{st}"] += 1
        absent = lambda s: s.lower() not in ctx.lower() and s.lower() not in q.lower()  # noqa: E731
        if t == "NAME":
            cands = [s for ai, s in pool[(st, len(a.split()))] if ai != it["art"] and absent(s) and not (words(s) & words(a))]
            rng.shuffle(cands)
        elif t == "NUMBER":
            lead = a[0]
            digs = [d for d in "123456789" if d != lead]
            rng.shuffle(digs)
            cands = [d + a[1:] for d in digs if absent(d + a[1:])]
        else:  # YEAR
            y = int(a)
            ds = [d for d in range(-60, 61) if abs(d) >= 3]
            rng.shuffle(ds)
            cands = [str(y + d) for d in ds if str(y + d)[:2] == a[:2] and absent(str(y + d)) and int(str(y + d)) <= 2025]
        # distractor D
        dpres = [x for x, s2, _ in it["par_answers"] if s2 == st and x != a and occ(ctx, x) and x.lower() not in q.lower()
                 and not (words(x) & words(a))]
        need_ft = t != "YEAR"

        def fts(strings):
            out = {}
            for k, tk in toks.items():
                out[k] = [first_content(tk, s)[0] for s in strings]
            return out

        def ft_distinct(strings):
            return all(len(set(v)) == len(v) and None not in v for v in fts(strings).values())

        def length_ok(s):
            P = {}
            for k, tk in toks.items():
                ok, p = span_ok(tk, ctx, it["start"], a, s)
                if not ok:
                    return None
                P[k] = p
            return P

        chosen, spans = [], {}
        for s in cands:
            if any(words(s) & words(c) for c in chosen):
                continue
            if need_ft and not ft_distinct([a] + chosen + [s]):
                continue
            P = length_ok(s)
            if P is None:
                continue
            chosen.append(s)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            cnt[f"drop_nosub_{st}"] += 1
            continue
        S, X = chosen
        Dopt, D_in = None, False
        for x in dpres:
            if not (words(x) & (words(S) | words(X))) and (not need_ft or ft_distinct([a, S, X, x])):
                Dopt, D_in = x, True
                break
        if Dopt is None:
            for s in cands:
                if s in chosen or any(words(s) & words(c) for c in [a, S, X]):
                    continue
                if need_ft and not ft_distinct([a, S, X, s]):
                    continue
                Dopt = s
                break
        if Dopt is None:
            cnt[f"drop_noD_{st}"] += 1
            continue
        order = [a, S, X, Dopt]
        random.Random(seed_of(it["id"] + "order")).shuffle(order)
        ntok = {k: len(span_ok(tk, ctx, it["start"], a, a)[1] or []) for k, tk in toks.items()} if False else None
        items.append(dict(it, S=S, X=X, D=Dopt, D_in=D_in, options=order, stratum="FT" if need_ft else "SP"))
        cnt[f"keep_{st}"] += 1
        cnt[f"keep_Din_{D_in}"] += 1
        if n_cap and len(items) >= n_cap:
            break
    return items, cnt


if __name__ == "__main__":
    TOK = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
           "llama": "unsloth/Meta-Llama-3.1-8B-Instruct", "gemma": "unsloth/gemma-2-9b-it"}
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else list(TOK)
    items, cnt = build(sys.argv[1], {k: TOK[k] for k in which})
    for k in sorted(cnt):
        print(f"{k:28s} {cnt[k]}")
    for it in items:
        it.pop("par_answers", None)
    json.dump(items, open(sys.argv[2], "w"))
    ft = [i for i in items if i["stratum"] == "FT"]
    print("items", len(items), "FT", len(ft), "articles(FT)", len({i['art'] for i in ft}), "paragraphs(FT)", len({(i['art'], i['par']) for i in ft}))
    print("context chars: median", sorted(len(i["context"]) for i in ft)[len(ft) // 2])
    random.seed(1)
    for it in random.sample(ft, 12):
        print(f"[{it['sub']:10s}] Q: {it['question'][:70]!r}\n    B={it['answer']!r} S={it['S']!r} X={it['X']!r} D={it['D']!r} D_in={it['D_in']}")
