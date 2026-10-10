"""Pilot scan of SQuAD v1.1 dev for Part A (stage 8): typing, uniqueness filters, counts. CPU only, no model."""
import collections
import json
import re
import sys

D = json.load(open(sys.argv[1]))["data"]
YEAR = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])$")
NUM = re.compile(r"^(\d{1,3}(,\d{3})+|\d+(\.\d+)?)$")
CONNECT = {"of", "de", "the", "and", "von", "van", "da", "du", "la", "le", "del", "di", "y", "al", "for", "on", "in"}
CAPW = re.compile(r"^[A-Z][A-Za-z\.\-'&]*$")
STOP = {"The", "A", "An", "It", "He", "She", "They", "This", "That", "These", "Those", "His", "Her", "Its", "Their", "In", "On", "At"}


def qclass(q):
    ql = q.lower()
    if re.search(r"\b(who|whom|whose)\b", ql):
        return "who"
    if re.search(r"\bwhere\b", ql) or re.search(r"\b(what|which) (city|country|state|town|region|county|nation|continent|island|river|province)\b", ql):
        return "where"
    if re.search(r"\bwhen\b|\bwhat year\b|\bwhich year\b|\bin what year\b", ql):
        return "when"
    if re.search(r"\bhow (many|much|long|old|far|large|big|tall|high)\b", ql):
        return "howmany"
    return "what"


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


def occurrences(text, a):
    return [m.start() for m in re.finditer(r"(?<![\w])" + re.escape(a) + r"(?![\w])", text)]


cnt = collections.Counter()
items = []
for art_i, art in enumerate(D):
    for par_i, par in enumerate(art["paragraphs"]):
        ctx = par["context"]
        for qa in par["qas"]:
            cnt["questions"] += 1
            texts = [x["text"] for x in qa["answers"]]
            a = collections.Counter(texts).most_common(1)[0][0]
            n_agree = texts.count(a)
            t = etype(a)
            cnt[f"type_{t}"] += 1
            if t is None:
                continue
            if n_agree < 2:
                cnt[f"drop_agree_{t}"] += 1
                continue
            occ = occurrences(ctx, a)
            if len(occ) != 1:
                cnt[f"drop_occ{min(len(occ), 2)}_{t}"] += 1
                continue
            if a.lower() in qa["question"].lower():
                cnt[f"drop_inq_{t}"] += 1
                continue
            qc = qclass(qa["question"])
            sub = t if t != "NAME" else {"who": "PERSON", "where": "PLACE"}.get(qc, "NAME_OTHER")
            items.append(dict(id=qa["id"], art=art_i, title=art["title"], par=par_i, context=ctx, question=qa["question"],
                              answer=a, start=occ[0], type=t, sub=sub, qclass=qc, n_words=len(a.split())))
            cnt[f"keep_{t}"] += 1
            cnt[f"keepsub_{sub}"] += 1
for k in sorted(cnt):
    print(f"{k:28s} {cnt[k]}")
json.dump(items, open(sys.argv[2], "w"))
print("items", len(items), "articles", len({i['art'] for i in items}), "paragraphs", len({(i['art'], i['par']) for i in items}))
import random
random.seed(0)
for it in random.sample(items, 25):
    print(f"[{it['sub']:10s}] Q: {it['question'][:80]!r} A: {it['answer']!r}")
