"""Counterfactual SQuAD v1.1 items for stage 8 part A (preregistration J): a passage whose answer entity e_B occurs once
is rewritten with a same-type entity of the same token length (in every study tokenizer), so the B, S, X and Z
passages differ only inside the entity's token span.

Rules (fixed in the entry; CPU only, no model):
  * e_B is the majority answer (given by >= 2 of the 3 annotators), typed YEAR / NUMBER / NAME, and NAME sub-typed by
    the question (who/whom/whose and person-shaped -> PERSON; where, or a place head noun -> PLACE; ORG and the rest
    are dropped). e_B occurs exactly once in the passage (word-bounded, case-sensitive) and not in the question
    (case-insensitive), and is not on the audited exclusion list (EXCLUDE).
  * Substitutes S, X, Z (Z only ever clamped, never shown): NAME from the pool of majority answers of the same subtype
    and word count in OTHER articles (PLACE only from questions with an explicit place head noun), minus EXCLUDE;
    NUMBER by replacing the leading digit; YEAR in the same century, 3 <= |dy| <= 60. Every substitute is absent from
    the passage and question and shares no word with e_B, the other substitutes or the distractor.
  * D (the fourth option): another majority answer of the same paragraph and subtype that occurs in the passage and
    not in the question (D_in), else an absent pool entity.
  * No partial mentions: no content word of e_B (>= 3 characters, not a connector) occurs in the passage outside e_B's
    span or in the question, and no content word of S, X, Z or D occurs in the passage or question (so no candidate is
    re-mentioned inside the passage). Items whose e_B fails this form the LEAK stratum (exploratory; e.g. a surname
    re-mentioned after the full name) and enter no primary population.
  * PLACE items are typed by the question's head noun into a place class (PLACE_CLASSES); substitutes come from pool
    entries of the same class. 'where' questions without a place head noun are dropped.
  * Per tokenizer: the S/X/Z passages have B's token count and differ only inside B's entity span; the first content
    tokens of B, S, X, Z and D are pairwise distinct (stratum FT; YEAR items form stratum SP without that rule).
EXCLUDE is the outcome of the pre-finalisation type audit of every pool and item entity (data/stage8a_type_audit.tsv),
done by reading the entity lists blind to any model output.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re

SQUAD_DEV_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

YEAR = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])$")
NUM = re.compile(r"^(\d{1,3}(,\d{3})+|\d+)$")
CONNECT = {"of", "de", "the", "and", "von", "van", "da", "du", "la", "le", "del", "di", "y", "al", "for", "on", "in"}
CAPW = re.compile(r"^[A-Z][A-Za-z\.\-'&]*$")
STOP = {"First", "Second", "Third", "Fourth", "Fifth", "One", "Two", "Three", "The", "A", "An", "It", "He", "She", "They",
        "This", "That", "These", "Those", "His", "Her", "Its", "Their", "In", "On", "At", "I", "We", "You"}
WH = re.compile(r"\b(who|whom|whose|where|when|what|which|how|why)\b", re.I)
AUX = set("is was are were did does do has had have can could will would should may might became become becomes of to "
          "in for on at by from with that which who".split())
PLACE_HEADS = {"country", "city", "town", "state", "county", "region", "continent", "river", "lake", "island", "province",
               "capital", "nation", "area", "village", "district"}
ORG_HEADS = {"company", "team", "network", "station", "organization", "organisation", "party", "university", "college",
             "school", "newspaper", "band", "club", "agency", "corporation", "channel", "league", "firm", "group"}
NONPERSON = set("""Union Party Council Church Company Corporation University College School Network Radio Television
Broadcasting Court Empire Kingdom Republic States Army Navy League Association Society Institute One North South East
West Central Northern Southern Eastern Western New United American British French German Royal National Super Bowl
Stadium Act War Prize Award Awards Museum Library Club Team City Valley River Island Mountains Range Sea Ocean Lake
Center Centre Institution Committee Department Ministry Assembly Parliament Group Foundation Order Arabs Muslims
Christians Catholics Protestants Huguenots Mongols Normans Dynasty Officer President Minister Pope King Queen Emperor
Catholic Roman Orthodox""".split())
PERSONW = re.compile(r"^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$|^[A-Z][a-z]*[A-Z][a-z]+$|^[A-Z]\.$")

PLACE_CLASSES = {"country": "country", "nation": "country", "city": "city", "town": "city", "village": "city",
                 "capital": "city", "state": "region", "province": "region", "county": "region", "region": "region",
                 "district": "region", "area": "region", "river": "water", "lake": "water", "island": "island",
                 "continent": "continent"}


# The granularity audit (pre-finalisation, entity lists only): places whose question's head noun names the wrong class.
CLASS_FIX = {"Beirut": "city", "Africa": "continent", "Asia": "continent", "North America": "continent",
             "Scandinavia": "region", "Japan": "country", "Samarkand": "city", "Deabolis": "city", "Lindau": "city"}


def place_class(q, a=None):
    """The place class of answer ``a`` to question ``q``: the audited class of ``a`` if it has one, else the class named
    by the question's head noun (None without a place head noun)."""
    if a in CLASS_FIX:
        return CLASS_FIX[a]
    _, hd = wh_head(q)
    if not hd:
        return None
    return PLACE_CLASSES.get(hd) or PLACE_CLASSES.get(hd.rstrip("s"))


def content_words(e):
    """The words of an entity that would identify it if mentioned alone: >= 3 characters, not a connector."""
    return [w for w in re.findall(r"[\w'\-]+", e) if len(w) >= 3 and w.lower() not in CONNECT
            and any(c.isalpha() for c in w)]


def leaks(text, e, skip=None):
    """Positions in ``text`` where a content word of ``e`` occurs (word-bounded, case-sensitive), outside the
    character range ``skip``."""
    out = []
    for w in content_words(e):
        for m in re.finditer(r"(?<![\w])" + re.escape(w) + r"(?![\w])", text):
            if skip is None or m.end() <= skip[0] or m.start() >= skip[1]:
                out.append(m.start())
    return sorted(out)


# The type audit (pre-finalisation; entity lists only, no model output): entities that are not of their assigned type.
EXCLUDE = {
    "PLACE": {
        "Baldwin": "surname", "Centrum": "not a place name", "Charles": "reads as a person (the Charles River)",
        "Conservation": "common noun", "Orange": "ambiguous (fruit, colour)", "Portuguese": "demonym",
        "Southern": "adjective", "U.S": "truncated abbreviation", "Political": "adjective",
        "Battle of Sainte-Foy": "event", "Book of Discipline": "book", "Germany and Scandinavia": "two places",
        "Somalia and Ethiopia": "two places", "Edison Machine Works": "company", "Wardenclyffe Tower": "building",
    },
    "PERSON": {
        "Death Wish Coffee": "company", "Economist Intelligence Unit": "organisation", "San Diego Chargers": "team",
        "Archangel Michael": "angel", "Arizona Cardinals": "team", "Cisco Systems": "company",
        "Cosgrove Hall": "studio", "Denver Broncos": "team", "Digital Spy": "website",
        "District Superintendents": "office", "Doctor Who": "television series", "English Heritage": "organisation",
        "General Electric": "company", "General Sejm": "assembly", "Han Chinese": "ethnic group",
        "Knaurs Lexikon": "encyclopedia", "Knight Ridder": "company", "Paramount Pictures": "company",
        "Pittard Sullivan": "company", "Pittsburgh Steelers": "team", "Polonia Warsaw": "team",
        "Prospect Park": "park", "Red Guards": "movement", "Science Magazine": "journal",
        "Westinghouse Electric": "company", "Joseph Thompsons": "plural surname", "Saint Nicolas": "church name",
        "Chicago Bears": "team", "Seattle Seahawks": "team",
    },
}


def excluded(sub: str, a: str) -> bool:
    return a in EXCLUDE.get(sub, {})


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


def wh_head(q):
    m = WH.search(q)
    if not m:
        return None, None
    wh, head = m.group(1).lower(), None
    if wh in ("what", "which"):
        for t in re.findall(r"[A-Za-z\-']+", q[m.end():])[:5]:
            if t.lower() in AUX:
                break
            head = t.lower()
    return wh, head


def person_shaped(a):
    w = a.split()
    return 2 <= len(w) <= 3 and all(PERSONW.match(x) for x in w) and not PERSONW.match(w[-1]).group(0).endswith(".") \
        and not any(x in NONPERSON for x in w) and w[-1][-1] != "."


def subtype(t, q, a=None):
    if t != "NAME":
        return t
    wh, head = wh_head(q)
    hs = head.rstrip("s") if head else None
    if wh in ("who", "whom", "whose"):
        return "PERSON" if person_shaped(a) else None
    if a is not None and any(x in NONPERSON - {"North", "South", "East", "West", "New", "United", "Central", "Northern",
                                                 "Southern", "Eastern", "Western"} for x in a.split()) and \
            (wh == "where" or hs in PLACE_HEADS):
        return None
    if wh == "where" or hs in PLACE_HEADS or head in PLACE_HEADS:
        return "PLACE"
    if hs in ORG_HEADS or head in ORG_HEADS:
        return "ORG"
    return None


def occ(text, a, flags=0):
    return [m.start() for m in re.finditer(r"(?<![\w])" + re.escape(a) + r"(?![\w])", text, flags)]


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def majority(qa):
    return collections.Counter(x["text"] for x in qa["answers"]).most_common(1)[0]


def words(s):
    return {w.lower() for w in re.findall(r"\w+", s)}


def load_squad(path):
    raw = open(path, "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    assert h == SQUAD_DEV_SHA256, f"{path}: sha256 {h} is not SQuAD v1.1 dev ({SQUAD_DEV_SHA256})"
    return json.loads(raw)["data"]


def base_and_pools(data):
    base, pool = [], collections.defaultdict(list)
    for ai, art in enumerate(data):
        for pi, par in enumerate(art["paragraphs"]):
            ctx, par_ans = par["context"], []
            for qa in par["qas"]:
                a, n = majority(qa)
                t = etype(a)
                st = subtype(t, qa["question"], a) if t else None
                if st is None or excluded(st, a):
                    continue
                par_ans.append((a, st))
                if t == "NAME" and n >= 2:
                    cls = place_class(qa["question"], a) if st == "PLACE" else None
                    if st == "PLACE" and cls is None:
                        continue
                    pool[(st, cls, len(a.split()))].append((ai, a))
            for qa in par["qas"]:
                a, n = majority(qa)
                t = etype(a)
                if t is None or n < 2:
                    continue
                o = occ(ctx, a)
                if len(o) != 1 or a.lower() in qa["question"].lower():
                    continue
                st = subtype(t, qa["question"], a)
                if st is None or st == "ORG" or excluded(st, a):
                    continue
                cls = place_class(qa["question"], a) if st == "PLACE" else None
                if st == "PLACE" and cls is None:
                    continue
                leak = bool(leaks(ctx, a, (o[0], o[0] + len(a))) or leaks(qa["question"], a))
                base.append(dict(id=qa["id"], art=ai, par=pi, title=art["title"], context=ctx, question=qa["question"],
                                 answer=a, start=o[0], type=t, sub=st, cls=cls, leak=leak, par_answers=par_ans))
    for k in pool:  # dedupe, keep the first article
        seen, out = set(), []
        for ai, a in pool[k]:
            if a not in seen:
                seen.add(a)
                out.append((ai, a))
        pool[k] = out
    return base, pool


def first_content(tok, s):
    """First token id of ' ' + s after 'Answer:' that is not whitespace-only, its index, and the continuation."""
    pre = tok("Answer:", add_special_tokens=False).input_ids
    ids = tok("Answer: " + s, add_special_tokens=False).input_ids
    assert ids[:len(pre)] == pre, (s, ids, pre)
    cont = ids[len(pre):]
    for i, t in enumerate(cont):
        if tok.decode([t]).strip():
            return t, i, cont
    return None, None, cont


def span_ok(tok, ctx, start, a, b):
    """ctx with a at start replaced by b: same token count, identical outside a's token span (offsets)."""
    ctx2 = ctx[:start] + b + ctx[start + len(a):]
    e1 = tok(ctx, add_special_tokens=False, return_offsets_mapping=True)
    e2 = tok(ctx2, add_special_tokens=False, return_offsets_mapping=True)
    if len(e1.input_ids) != len(e2.input_ids):
        return False, None
    P = [i for i, (s, e) in enumerate(e1.offset_mapping) if e > start and s < start + len(a)]
    diff = [i for i, (x, y) in enumerate(zip(e1.input_ids, e2.input_ids)) if x != y]
    if not diff or not set(diff) <= set(P):
        return False, None
    return True, P


def build_items(path, toks: dict):
    """``toks``: {name: tokenizer}. Returns (items, counts); each item has S, X, Z, D, D_in, options, stratum."""
    base, pool = base_and_pools(load_squad(path))
    cnt, items = collections.Counter(), []
    for it in base:
        a, t, st, ctx, q = it["answer"], it["type"], it["sub"], it["context"], it["question"]
        rng = random.Random(seed_of(it["id"]))
        cnt[f"cand_{st}"] += 1
        absent = lambda s: (s.lower() not in ctx.lower() and s.lower() not in q.lower()  # noqa: E731
                            and not leaks(ctx, s) and not leaks(q, s))
        if t == "NAME":
            cands = [s for ai, s in pool[(st, it["cls"], len(a.split()))]
                     if ai != it["art"] and absent(s) and not (words(s) & words(a))]
            rng.shuffle(cands)
        elif t == "NUMBER":
            digs = [d for d in "123456789" if d != a[0]]
            rng.shuffle(digs)
            cands = [d + a[1:] for d in digs if absent(d + a[1:])]
        else:
            y = int(a)
            ds = [d for d in range(-60, 61) if abs(d) >= 3]
            rng.shuffle(ds)
            cands = [str(y + d) for d in ds if str(y + d)[:2] == a[:2] and absent(str(y + d)) and int(str(y + d)) <= 2025]
        dpres = [x for x, s2 in it["par_answers"] if s2 == st and x != a and occ(ctx, x) and x.lower() not in q.lower()
                 and not (words(x) & words(a))]
        need_ft = t != "YEAR"

        def ft_distinct(strings):
            return all(len(set(v)) == len(v) and None not in v
                       for v in ([first_content(tk, s)[0] for s in strings] for tk in toks.values()))

        def length_ok(s):
            return all(span_ok(tk, ctx, it["start"], a, s)[0] for tk in toks.values())

        chosen = []
        for s in cands:
            if any(words(s) & words(c) for c in chosen):
                continue
            if need_ft and not ft_distinct([a] + chosen + [s]):
                continue
            if not length_ok(s):
                continue
            chosen.append(s)
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            cnt[f"drop_nosub_{st}"] += 1
            continue
        S, X, Z = chosen
        Dopt, D_in = None, False
        for x in dpres:
            if not (words(x) & (words(S) | words(X) | words(Z))) and (not need_ft or ft_distinct([a, S, X, Z, x])):
                Dopt, D_in = x, True
                break
        if Dopt is None:
            for s in cands:
                if s in chosen or any(words(s) & words(c) for c in [a, S, X, Z]):
                    continue
                if need_ft and not ft_distinct([a, S, X, Z, s]):
                    continue
                Dopt = s
                break
        if Dopt is None:
            cnt[f"drop_noD_{st}"] += 1
            continue
        order = [a, S, X, Dopt]
        random.Random(seed_of(it["id"] + "order")).shuffle(order)
        rec = {k: v for k, v in it.items() if k != "par_answers"}
        stratum = "SP" if not need_ft else "LEAK" if it["leak"] else "FT"
        items.append(dict(rec, S=S, X=X, Z=Z, D=Dopt, D_in=D_in, options=order, stratum=stratum))
        cnt[f"keep_{st}"] += 1
    return items, cnt


def split_articles(items, seed=20261010, n_rank=12):
    """(R titles, E titles): a seeded sample of n_rank article titles ranks heads; the rest evaluate."""
    titles = sorted({i["title"] for i in items})
    R = set(random.Random(seed).sample(titles, n_rank))
    return R, set(titles) - R


def cap(items, per_article, per_paragraph=2, seed=1):
    """Items in a seeded shuffle of the id-sorted list, at most per_paragraph per paragraph, per_article per article."""
    xs = sorted(items, key=lambda i: i["id"])
    random.Random(seed).shuffle(xs)
    na, npar, out = collections.Counter(), collections.Counter(), []
    for i in xs:
        if na[i["title"]] < per_article and npar[(i["title"], i["par"])] < per_paragraph:
            out.append(i)
            na[i["title"]] += 1
            npar[(i["title"], i["par"])] += 1
    return out
