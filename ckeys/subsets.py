"""Named-subset re-mention arms of preregistration G, part (c): which candidates a later sentence or list names.

Per core the roles are B (base, at the writing token p), S (source), X (``pick_x``) and o1, o2, o3, the three
remaining candidates in an order seeded by the core. Named words are always written in the canonical LOCATIONS order,
so S and X occupy varying positions and the k = 6 arms are byte-identical to POST (S6) and AFTER (L6). The arms are
core-dependent and registered in the ckeys.encoding registry (``import_arm_modules("ckeys.subsets")``); ``item["arm_meta"]``
holds the named set. ``check_rules`` is the preflight of scripts/gpu_stage5.sh.
"""
from __future__ import annotations

import hashlib
import json
import random

from .encoding import PREFIX, raw_prompt, register_arm
from .story import LOCATIONS, make_cores, pick_x, record

# arm -> (family, role members); k = number of named candidates
SUBSET_ARMS = {
    "S2": ("sentence", ("S", "X")), "S3": ("sentence", ("B", "S", "X")), "S3out": ("sentence", ("B", "o1", "o2")),
    "S3half": ("sentence", ("B", "S", "o1")), "S4": ("sentence", ("B", "S", "X", "o1")),
    "S4out": ("sentence", ("S", "X", "o1", "o2")), "S6": ("sentence", ("B", "S", "X", "o1", "o2", "o3")),
    "L2": ("list", ("S", "X")), "L3": ("list", ("B", "S", "X")), "L3out": ("list", ("B", "o1", "o2")),
    "L4": ("list", ("B", "S", "X", "o1")), "L4out": ("list", ("S", "X", "o1", "o2")),
    "L6": ("list", ("B", "S", "X", "o1", "o2", "o3")),
}
SENTENCE_ARMS = [a for a, (f, _) in SUBSET_ARMS.items() if f == "sentence"]
LIST_ARMS = [a for a, (f, _) in SUBSET_ARMS.items() if f == "list"]
K6 = {"S6": "POST", "L6": "AFTER"}  # byte-identical standard arms


def others(core: dict, X: str) -> list[str]:
    """The three candidates that are neither B, S nor X, in an order fixed by the core (seeded shuffle)."""
    pool = [l for l in LOCATIONS if l not in (core["base"], core["source"], X)]
    assert len(pool) == 3
    seed = int(hashlib.sha256((json.dumps(core, sort_keys=True) + "|others").encode()).hexdigest()[:8], 16)
    random.Random(seed).shuffle(pool)
    return pool


def named_set(arm: str, core: dict, X: str) -> tuple[str, ...]:
    """Candidates named by ``arm`` for this core, in the canonical LOCATIONS order."""
    o = others(core, X)
    role = {"B": core["base"], "S": core["source"], "X": X, "o1": o[0], "o2": o[1], "o3": o[2]}
    members = {role[r] for r in SUBSET_ARMS[arm][1]}
    return tuple(l for l in LOCATIONS if l in members)


def sentence(named: tuple[str, ...]) -> str:
    items = [f"a {l}" for l in named]
    return "The room has " + (items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]) + "."


def mention_span(arm: str, named: tuple[str, ...]) -> str:
    return sentence(named) if SUBSET_ARMS[arm][0] == "sentence" else "Choices: " + ", ".join(named)


def subset_prompt(arm: str, story: str, query: str, named: tuple[str, ...]) -> str:
    if SUBSET_ARMS[arm][0] == "sentence":
        return PREFIX + story + " " + sentence(named) + "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:"
    return PREFIX + story + "\nQuestion: " + query + "\nChoices: " + ", ".join(named) + "\nAnswer with one word.\nAnswer:"


def _register(arm):
    register_arm(arm, lambda story, query, core, X: subset_prompt(arm, story, query, named_set(arm, core, X)), needs_core=True,
                 span=lambda story, query, core, X: mention_span(arm, named_set(arm, core, X)),
                 meta=lambda core, X: {"named": list(named_set(arm, core, X)), "k": len(SUBSET_ARMS[arm][1])})


for _a in SUBSET_ARMS:
    _register(_a)


def check_rules(n: int = 2000, tok=None, seed: int = 123) -> str:
    """Preflight: per arm the named set has the right size and role membership in canonical order, every prompt is
    the NONE prompt up to and including the story (so nothing before p changes), S6 == POST and L6 == AFTER byte for
    byte. With ``tok`` also at the token level: the chat-wrapped B/S/X prompts of every arm have equal length, differ
    at exactly NONE's p and equal NONE's tokens up to p."""
    from .encoding import encode
    for core in make_cores(n, random.Random(seed)):
        X = pick_x(core)
        B, S = core["base"], core["source"]
        r = record(core, "direct", B)
        none = raw_prompt("NONE", r["story"], r["query"])
        for arm, (fam, members) in SUBSET_ARMS.items():
            named = named_set(arm, core, X)
            assert len(named) == len(members) == len(set(named)), arm
            assert all((role in members) == (loc in named) for role, loc in (("B", B), ("S", S), ("X", X))), arm
            assert list(named) == [l for l in LOCATIONS if l in named], (arm, named)
            raw = subset_prompt(arm, r["story"], r["query"], named)
            assert raw.startswith(PREFIX + r["story"]) and none.startswith(PREFIX + r["story"]), arm
            assert raw.count(mention_span(arm, named)) == 1, arm
        assert subset_prompt("S6", r["story"], r["query"], named_set("S6", core, X)) == raw_prompt("POST", r["story"], r["query"])
        assert subset_prompt("L6", r["story"], r["query"], named_set("L6", core, X)) == raw_prompt("AFTER", r["story"], r["query"])
        if tok is not None:
            recs = {k: record(core, "direct", loc) for k, loc in (("B", B), ("S", S), ("X", X))}
            ids = {k: encode(tok, raw_prompt("NONE", q["story"], q["query"]))[0] for k, q in recs.items()}
            p = (ids["B"] != ids["S"]).nonzero().flatten().tolist()
            assert len(p) == 1 and (ids["B"] != ids["X"]).nonzero().flatten().tolist() == p, p
            p = p[0]
            for arm in SUBSET_ARMS:
                e = {k: encode(tok, subset_prompt(arm, q["story"], q["query"], named_set(arm, core, X)))[0] for k, q in recs.items()}
                assert len({tuple(v.shape) for v in e.values()}) == 1, arm
                d = (e["B"] != e["S"]).nonzero().flatten().tolist()
                assert d == [p] == (e["B"] != e["X"]).nonzero().flatten().tolist(), (arm, d, p)
                assert all(e[k][: p + 1].tolist() == ids[k][: p + 1].tolist() for k in e), arm
    return f"subset rules ok on {n} cores" + ("" if tok is None else f" (token level, p checked)")


if __name__ == "__main__":
    print(check_rules())
