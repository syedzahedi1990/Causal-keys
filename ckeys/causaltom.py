"""Prakash et al. (2026, ICLR) CausalToM material for part (b) of preregistration P-2026-10-05-H: a port of their
template-2 story generator, ``Dataset.__getitem__`` and ``get_reversed_sentence_counterfacts`` (release
https://github.com/Nix07/mind at RELEASE_SHA), the readout formats, token positions and the ID donor.

Nothing of the release is stored here: the story templates, entity lists and the instruction string are read at run time
from the directory named by PRAKASH_REPO (a checkout of RELEASE_SHA), and the sha256 of every release file used is
asserted (FILES). ``pool(rel)`` reproduces the release's own call (random.seed(10), n = 320, the argument
2 * (80 + 80) of their prepare_dataset) byte for byte: the clean|cf prompt list hashes to POOL_SHA256. n must stay 320,
because the q draws follow all entity draws in the random stream, so another n changes the questions.

Formats differ only after the story: NO-MENTION (their question), QNAMES (the question names s1, s2, S, X), OPTIONS-AFTER
(a choices list and instruction), LETTERS-AFTER (lettered choices, read on " A".." D"); QNAMES2 (exploratory) names the
two story states. S and X are two drinks absent from the story, drawn per pair with the candidate order from the sha256
of the clean story; S is also the ID donor's word (s_q -> S) and the natural clamp's S.
Positions are derived from the offsets of the tokenisation their code uses (add_special_tokens=True) and asserted:
Qwen2.5 state words at 154 and 166 followed by '.' (13) and '.\\n' (624); Llama-3 one later (BOS) with '.\\n' = 627.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

RELEASE_URL = "https://github.com/Nix07/mind"
RELEASE_SHA = "0579347e3cf963d13d55edf041c7b595b0dcd88b"
FILES = {  # release path -> sha256 at RELEASE_SHA (data read at run time; the two code files are the source of the port)
    "data/story_templates.json": "e89df933496ed130aae460af09b9bd6969dd4747b3eabfda15bb6d3cbc6f2bfd",
    "data/synthetic_entities/characters.json": "ef6a306307382202b73ea9f8550414b44787f95eca43b0934e3abb6e1edc43fe",
    "data/synthetic_entities/bottles.json": "ad22ef58cfcdc853392bde2a01719b9075ef6d7343620bd243a1b0ecc9dc3e54",
    "data/synthetic_entities/drinks.json": "055ae58e96750dce60ba9d572f5e6bb155dea6831e5d6c93ea1f2497c68cc94d",
    "src/dataset.py": "915c1d3c86abf13c5d682cd948081ae27b7b2ae9d1e0538ee2d5a45d546b100e",
    "notebooks/causalToM_novis/utils.py": "91a0d58a7bac24de8f07d16d2b61c5de35b7bab85f994be8b6f2b48fd9e34d34",
}
POOL_SHA256 = "4451da1a1bcde9ceac0508a839af9c1e1a7f0ab9c3c875cd46d35e3dd3c55099"
N_POOL, POOL_SEED, TEMPLATE = 320, 10, 2
DEFAULT_REPO = "/home/user/nix07/mind"

FORMATS = ("NO-MENTION", "QNAMES", "OPTIONS-AFTER", "LETTERS-AFTER")
EXTRA_FORMATS = ("QNAMES2",)
LETTERS = ("A", "B", "C", "D")
# constants of the release (run_patching_exp_utils.py: [155,156,167,168] for Llama-3, minus 1 for Qwen2) and our lengths
POS = {0: (154, 155, 166, 167), 1: (155, 156, 167, 168)}            # key: number of BOS tokens
PUNCT = {0: (13, 624), 1: (13, 627)}                                 # '.' and '.\n'
LENGTHS = {0: {"NO-MENTION": 180, "QNAMES": 186, "OPTIONS-AFTER": 196, "LETTERS-AFTER": 203, "QNAMES2": 182},
           1: {"NO-MENTION": 181, "QNAMES": 187, "OPTIONS-AFTER": 197, "LETTERS-AFTER": 204, "QNAMES2": 183}}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Release:
    root: str
    templates: dict
    characters: list
    objects: list
    states: list
    instruction: str
    hashes: dict


def load(repo: str | None = None) -> Release:
    """Read the release at ``repo`` (default $PRAKASH_REPO, else DEFAULT_REPO) and assert every file's sha256."""
    root = Path(repo or os.environ.get("PRAKASH_REPO") or DEFAULT_REPO)
    hashes = {}
    for f, h in FILES.items():
        assert (root / f).is_file(), f"release file {root / f} missing (check out {RELEASE_URL} at {RELEASE_SHA})"
        hashes[f] = sha256_file(root / f)
        assert hashes[f] == h, f"{f}: sha256 {hashes[f]} != {h} (not the release at {RELEASE_SHA})"
    D = root / "data" / "synthetic_entities"
    ents = [json.loads((D / f).read_text()) for f in ("characters.json", "bottles.json", "drinks.json")]
    tree = ast.parse((root / "src" / "dataset.py").read_text())   # the default of Dataset.instruction
    instr = next(n.value.value for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "Dataset" for n in c.body
                 if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "instruction")
    return Release(str(root), json.loads((root / "data" / "story_templates.json").read_text()), *ents, instr, hashes)


# --------------------------------------------------------------------------- port of src/dataset.py (template 2)
def make_story(rel: Release, t: int, chars, objs, states) -> dict:
    """Sample.__post_init__ / set_story: placeholders replaced in their order; beliefs of templates 0, 2, 3."""
    assert len(set(states)) == len(states) and len(set(objs)) == len(objs) and len(set(chars)) == len(chars)
    ph = rel.templates["placeholders"]["entity"]
    tpl = rel.templates["templates"][t]
    s = tpl["context"]
    for i, c in enumerate(chars):
        s = s.replace(ph["character"][i], c)
    s = s.replace(ph["container"][0], objs[0]).replace(ph["container"][1], objs[1])
    s = s.replace(ph["state"][0], states[0]).replace(ph["state"][1], states[1])
    assert "<" not in s and ">" not in s
    world = {objs[0]: states[0], objs[1]: states[1]}
    belief = [dict(world), dict(world)]
    if t in (0, 2, 3):
        belief[0][objs[1]] = belief[1][objs[0]] = "unknown"
    elif t == 1:
        belief[1][objs[0]] = "unknown"
    return dict(template=tpl, characters=list(chars), objects=list(objs), states=list(states), story=s, belief=belief)


def getitem(rel: Release, smp: dict, set_container=None, set_character=None, rng=random) -> dict:
    """Dataset.__getitem__: the raw prompt 'Instruction: ...\\n\\nStory: ...\\nQuestion: ...\\nAnswer:'."""
    prompt = f"Instruction: {rel.instruction.strip()}\n\n" + f"Story: {smp['story'].strip()}\n"
    c = rng.choice([0, 1]) if set_character is None else set_character
    actor, beliefs = smp["characters"][c], smp["belief"][c]
    o = rng.choice([0, 1]) if set_container is None else set_container
    container = smp["objects"][o]
    ans = beliefs.get(container, "unknown")
    phq = rel.templates["placeholders"]["question"]
    question = smp["template"]["question"].replace(phq["character"], actor).replace(phq["container"], container)
    prompt += f"Question: {question}\n" + "Answer:"
    return dict(characters=smp["characters"], objects=smp["objects"], states=smp["states"], story=smp["story"],
                question=question, target=ans, prompt=prompt, character_idx=c, object_idx=o, template_idx=TEMPLATE)


def reversed_sentence_counterfacts(rel: Release, n: int, rng) -> list[dict]:
    """notebooks/causalToM_novis/utils.py get_reversed_sentence_counterfacts: every entity draw first, then every q."""
    clean, cf = [], []
    for _ in range(n):
        chars, objs, states = rng.sample(rel.characters, 2), rng.sample(rel.objects, 2), rng.sample(rel.states, 2)
        clean.append(make_story(rel, TEMPLATE, chars, objs, states))
        cf.append(make_story(rel, TEMPLATE, chars[::-1], objs[::-1], states[::-1]))
    out = []
    for i in range(n):
        q = rng.choice([0, 1])
        a, b = getitem(rel, clean[i], q, q, rng), getitem(rel, cf[i], 1 ^ q, 1 ^ q, rng)
        out.append({f"{k}_{f}": d[g] for k, d in (("clean", a), ("counterfactual", b))
                    for f, g in (("characters", "characters"), ("objects", "objects"), ("states", "states"), ("story", "story"),
                                 ("question", "question"), ("prompt", "prompt"), ("ans", "target"))}
                   | {"target": " " + clean[i]["states"][1 ^ q]})
    return out


def pool_hash(pairs) -> str:
    return hashlib.sha256("\n".join(p["clean_prompt"] + "|" + p["counterfactual_prompt"] for p in pairs).encode()).hexdigest()


def pool(rel: Release) -> list[dict]:
    """The release's pool: random.seed(10), n = 320 (a fresh Random(10) gives the module-level random's stream)."""
    pairs = reversed_sentence_counterfacts(rel, N_POOL, random.Random(POOL_SEED))
    h = pool_hash(pairs)
    assert h == POOL_SHA256, f"pool sha256 {h} != {POOL_SHA256}: the port does not reproduce the release"
    return pairs


# --------------------------------------------------------------------------- formats, items, positions
def meta(rel: Release, pr: dict) -> dict:
    """q, s_q, the swap target (their label), S, X and the candidate order C (seeded by the sha256 of the clean story)."""
    st = pr["clean_states"]
    q = st.index(pr["clean_ans"])
    assert pr["counterfactual_ans"] == pr["clean_ans"] and pr["target"] == " " + st[1 - q]
    assert pr["counterfactual_question"] == pr["clean_question"]
    rng = random.Random(int(hashlib.sha256(pr["clean_story"].encode()).hexdigest(), 16))
    S, X = rng.sample([d for d in rel.states if d not in st], 2)
    cand = [st[0], st[1], S, X]
    rng.shuffle(cand)
    return dict(q=q, s_q=st[q], other=st[1 - q], S=S, X=X, cand=cand, states=list(st),
                character=pr["clean_characters"][q], container=pr["clean_objects"][q])


def tail(fmt: str, question: str, m: dict) -> str:
    c = m["cand"]
    if fmt == "NO-MENTION":
        return f"Question: {question}\nAnswer:"
    if fmt == "QNAMES":
        return f"Question: Does {m['character']} believe the {m['container']} contains {', '.join(c[:-1])} or {c[-1]}?\nAnswer:"
    if fmt == "QNAMES2":
        return f"Question: Does {m['character']} believe the {m['container']} contains {m['states'][0]} or {m['states'][1]}?\nAnswer:"
    if fmt == "OPTIONS-AFTER":
        return f"Question: {question}\nChoices: {', '.join(c)}\nAnswer with exactly one choice.\nAnswer:"
    if fmt == "LETTERS-AFTER":
        return f"Question: {question}\nChoices: " + ", ".join(f"{L}) {w}" for L, w in zip(LETTERS, c)) + "\nAnswer with one letter.\nAnswer:"
    raise ValueError(fmt)


def prompt(rel: Release, story: str, question: str, fmt: str, m: dict) -> str:
    return f"Instruction: {rel.instruction.strip()}\n\nStory: {story.strip()}\n" + tail(fmt, question, m)


def state_spans(story: str) -> list[tuple[int, int]]:
    """Character spans of the two state words ('fills it with <state>.') in the story."""
    sp = [mt.span(1) for mt in re.finditer(r"fills it with (\w+)\.", story)]
    assert len(sp) == 2, story
    return sp


def swap_state(story: str, which: int, word: str) -> str:
    a, b = state_spans(story)[which]
    return story[:a] + word + story[b:]


def texts(rel: Release, pr: dict, fmt: str, m: dict | None = None) -> dict:
    """B (clean), C (counterfactual), S (= the ID donor D: s_q -> S) and X (s_q -> X) prompts of one pair in ``fmt``."""
    m = m or meta(rel, pr)
    qs = pr["clean_question"]
    t = {"B": prompt(rel, pr["clean_story"], qs, fmt, m), "C": prompt(rel, pr["counterfactual_story"], qs, fmt, m),
         "S": prompt(rel, swap_state(pr["clean_story"], m["q"], m["S"]), qs, fmt, m),
         "X": prompt(rel, swap_state(pr["clean_story"], m["q"], m["X"]), qs, fmt, m)}
    if fmt == "NO-MENTION":
        assert t["B"] == pr["clean_prompt"] and t["C"] == pr["counterfactual_prompt"], "NO-MENTION must be their prompt"
    return t


def word_token(tok, enc, text: str, a: int, b: int) -> int:
    """Index of the single token covering text[a:b] (the word with its leading space)."""
    hit = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x < b and y > a]
    assert len(hit) == 1 and tok.decode([enc["input_ids"][hit[0]]]) == " " + text[a:b], (text[a:b], hit)
    return hit[0]


def encode(tok, text: str) -> dict:
    """Their tokenisation: add_special_tokens=True (Llama-3 prepends BOS, Qwen2.5 adds nothing)."""
    return tok(text, return_offsets_mapping=True)


def n_bos(tok, ids) -> int:
    return int(tok.bos_token_id is not None and ids[0] == tok.bos_token_id)


def locate(tok, rel: Release, pr: dict, fmt: str, m: dict | None = None) -> dict:
    """Token ids of B, C, S, X and the positions, each derived from offsets and asserted against the constants.
    Returns ids, P = [s1, s1+1, s2, s2+1], p = position of s_q, the BIND map (dst <- src) and the ID span."""
    m = m or meta(rel, pr)
    t = texts(rel, pr, fmt, m)
    enc = {k: encode(tok, v) for k, v in t.items()}
    ids = {k: e["input_ids"] for k, e in enc.items()}
    nb = n_bos(tok, ids["B"])
    T = len(ids["B"])
    assert all(len(v) == T for v in ids.values()), {k: len(v) for k, v in ids.items()}
    if fmt in LENGTHS[nb]:
        assert T == LENGTHS[nb][fmt], (fmt, T, LENGTHS[nb][fmt])
    off = len(f"Instruction: {rel.instruction.strip()}\n\nStory: ")
    P = {}
    for k, story in (("B", pr["clean_story"]), ("C", pr["counterfactual_story"])):
        assert t[k][off:off + len(story)] == story
        P[k] = [word_token(tok, enc[k], t[k], off + a, off + b) for a, b in state_spans(story)]
    w1, w2 = P["B"]
    pos = [w1, w1 + 1, w2, w2 + 1]
    assert tuple(pos) == POS[nb] and P["C"] == P["B"], (pos, P["C"], POS[nb])
    assert (ids["B"][w1 + 1], ids["B"][w2 + 1]) == PUNCT[nb] == (ids["C"][w1 + 1], ids["C"][w2 + 1])
    assert ids["C"][w1] == ids["B"][w2] and ids["C"][w2] == ids["B"][w1]   # the cf's first span holds clean's second word
    p = pos[2 * m["q"]]
    for k in ("S", "X"):
        d = [i for i in range(T) if ids[k][i] != ids["B"][i]]
        assert d == [p], (k, d, p)
        assert tok.decode([ids[k][p]]) == " " + m[k]
    return dict(ids=ids, T=T, n_bos=nb, P=pos, p=p, Pw=[w1, w2], bind_dst=[w2, w2 + 1, w1, w1 + 1], bind_src=pos,
                id_span=[p, p + 1], meta=m)


def readout_words(fmt: str, m: dict) -> dict:
    """Readout string of every role: the word, or its letter under LETTERS-AFTER (roles s1, s2, S, X, s_q, other)."""
    w = {"s1": m["states"][0], "s2": m["states"][1], "S": m["S"], "X": m["X"], "s_q": m["s_q"], "other": m["other"]}
    return {k: LETTERS[m["cand"].index(v)] for k, v in w.items()} if fmt == "LETTERS-AFTER" else w


def single_id(tok, word: str) -> int:
    t = tok.encode(" " + word, add_special_tokens=False)
    assert len(t) == 1, f"{word!r} is not one token with a leading space"
    return t[0]


def correct(tok, token_id: int, target: str) -> bool:
    """Their criterion: decode(argmax).lower().strip() == target.lower().strip()."""
    return tok.decode([int(token_id)]).lower().strip() == target.lower().strip()


if __name__ == "__main__":   # PRAKASH_REPO=<checkout> python -m ckeys.causaltom: the release hashes and the pool hash
    r = load()
    print(f"release {RELEASE_SHA} at {r.root}: " + ", ".join(f"{f} {h[:12]}" for f, h in r.hashes.items()))
    print(f"pool sha256 {pool_hash(pool(r))} (n = {N_POOL}, seed {POOL_SEED}): OK")
