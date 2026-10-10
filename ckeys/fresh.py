"""Stage 8 part B (preregistration J, P-2026-10-10-J): fresh story populations, the second location lexicon, the
preregistered after/before sentences and their null versions, per-core candidate order, and the prompt builders and
answer parsers of the fresh-sample replication.

Populations (cores compared by their full tuple FIELDS; U = the union of make_cores(1000, Random(s)), s = 0..3, which
contains every templated core of stages 1-7):
  F   the first 150 cores of the stream make_cores(., Random(20261013)) that are in neither U nor earlier in the stream
  C   the first 30 cores of the stream make_cores(., Random(20261014)) in neither U, F nor earlier in the stream
      (calibration only: frame discovery, never in a statistic)
  S0  make_cores(150, Random(0)), the cores of stages 1 and 3b, re-measured on purpose (JB6); original wording
A core is drawn in the canonical lexicon (story.LOCATIONS) and carries three attributes fixed by its hash
h = sha256(json.dumps(core tuple)) (F and C; S0 keeps lexicon 1, the canonical order and the ROOM sentence):
  lexicon   the half of the population with the smallest h (ranks 0 .. n/2 - 1) uses LEX2, the rest LEX1; a core's
            locations are rendered by index (LOCATIONS[i] -> LEX2[i]), so every rule of make_cores and pick_x holds
  sentence  SENTENCES[rank % 8], the same sentence in the POST and PRE arms (and its null version in POST-NULL)
  order     random.Random(int(h[:16], 16)).sample(range(6), 6): the order of the six candidates in the list and in the
            sentence, the same in every arm of the core
The disjointness of F and C from U is checked on the canonical tuple, which is stricter than on the rendered one.

Arms (raw user-turn prompts; ckeys.encoding.encode adds the chat wrapper and the "Answer:" prefill): AFTER (list after
the question), BEFORE (list before the story), NONE, POST (the core's sentence after the story), PRE (before it),
POST-NULL (the core's sentence with six nouns from neither lexicon, NULL_NOUNS, in the core's order), and, for S0 only,
P1 and LETTER. With lexicon 1, the canonical order and the ROOM sentence every builder equals
ckeys.encoding.raw_prompt byte for byte (tests/test_fresh.py).

Every sentence has six single-token slots; in every Part-B tokenizer each one is within +-3 tokens of ROOM (it is
+0 to +2), with either lexicon or the null nouns in any order (tests/test_fresh.py; the per-model tokenizer check of
experiments/fresh_factorial.py repeats it on the verified files).
"""
from __future__ import annotations

import functools
import hashlib
import json
import random
import re
from dataclasses import dataclass

from .encoding import LETTERS, ROOM
from .story import LOCATIONS, PREFIX, make_cores, pick_x, record
from .surface import FRAMES_E_FIXED, FRAMES_SIGMA, FormSet

FIELDS = ("agent", "other", "object", "distractor", "initial", "distractor_location", "base", "source")
LEX1 = LOCATIONS
LEX2 = ("bin", "crate", "tray", "jar", "bucket", "chest")
LEXICONS = {1: LEX1, 2: LEX2}
NULL_NOUNS = ("rug", "clock", "mirror", "poster", "radio", "globe")   # neither lexicon, no story object, consonant-initial
ROOM_TEMPLATE = "The room has a {0}, a {1}, a {2}, a {3}, a {4} and a {5}."
SENTENCES = (   # the eight preregistered neutral sentences (after the story in POST, before it in PRE)
    "There is a {0}, a {1}, a {2}, a {3}, a {4} and a {5} in the house.",
    "The hallway also has a {0}, a {1}, a {2}, a {3}, a {4} and a {5}.",
    "In the kitchen there are a {0}, a {1}, a {2}, a {3}, a {4} and a {5}.",
    "A {0}, a {1}, a {2}, a {3}, a {4} and a {5} stand along the wall.",
    "The house contains a {0}, a {1}, a {2}, a {3}, a {4} and a {5}.",
    "Nearby there are a {0}, a {1}, a {2}, a {3}, a {4} and a {5}.",
    "The attic holds a {0}, a {1}, a {2}, a {3}, a {4} and a {5}.",
    "Along one wall sit a {0}, a {1}, a {2}, a {3}, a {4} and a {5}.",
)
assert ROOM_TEMPLATE.format(*LOCATIONS) == ROOM

U_SEEDS, U_N = (0, 1, 2, 3), 1000
U_SHA256 = "abd1f0530a3d08a2058743f59102f8dd6dd360af06fc65dd3277c5b5eb176c3d"     # sorted distinct tuples of U
S0_SHA256 = "fdd1bf1ba4d8d657f663f786c3ff92d0145e41cf02e123e602d594b183110121"    # S0 tuples in drawn order
POPULATIONS = {"F": (150, 20261013), "C": (30, 20261014), "S0": (150, 0)}
# sha256 of json.dumps([[*tuple, lexicon, sentence, order], ...]) in population order (pop_hash), pinned at the build
POP_SHA256 = {"F": "e87047c9c877a21db89bf5081d082de748ea5d77b620e33135d178c3bf24b14f",
              "C": "a625fd13d1dd67bc0c01c3a173807c7b71ee8347451c139d93ffc20f1c6486e9",
              "S0": "48bb0a3ad22463ee1831cabaef714d87b2a7cb7f721e23d877111f19ba0d6900"}

ARMS_F = ("AFTER", "BEFORE", "NONE", "POST", "PRE", "POST-NULL")   # F and C (B-9, LEAN)
ARMS_S0 = ("P1", "AFTER", "BEFORE", "POST", "PRE", "NONE")         # S0 (JB6, JB-G2)
ARMS_CAL = ARMS_F + ("P1",)                                        # frame discovery on C
ARMS_ALL = ARMS_F + ("P1", "LETTER")


def core_tuple(c: dict) -> tuple:
    return tuple(c[f] for f in FIELDS)


def core_hash(c: dict) -> str:
    return hashlib.sha256(json.dumps(list(core_tuple(c))).encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def used_cores() -> frozenset:
    """U: every core tuple of make_cores(1000, Random(s)) for s in 0..3."""
    return frozenset(core_tuple(c) for s in U_SEEDS for c in make_cores(U_N, random.Random(s)))


def u_hash(U=None) -> str:
    U = used_cores() if U is None else U
    return hashlib.sha256(json.dumps(sorted(list(t) for t in U)).encode()).hexdigest()


def fresh_cores(n: int, seed: int, exclude=()) -> list[dict]:
    """The first n cores of the stream make_cores(., Random(seed)) whose tuple is not in U, not in ``exclude`` (tuples)
    and not earlier in the stream (make_cores(n, rng) and n calls of make_cores(1, rng) draw the same stream)."""
    rng, bad, out = random.Random(seed), set(used_cores()) | set(exclude), []
    while len(out) < n:
        c = make_cores(1, rng)[0]
        t = core_tuple(c)
        if t in bad:
            continue
        bad.add(t)
        out.append(c)
    return out


@dataclass(frozen=True)
class Item:
    """One population core with its rendering: ``core`` in the canonical lexicon, ``lex`` 1 or 2, ``order`` the six
    canonical indices in listing order, ``sent`` an index of SENTENCES or None (ROOM, S0)."""
    pop: str
    idx: int
    core: dict
    lex: int
    order: tuple
    sent: int | None

    @property
    def words(self) -> tuple:
        """The six candidate words in canonical index order."""
        return LEXICONS[self.lex]

    def w(self, loc: str) -> str:
        """The rendered word of a canonical location."""
        return self.words[LOCATIONS.index(loc)]

    @property
    def rcore(self) -> dict:
        c = dict(self.core)
        for f in ("initial", "distractor_location", "base", "source"):
            c[f] = self.w(c[f])
        return c

    @property
    def x_canon(self) -> str:
        return pick_x(self.core)

    @property
    def track(self) -> dict:
        """Rendered words of S, B, X and the initial location (the tracked candidates of format_factorial)."""
        c = self.core
        return {"S": self.w(c["source"]), "B": self.w(c["base"]), "X": self.w(self.x_canon), "init": self.w(c["initial"])}

    @property
    def names(self) -> dict:
        c = self.core
        return {"a": c["agent"], "b": c["other"], "o": c["object"], "d": c["distractor"]}

    @property
    def cluster(self) -> tuple:
        """The bootstrap cluster: (lexicon, ordered location pair)."""
        return (self.lex, self.core["base"], self.core["source"])

    @property
    def key(self) -> str:
        return f"{self.pop}{self.idx:03d}"

    def spec(self) -> list:
        return [*core_tuple(self.core), self.lex, self.sent, list(self.order)]

    def meta(self) -> dict:
        return {"key": self.key, "core": self.core, "lex": self.lex, "order": list(self.order), "sent": self.sent,
                "words": list(self.words), "track": self.track, "cluster": list(self.cluster)}


def attributes(cores: list[dict]) -> list[tuple]:
    """(lexicon, sentence, order) per core: lexicon 2 for the half with the smallest hash, sentence = hash rank % 8,
    order seeded by the hash."""
    hs = [core_hash(c) for c in cores]
    rank = {i: r for r, i in enumerate(sorted(range(len(cores)), key=lambda i: hs[i]))}
    half = len(cores) // 2
    return [(2 if rank[i] < half else 1, rank[i] % len(SENTENCES), tuple(random.Random(int(hs[i][:16], 16)).sample(range(6), 6)))
            for i in range(len(cores))]


@functools.lru_cache(maxsize=None)
def population(name: str) -> tuple:
    """The Items of population F, C or S0 (in drawn order)."""
    n, seed = POPULATIONS[name]
    if name == "S0":
        return tuple(Item("S0", i, c, 1, tuple(range(6)), None) for i, c in enumerate(make_cores(n, random.Random(seed))))
    exclude = {core_tuple(it.core) for it in population("F")} if name == "C" else set()
    cores = fresh_cores(n, seed, exclude)
    return tuple(Item(name, i, c, lex, order, sent) for i, (c, (lex, sent, order)) in enumerate(zip(cores, attributes(cores))))


def pop_hash(items) -> str:
    return hashlib.sha256(json.dumps([it.spec() for it in items]).encode()).hexdigest()


# --------------------------------------------------------------------------- prompts
def listing(it: Item) -> str:
    return "Choices: " + ", ".join(it.words[i] for i in it.order)


def letter_listing(it: Item) -> str:
    return "Choices: " + ", ".join(f"{L}) {it.words[i]}" for L, i in zip(LETTERS, it.order))


def sentence(it: Item, null: bool = False) -> str:
    t = ROOM_TEMPLATE if it.sent is None else SENTENCES[it.sent]
    nouns = NULL_NOUNS if null else it.words
    return t.format(*[nouns[i] for i in it.order])


def raw_prompt(it: Item, arm: str, location: str) -> str:
    """The user-turn prompt of ``arm`` for the item with the moved-to location ``location`` (canonical)."""
    rec = record(it.rcore, "direct", it.w(location))
    story, query = rec["story"], rec["query"]
    one = "\nAnswer with one word.\nAnswer:"
    if arm == "AFTER":
        return PREFIX + story + "\nQuestion: " + query + "\n" + listing(it) + one
    if arm == "BEFORE":
        return "Read the story and answer the question.\n" + listing(it) + "\n\nStory: " + story + "\nQuestion: " + query + one
    if arm == "NONE":
        return PREFIX + story + "\nQuestion: " + query + one
    if arm == "POST":
        return PREFIX + story + " " + sentence(it) + "\nQuestion: " + query + one
    if arm == "PRE":
        return PREFIX + sentence(it) + " " + story + "\nQuestion: " + query + one
    if arm == "POST-NULL":
        return PREFIX + story + " " + sentence(it, null=True) + "\nQuestion: " + query + one
    if arm == "P1":
        return PREFIX + story + "\nQuestion: " + query + "\n" + listing(it) + "\nAnswer with exactly one choice.\nAnswer:"
    if arm == "LETTER":
        return PREFIX + story + "\nQuestion: " + query + "\n" + letter_listing(it) + "\nAnswer with one letter.\nAnswer:"
    raise ValueError(arm)


def span_text(it: Item, arm: str) -> str | None:
    """The re-mention text of the arm (list, sentence or null sentence), or None."""
    return {"AFTER": listing, "BEFORE": listing, "P1": listing, "LETTER": letter_listing, "POST": sentence,
            "PRE": sentence, "POST-NULL": lambda i: sentence(i, True)}.get(arm, lambda i: None)(it)


# --------------------------------------------------------------------------- answers, forms and frames
@functools.lru_cache(maxsize=None)
def cand_re(words: tuple) -> re.Pattern:
    """The item's candidate pattern, as ckeys.generate.CAND_RE for its words (case-insensitive, plurals allowed)."""
    return re.compile(r"(?i)\b(" + "|".join(words) + r")")


def parse(text: str, words) -> str:
    """The first of ``words`` named in ``text`` (lower case), else 'other'."""
    m = cand_re(tuple(words)).search(text)
    return m.group(1).lower() if m else "other"


def stop_fn(tok, words):
    """greedy's stop(r, ids): True once the decoded text names a candidate followed by a non-letter, or ends with a
    complete candidate word (ckeys.generate.candidate_stop for the item's words)."""
    pat = cand_re(tuple(words))

    def f(_r, ids):
        s = tok.decode(ids)
        m = pat.search(s)
        return bool(m) and (m.end() < len(s) or s.endswith(m.group(0)))
    return f


def gen_text(tok, ids, gen) -> str:
    """The text a generation adds to the prompt, decoded together with the end of the prompt (sentencepiece decoders
    drop the leading space of a sequence's first token)."""
    ids = list(ids)[-16:]
    pre, full = tok.decode(ids), tok.decode(ids + list(gen))
    return full[len(pre):] if full.startswith(pre) else tok.decode(list(gen))


def extract_frame(text: str, names: dict, words) -> str | None:
    """ckeys.generate.extract_frame for the item's words: the text before the first candidate, the item's names
    replaced by {a}, {b}, {o}, {d}; None without a candidate, or with a newline, or longer than 60 characters."""
    m = cand_re(tuple(words)).search(text)
    if not m:
        return None
    fr = text[:m.start()]
    if "\n" in fr or len(fr) > 60:
        return None
    for k in sorted(names, key=lambda k: -len(names[k])):
        if names[k]:
            fr = re.sub(r"\b" + re.escape(names[k]) + r"\b", "{" + k + "}", fr)
    return fr


def frame_sets(disc=()) -> dict:
    """The nested form sets of the three scorings' trie: sigma (12 forms) and E (the 32 fixed forms plus the discovered
    frames); L, the lower-case ' w', is a member of both."""
    return {"sigma": FRAMES_SIGMA, "E": tuple(FRAMES_E_FIXED) + tuple(f for f in disc if f not in FRAMES_E_FIXED)}


def form_set(tok, it: Item, disc=()):
    """The item's FormSet over its six words (canonical order) and its names; a discovered frame whose forms break the
    disjointness rule for this item (one sequence a proper prefix of another) is dropped for this item, the last
    admitted first. Returns (FormSet, dropped frames)."""
    disc, dropped = list(disc), []
    while True:
        try:
            return FormSet(tok, it.words, frame_sets(disc), names=it.names), dropped
        except ValueError:
            if not disc:
                raise
            dropped.insert(0, disc.pop())
