"""Stage 9 (docs/stage9/SPEC.md, predictions docs/STAGE9_PREDICTIONS.md): populations, onset grids, the letter parser and
stop rule of the stories arm, and the run metadata every stage-9 results file records. The API is fixed by SPEC section
5.5; the other stage-9 modules (experiments/stage9_stories.py, stage9_passages.py, stage9_edits.py and the scorer) import
it. Everything here is pure apart from the cached exclusion set and the git / file reads of ``run_meta``.

Populations (story cores compared by the full tuple ckeys.fresh.FIELDS, in the canonical lexicon):
  X9  U (ckeys.fresh.used_cores(): seeds 0-3, 1000 each) and the first N_STREAM = 3000 draws of make_cores(1, Random(s))
      for every seed s in STREAM_SEEDS (every seed that drew story cores before stage 9). 45,832 tuples.
  N9  ckeys.fresh.fresh_cores(150, 9101, X9): the first 150 cores of the Random(9101) stream in neither U nor X9 nor
      earlier in the stream (161 draws), with the attributes of ckeys.fresh.attributes (lexicon, sentence, order), as
      F. Items fresh.Item("N9", i, core, lex, order, sent); pinned by fresh.pop_hash and by the tuples' sha256.
  C   ckeys.fresh.population("C"): the 30 stage-8b calibration cores, used only to fix l_b before N9 is scored.
  E9  ckeys.edits._take(9102, 80, U | X9 | tuples(N9), ckeys.edits.cond_E) (173 draws), canonical lexicon, order and
      ROOM sentence: fresh.Item("E9", i, core, 1, (0, 1, 2, 3, 4, 5), None); pinned by ckeys.edits.pop_hash.

Onsets: ``story_onsets(L)`` = every second block from 2 to L - 2; ``passage_onsets(L)`` = round(x L) for x in PASSAGE_X.

LETTER answers (SPEC 9.2): s = the text after ``lstrip()`` and the skip of leading markdown, brackets and quotes (the
skip set of ckeys.natural_formats.letter_of). A letter answer is s[0] in A-F not followed by a letter; it names the word
the item's listing gives that letter, ``it.words[it.order["ABCDEF".index(s[0])]]``. Otherwise the word parser of
ckeys.fresh decides (a named candidate, or "other").
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

from . import edits, fresh
from .story import make_cores

ROOT = Path(__file__).resolve().parents[1]
NOTE = "docs/STAGE9_PREDICTIONS.md"
SPEC = "docs/stage9/SPEC.md"
FREEZE_SUBJECT = "Freeze stage-9 predictions"
# run_meta's git_dirty reads only these paths: the code directories the guard freezes (scripts/stage9_common.sh
# S9_CODE_DIRS), the prediction note and the spec directory
DIRTY_PATHS = ("ckeys", "experiments", "analysis", "scripts", "tests", "data", NOTE, "docs/stage9/SPEC.md")

SEEDS = {"N9": 9101, "E9": 9102}
N_POP = {"N9": 150, "E9": 80}
STREAM_SEEDS = (7, 8, 9, 99, 101, 202, 20261011, 20261013, 20261014, 8101, 8102, 8103, 8104, 81, 84)
N_STREAM = 3000
X9_SIZE = 45832
DRAWS = {"N9": 161, "E9": 173}
POP_SHA256 = {"N9": "0cf411ab8fe99ec495adfb30944fad69c6b65964fabde849b848cc6358d38da3",          # fresh.pop_hash(items)
              "N9_tuples": "5b87f7732bd94b7143f3dd94fe65d58cb59884e0fe0e5fe4b3432a765500a2aa",
              "E9": "e4daeebd5227804ed0931302ab93d72b42b2a2d7af6222d922d8e555945c2453",          # edits.pop_hash(cores)
              "C": fresh.POP_SHA256["C"]}
STORY_FORMATS = ("NONE", "BEFORE", "POST-NULL", "POST", "AFTER", "LETTER")   # run order
EDIT_FORMATS = ("NONE", "AFTER", "LETTER", "POST")
PASSAGE_FORMATS = ("NOM", "OPTB", "OPTA")            # LETA: the deadline-gated extra step
EDIT_BLOCKS, DOSE_BLOCKS = (3, 7, 11, 15), (3, 7)
KV_ONLY_ONSETS = (4, 8, 12, 16)                      # stories: the stage-8c onsets l + 1 of blocks 3, 7, 11, 15
PASSAGE_X = (0.15, 0.3, 0.45, 0.6)
BOOT_SEED, NB = 20261015, 10000

# the committed stage-8 artefacts stage 9 reads (SPEC section 4), path relative to the repository root -> sha256
ARTEFACT_SHA256 = {
    "results/gpu_stage8b/frames/mistral7.json": "9434618bf9d10464c1daafa9fe8b6fd45cb11596b8cb668e75eebdfdb6ccb953",
    "results/gpu_stage8b/frames/llama8.json": "da3daabc1b59219f4484fee43a2c81e19dbbc4419ad4bc622d3206fe94570486",
    "results/gpu_stage8b/frames/qwen7.json": "ae18b810014aed61658ed99b73c9cd429e144ca8a65a3135502fe213a96a3f4a",
    "results/gpu_stage8b/g3/mistral7.json": "dc70f763b0d8f36c65fa7cc24a833a6f2beb2e65d32dee6dce518b48aed67261",
    "results/gpu_stage8b/g3/llama8.json": "da2d4c8a88cf1d394519cf1e57d60751cc13fe88ed9342010e069d651f4431b7",
    "results/gpu_stage8b/g3/qwen7.json": "9b0810b4b73f65d5ba54fb4a86eec06b9c0533375e285296c49fac816778d08e",
    "results/gpu_stage8a/frames/mistral7.json": "e3b5491224639310f539aa19638a81219a33f9985ac99da2492b7dc810919290",
    "results/gpu_stage8a/frames/llama8.json": "10a02e134ee9fecf84bbffcd2a83e0c2c529678c90a9eb122d5beed7c8228e75",
    "results/gpu_stage8a/frames/qwen7.json": "cb8d855f94db4179060fc1ab8b6eca80e8079ed09453469b86b91925f43fa86c",
    "results/gpu_stage8c/calib/qwen7.pt": "5b379fcb2ab35a9de1a4012d77ff450d64a43bb321a290d6b1878ce9da218b07",
    "results/gpu_stage8c/calib/mistral7.pt": "351d52871ec83c04738072c2a3dfddf4ef13697776410fcb2496fe119434cb1b",
    "results/gpu_stage8c/calib/llama8.pt": "26085156a0f1ff6c657a1bb3c647675586c16d6740402166223a48fc10565bcb",
    "results/gpu_stage8c/das/qwen7.pt": "aab335615a7d75bc74331a6681c79cff2241d94ea0c57a859c28fa94df80cc00",
    "results/gpu_stage8c/das/mistral7.pt": "0c2069fd7180d9aed727c60ab30579c281f01db115007b10a311df02caa3f912",
    "results/gpu_stage8a/squad/dev-v1.1.json": "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
}

_ABC = "ABCDEF"                     # the LETTER answer letters (ckeys.encoding.LETTERS as a string)
LETTER_SKIP = "*_`([\"' "          # ckeys.natural_formats.letter_of's skip set (after lstrip)

_T0 = datetime.datetime.now(datetime.timezone.utc)


# --------------------------------------------------------------------------- populations
def tuples(cores) -> list[tuple]:
    return [fresh.core_tuple(c) for c in cores]


@functools.lru_cache(maxsize=None)
def excluded() -> frozenset:
    """X9: U and the first N_STREAM draws of make_cores(1, Random(s)) for every s in STREAM_SEEDS (make_cores(3000, rng)
    draws the same 3000 cores)."""
    X = set(fresh.used_cores())
    for s in STREAM_SEEDS:
        X |= {fresh.core_tuple(c) for c in make_cores(N_STREAM, random.Random(s))}
    return frozenset(X)


def draw(seed: int, n: int, excl, cond=lambda c: True) -> tuple[list[dict], int]:
    """The first n cores of the make_cores(1, Random(seed)) stream not in ``excl``, not earlier in the stream and
    satisfying ``cond`` (the rule of ckeys.fresh.fresh_cores and ckeys.edits._take), and the number of draws."""
    rng, seen, out, k = random.Random(seed), set(), [], 0
    while len(out) < n:
        c = make_cores(1, rng)[0]
        k += 1
        t = fresh.core_tuple(c)
        if t in excl or t in seen:
            continue
        seen.add(t)
        if cond(c):
            out.append(c)
    return out, k


@functools.lru_cache(maxsize=None)
def _story_cores() -> tuple:
    cores = fresh.fresh_cores(N_POP["N9"], SEEDS["N9"], excluded())
    h = hashlib.sha256(json.dumps([list(t) for t in tuples(cores)]).encode()).hexdigest()
    assert h == POP_SHA256["N9_tuples"], f"N9 tuples {h} differ from the pin"
    return tuple(cores)


def story_cores() -> list[dict]:
    """The N9 cores in drawn order (tuples' sha256 asserted)."""
    return [dict(c) for c in _story_cores()]


@functools.lru_cache(maxsize=None)
def stories() -> tuple:
    """The N9 fresh.Items (lexicon, sentence and order as F; fresh.pop_hash asserted)."""
    cores = story_cores()
    items = tuple(fresh.Item("N9", i, c, lex, order, sent)
                  for i, (c, (lex, sent, order)) in enumerate(zip(cores, fresh.attributes(cores))))
    assert fresh.pop_hash(items) == POP_SHA256["N9"], f"N9 {fresh.pop_hash(items)} differs from the pin"
    return items


def calibration() -> tuple:
    """fresh.population("C"), the 30 stage-8b calibration cores (pop_hash asserted)."""
    items = fresh.population("C")
    assert fresh.pop_hash(items) == POP_SHA256["C"], "C differs from its pin"
    return items


@functools.lru_cache(maxsize=None)
def _edit_cores() -> tuple:
    excl = set(fresh.used_cores()) | set(excluded()) | set(tuples(_story_cores()))
    cores = edits._take(SEEDS["E9"], N_POP["E9"], excl, edits.cond_E)
    assert edits.pop_hash(cores) == POP_SHA256["E9"], f"E9 {edits.pop_hash(cores)} differs from the pin"
    return tuple(cores)


def edit_cores() -> list[dict]:
    """The E9 cores in drawn order (edits.pop_hash asserted)."""
    return [dict(c) for c in _edit_cores()]


def edit_items() -> tuple:
    """E9 as fresh.Items: canonical lexicon (1), canonical order, the ROOM sentence (sent None)."""
    return tuple(fresh.Item("E9", i, c, 1, tuple(range(6)), None) for i, c in enumerate(edit_cores()))


# --------------------------------------------------------------------------- onsets
def story_onsets(nL: int) -> list[int]:
    """Every second block from 2 to L - 2: 32 -> 2..30 (15), 28 -> 2..26 (13), 24 -> 2..22 (11)."""
    return list(range(2, nL - 1, 2))


def passage_onsets(nL: int) -> list[int]:
    """round(x L) for x in PASSAGE_X: 32 -> [5, 10, 14, 19]; 28 -> [4, 8, 13, 17]; 24 -> [4, 7, 11, 14]."""
    return [round(x * nL) for x in PASSAGE_X]


# --------------------------------------------------------------------------- LETTER answers
def letter_text(text: str) -> str:
    """s of SPEC 9.2: the text after lstrip() and the skip of leading markdown, brackets and quotes."""
    return text.lstrip().lstrip(LETTER_SKIP)


def parse_letter(text: str, it) -> tuple[str, str]:
    """(answer, how) of a LETTER generation: (the word the letter names, "letter") if s starts with a letter A-F not
    followed by another letter; else (the first candidate word named, "word") or ("other", "none"). An article and a
    word ("A drawer") is therefore option A, as SPEC 9.2 reads it; the stories arm stores the word parse beside the
    answer ("word") so such conflicts can be counted."""
    s = letter_text(text)
    if s and s[0] in _ABC and (len(s) == 1 or not s[1].isalpha()):
        return it.words[it.order[_ABC.index(s[0])]], "letter"
    w = fresh.parse(text, it.words)
    return w, ("word" if w != "other" else "none")


def letter_stop(tok, it):
    """greedy's stop(r, ids) under LETTER: s (SPEC 9.2) has at least two characters, no U+FFFD (an incomplete multi-byte
    character) in its first two, starts with a letter A-F and its second character is not a letter; or the word stop
    (ckeys.fresh.stop_fn over the item's words) fires."""
    word = fresh.stop_fn(tok, it.words)

    def f(r, ids):
        s = letter_text(tok.decode(ids))
        if len(s) >= 2 and "\ufffd" not in s[:2] and s[0] in _ABC and not s[1].isalpha():
            return True
        return word(r, ids)
    return f


# --------------------------------------------------------------------------- provenance helpers
def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _utc(t: datetime.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def run_meta(test: bool | None = None) -> dict:
    """SPEC 11.1's run fields: git_commit, git_dirty (tracked files modified in DIRTY_PATHS: the code directories, the
    prediction note and the spec; a modified paper/ or other file outside them does not count), freeze_commit
    ($STAGE9_FREEZE, exported by the guard; None in TEST_MODE), the sha256 of the prediction note and of the spec in the
    working tree (None if absent), started_utc (when this module was imported, i.e. the process start) and finished_utc
    (now: call it when the file is written). ``test`` None reads TEST_MODE from the environment."""
    test = bool(os.environ.get("TEST_MODE")) if test is None else test
    st = _git("status", "--porcelain", "--untracked-files=no", "--", *DIRTY_PATHS)
    sha = {k: (file_sha(ROOT / p) if (ROOT / p).exists() else None) for k, p in (("predictions_sha256", NOTE),
                                                                               ("spec_sha256", SPEC))}
    return {"git_commit": _git("rev-parse", "HEAD"), "git_dirty": None if st is None else bool(st),
            "freeze_commit": None if test else (os.environ.get("STAGE9_FREEZE") or None), **sha,
            "started_utc": _utc(_T0), "finished_utc": _utc(datetime.datetime.now(datetime.timezone.utc))}
