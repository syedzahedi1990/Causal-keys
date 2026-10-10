"""The 24 frozen neutral sentences of stage 8, part C (preregistration J): the contexts of the steering vectors E2 (the
English location word) and E5 (its French, German or synonym form, ckeys/variants.py _F), and of the 5-dimensional
English lexical span L_l used by the LEX / NONLEX split.

Each sentence has exactly one slot {x}, filled with one word. No sentence describes movement, belief, a question or
containment, and no sentence names a location word of the story template outside its slot. A sentence is the user turn
of the model's chat template (system turn "You are a helpful assistant.", generation prompt, no prefill:
ckeys.encoding.encode(tok, sentence, prefill="")). The position read is the last token of " " + form (the word with its
leading space), found as a contiguous token subsequence that must occur exactly once.

SENTENCES_SHA256 = sha256(json.dumps(list(SENTENCES))) is pinned here and in the entry; tests/test_stage8_edits.py checks it.
Sentences 1-12 are those of the design pilot (scratchpad partC/pilot_c.py NEUTRAL); 13-24 were written for the entry.
"""
from __future__ import annotations

import hashlib
import json

import torch

from .encoding import encode
from .story import LOCATIONS
from .variants import _F

SENTENCES = (
    "The old {x} in the hallway was painted green last spring.",
    "She bought a new {x} at the market on Saturday.",
    "Every {x} in the shop had a small price tag.",
    "My uncle builds a wooden {x} for every new neighbour.",
    "A dusty {x} stood quietly in the corner of the attic.",
    "The catalogue lists one {x} under the heading of home goods.",
    "He wrote the word {x} on the whiteboard during the lesson.",
    "The museum displays a carved {x} from the eighteenth century.",
    "Our neighbour sells a sturdy {x} for a fair price.",
    "The designer sketched a modern {x} for the new apartment.",
    "Children like to draw a big {x} with bright colours.",
    "The hotel room had a large {x} near the window.",
    "The carpenter repaired a broken {x} last winter.",
    "A bright red {x} appeared in the advertisement.",
    "The artist painted a small {x} on the canvas.",
    "Grandmother admired an antique {x} at the fair.",
    "The store window showed a shiny {x} beside the door.",
    "Their teacher described a large {x} during the art class.",
    "The magazine photographed a modern {x} for its cover.",
    "My friend restored an old {x} with fresh varnish.",
    "The auction offered a rare {x} from a famous collection.",
    "A clean {x} stood next to the office desk on Monday.",
    "The workshop produced a sturdy {x} every week.",
    "Visitors noticed a tall {x} near the entrance.",
)
SENTENCES_SHA256 = "593ce65ce4eb5b468527c6447601cc6670f9576e6434398dc105fb6c21bcce2a"

# the six forms of each family, in LOCATIONS order (E2: the English word; E5: the variants of ckeys/variants.py)
FORMS = {"EN": tuple(LOCATIONS), "FR": _F["FR"], "DE": _F["DE"], "SYN": _F["SYN"]}
MIN_SENTENCES = 20      # a family is usable in a tokenizer only if at least this many sentences place all six forms cleanly


def sentences_sha256() -> str:
    return hashlib.sha256(json.dumps(list(SENTENCES)).encode()).hexdigest()


def form_tokens(tok, form: str) -> list[int]:
    return tok(" " + form, add_special_tokens=False).input_ids


def token_pos(tok, ids, form: str) -> int:
    """The position of the last token of " " + form in ``ids`` ([1, T] or a list); asserts exactly one occurrence."""
    seq = ids[0].tolist() if isinstance(ids, torch.Tensor) else list(ids)
    w = form_tokens(tok, form)
    hits = [i + len(w) - 1 for i in range(len(seq) - len(w) + 1) if seq[i:i + len(w)] == w]
    assert len(hits) == 1, (form, hits)
    return hits[0]


def sentence_ids(tok, sentence: str, form: str) -> tuple[torch.Tensor, int]:
    """(ids [1, T], position of the form's last token) of one neutral sentence with ``form`` in its slot."""
    ids = encode(tok, sentence.format(x=form), prefill="")
    return ids, token_pos(tok, ids, form)


def clean_sentences(tok, family: str) -> list[int]:
    """Indices of the sentences in which all six forms of ``family`` occur exactly once as their standalone token
    sequence (the same sentences are then used for all six forms, so a mean difference holds the context fixed)."""
    ok = []
    for i, s in enumerate(SENTENCES):
        try:
            for f in FORMS[family]:
                sentence_ids(tok, s, f)
        except AssertionError:
            continue
        ok.append(i)
    return ok


def check(tok) -> dict:
    """Per family: the clean sentences and whether the family is usable (>= MIN_SENTENCES); E2 (EN) must use all 24."""
    rep = {}
    for fam in FORMS:
        idx = clean_sentences(tok, fam)
        rep[fam] = {"clean": idx, "n": len(idx), "usable": len(idx) >= MIN_SENTENCES,
                    "n_tokens": {f: len(form_tokens(tok, f)) for f in FORMS[fam]}}
    return rep
