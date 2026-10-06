"""Belief-tracking story template, ported from Paper 1's released generator (gpu/runtime/story_task.py).

Base and source stories differ only in the critical location named by the move event. The base story
mentions exactly three locations: the object's initial location, the distractor object's location and
the critical location. ``choices=False`` drops the answer-choices line so that no other location
appears in context.
"""
from __future__ import annotations

import hashlib
import json
import random

LOCATIONS = ("box", "basket", "shelf", "drawer", "cabinet", "closet")
PAIR_SWAP = dict(zip(LOCATIONS, ("basket", "box", "drawer", "shelf", "closet", "cabinet")))
SIX_CYCLE = dict(zip(LOCATIONS, LOCATIONS[1:] + LOCATIONS[:1]))
AGENTS = ("Alice", "Bob")
OBJECTS = ("candle", "ticket", "toy", "letter", "notebook", "map", "key", "coin", "ring", "book",
           "hat", "cup", "ball", "pen", "watch", "phone", "card", "bag", "lamp", "shoe")
PREFIX = "Read the story and answer the question.\n\nStory: "

QUERIES = {
    "direct": "Where does {a} believe the {o} is?",
    "world": "Where is the {o} actually located?",
    "other_agent": "Where does {b} believe the {o} is?",
    "irrelevant_object": "Where does {a} believe the {d} is?",
}


def make_cores(n: int, rng: random.Random, distinct: bool = True) -> list[dict]:
    cores = []
    while len(cores) < n:
        a, b = rng.sample(AGENTS, 2)
        o, d = rng.sample(OBJECTS, 2)
        init, dloc, base, src = (rng.choice(LOCATIONS) for _ in range(4))
        if distinct and (len({base, src, init}) < 3 or dloc in (base, src)):
            continue
        cores.append(dict(agent=a, other=b, object=o, distractor=d, initial=init,
                          distractor_location=dloc, base=base, source=src))
    return cores


def pick_x(core: dict) -> str:
    """Third location X, absent from the story and from {base, source}; seeded by the core."""
    used = {core["base"], core["source"], core["initial"], core["distractor_location"]}
    pool = [l for l in LOCATIONS if l not in used]
    seed = int(hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:8], 16)
    return random.Random(seed).choice(pool)


def record(core: dict, view: str, location: str) -> dict:
    a, b, o, d = core["agent"], core["other"], core["object"], core["distractor"]
    initial = " ".join(f"Everyone initially sees that the {obj} is in the {where}."
                       for obj, where in sorted([(o, core["initial"]), (d, core["distractor_location"])]))
    event = f"{a} watches as the {o} is moved to the {location}. {b} does not see this happen."
    answer = {"other_agent": core["initial"], "irrelevant_object": core["distractor_location"]}.get(view, location)
    return {"story": initial + " " + event, "query": QUERIES[view].format(a=a, b=b, o=o, d=d), "answer": answer}


def prompt(rec: dict, choices: bool | str = True) -> str:
    """``choices``: True/"after" lists choices after the question, "before" lists them before the story
    (so choice tokens cannot attend to story tokens), False omits them."""
    listing = "Choices: " + ", ".join(LOCATIONS)
    if choices == "before":
        return ("Read the choices, then the story, and answer the question.\n" + listing + "\n\nStory: "
                + rec["story"] + "\nQuestion: " + rec["query"] + "\nAnswer with exactly one choice.\nAnswer:")
    text = PREFIX + rec["story"] + "\nQuestion: " + rec["query"]
    if choices:
        text += "\n" + listing + "\nAnswer with exactly one choice."
    else:
        text += "\nAnswer with one word."
    return text + "\nAnswer:"
