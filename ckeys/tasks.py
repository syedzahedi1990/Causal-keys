"""Additional in-context state tasks with the same format arms as Paper 1's belief task (generality, stage 3).

Each task writes a state value at a single critical token in the final story sentence; base and source
stories differ only at that token. Format arms mirror ckeys.encoding.raw_prompt:
  P1      options listed after the question, "Answer with exactly one choice."
  NONE    no options, "Answer with one word."
  BEFORE  options listed before the story
  POST    a neutral sentence re-mentioning every candidate, placed after the story, free-form answer
  LETTER  lettered options after the question, answer is a letter
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

PREFIX = "Read the story and answer the question.\n\nStory: "
LETTERS = "ABCDEFGHIJ"
NAMES = ("Alice", "Bob", "Carol", "David", "Emma", "Frank")


@dataclass(frozen=True)
class Task:
    name: str
    values: tuple
    things: tuple
    initial: str          # "{thing} ... {value}" sentence for a thing's starting state
    update: str           # "{agent} ... {thing} ... {value}" sentence ending with the critical value
    query: str
    remention: str        # neutral sentence listing every candidate
    unit: str = "one word"
    meta: dict = field(default_factory=dict)

    def make_cores(self, n: int, rng: random.Random) -> list[dict]:
        cores = []
        while len(cores) < n:
            agent = rng.choice(NAMES)
            thing, dthing = rng.sample(self.things, 2)
            init, dval, base, src = (rng.choice(self.values) for _ in range(4))
            if len({base, src, init}) < 3 or dval in (base, src):
                continue
            cores.append(dict(agent=agent, object=thing, distractor=dthing, initial=init,
                              distractor_location=dval, base=base, source=src))
        return cores

    def story(self, core: dict, value: str) -> str:
        first = sorted([(core["object"], core["initial"]), (core["distractor"], core["distractor_location"])])
        intro = " ".join(self.initial.format(thing=t, value=v) for t, v in first)
        return intro + " " + self.update.format(agent=core["agent"], thing=core["object"], value=value)

    def question(self, core: dict) -> str:
        return self.query.format(thing=core["object"])

    def alphabet(self, arm: str) -> tuple:
        return tuple(LETTERS[: len(self.values)]) if arm == "LETTER" else self.values

    def raw_prompt(self, arm: str, story: str, query: str) -> str:
        listing = "Choices: " + ", ".join(self.values)
        letters = "Choices: " + ", ".join(f"{L}) {v}" for L, v in zip(LETTERS, self.values))
        if arm == "P1":
            return PREFIX + story + "\nQuestion: " + query + "\n" + listing + "\nAnswer with exactly one choice.\nAnswer:"
        if arm == "NONE":
            return PREFIX + story + "\nQuestion: " + query + f"\nAnswer with {self.unit}.\nAnswer:"
        if arm == "BEFORE":
            return ("Read the story and answer the question.\n" + listing + "\n\nStory: " + story +
                    "\nQuestion: " + query + f"\nAnswer with {self.unit}.\nAnswer:")
        if arm == "POST":
            return PREFIX + story + " " + self.remention + "\nQuestion: " + query + f"\nAnswer with {self.unit}.\nAnswer:"
        if arm == "LETTER":
            return PREFIX + story + "\nQuestion: " + query + "\n" + letters + "\nAnswer with one letter.\nAnswer:"
        raise ValueError(arm)


def _list(vals):
    return ", ".join(vals[:-1]) + " and " + vals[-1]


COLORS = ("red", "blue", "green", "yellow", "black", "white")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

TASKS = {
    "paint": Task(
        name="paint", values=COLORS,
        things=("fence", "door", "bench", "shed", "gate", "chair", "table", "wall", "boat", "bike"),
        initial="The {thing} was painted {value}.",
        update="Later, {agent} repainted the {thing} {value}.",
        query="What color is the {thing} now?",
        remention=f"The shop sells {_list(COLORS)} paint."),
    "schedule": Task(
        name="schedule", values=DAYS,
        things=("meeting", "dinner", "appointment", "party", "class", "interview", "concert", "match", "lecture", "trip"),
        initial="The {thing} was scheduled for {value}.",
        update="Later, {agent} moved the {thing} to {value}.",
        query="On which day is the {thing} now?",
        remention=f"The calendar shows {_list(DAYS)}."),
}
