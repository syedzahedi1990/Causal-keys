"""Prompt encoding matched to Paper 1 (gpu/runtime/mistral_engine.encode_record), plus format arms.

Paper 1 wraps the raw prompt as the user turn after a "You are a helpful assistant." system turn,
adds the generation prompt, and appends an "Answer:" assistant prefill. Format arms vary only where
(and whether) the candidate answers are re-mentioned relative to the critical state token.
"""
from __future__ import annotations

import torch

from .story import LOCATIONS, PREFIX

LETTERS = ("A", "B", "C", "D", "E", "F")
LISTING = "Choices: " + ", ".join(LOCATIONS)
LETTER_LISTING = "Choices: " + ", ".join(f"{L}) {loc}" for L, loc in zip(LETTERS, LOCATIONS))
ROOM = "The room has a box, a basket, a shelf, a drawer, a cabinet and a closet."

# arm -> (raw prompt builder, candidate alphabet)
def raw_prompt(arm: str, story: str, query: str) -> str:
    if arm == "P1":       # Paper 1's exact format
        return PREFIX + story + "\nQuestion: " + query + "\n" + LISTING + "\nAnswer with exactly one choice.\nAnswer:"
    if arm == "AFTER":    # listing after the state token, free-form instruction
        return PREFIX + story + "\nQuestion: " + query + "\n" + LISTING + "\nAnswer with one word.\nAnswer:"
    if arm == "BEFORE":   # listing before the story: listing tokens cannot attend to the state token
        return ("Read the story and answer the question.\n" + LISTING + "\n\nStory: " + story +
                "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:")
    if arm == "NONE":     # no re-mention
        return PREFIX + story + "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:"
    if arm == "POST":     # neutral (non-MC) re-mention after the state token
        return PREFIX + story + " " + ROOM + "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:"
    if arm == "PRE":      # neutral re-mention before the state token
        return PREFIX + ROOM + " " + story + "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:"
    if arm == "LETTER":   # lettered options after the state token, answer is a letter
        return PREFIX + story + "\nQuestion: " + query + "\n" + LETTER_LISTING + "\nAnswer with one letter.\nAnswer:"
    raise ValueError(arm)


ARMS = ("P1", "AFTER", "BEFORE", "NONE", "POST", "PRE", "LETTER")


def alphabet(arm: str) -> tuple[str, ...]:
    return LETTERS if arm == "LETTER" else LOCATIONS


def encode(tok, raw: str, system: str = "You are a helpful assistant.", prefill: str = "Answer:") -> torch.Tensor:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": raw}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text + prefill, add_special_tokens=False, return_tensors="pt").input_ids


def candidate_ids(tok, arm: str) -> list[int]:
    ids = []
    for c in alphabet(arm):
        t = tok.encode(" " + c, add_special_tokens=False)
        assert len(t) == 1, f"candidate {c!r} is not a single token"
        ids.append(t[0])
    return ids
