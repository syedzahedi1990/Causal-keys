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


WRAPPER_USED = {"system_merged": False}


def chat_text(tok, raw: str, system: str = "You are a helpful assistant.", prefill: str = "Answer:") -> str:
    """Paper 1 wrapper: system turn + user turn + generation prompt (thinking disabled) + prefill.

    Templates that reject a system role (e.g. Gemma-2) get the system text merged into the user turn;
    this is recorded in WRAPPER_USED. ``enable_thinking`` is ignored by templates that do not use it.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": raw}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception as e:  # jinja2 TemplateError for unsupported system role
        if not system or "system" not in str(e).lower():
            raise
        WRAPPER_USED["system_merged"] = True
        text = tok.apply_chat_template([{"role": "user", "content": system + "\n\n" + raw}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    assert text.count(raw) == 1, "wrapper must contain the raw prompt exactly once"
    return text + prefill


def encode(tok, raw: str, system: str = "You are a helpful assistant.", prefill: str = "Answer:") -> torch.Tensor:
    return tok(chat_text(tok, raw, system, prefill), add_special_tokens=False, return_tensors="pt").input_ids


def candidate_ids(tok, arm: str) -> list[int]:
    ids = []
    for c in alphabet(arm):
        t = tok.encode(" " + c, add_special_tokens=False)
        assert len(t) == 1, f"candidate {c!r} is not a single token"
        ids.append(t[0])
    return ids
