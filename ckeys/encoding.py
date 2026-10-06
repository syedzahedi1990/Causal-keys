"""Prompt encoding matched to Paper 1 (gpu/runtime/mistral_engine.encode_record), plus format arms.

Paper 1 wraps the raw prompt as the user turn after a "You are a helpful assistant." system turn,
adds the generation prompt, and appends an "Answer:" assistant prefill. Format arms vary only where
(and whether) the candidate answers are re-mentioned relative to the critical state token.

Arm registry: ``register_arm`` adds arms from other modules (subset, variant, ...) without editing the
experiment scripts; ``build_prompt(arm, story, query, core, X)`` dispatches any registered arm, and
``raw_prompt`` keeps its core-free signature for the standard arms and core-independent registered arms.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

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
    if arm in ARM_BUILDERS and not ARM_BUILDERS[arm].needs_core:
        return ARM_BUILDERS[arm].build(story, query)
    raise ValueError(arm)


ARMS = ("P1", "AFTER", "BEFORE", "NONE", "POST", "PRE", "LETTER")


@dataclass(frozen=True)
class Arm:
    build: Callable[..., str]                 # (story, query) -> raw prompt; (story, query, core, X) if needs_core
    needs_core: bool = False
    span: Callable[..., str] | None = None    # same arguments -> the re-mention text (row groups of the splice)
    letters: bool = False                     # the answer alphabet is LETTERS, not LOCATIONS
    forms: Callable | None = None             # tok -> six token ids, the arm's own form of each location
    meta: Callable | None = None              # (core, X) -> dict stored with every item of the arm


ARM_BUILDERS: dict[str, Arm] = {}
ALL_ARMS: list[str] = list(ARMS)


def register_arm(name: str, build: Callable[..., str], needs_core: bool = False, span=None, letters: bool = False,
                 forms=None, meta=None) -> None:
    assert name not in ARM_BUILDERS or ARM_BUILDERS[name].build is build, f"arm {name} already registered"
    ARM_BUILDERS[name] = Arm(build, needs_core, span, letters, forms, meta)
    if name not in ALL_ARMS:
        ALL_ARMS.append(name)


def build_prompt(arm: str, story: str, query: str, core: dict | None = None, X: str | None = None) -> str:
    a = ARM_BUILDERS[arm]
    return a.build(story, query, core, X) if a.needs_core else a.build(story, query)


def arm_span(arm: str, story: str, query: str, core: dict | None = None, X: str | None = None) -> str | None:
    a = ARM_BUILDERS[arm]
    return None if a.span is None else a.span(story, query, core, X) if a.needs_core else a.span(story, query)


def import_arm_modules(spec: str) -> None:
    """Import the comma-separated modules that register further arms (e.g. ``ckeys.subsets,ckeys.variants``)."""
    for m in filter(None, spec.split(",")):
        importlib.import_module(m)


for _a in ARMS:  # the standard arms, with their re-mention spans
    register_arm(_a, (lambda arm: lambda story, query: raw_prompt(arm, story, query))(_a),
                 span={"P1": lambda *_: LISTING, "AFTER": lambda *_: LISTING, "BEFORE": lambda *_: LISTING,
                       "LETTER": lambda *_: LETTER_LISTING, "POST": lambda *_: ROOM, "PRE": lambda *_: ROOM}.get(_a),
                 letters=_a == "LETTER")


def alphabet(arm: str) -> tuple[str, ...]:
    return LETTERS if arm == "LETTER" or (arm in ARM_BUILDERS and ARM_BUILDERS[arm].letters) else LOCATIONS


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


def encode_raw(tok, raw: str, bos: bool | None = None) -> torch.Tensor:
    """No chat template (GPT-2, base models, raw wrappers); ``bos`` None = prepend the BOS token iff the tokenizer
    has one (GPT-2 <|endoftext|>, Mistral <s>; Qwen has none)."""
    ids = tok(raw, add_special_tokens=False).input_ids
    if bos or (bos is None and tok.bos_token_id is not None):
        ids = [tok.bos_token_id] + ids
    return torch.tensor([ids])


def encode_any(tok, raw: str, chat: bool = True, prefill: str = "Answer:", bos: bool | None = None,
               system: str = "You are a helpful assistant.") -> torch.Tensor:
    return encode(tok, raw, system, prefill) if chat else encode_raw(tok, raw, bos)


def candidate_ids(tok, arm: str) -> list[int]:
    ids = []
    for c in alphabet(arm):
        t = tok.encode(" " + c, add_special_tokens=False)
        assert len(t) == 1, f"candidate {c!r} is not a single token"
        ids.append(t[0])
    return ids
