"""Prompt formats of stage 8 part A (preregistration J): one SQuAD item, six ways of mentioning the candidate answers.

The raw prompt is wrapped by ckeys.encoding.chat_text (system "You are a helpful assistant.", the prompt as the user
turn, the generation prompt with thinking disabled, the assistant prefill "Answer:").
  NOM   no mention:               passage, question, free-form instruction
  OPTA  options after:            passage, question, "Options: o1; o2; o3; o4", multiple-choice instruction
  OPTB  options before:           the same options line before the passage (instruction-matched to OPTA)
  MENA  natural mention after:    "Related articles mention o1, o2, o3 and o4." after the passage (matched to NOM)
  MENB  natural mention before:   the same sentence before the passage
  LETA  lettered options after:   "A. o1" ... "D. o4", answer with the letter
"""
from __future__ import annotations

import re
import string

from .encoding import chat_text

FORMATS = ("NOM", "OPTA", "OPTB", "MENA", "MENB", "LETA")
PRE = "Read the passage and answer the question.\n\n"
FREE = "Answer with the exact words from the passage."
MC = "Answer with exactly one of the options."


def and_list(xs):
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def parts(fmt, it):
    """(pre, post): raw prompt = pre + passage + post."""
    q, opts = it["question"].strip(), it["options"]
    ol = "Options: " + "; ".join(opts)
    if fmt == "NOM":
        return PRE + "Passage: ", "\nQuestion: " + q + "\n" + FREE
    if fmt == "OPTA":
        return PRE + "Passage: ", "\nQuestion: " + q + "\n" + ol + "\n" + MC
    if fmt == "OPTB":
        return PRE + ol + "\n\nPassage: ", "\nQuestion: " + q + "\n" + MC
    if fmt == "MENA":
        return PRE + "Passage: ", " Related articles mention " + and_list(opts) + ".\nQuestion: " + q + "\n" + FREE
    if fmt == "MENB":
        return PRE + "Passage: Related articles mention " + and_list(opts) + ". ", "\nQuestion: " + q + "\n" + FREE
    if fmt == "LETA":
        lo = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts))
        return PRE + "Passage: ", "\nQuestion: " + q + "\nOptions:\n" + lo + "\nAnswer with the letter of the correct option."
    raise ValueError(fmt)


def passage(it, ent):
    return it["context"][:it["start"]] + ent + it["context"][it["start"] + len(it["answer"]):]


def encode_item(tok, it, fmt, ent):
    """(chat text, ids, offsets, (c0, c1) the entity's characters in the chat text) with ``ent`` in the passage."""
    pre, post = parts(fmt, it)
    raw = pre + passage(it, ent) + post
    text = chat_text(tok, raw)
    c0 = text.index(raw) + len(pre) + it["start"]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return text, enc.input_ids, enc.offset_mapping, (c0, c0 + len(ent))


def span_positions(offsets, c0, c1):
    return [i for i, (s, e) in enumerate(offsets) if e > c0 and s < c1]


def cont_ids(tok, text, ids, ent):
    """(continuation ids of ' ' + ent after the prompt, index of the first non-whitespace token); prefix-stable."""
    full = tok(text + " " + ent, add_special_tokens=False).input_ids
    assert full[:len(ids)] == list(ids), "prefix not stable"
    c = full[len(ids):]
    return c, next(i for i, t in enumerate(c) if tok.decode([t]).strip())


def letter_ids(tok, text, ids):
    """{letter: id} of ' A' .. ' D' after the prompt (each must be one token)."""
    out = {}
    for L in "ABCD":
        c, j = cont_ids(tok, text, ids, L)
        assert j == len(c) - 1, f"letter {L} is not a single token after whitespace: {c}"
        out[L] = (c[:j], c[j])
    assert len({tuple(w) for w, _ in out.values()}) == 1
    return out


def passage_range(text, it, fmt, ent):
    """(p0, p1): the passage's characters in the chat text."""
    pre, post = parts(fmt, it)
    p = passage(it, ent)
    p0 = text.index(pre + p + post) + len(pre)
    return p0, p0 + len(p)


def option_rows(text, offsets, it, fmt, ent):
    """{option string: token rows of that option's mention outside the passage} (options line, sentence or lettered
    list; the passage itself names e_B, and D when D_in)."""
    p0, p1 = passage_range(text, it, fmt, ent)
    out = {}
    for o in it["options"]:
        rows = set()
        for m in re.finditer(r"(?<![\w])" + re.escape(o) + r"(?![\w])", text):
            if m.end() <= p0 or m.start() >= p1:
                rows |= set(span_positions(offsets, m.start(), m.end()))
        assert rows or fmt in ("NOM",), (fmt, o)
        out[o] = sorted(rows)
    return out


def norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def matches(gen, ent):
    g, e = norm(gen.split("\n")[0]), norm(ent)
    return g == e or g.startswith(e + " ")


def check_item(tok, it, formats=FORMATS):
    """None if the item is valid for ``tok`` in every format, else the first failure: the B/S/X/Z prompts have equal
    length and differ only inside B's entity span; the continuation of ' ' + e after the prefill is prefix-stable with a
    whitespace prefix w shared by B, S, X, Z and D; the decision tokens are pairwise distinct (stratum FT)."""
    ents = {Y: it[k] for Y, k in (("B", "answer"), ("S", "S"), ("X", "X"), ("Z", "Z"))}
    for f in formats:
        enc = {Y: encode_item(tok, it, f, e) for Y, e in ents.items()}
        tb, ib, ob, (c0, c1) = enc["B"]
        P = span_positions(ob, c0, c1)
        for Y in "SXZ":
            iy = enc[Y][1]
            if len(iy) != len(ib):
                return f"{f}: length {Y}"
            diff = [i for i, (x, y) in enumerate(zip(ib, iy)) if x != y]
            if not diff or not set(diff) <= set(P):
                return f"{f}: diff outside span {Y}"
        if f == "LETA":
            try:
                letter_ids(tok, tb, ib)
            except AssertionError as ex:
                return f"{f}: letters {ex}"
            continue
        try:
            decs = [cont_ids(tok, tb, ib, it[k]) for k in ("answer", "S", "X", "Z", "D")]
        except AssertionError:
            return f"{f}: prefix"
        if len({tuple(c[:j]) for c, j in decs}) != 1:
            return f"{f}: w not shared"
        if it["stratum"] == "FT" and len({c[j] for c, j in decs}) != 5:
            return f"{f}: decision tokens not distinct"
    return None
