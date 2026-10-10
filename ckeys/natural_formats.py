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


# --------------------------------------------------------------------------- additions for the stage-8 part-A run
# Closed-book prompts (the prior controls of J-A-2): the OPTA / LETA question, options and instruction without the
# passage, with the critic pilot's lead sentence. Answer frames (A-8): the per-model text between the "Answer:" prefill
# and the entity, fixed before the factorial from greedy ID generations on R items; every continuation is then scored
# after prompt + frame. Frame " " reproduces cont_ids (tests/test_natural_clamp.py).
CB_PRE = "Answer the question.\n\n"
CB_FORMATS = ("CBOPT", "CBLET")
LETTER_FORMATS = ("LETA", "CBLET")
FRAMES = ("", " ", " **", "**", " The ", " the ")
LETTER_INSTR = "Answer with the letter of the correct option."


def cb_raw(fmt, it):
    """The closed-book raw prompt: CBOPT (OPTA's options line and instruction) or CBLET (LETA's lettered list)."""
    q, opts = it["question"].strip(), it["options"]
    if fmt == "CBOPT":
        return CB_PRE + "Question: " + q + "\nOptions: " + "; ".join(opts) + "\n" + MC
    if fmt == "CBLET":
        lo = "\n".join(f"{L}. {o}" for L, o in zip("ABCD", opts))
        return CB_PRE + "Question: " + q + "\nOptions:\n" + lo + "\n" + LETTER_INSTR
    raise ValueError(fmt)


def encode_cb(tok, it, fmt):
    """(chat text, ids) of a closed-book prompt."""
    text = chat_text(tok, cb_raw(fmt, it))
    return text, tok(text, add_special_tokens=False).input_ids


def frame_cont(tok, text, ids, ent, frame=" "):
    """(c, j): the continuation ids of ``frame + ent`` after the prompt (prefix-stable) and j, the number of leading
    tokens of c that lie inside the frame's characters (character offsets). The frame characters c[:j] leave uncovered
    must be whitespace (merged into the decision token c[j]), and c[j] must not decode to whitespace; AssertionError
    otherwise (the item is then not valid under that frame)."""
    enc = tok(text + frame + ent, add_special_tokens=False, return_offsets_mapping=True)
    full, off = enc.input_ids, enc.offset_mapping
    assert full[:len(ids)] == list(ids), "prefix not stable"
    c, oc = full[len(ids):], off[len(ids):]
    f1 = len(text) + len(frame)
    j = 0
    while j < len(c) and oc[j][1] <= f1:
        j += 1
    assert j < len(c), "no entity token after the frame"
    covered = max([len(text)] + [e for _, e in oc[:j]])
    assert (text + frame)[covered:f1].strip() == "", f"frame {frame!r} is not separable from the entity"
    assert tok.decode([c[j]]).strip(), "the decision token is whitespace"
    return c, j


def decision_ids(tok, text, ids, it, fmt, frame=" ", strict=True):
    """Frame-aware decision tokens of one prompt: {"w": tokens before the decision (shared), "j": len(w), "c": {Y:
    continuation}, "dec": {Y: decision token id}}. Entity formats: Y in B, S, X, Z, D (closed-book CBOPT: the four
    options B, S, X, D), the continuation of frame + entity. Letter formats (LETA, CBLET): Y in B, S, X, D, the
    continuation of frame + the letter of Y's option, which must be one token after w. AssertionError when w is not
    shared, a letter is not one token, or (strict) the decision tokens are not pairwise distinct."""
    ents = {Y: it[k] for Y, k in (("B", "answer"), ("S", "S"), ("X", "X"), ("Z", "Z"), ("D", "D"))}
    if fmt in CB_FORMATS or fmt in LETTER_FORMATS:
        ents = {Y: e for Y, e in ents.items() if e in it["options"]}
    if fmt in LETTER_FORMATS:
        ents = {Y: "ABCD"[it["options"].index(e)] for Y, e in ents.items()}
    cj = {Y: frame_cont(tok, text, ids, e, frame) for Y, e in ents.items()}
    ws = {tuple(c[:j]) for c, j in cj.values()}
    assert len(ws) == 1, "w not shared"
    j = next(iter(cj.values()))[1]
    if fmt in LETTER_FORMATS:
        assert all(len(c) == j + 1 for c, _ in cj.values()), "a letter is not one token after w"
    dec = {Y: c[j] for Y, (c, _) in cj.items()}
    if strict:
        assert len(set(dec.values())) == len(dec), "decision tokens not distinct"
    return {"w": list(next(iter(ws))), "j": j, "c": {Y: c for Y, (c, _) in cj.items()}, "dec": dec}


def letter_of(gen):
    """The option letter a generation starts with (leading whitespace and markdown/brackets skipped; the letter must
    not be followed by another letter), else None."""
    s = gen.lstrip().lstrip("*_`([\"' ")
    return s[0] if s and s[0] in "ABCD" and (len(s) == 1 or not s[1].isalpha()) else None


def answer_of(gen, it, fmt):
    """Which entity a generation gives: "B", "S", "X", "Z", "D" (``matches`` on the entity; letter formats: the option
    of the letter it starts with) or "other"."""
    ents = {"B": it["answer"], "S": it["S"], "X": it["X"], "Z": it["Z"], "D": it["D"]}
    if fmt in LETTER_FORMATS:
        L = letter_of(gen)
        if L is None:
            return "other"
        e = it["options"]["ABCD".index(L)]
        return next(Y for Y, x in ents.items() if x == e)
    return next((Y for Y, e in ents.items() if matches(gen, e)), "other")


def frame_of(gen, ent):
    """The text a generation puts before the entity (case-sensitive first occurrence), or None."""
    i = gen.find(ent)
    return gen[:i] if i >= 0 else None


def choose_frame(prefixes):
    """The most frequent of FRAMES among ``prefixes`` (ties: the earlier in FRAMES); " " when none is one of FRAMES.
    Returns (frame, {frame: count})."""
    counts = {f: sum(p == f for p in prefixes) for f in FRAMES}
    best = max(FRAMES, key=lambda f: (counts[f], -FRAMES.index(f)))
    return (best if counts[best] else " "), counts
