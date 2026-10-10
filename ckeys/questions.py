"""Stage 8, part D of P-2026-10-10-J (docs/PREREGISTRATION.md): the two question arms Q_IN / Q_OUT of the belief task and
the chat-wrapped in-sentence IOI arm INLINE_CHAT, plus the row layout of the IOI prompts the part needs.

Q_IN and Q_OUT are OPTIONS-AFTER (P1) with only the question line replaced:
  PREFIX + story + "\\nQuestion: Which of the choices is (not) mentioned in the story?\\n" + LISTING
         + "\\nAnswer with exactly one choice.\\nAnswer:"
They are registered with ckeys.encoding.register_arm (span = LISTING), so experiments/format_factorial.run_item and
row_restricted_keys work unchanged with ``--arm-modules ckeys.questions``. The story's own query is ignored.

INLINE_CHAT is ckeys.ioi's INLINE sentence (the four names as a spaced parenthetical after the IO mention) inside the
chat wrapper and instruction of the AFTER arm, without the "Choices:" line:
  HEAD + "\\n\\nSentence: " + inline sentence + "\\n" + INSTR + "\\nAnswer:"   (chat_text, "Answer:" prefill)
so that INLINE against AFTER holds the wrapper fixed. ``ioi_run_item`` is experiments/ioi_factorial.run_item for any of
the IOI arms of ckeys.ioi and for INLINE_CHAT (the same rows, captures and per-item record; the shared code path is
checked against ioi_factorial.run_item in tests/test_questions.py). ``ioi_layout`` gives the text, ids, offsets, the
writing token p and the row groups of ckeys.ioi.RowTask (INLINE_CHAT uses INLINE's groups), with the listed name rows
named by role.
"""
from __future__ import annotations

import torch

from . import ioi
from .encoding import LISTING, encode_any, register_arm
from .story import PREFIX

Q_IN = "Which of the choices is mentioned in the story?"
Q_OUT = "Which of the choices is not mentioned in the story?"
QUESTIONS = {"Q_IN": Q_IN, "Q_OUT": Q_OUT}
IOI_ARMS = ("INLINE", "INLINE_CHAT", "AFTER", "BEFORE")


def q_prompt(arm: str, story: str) -> str:
    return PREFIX + story + "\nQuestion: " + QUESTIONS[arm] + "\n" + LISTING + "\nAnswer with exactly one choice.\nAnswer:"


for _a in QUESTIONS:
    register_arm(_a, (lambda arm: lambda story, query: q_prompt(arm, story))(_a), span=lambda *_: LISTING)


# --------------------------------------------------------------------------- IOI
def ioi_raw(arm: str, core: dict, io: str, chat: bool) -> str:
    """The raw prompt of an IOI arm for the run whose IO name is ``io`` (``chat``: instruct models)."""
    if arm == "INLINE_CHAT":
        return ioi.HEAD + "\n\nSentence: " + ioi.inline(core, io, False) + "\n" + ioi.INSTR + "\nAnswer:"
    return ioi.raw_prompt(arm, core, io, ioi.arm_chat(arm, chat))


def ioi_chat(arm: str, chat: bool) -> bool:
    return chat if arm == "INLINE_CHAT" else ioi.arm_chat(arm, chat)


def ioi_encode_runs(tok, core: dict, arm: str, chat: bool, bos=None) -> dict | None:
    """ckeys.ioi.encode_runs for every arm of IOI_ARMS: {"B", "S", "X": [1, T]} or None (lengths differ, or the runs
    differ at more than one position)."""
    if arm != "INLINE_CHAT":
        return ioi.encode_runs(tok, core, arm, chat, bos)
    ids = {k: encode_any(tok, ioi_raw(arm, core, core[r], chat), chat, bos=bos) for k, r in (("B", "io_b"), ("S", "io_s"), ("X", "io_x"))}
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    if len(diff) != 1 or (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() != diff:
        return None
    return ids


def ioi_valid(tok, core: dict, arm: str, chat: bool, bos=None) -> str | None:
    """None if the core is usable in this arm and tokenizer, else the reason (lengths, positions, a name that is not one
    token, the occurrence counts of ckeys.ioi.check_occurrences, the row groups)."""
    try:
        ids = ioi_encode_runs(tok, core, arm, chat, bos)
        if ids is None:
            return "B/S/X encodings differ in length or at more than one position"
        ioi.check_occurrences(tok, core, "INLINE" if arm == "INLINE_CHAT" else arm, ids["B"])
        ioi_layout(tok, core, arm, chat, bos)
    except AssertionError as ex:
        return f"assertion: {ex}"
    return None


def ioi_layout(tok, core: dict, arm: str, chat: bool, bos=None) -> dict:
    """Text, ids (B, S, X), offsets, p, T and the RowTask groups of one IOI prompt; ``named`` maps the roles io_b, io_s,
    io_x, subj to their row in the listed names ("options")."""
    from experiments.row_restricted_keys import encode_with_offsets
    c = ioi_chat(arm, chat)
    ids = ioi_encode_runs(tok, core, arm, chat, bos)
    assert ids is not None, (arm, core)
    text, ib, off = encode_with_offsets(tok, ioi_raw(arm, core, core["io_b"], chat), chat=c, bos=bos)
    assert torch.equal(ib, ids["B"]), (arm, "offset encoding differs from the run encoding")
    p = (ids["B"][0] != ids["S"][0]).nonzero().item()
    g = ioi.RowTask(chat=chat)
    g._arm = "INLINE" if arm == "INLINE_CHAT" else arm
    groups = g.groups(tok, text, ib, off, p, core, g._arm)
    tid = ioi.name_ids(tok, core)
    roles = {"io_b": tid["B"], "io_s": tid["S"], "io_x": tid["X"], "subj": tid["Subj"]}
    named = {r: [i for i in groups["options"] if ib[0, i].item() == v] for r, v in roles.items()}
    assert all(len(v) == 1 for v in named.values()), (arm, core, named)
    named = {r: v[0] for r, v in named.items()}
    if arm == "BEFORE":
        assert all(i < p for i in groups["options"]), (arm, groups["options"], p)
    else:
        assert all(i > p for i in groups["options"]), (arm, groups["options"], p)
    return dict(core=core, arm=arm, chat=c, text=text, ids=ids, off=off, p=p, T=ib.shape[1], groups=groups,
                named=named, tid=tid)


@torch.no_grad()
def ioi_run_item(model, tok, core, arm, chat, bos, device):
    """experiments/ioi_factorial.run_item, extended to INLINE_CHAT (the same ten clamp rows, three clean passes and
    per-item record; ``check_occurrences`` as INLINE)."""
    from experiments import ioi_factorial as iof
    if arm != "INLINE_CHAT":
        return iof.run_item(model, tok, core, arm, chat, bos, device)
    ids = ioi_encode_runs(tok, core, arm, chat, bos)
    if ids is None:
        return None
    ioi.check_occurrences(tok, core, "INLINE", ids["B"])
    return ioi_item_from_ids(model, tok, core, arm, chat, ids, device)


@torch.no_grad()
def ioi_item_from_ids(model, tok, core, arm, chat, ids, device):
    """The body of experiments/ioi_factorial.run_item on given encodings {"B", "S", "X": [1, T]} (tests/test_flag.py checks
    it against run_item itself on INLINE)."""
    from experiments import ioi_factorial as iof
    from .clamp import capture_kv, clamp_kv, stack_rows
    from .interventions import blocks
    ids = {k: v.to(device) for k, v in ids.items()}
    pos = (ids["B"][0] != ids["S"][0]).nonzero().item()
    nL, tid = len(blocks(model)), ioi.name_ids(tok, core)
    kv, clean = {}, {}
    for name in ("B", "S", "X"):
        with capture_kv(model, [pos], range(nL)) as t:
            lp = iof.last_logprobs(model, ids[name])[0]
        kv[name] = {k: v[0] for k, v in t.items()}
        clean[name] = iof.stats(lp, tid)
    rows = iof.row_specs(nL)
    tabs = stack_rows(kv, [lambda l, ch, r=r: (r[0] if ch == "k" else r[1]) if l >= r[2] else "B" for r in rows], range(nL))
    with clamp_kv(model, [pos], tabs, range(nL)):
        lp = iof.last_logprobs(model, ids["B"].expand(len(rows), -1))
    out = {f"{iof.LABEL[(k, v)]}@{l0}": {"m": (lp[i, tid["S"]] - lp[i, tid["B"]]).item(), "lp": {t: lp[i, j].item() for t, j in tid.items()}}
           for i, (k, v, l0) in enumerate(rows)}
    floor_B = abs(out["ID@0"]["m"] - clean["B"]["m"])
    floor_S = max(abs(out["KV_S@0"]["lp"][t] - clean["S"]["lp"][t]) for t in iof.TRACK)
    return {"core": core, "arm": arm, "chat": chat, "pos": pos, "len": ids["B"].shape[1], "n_layers": nL,
            "pattern": core["pattern"], "template": core["template"], "clean": clean, "m": out, "floor_B": floor_B, "floor_S": floor_S}
