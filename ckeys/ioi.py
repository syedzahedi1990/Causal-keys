"""Indirect Object Identification (Wang et al. 2022) cores and format arms for the IOI part (e) of preregistration G.

A core has four distinct names: the indirect object of the base run IO_B, the source IO_S and the third name IO_X
(the three runs differ only at the IO mention p) and the subject S, plus one of the 15 Wang et al. templates
(BABA form, truncated before the final [A]; ABBA swaps the first [B] and [A]; cores alternate ABBA/BABA), a place,
an object and a per-core permutation that fixes the order of the four names in every list. The place and object
pools are the reduced ones (19 and 18; 'cafe', 'necklace', 'snack' split in Mistral-v0.3).

Arms: PLAIN (the sentence, answer at ' to'); AFTER / BEFORE / QUESTION (chat-wrapped on instruct models, raw
otherwise); INLINE / INLINE_BEFORE (raw; the list as a spaced parenthetical '( Ruth, Erik, Charles, Joan)' after
the IO mention, located by a whole-word regex, or before the sentence). ``RowTask`` serves
``experiments/row_restricted_keys.py --task ioi``.
"""
from __future__ import annotations

import random
import re

from .encoding import encode_any

TEMPLATES = (
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to",
)
PLACES = ("store", "garden", "restaurant", "school", "hospital", "office", "house", "station", "park", "beach",
          "market", "library", "church", "hotel", "museum", "airport", "farm", "bank", "gym")
OBJECTS = ("ring", "kiss", "bone", "basketball", "computer", "drink", "book", "ball", "gift", "pen", "cup", "hat",
           "coin", "key", "letter", "ticket", "phone", "bag")
NAMES = tuple(
    "Mary John James Robert Michael William David Richard Joseph Thomas Charles Daniel Matthew Anthony Mark Paul "
    "Steven Andrew George Joshua Kevin Brian Edward Ronald Jason Ryan Jacob Gary Nicholas Eric Stephen Jonathan "
    "Larry Justin Scott Frank Benjamin Samuel Gregory Raymond Alexander Patrick Jack Dennis Jerry Tyler Aaron "
    "Henry Douglas Peter Adam Nathan Walter Kyle Harold Carl Jeremy Keith Roger Arthur Terry Sean Christian Austin "
    "Joe Albert Jesse Billy Bruce Bryan Ralph Roy Louis Philip Bobby Johnny Jennifer Linda Elizabeth Barbara Susan "
    "Jessica Sarah Karen Nancy Lisa Margaret Emily Michelle Carol Amanda Rebecca Laura Amy Angela Helen Anna "
    "Nicole Ruth Emma Catherine Virginia Rachel Maria Julie Victoria Kelly Joan Lauren Frances Martha Andrea "
    "Hannah Ann Jean Alice Sara Julia Marie Madison Grace Rose Diana Jane Tom Bob Bill Jim Dan Sam Alex Max Ben "
    "Luke Chris Mike Steve Dave Ken Ron Don Tim Jeff Greg Phil Matt Nick Josh Zach Jake Kate Lucy Molly Ellen "
    "Claire Lily Sophie Isabel Charlotte Eva Beth Liz Meg Pat Ray Lee Jay Guy Hugh Ian Neil Karl Kurt Leo Lewis "
    "Oliver Oscar Simon Stuart Victor Vincent Warren Wayne Carlos Jose Juan Luis Miguel Pedro Marco Mario Paulo "
    "Bruno Hans Erik Lars Ivan".split())

ARMS = ("PLAIN", "AFTER", "BEFORE", "QUESTION", "INLINE", "INLINE_BEFORE")
CHAT_ARMS = ("AFTER", "BEFORE", "QUESTION")      # chat-wrapped on instruct models, raw otherwise
LIST_ARMS = ("AFTER", "BEFORE", "INLINE", "INLINE_BEFORE")   # the IO name occurs twice
ROLES = ("io_b", "io_s", "io_x", "subj")
HEAD = "Complete the sentence with the right name."
INSTR = "Answer with one name."
QUESTION = "Question: Which name completes the sentence?"


def make_cores(n: int, rng: random.Random) -> list[dict]:
    cores = []
    for i in range(n):
        io_b, io_s, io_x, subj = rng.sample(NAMES, 4)
        order = [0, 1, 2, 3]
        rng.shuffle(order)  # slot order of (IO_B, IO_S, IO_X, S) in every list of the core
        cores.append(dict(template=rng.randrange(len(TEMPLATES)), pattern="ABBA" if i % 2 == 0 else "BABA",
                          place=rng.choice(PLACES), object=rng.choice(OBJECTS), io_b=io_b, io_s=io_s, io_x=io_x,
                          subj=subj, order=order))
    return cores


def sentence(core: dict, io: str) -> str:
    t = TEMPLATES[core["template"]]
    if core["pattern"] == "ABBA":  # swap the first [B] and [A]
        t = t.replace("[B]", "[TMP]", 1).replace("[A]", "[B]", 1).replace("[TMP]", "[A]", 1)
    return (t.replace("[B]", core["subj"]).replace("[A]", io).replace("[PLACE]", core["place"])
            .replace("[OBJECT]", core["object"]))


def names_in_order(core: dict) -> list[str]:
    names = [core[r] for r in ROLES]
    return [names[i] for i in core["order"]]


def listing(core: dict) -> str:
    return "Choices: " + ", ".join(names_in_order(core))


def parenthetical(core: dict) -> str:
    return "( " + ", ".join(names_in_order(core)) + ")"   # spaced: '( Ruth' keeps ' Ruth' as one token


def inline(core: dict, io: str, before: bool) -> str:
    s, par = sentence(core, io), parenthetical(core)
    if before:
        return par + " " + s
    i = re.search(r"\b" + re.escape(io) + r"\b", s).end()   # whole word: Sam is inside Samuel
    m = re.search(r"\. |, ", s[i:])
    assert m, s
    return s[:i + m.start()] + " " + par + s[i + m.start():]


def raw_prompt(arm: str, core: dict, io: str, chat: bool) -> str:
    s = sentence(core, io)
    if arm == "PLAIN":
        return s
    if arm == "INLINE":
        return inline(core, io, False)
    if arm == "INLINE_BEFORE":
        return inline(core, io, True)
    if not chat:  # raw completion (GPT-2, base models)
        if arm == "AFTER":
            return s + "\n" + listing(core) + "\nAnswer:"
        if arm == "BEFORE":
            return listing(core) + "\n" + s + "\nAnswer:"
        if arm == "QUESTION":
            return s + "\n" + QUESTION + "\nAnswer:"
    else:
        if arm == "AFTER":
            return HEAD + "\n\nSentence: " + s + "\n" + listing(core) + "\n" + INSTR + "\nAnswer:"
        if arm == "BEFORE":
            return HEAD + "\n" + listing(core) + "\n\nSentence: " + s + "\n" + INSTR + "\nAnswer:"
        if arm == "QUESTION":
            return HEAD + "\n\nSentence: " + s + "\n" + QUESTION + "\n" + INSTR + "\nAnswer:"
    raise ValueError(arm)


def arm_chat(arm: str, chat: bool) -> bool:
    return chat and arm in CHAT_ARMS


def name_id(tok, name: str) -> int:
    t = tok.encode(" " + name, add_special_tokens=False)
    assert len(t) == 1, f"name {name!r} is not a single token"
    return t[0]


def name_ids(tok, core: dict) -> dict:
    """Tracked tokens: S = IO_S, B = IO_B, X = IO_X, Subj = the subject."""
    return {"S": name_id(tok, core["io_s"]), "B": name_id(tok, core["io_b"]), "X": name_id(tok, core["io_x"]),
            "Subj": name_id(tok, core["subj"])}


def encode_runs(tok, core: dict, arm: str, chat: bool, bos=None) -> dict | None:
    """``{"B", "S", "X": [1, T]}`` of the three runs, or None if they differ in length or at more than one position."""
    c = arm_chat(arm, chat)
    ids = {k: encode_any(tok, raw_prompt(arm, core, core[r], c), c, bos=bos) for k, r in (("B", "io_b"), ("S", "io_s"), ("X", "io_x"))}
    if len({tuple(v.shape) for v in ids.values()}) > 1:
        return None
    diff = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
    if len(diff) != 1 or (ids["B"][0] != ids["X"][0]).nonzero().flatten().tolist() != diff:
        return None
    return ids


def check_occurrences(tok, core: dict, arm: str, ids_b) -> None:
    """The IO id occurs once (PLAIN, QUESTION) or twice (list arms); S's id twice or three times (with a list)."""
    tid = name_ids(tok, core)
    n_io, n_s = (ids_b[0] == tid["B"]).sum().item(), (ids_b[0] == tid["Subj"]).sum().item()
    assert n_io == (2 if arm in LIST_ARMS else 1), (arm, core, n_io)
    assert n_s in ((2, 3) if arm in LIST_ARMS else (2,)), (arm, core, n_s)


class RowTask:
    """``experiments/row_restricted_keys.py --task ioi``: B = IO_B run, S = IO_S run, targets (id_S, id_B).

    ``chat`` applies to the AFTER / BEFORE / QUESTION arms only (the ``chat`` attribute follows the arm of the last
    ``prompts`` call, which ``run`` makes before encoding); ``RowTask(chat=False)`` for GPT-2 and base models.
    Groups: ``sentence_tail`` (sentence rows after p), ``options`` (the four listed names), ``options_sx`` (the listed
    IO_S and IO_X), ``choices`` (the whole list span) and ``tail`` (everything after the list, or after the sentence
    when nothing is listed after it: instruction, chat tokens, prefill, answer position).
    """
    bos = None

    def __init__(self, chat=True):
        self.chat_models, self._arm = chat, None

    @property
    def chat(self):
        return arm_chat(self._arm, self.chat_models)

    def cores(self, n, seed=0):
        return make_cores(n, random.Random(seed))

    def prompts(self, core, arm):
        self._arm = arm
        return raw_prompt(arm, core, core["io_b"], self.chat), raw_prompt(arm, core, core["io_s"], self.chat)

    def targets(self, tok, core, arm):
        return name_id(tok, core["io_s"]), name_id(tok, core["io_b"])

    def groups(self, tok, text, ids, off, p, core, arm):
        from experiments.row_restricted_keys import rows_in
        body = raw_prompt(arm, core, core["io_b"], False) if arm.startswith("INLINE") else sentence(core, core["io_b"])
        span = parenthetical(core) if arm.startswith("INLINE") else listing(core) if arm in ("AFTER", "BEFORE") else None
        b0, b1 = text.index(body), text.index(body) + len(body)
        g, tid = {}, name_ids(tok, core)
        if span:
            c0 = text.index(span)
            assert text.count(span) == 1, (arm, core)
            g["choices"] = rows_in(off, c0, c0 + len(span))
            g["options"] = [i for i in g["choices"] if ids[0, i].item() in tid.values()]
            g["options_sx"] = [i for i in g["choices"] if ids[0, i].item() in (tid["S"], tid["X"])]
            assert len(g["options"]) == 4 and len(g["options_sx"]) == 2, (arm, core)
            b1 = max(b1, c0 + len(span))
        used = set(g.get("choices", []))
        g["sentence_tail"] = [i for i in rows_in(off, b0, b1) if i > p and i not in used]
        used |= set(g["sentence_tail"])
        g["tail"] = [i for i in rows_in(off, b1, len(text)) if i > p and i not in used]
        return g


def _lse(d: dict) -> float:
    import math
    m = max(d.values())
    return m + math.log(sum(math.exp(v - m) for v in d.values()))


def identity_measures(item: dict) -> dict:
    """The per-item quantities of part (e) from one ``experiments/ioi_factorial.py`` item, every clamped row measured
    against the batched self-clamp row ID@0: ID_C = 1/2 [Delta_S(C_S) - Delta_S(C_X) + Delta_X(C_X) - Delta_X(C_S)]
    with Delta_Y(Z) = lp(Y | C_C(Z)) - lp(Y | ID) for C in K, V, KV (``idK``, ``idV``, ``idKV``; ``id4*`` on the
    four-way renormalised log-probs), d_C = m(C_C(S)) - m(ID) per onset (``d``), d_C^LD on LD_S = lp(IO_S) - lp(S)
    (``dLD``), and the clean-run competence (``two_B``, ``four_B``, ``LD_B``, ``mass_B``, ``two_S``, ``four_S``)."""
    import math
    m, ref = item["m"], item["m"]["ID@0"]["lp"]
    ref4 = {t: v - _lse(ref) for t, v in ref.items()}

    def dl(row, t, four=False):
        lp = m[row]["lp"]
        return (lp[t] - _lse(lp) - ref4[t]) if four else lp[t] - ref[t]

    def ident(ch, four=False):
        return 0.5 * ((dl(f"{ch}_S@0", "S", four) - dl(f"{ch}_X@0", "S", four)) + (dl(f"{ch}_X@0", "X", four) - dl(f"{ch}_S@0", "X", four)))

    cb, cs = item["clean"]["B"], item["clean"]["S"]
    out = {"idK": ident("K"), "idV": ident("V"), "idKV": ident("KV"), "id4K": ident("K", True), "id4V": ident("V", True),
           "id4KV": ident("KV", True), "d": {k: m[k]["m"] - m["ID@0"]["m"] for k in m if k != "ID@0"},
           "dLD": {k: (m[k]["lp"]["S"] - m[k]["lp"]["Subj"]) - (ref["S"] - ref["Subj"]) for k in m if k != "ID@0"},
           "L": item["n_layers"], "pattern": item["core"]["pattern"], "lpB": cb["lp"], "floor_B": item["floor_B"], "floor_S": item["floor_S"],
           "LD_B": cb["lp"]["B"] - cb["lp"]["Subj"], "mass_B": sum(math.exp(v) for v in cb["lp"].values()),
           "four_B": max(cb["lp"], key=cb["lp"].get) == "B", "four_S": max(cs["lp"], key=cs["lp"].get) == "S"}
    out["two_B"], out["two_S"] = out["LD_B"] > 0, cs["lp"]["S"] - cs["lp"]["Subj"] > 0
    return out
