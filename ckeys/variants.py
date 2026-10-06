"""Non-identical re-mention arms of preregistration G, part (d): the SENTENCE-AFTER sentence (POST) with the six
locations in another surface form, structure-matched floors (six non-candidate nouns), and the list family.

All arms are core-independent and registered in the ckeys.encoding registry (``import_arm_modules("ckeys.variants")``).
``forms`` gives every arm its six form tokens (first piece of " " + form), so the factorial records Sf/Xf/Bf log-probs
next to S/X/B; ``form_spans`` gives the full token sequences for the attention probe. The English answer scoring and
the story are unchanged; only the sentence or list after the writing token differs.
"""
from __future__ import annotations

from .encoding import LISTING, PREFIX, ROOM, candidate_ids, raw_prompt, register_arm
from .story import LOCATIONS, OBJECTS

SENTENCES = {
    "POST": ROOM,
    "POST_THE": "The room has the box, the basket, the shelf, the drawer, the cabinet and the closet.",
    "POST_MODIF": "The room has a storage box, a wicker basket, a wooden shelf, a desk drawer, a filing cabinet and a walk-in closet.",
    "POST_TITLE": "The room has a Box, a Basket, a Shelf, a Drawer, a Cabinet and a Closet.",
    "POST_UPPER": "The room has a BOX, a BASKET, a SHELF, a DRAWER, a CABINET and a CLOSET.",
    "POST_PLURAL": "The room has boxes, baskets, shelves, drawers, cabinets and closets.",
    "POST_SYN": "The room has a crate, a hamper, a ledge, a compartment, a cupboard and a wardrobe.",
    "POST_FRMIX": "The room has a boîte, a panier, an étagère, a tiroir, an armoire and a placard.",
    "POST_DEMIX": "The room has a Kiste, a Korb, a Regal, a Schublade, a Schrank and a Kammer.",
    "POST_FR": "La pièce contient une boîte, un panier, une étagère, un tiroir, une armoire et un placard.",
    "POST_DE": "Der Raum hat eine Kiste, einen Korb, ein Regal, eine Schublade, einen Schrank und eine Kammer.",
    "POST_OTHER": "The room has a table, a chair, a desk, a bed, a sofa and a rug.",
    "POST_FR_OTHER": "La pièce contient une table, une chaise, un bureau, un lit, un canapé et un tapis.",
    "POST_DE_OTHER": "Der Raum hat einen Tisch, einen Stuhl, einen Schreibtisch, ein Bett, ein Sofa und einen Teppich.",
}
_F = {
    "TITLE": ("Box", "Basket", "Shelf", "Drawer", "Cabinet", "Closet"),
    "UPPER": ("BOX", "BASKET", "SHELF", "DRAWER", "CABINET", "CLOSET"),
    "PLURAL": ("boxes", "baskets", "shelves", "drawers", "cabinets", "closets"),
    "SYN": ("crate", "hamper", "ledge", "compartment", "cupboard", "wardrobe"),
    "FR": ("boîte", "panier", "étagère", "tiroir", "armoire", "placard"),
    "DE": ("Kiste", "Korb", "Regal", "Schublade", "Schrank", "Kammer"),
    "OTHER": ("table", "chair", "desk", "bed", "sofa", "rug"),
    "FR_OTHER": ("table", "chaise", "bureau", "lit", "canapé", "tapis"),
    "DE_OTHER": ("Tisch", "Stuhl", "Schreibtisch", "Bett", "Sofa", "Teppich"),
}
FORMS = {"POST": LOCATIONS, "POST_THE": LOCATIONS, "POST_MODIF": LOCATIONS, "AFTER": LOCATIONS} | \
    {"POST_" + k: v for k, v in _F.items() if k not in ("FRMIX", "DEMIX")} | \
    {"POST_FRMIX": _F["FR"], "POST_DEMIX": _F["DE"]} | {"AFTER_" + k: _F[k] for k in ("SYN", "FR", "DE", "OTHER")}
LISTS = {a: FORMS[a] for a in ("AFTER", "AFTER_SYN", "AFTER_FR", "AFTER_DE", "AFTER_OTHER")}
SPAN_FORMS = {"POST_MODIF": ("storage box", "wicker basket", "wooden shelf", "desk drawer", "filing cabinet", "walk-in closet")}
VARIANT_ARMS = tuple(a for a in list(SENTENCES) + list(LISTS) if a not in ("POST", "AFTER"))
# family of competence prompts (experiments/form_competence.py) that gates each arm; None = exact form, no gate
FAMILY = {"POST_TITLE": "TITLE", "POST_UPPER": "UPPER", "POST_PLURAL": "PLURAL", "POST_SYN": "SYN", "POST_FRMIX": "FR",
          "POST_DEMIX": "DE", "POST_FR": "FR", "POST_DE": "DE", "AFTER_SYN": "SYN", "AFTER_FR": "FR", "AFTER_DE": "DE"}
# floor and anchor of r_K(v) = [ID_K(v) - ID_K(floor)] / [ID_K(anchor) - ID_K(floor)]
FLOOR = {a: ("POST_OTHER", "POST") for a in SENTENCES if a not in ("POST", "POST_OTHER", "POST_FR_OTHER", "POST_DE_OTHER")} | \
    {"POST_FR": ("POST_FR_OTHER", "POST"), "POST_DE": ("POST_DE_OTHER", "POST")} | \
    {a: ("AFTER_OTHER", "AFTER") for a in ("AFTER_SYN", "AFTER_FR", "AFTER_DE")}
# first-token collisions among the six forms, per tokenizer family (seed-0 counts in tests/test_variants.py)
COLLISIONS = {
    "qwen": {"DE": {("box", "basket"), ("drawer", "cabinet")}},
    "olmo": {"DE": {("box", "basket"), ("drawer", "cabinet")}},
    "mistral": {"DE": {("drawer", "cabinet")}, "UPPER": {("cabinet", "closet")}},
}
COLLIDING_ARMS = {"DE": ("POST_DE", "POST_DEMIX", "AFTER_DE"), "UPPER": ("POST_UPPER",)}

assert SENTENCES["POST"] == ROOM and "Choices: " + ", ".join(LISTS["AFTER"]) == LISTING
assert all(w not in OBJECTS and w not in LOCATIONS for w in _F["OTHER"])


def variant_prompt(arm: str, story: str, query: str) -> str:
    if arm in SENTENCES:
        return PREFIX + story + " " + SENTENCES[arm] + "\nQuestion: " + query + "\nAnswer with one word.\nAnswer:"
    return PREFIX + story + "\nQuestion: " + query + "\nChoices: " + ", ".join(LISTS[arm]) + "\nAnswer with one word.\nAnswer:"


def variant_span(arm: str) -> str:
    return SENTENCES[arm] if arm in SENTENCES else "Choices: " + ", ".join(LISTS[arm])


assert variant_prompt("AFTER", "s", "q") == raw_prompt("AFTER", "s", "q") and variant_prompt("POST", "s", "q") == raw_prompt("POST", "s", "q")


def form_ids(tok, arm: str) -> list[int]:
    """First token of " " + form per location; the candidate ids for any arm without forms (never raises)."""
    if arm not in FORMS:
        return candidate_ids(tok, arm)
    return [tok.encode(" " + f, add_special_tokens=False)[0] for f in FORMS[arm]]


def form_spans(tok, arm: str) -> list[list[int]]:
    """Full token sequence of " " + form per location (the MODIF phrases), for the attention probe."""
    return [tok.encode(" " + f, add_special_tokens=False) for f in SPAN_FORMS.get(arm, FORMS[arm])]


def tokenizer_family(name: str) -> str:
    n = name.lower()
    return "mistral" if "mistral" in n else "olmo" if "olmo" in n else "qwen" if "qwen" in n else "other"


def collisions(tok, arm: str) -> set[tuple[str, str]]:
    """Pairs of locations whose forms share a first token under this arm (computed from the tokenizer)."""
    f = form_ids(tok, arm)
    return {(LOCATIONS[i], LOCATIONS[j]) for i in range(6) for j in range(i + 1, 6) if f[i] == f[j]}


def excluded(core: dict, X: str, pairs: set[tuple[str, str]]) -> bool:
    """A core leaves the form-scored, any-form and attention populations when any pairwise coincidence holds
    among f_B, f_S, f_X."""
    w = (core["base"], core["source"], X)
    return any({a, b} <= set(w) for a, b in pairs)


def collision_pairs(model_name: str, arm: str) -> set[tuple[str, str]]:
    fam = tokenizer_family(model_name)
    for k, arms in COLLIDING_ARMS.items():
        if arm in arms:
            return set(COLLISIONS.get(fam, {}).get(k, set()))
    return set()


def _register(arm):
    register_arm(arm, lambda story, query: variant_prompt(arm, story, query), span=lambda story, query: variant_span(arm),
                 forms=lambda tok: form_ids(tok, arm))


for _a in VARIANT_ARMS:
    _register(_a)
