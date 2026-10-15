"""Stage 9, passages arm (docs/stage9/SPEC.md sections 5.6, 6, 7, 9.3, 11.6; predictions docs/STAGE9_PREDICTIONS.md P1-P3,
F2): the format x depth window on counterfactual SQuAD passages.

Items: data/stage8a_items.json with their passages from the SQuAD v1.1 dev file (ckeys.squad_items.load_squad asserts
its sha256; experiments.natural_factorial.load_items). E = the 185 split-E items valid for the model's tokenizer and the
stage-8a answer frame in every one of the six stage-8a formats (ckeys.natural_rows.valid_all, the stage-8a rule; the file
records sha_ids of the valid ids, which the check compares with the stage-8a factorial's population_sha256); R = the 71
split-R items under the same rule (a held-out check that decides nothing). Clusters: the article (it["art"]).

Formats (ckeys.natural_formats through ckeys.natural_rows.prep, chat wrapper with the "Answer:" prefill): NOM (free
form), OPTB (options before the passage), OPTA (options after); LETA (lettered options after) is the deadline-gated extra
step. Each format captures its own B and S tables (OPTB changes the prefix).

Rows (SPEC 7.1; every row is on the B prompt and writes K and V at every position of the answer span P in EVERY layer:
in layer l' and channel ch the captured table of the donor run, B's own where the donor is B and S's where it is S):
  none, none2      never S (the identity; first and last row of the batch: an in-batch determinism check; the
                   BF16 floor is none against clean_B, below)
  KV@0             always S (the full patch, the S run exactly)
  late@l, early@l  S on [l, L) / on [0, l), for l in ckeys.stage9.passage_onsets(L) = round(x L), x in 0.15, 0.3, 0.45, 0.6
  K@0, V@0         S in the key / value channel only, every layer
  Klate@l, Vlate@l S in the key / value channel on [l, L) only (diagnostic)
in the order none, KV@0, late@l..., early@l..., K@0, V@0, Klate@l..., Vlate@l..., none2 (21 rows).

Per item and format: the capture batch (ckeys.natural_rows.capture: B, S, X, Z prompts + w, K/V at P in every layer),
the scoring batch (ckeys.natural_rows.score_rows, every row on B prompt + c_S teacher-forced: the decision-token
log-probabilities lp of B, S, X, Z, D (LETA: the letters of B, S, X, D) after the frame tokens w, the option mass, the
full-vocabulary argmax) and the generation batch (ckeys.natural_rows.generate_rows: greedy, 48 new tokens, the rows plus
the unclamped S run; entity formats without the stage-8a W + 2 word cut: a row ends at EOS or an end-of-turn token, at a
newline or at 48 tokens; LETA keeps decided_stop's two-character letter rule; the text is read up to the first special
token). Two answer classes per row: who_strict (ckeys.natural_formats.answer_of, the stage-8a rule) and who_contain (SPEC
9.3: the entity whose normalised text occurs earliest in the normalised first line as a whole-word substring, ties to the
longer; LETA: who_strict). The containment normaliser (norm9) is the stage-8a norm extended so that punctuation never
hides a contained answer: a possessive 's (straight or curly apostrophe) at the end of a word is dropped, every Unicode
punctuation character (category P*, and the ASCII symbols the stage-8a norm drops) is removed, except dashes (category
Pd) and the slash, which separate words and become spaces; articles are dropped and whitespace collapsed. The same
normaliser is applied to the generated text and to every entity (B, S, X, Z, D alike). Containment is primary; strict is
the sensitivity analysis. Each record also stores "clean_B": the decision-token log-probabilities of the B prompt's own
pass in the capture batch (B, S, X, Z + w; another batch shape than the 21-row scoring batch), against which the scorer
reads the BF16 floor of the none row (the none2 row is an in-batch determinism check: the same ids and tables in one
forward come out bitwise equal, also in BF16).

Stages (each a separate process; BF16, sdpa, use_cache=False in every scoring pass; scripts/gpu_stage9.sh):
  check  tokenizer only, no weights: the frame file's pinned sha256, the items' sha256, the SQuAD file and the items'
         passages, the valid E and R sets and their sha256 (outside TEST_MODE the E sha256 must equal the stage-8a
         factorial's population_sha256 and E must have 185 items), the onsets of the model's depth ->
         OUT/check/<tag>_passages.json (also when a check fails: "pass": false with the reasons); exit 1 when one fails
  eval   per split (E, then R) and format (--formats order) the three batches above -> OUT/passages/<tag>.json, or
         OUT/passages/<tag>_LETA.json for --formats LETA. The file is written atomically after every (split, format),
         its pieces in the fixed order E, R x NOM, OPTB, OPTA (a resumed file equals a single session's). A rerun
         keeps every (split, format) piece of an existing file with the same commit, model files, frame, items, valid
         set and rows, also pieces outside this run's --splits/--formats (provenance "carried": a narrower rerun never
         drops results). Before each piece the time it needs is projected from the measured seconds per item, and a
         piece that would end after STAGE8_DEADLINE minus --reserve-min is not started, nor is any later piece (exit 3:
         partial, the next session runs the missing pieces in the same order; the first piece a process runs without a
         measured rate is gated by the deadline itself). The weights load only when a piece starts (the piece is gated
         again after the load): a step past the deadline, or one whose pieces are all kept, writes its file from the
         tokenizer and the config alone.
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct FP32 on the CPU, passages windowed to <= 320 characters
(experiments.natural_factorial.window), frame " " (no frames file), n = 2 E and 2 R items, outputs tagged TEST_<key>.
Batch sizes (A100-80GB; passages up to ~830 tokens): per item and format one 4-row capture forward, one 21-row scoring
forward and one 22-row generation batch (<= 48 new tokens).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from ckeys import stage9 as s9
from ckeys.encoding import WRAPPER_USED
from ckeys.interventions import blocks
from ckeys.natural_formats import FORMATS, LETTER_FORMATS, answer_of
from ckeys.natural_rows import Invalid, capture, generate_rows, prep, score_rows, valid_all
from ckeys.squad_items import SQUAD_DEV_SHA256
from experiments.format_factorial import provenance as ff_provenance
from experiments.ioi_factorial import device_name
from experiments.natural_factorial import deadline_passed, load_items, load_model, sha_ids, split
from experiments.stage6_heads import write_atomic
from experiments.stage8_edits import verified

ROOT = Path(__file__).resolve().parents[1]
TINY = "Qwen/Qwen2.5-0.5B-Instruct"
ITEMS = ROOT / "data" / "stage8a_items.json"
ITEMS_SHA256 = "40874aa1033d1af829f5642916282a5440df2dad011c991b4ac2704a2c25b591"
SQUAD = "results/gpu_stage8a/squad/dev-v1.1.json"
FRAMES = "results/gpu_stage8a/frames/{key}.json"          # the stage-8a answer frames (pinned in ckeys.stage9.ARTEFACT_SHA256)
STAGE8A = "results/gpu_stage8a/factorial/{key}.json"      # its provenance.population_sha256: the valid E set of stage 8a
KEYS = ("mistral7", "llama8", "qwen7")
LETA = "LETA"
ALL_FORMATS = tuple(s9.PASSAGE_FORMATS) + (LETA,)
SPLITS = ("E", "R")
N_E = 185                                                 # valid E items at the three models in stage 8a
MAX_NEW = 48
ENTS = (("B", "answer"), ("S", "S"), ("X", "X"), ("Z", "Z"), ("D", "D"))


def log(s):
    print(s, flush=True)


def r5(x):
    return round(float(x), 5)


# --------------------------------------------------------------------------- rows
@dataclass(frozen=True)
class Win:
    """A window row (SPEC 7.1): ``donor(l, ch, i)`` names the run whose captured table row r writes in layer l, channel
    ch, at span position i (ignored: every position of P takes the same donor)."""
    kind: str        # "none", "KV", "late", "early", "K", "V", "Klate", "Vlate"
    onset: int = 0

    def takes_s(self, l: int, ch: str) -> bool:
        k, on = self.kind, self.onset
        if k == "none":
            return False
        if k == "KV":
            return True
        if k == "late":
            return l >= on
        if k == "early":
            return l < on
        if k in ("K", "V"):
            return ch == k.lower()
        if k in ("Klate", "Vlate"):
            return ch == k[0].lower() and l >= on
        raise ValueError(k)

    def donor(self, l: int, ch: str, i: int = 0) -> str:
        return "S" if self.takes_s(l, ch) else "B"


def window_rows(nL: int) -> dict:
    """{name: Win} in the batch order of SPEC 7.1 (21 rows): none, KV@0, late@l, early@l, K@0, V@0, Klate@l, Vlate@l,
    none2, with l in ckeys.stage9.passage_onsets(nL)."""
    on = s9.passage_onsets(nL)
    rows = {"none": Win("none"), "KV@0": Win("KV")}
    rows |= {f"late@{l}": Win("late", l) for l in on}
    rows |= {f"early@{l}": Win("early", l) for l in on}
    rows |= {"K@0": Win("K"), "V@0": Win("V")}
    rows |= {f"Klate@{l}": Win("Klate", l) for l in on}
    rows |= {f"Vlate@{l}": Win("Vlate", l) for l in on}
    rows["none2"] = Win("none")
    return rows


# --------------------------------------------------------------------------- answers
_ASCII_PUNCT = frozenset(string.punctuation)          # what ckeys.natural_formats.norm drops (some are symbols, not P*)
_SEPARATORS = frozenset("/")                           # with every dash (category Pd): becomes a space
_POSSESSIVE = re.compile(r"(?<=\w)['\u2019\u02bc]s(?!\w)")


def _punct(c: str) -> bool:
    return c in _ASCII_PUNCT or unicodedata.category(c).startswith("P")


def norm9(s: str) -> str:
    """The containment normaliser (SPEC 9.3): lower case; a possessive 's / \u2019s / \u02bcs at the end of a word
    dropped; dashes (Unicode category Pd) and the slash replaced by a space; every other punctuation character (Unicode
    category P*, and the ASCII symbols of string.punctuation that ckeys.natural_formats.norm drops) removed; the
    articles a, an, the dropped; whitespace collapsed. On ASCII text without a possessive, dash or slash it equals
    ckeys.natural_formats.norm."""
    s = _POSSESSIVE.sub("", s.lower())
    s = "".join(" " if (c in _SEPARATORS or unicodedata.category(c) == "Pd") else "" if _punct(c) else c for c in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def who_contain(text: str, it, fmt: str) -> str:
    """SPEC 9.3: with g = norm9(first line) and e_Y = norm9(entity Y), Y in B, S, X, Z, D, the entity whose e_Y occurs in
    g as a whole-word substring at the smallest start index (ties: the longer e_Y; then the order B, S, X, Z, D), or
    "other". Letter formats: who_strict."""
    if fmt in LETTER_FORMATS:
        return answer_of(text, it, fmt)
    g = norm9(text.split("\n")[0])
    best = None
    for k, (Y, key) in enumerate(ENTS):
        e = norm9(it[key])
        if not e:
            continue
        m = re.search(r"(?<!\S)" + re.escape(e) + r"(?!\S)", g)
        if m and (best is None or (m.start(), -len(e), k) < best[0]):
            best = ((m.start(), -len(e), k), Y)
    return best[1] if best else "other"


def answers(text: str, it, fmt: str) -> dict:
    return {"who_strict": answer_of(text, it, fmt), "who_contain": who_contain(text, it, fmt)}


# --------------------------------------------------------------------------- items
def valid_split(tok, items, name, frame, n=0):
    """(valid items, [[id, reason]] skipped) of split ``name``: valid in every stage-8a format for this tokenizer and
    frame (the stage-8a rule), in rank order; with n > 0 the first n valid items."""
    valid, skipped = [], []
    for it in split(items, name):
        why = valid_all(tok, it, FORMATS, frame)
        (skipped.append([it["id"], why]) if why else valid.append(it))
        if n and len(valid) == n:
            break
    return valid, skipped


def read_frame(a):
    """(frame, frames file, its sha256): the stage-8a frame of the model (sha256 asserted against the pin); TEST_MODE
    and keys without a stage-8a frames file pin: " " with no file."""
    if a.test:
        return " ", None, None
    rel = FRAMES.format(key=a.key)
    pin = s9.ARTEFACT_SHA256.get(rel)
    assert pin, f"no pinned stage-8a frames file for key {a.key}"
    sha = s9.file_sha(ROOT / rel)
    assert sha == pin, f"{rel}: sha256 {sha} differs from the pin {pin}"
    return json.loads((ROOT / rel).read_text())["frame"], rel, sha


def stage8a_population(a):
    """provenance.population_sha256 of the stage-8a factorial file of the model (None in TEST_MODE or if absent)."""
    f = ROOT / STAGE8A.format(key=a.key)
    if a.test or not f.exists():
        return None
    return json.loads(f.read_text())["provenance"].get("population_sha256")


# --------------------------------------------------------------------------- one item in one format
@torch.no_grad()
def run_item(model, tok, it, fmt, frame, rows, nL, dev, name, max_new=MAX_NEW):
    """The capture, scoring and generation batches of one item in one format (record of SPEC 11.6); "clean_B" is the B
    prompt's decision-token log-probabilities from the capture batch (the reference of the BF16 floor)."""
    d = prep(tok, it, fmt, frame)
    kv, runs = capture(model, d, range(nL), dev)
    sc = score_rows(model, d, kv, rows, nL, dev, "S")
    gens = generate_rows(model, tok, it, d, kv, rows, nL, max_new, extra={"S_run": "S"}, stop=fmt in LETTER_FORMATS)
    out = {}
    for n in rows:
        g, s = gens[n], sc[n]
        out[n] = {"lp": {Y: r5(v) for Y, v in s["lp"].items()}, "mass": r5(s["mass"]), "argmax": s["argmax"],
                  "text": g["text"], "ids": g["ids"], "n_tok": len(g["ids"])} | answers(g["text"], it, fmt)
        assert out[n]["who_strict"] == g["who"]
    g = gens["S_run"]
    return dict(id=it["id"], art=it["art"], title=it["title"], sub=it["sub"], D_in=it["D_in"], stratum=it["stratum"],
                split=name, fmt=fmt, nP=len(d["P"]), P=d["P"], T=d["T"], j=d["j"], w=d["w"], dec=d["dec"], rows=out,
                S_run={"text": g["text"], "ids": g["ids"], "n_tok": len(g["ids"])} | answers(g["text"], it, fmt),
                clean_B={"lp": {Y: r5(v) for Y, v in runs["B"].items()}})


# --------------------------------------------------------------------------- provenance
def eval_device(a):
    """The device experiments.natural_factorial.load_model puts the model on."""
    return "cuda" if torch.cuda.is_available() and not a.test else "cpu"


def model_fields(a, cfg, dev):
    """The model fields of the provenance from the config and load_model's rules (dtype a.dtype; eager attention for
    gemma2, sdpa otherwise), known before the weights load; the loaded model is checked against them (check_loaded)."""
    return {"dtype": str(getattr(torch, a.dtype)),
            "attn_implementation": "eager" if cfg.model_type.startswith("gemma2") else "sdpa",
            "device": device_name(dev), "n_layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
            "n_heads": cfg.num_attention_heads, "n_kv_heads": cfg.num_key_value_heads}


def check_loaded(model, dev, P):
    """The loaded model has the dtype, attention, depth and widths the provenance recorded before the load."""
    cfg = model.config
    got = {"dtype": str(next(model.parameters()).dtype), "attn_implementation": cfg._attn_implementation,
           "device": device_name(dev), "n_layers": len(blocks(model)), "hidden": cfg.hidden_size,
           "n_heads": cfg.num_attention_heads, "n_kv_heads": cfg.num_key_value_heads}
    bad = {k: (P.get(k), v) for k, v in got.items() if P.get(k) != v}
    assert not bad, f"the loaded model differs from the provenance written before the load: {bad}"


def base_prov(a, stage, cfg=None, dev=None, nL=None):
    """SPEC 11.1. ``revision``: --revision, else the verified directory's (as the stories arm); model fields from the
    config (``cfg``) and the load rules when given."""
    ver = verified(a.model)
    p = ff_provenance(a) | s9.run_meta(a.test) | {
        "stage9": True, "arm": "passages", "stage": stage, "model": a.model, "model_key": a.key,
        "revision": a.revision or (ver or {}).get("revision"), "verified": ver, "test_mode": a.test,
        "items_sha256": s9.file_sha(a.items), "squad_sha256": SQUAD_DEV_SHA256, "max_new": a.max_new,
        "reserve_min": a.reserve_min, "timings": {}}
    if cfg is not None:
        p |= model_fields(a, cfg, dev)
    if nL is not None:
        p["n_layers"] = nL
        p["onsets"] = p["kv_only_onsets"] = s9.passage_onsets(nL)
    p["wrapper"] = dict(WRAPPER_USED)
    return p


def items_ok(a):
    sha = s9.file_sha(a.items)
    return sha == ITEMS_SHA256, sha


# --------------------------------------------------------------------------- stage check (tokenizer only)
def stage_check(a):
    t0 = time.time()
    fails = []
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    nL = cfg.num_hidden_layers
    ok, sha = items_ok(a)
    if not ok:
        fails.append(f"items sha256 {sha} != {ITEMS_SHA256}")
    try:
        frame, ffile, fsha = read_frame(a)
    except (AssertionError, OSError, KeyError) as ex:
        fails.append(f"frame: {ex}")
        frame, ffile, fsha = " ", None, None
    items = None
    if ok:                                                  # a wrong items file is not read
        try:
            items = load_items(a)                           # asserts the SQuAD sha256 and every answer in its passage
        except (AssertionError, OSError, KeyError, IndexError, ValueError) as ex:
            fails.append(f"items/SQuAD: {type(ex).__name__}: {ex}")
    valid, skipped = {name: [] for name in SPLITS}, {name: [] for name in SPLITS}
    for name in SPLITS if items is not None else ():
        v, sk = valid_split(tok, items, name, frame, a.n)
        valid[name], skipped[name] = [it["id"] for it in v], sk
    vsha = {name: sha_ids(ids) for name, ids in valid.items()}
    s8 = stage8a_population(a)
    if not a.test:
        if s8 is None:
            fails.append(f"no stage-8a population_sha256 for {a.key} ({STAGE8A.format(key=a.key)})")
        if items is not None and a.n == 0:                  # (without items the reason is already recorded)
            if len(valid["E"]) != N_E:
                fails.append(f"E: {len(valid['E'])} valid items, stage 8a had {N_E}")
            if s8 is not None and vsha["E"] != s8:
                fails.append(f"E: valid sha256 {vsha['E']} != stage 8a's {s8}")
    on = s9.passage_onsets(nL)
    if not (0 < on[0] and all(x < y for x, y in zip(on, on[1:])) and on[-1] < nL):
        fails.append(f"onsets {on} at L = {nL}")
    rows = window_rows(nL)
    if len(rows) != 21:
        fails.append(f"{len(rows)} rows, not 21")
    P = base_prov(a, "check", nL=nL) | {
        "frame": frame, "frames_file": ffile, "frames_sha256": fsha, "valid_sha256": vsha, "stage8a_population_sha256": s8,
        "population": list(SPLITS), "population_sha256": vsha, "n_items": {k: len(v) for k, v in valid.items()},
        "skipped_items": skipped, "formats": list(ALL_FORMATS)}
    P["timings"]["check"] = round(time.time() - t0, 1)
    res = {"provenance": P, "pass": not fails, "fails": fails, "skipped": skipped, "valid": valid, "valid_sha256": vsha,
           "n_valid": {k: len(v) for k, v in valid.items()}, "stage8a_population_sha256": s8, "frame": frame,
           "onsets": on, "rows": list(rows)}
    path = Path(a.out) / "check" / f"{a.tag}_passages.json"
    write_atomic(res, path)
    log(f"[{a.tag}] passages check {'passed' if not fails else 'FAILED: ' + '; '.join(fails)}: valid E {len(valid['E'])}, "
        f"R {len(valid['R'])}; E sha256 {vsha['E'][:12]} (stage 8a {str(s8)[:12]}); frame {frame!r}; onsets {on} -> {path}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------- stage eval
def out_path(a):
    return Path(a.out) / "passages" / (f"{a.tag}_LETA.json" if a.formats == [LETA] else f"{a.tag}.json")


SAME = ("git_commit", "model_key", "revision", "dtype", "attn_implementation", "n_layers", "frame", "frames_sha256",
        "items_sha256", "squad_sha256", "max_new", "test_mode")


def model_files(p):
    """The verified repository and revision of the model files (VERIFIED.json's sha256 changes with every fetch: it
    records the fetch time)."""
    v = p.get("verified") or {}
    return v.get("repo"), v.get("revision")


def kept_pieces(old, P, rows):
    """{(split, fmt)} of an earlier file that this run keeps: same commit, model files, dtype, frame, items and rows, and
    the same valid set of the piece's split."""
    if not old or old.get("rows") != list(rows):
        return set()
    op = old.get("provenance", {})
    if any(op.get(k) != P.get(k) for k in SAME) or model_files(op) != model_files(P):
        return set()
    keep = set()
    for name, fm in old.get("splits", {}).items():
        if op.get("valid_sha256", {}).get(name) != P["valid_sha256"].get(name):
            continue
        keep |= {(name, f) for f in fm}
    return keep


def piece_order(a, keep):
    """(splits, formats) of the file in the fixed order of SPLITS and ALL_FORMATS (the pipeline's run order), whatever the
    request order and however the pieces are spread over sessions: the requested ones and those of kept pieces outside
    the request (an earlier run's)."""
    names = [s for s in SPLITS if s in a.splits or any(n == s for n, _ in keep)]
    fmts = [f for f in ALL_FORMATS if f in a.formats or any(g == f for _, g in keep)]
    return names, fmts


@torch.no_grad()
def stage_eval(a):
    """The weights load only when a piece starts: a step past the deadline, or one whose pieces are all kept, writes or
    keeps its file from the tokenizer and the config alone (the loaded model is then checked against that provenance)."""
    t0 = time.time()
    frame, ffile, fsha = read_frame(a)
    ok, sha = items_ok(a)
    assert ok, f"items sha256 {sha} != {ITEMS_SHA256}"
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    dev = eval_device(a)
    nL = cfg.num_hidden_layers
    rows = window_rows(nL)
    items = load_items(a)
    valid, skipped = {}, {}
    for name in SPLITS:                                     # both splits' valid sets are recorded; a.splits are run
        valid[name], skipped[name] = valid_split(tok, items, name, frame, a.n)
    vsha = {name: sha_ids([it["id"] for it in v]) for name, v in valid.items()}
    P = base_prov(a, "eval", cfg, dev, nL) | {
        "frame": frame, "frames_file": ffile, "frames_sha256": fsha, "valid_sha256": vsha,
        "stage8a_population_sha256": stage8a_population(a), "population": list(a.splits), "population_sha256": vsha,
        "n_items": {k: len(v) for k, v in valid.items()}, "skipped_items": skipped, "formats": list(a.formats),
        "splits": list(a.splits), "formats_not_run": [], "kept": [], "carried": [], "complete": False}
    path = out_path(a)
    old = json.loads(path.read_text()) if path.exists() else None
    keep = kept_pieces(old, P, rows)                        # every compatible piece, also those outside the request
    names, fmts = piece_order(a, keep)
    J = {"provenance": P, "rows": list(rows), "frame": frame, "splits": {name: {} for name in names}}

    def put(name, f, recs):                                 # the file keeps the fixed order, whatever order pieces come in
        J["splits"][name][f] = recs
        J["splits"] = {n: {g: J["splits"][n][g] for g in fmts if g in J["splits"][n]} for n in names}

    for name in names:
        for f in fmts:
            if (name, f) in keep:
                put(name, f, old["splits"][name][f])
                P["kept"].append(f"{name}/{f}")
                if name not in a.splits or f not in a.formats:
                    P["carried"].append(f"{name}/{f}")
                P["timings"][f"{name}/{f}"] = old["provenance"].get("timings", {}).get(f"{name}/{f}")
    if P["kept"]:
        log(f"[{a.tag}] resuming {path}: kept {P['kept']}" + (f" (outside this request: {P['carried']})" if P["carried"] else ""))
    rates = [P["timings"][f"{name}/{f}"] / len(J["splits"][name][f]) for name in J["splits"] for f in J["splits"][name]
             if J["splits"][name][f] and P["timings"].get(f"{name}/{f}")]
    per_item, status, model = max(rates, default=None), 0, None   # seconds per item (kept pieces, then measured)

    def late(name, fmt, need):                              # after one piece is left out, so are the later ones
        if status == 3 or deadline_passed(need, a.reserve_min):
            P["formats_not_run"].append(f"{name}/{fmt}")
            log(f"[{a.tag}] {name}/{fmt} not started: projected {need:.0f}s past STAGE8_DEADLINE minus {a.reserve_min} min")
            return True
        return False

    for name in a.splits:
        for fmt in a.formats:
            if fmt in J["splits"][name]:
                continue
            need = (per_item or 0.0) * len(valid[name])
            if late(name, fmt, need):
                status = 3
                continue
            if model is None:                               # the first piece to run loads the weights, then is gated again
                t1 = time.time()
                model, _, ldev = load_model(a)
                assert ldev == dev, (ldev, dev)
                check_loaded(model, dev, P)
                P["timings"]["load"] = round(time.time() - t1, 1)
                if late(name, fmt, need):
                    status = 3
                    continue
            t1 = time.time()
            recs = []
            for it in valid[name]:
                try:
                    recs.append(run_item(model, tok, it, fmt, frame, rows, nL, dev, name, a.max_new))
                except Invalid as ex:          # cannot happen after valid_all; recorded for safety
                    skipped[name].append([it["id"], f"{fmt}: {ex}"])
            sec = time.time() - t1
            if valid[name]:
                rate = sec / len(valid[name])
                per_item = rate if per_item is None else max(per_item, rate)
            put(name, fmt, recs)
            P["timings"][f"{name}/{fmt}"] = round(sec, 1)
            P["finished_utc"] = s9.run_meta(a.test)["finished_utc"]
            P["wrapper"] = dict(WRAPPER_USED)
            write_atomic(J, path)
            log(f"[{a.tag}] {name}/{fmt}: {len(recs)} items ({sec:.0f}s)")
    P["complete"] = not P["formats_not_run"]
    P["timings"]["eval"] = round(time.time() - t0, 1)
    P["finished_utc"] = s9.run_meta(a.test)["finished_utc"]
    write_atomic(J, path)
    log(f"wrote {path} ({time.time() - t0:.0f}s" + ("" if model is not None else ", weights not loaded") + ")"
        + (f"; not run at the deadline: {P['formats_not_run']}" if status else ""))
    return status


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "eval"])
    ap.add_argument("--model", default=None, help="a local verified directory (s8_fetch) or a Hub id")
    ap.add_argument("--key", required=True, choices=KEYS, help="the model key of scripts/stage8_models.json")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--squad", default=str(ROOT / SQUAD), help="SQuAD v1.1 dev-v1.1.json (sha256 asserted)")
    ap.add_argument("--items", default=str(ITEMS))
    ap.add_argument("--formats", default=",".join(s9.PASSAGE_FORMATS), help="comma-separated; LETA alone writes <tag>_LETA.json")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--n", type=int, default=0, help="0 = every valid item (TEST_MODE: 2)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--reserve-min", type=float, default=0.0, help="minutes the pipeline needs after this step (deadline)")
    ap.add_argument("--out", default=None, help="default results/gpu_stage9 (TEST_MODE: results/gpu_stage9_test)")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, windowed passages, n = 2")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    if a.test:
        a.model, a.revision, a.dtype, a.n = TINY, None, "float32", a.n or 2
    assert a.model, "--model is required"
    a.out = a.out or str(ROOT / ("results/gpu_stage9_test" if a.test else "results/gpu_stage9"))
    a.formats = [f for f in a.formats.split(",") if f]
    a.splits = [s for s in a.splits.split(",") if s]
    assert a.formats and set(a.formats) <= set(ALL_FORMATS), f"formats {a.formats} not in {ALL_FORMATS}"
    assert LETA not in a.formats or a.formats == [LETA], "LETA runs alone (its own file, <tag>_LETA.json)"
    assert len(set(a.formats)) == len(a.formats) and a.splits and set(a.splits) <= set(SPLITS) and len(set(a.splits)) == len(a.splits)
    assert a.max_new == MAX_NEW or a.test, "the 48-token generation is fixed outside TEST_MODE"
    a.tag = ("TEST_" if a.test else "") + a.key
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    rc = {"check": stage_check, "eval": stage_eval}[a.stage](a)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())   # 3: complete up to the deadline, pieces not run (scripts/stage8_common.sh); 1: a check failed
