"""Stage 9, stories arm (docs/stage9/SPEC.md sections 5-7, 9, 11.3-11.4; predictions docs/STAGE9_PREDICTIONS.md): the
edit window at the writing token by format and depth, on the 150 new fresh story cores N9, with the onset l_b fixed on
the 30 stage-8b calibration cores C first.

Per item and format (NONE, BEFORE, POST-NULL, POST, AFTER, LETTER; ckeys.fresh.raw_prompt, the B, S and X prompts of one
length differing at one position p, experiments.fresh_factorial.encode_item): one forward of the B, S, X prompts with
the trie captures K and V at p in every layer (ckeys.clamp.capture_kv); then every window row (SPEC 7.1) is a table
assignment on the B prompt that writes, in EVERY layer l' and channel ch, the captured K or V of the donor run
donor(row, l', ch) in {B, S}:
  none, none2       B everywhere (identity; first and last row of the batch). Same ids and tables in one batch, so
                    the two are bit-identical even in BF16: the BF16 floor is measured against the clean B run of the
                    capture pass (a batch of 3; its E and L stored as "clean_B"), as KV@0 is against the clean S run
  KV@0              S everywhere (the full patch: the S run)
  late@l / early@l  S in layers l' >= l / l' < l, B elsewhere (l in stage9.story_onsets(L): every second block 2 .. L-2)
  K@0 / V@0         S in the key / value channel of every layer
  Klate@l / Vlate@l S in the key / value channel of layers l' >= l (l in stage9.KV_ONLY_ONSETS = 4, 8, 12, 16)
Rows are scored by ONE forward over the prompt and the token trie (ckeys.surface.score; the trie passed J-B-G3 at the
three models): E = log-sum over the 32 fixed E forms (stage-8b frames file: none discovered), L = log p(" w"), six-vectors
in canonical word order (LETTER: letters " X", "X", " **X", "**X" mapped back to the word each letter names in the item's
listing). Generation (ckeys.generate.greedy, 16 tokens, the KV cache on, the rows clamped in the prompt pass): one batch
of the rows on the B prompt plus the clean S prompt as an unclamped extra row; a row ends at EOS / end of turn, at a
newline, at 16 tokens or by the stop rule (word formats: ckeys.fresh.stop_fn; LETTER: ckeys.stage9.letter_stop). Answers:
ckeys.fresh.parse (word formats) or ckeys.stage9.parse_letter (LETTER). Under LETTER every generated row also stores
"word" = ckeys.fresh.parse of the same text, so a letter answer that disagrees with a candidate word in the same text
(for example " A drawer": an article read as option A by SPEC 9.2) can be counted afterwards (a sensitivity number; the
answer stays SPEC 9.2's).

Stages (one process each; BF16, sdpa, use_cache=False in every scoring pass; scripts/gpu_stage9.sh):
  check  tokenizer only, no weights: the N9 and C pins; every N9 and C item valid in the six formats; the word form sets
         (with the stage-8b frames) build and decode back; the letter form set builds and decodes back; the stage-8b
         frames and G3 files on disk equal their pins (G3 scorer "trie") -> OUT/check/<tag>_stories.json; exit 1 on a
         failure
  cal    C (30 cores), the six formats (AFTER required: exit 1 without it), every row and the generation ->
         OUT/lb/<tag>.json with l_b = the first onset whose late@l AFTER flip (generated answer = S over all C cores, no
         competence filter) is <= 0.2, or None. The lb file is written only when every format is done; until then the
         progress is kept in OUT/lb/<tag>.json.partial (resumed by format; exit 3 at the deadline)
  eval   N9 (150 items), the six formats -> OUT/stories/<tag>.json; reads the lb file first (its sha256 and l_b
         recorded; exit 1 before the model is loaded when the lb file was written at another git commit or model
         revision: redo the cal step with FORCE_STEPS=cal_<key>); the file is rewritten after every format; a rerun
         keeps the formats of an existing file with the same git_commit, revision and lb_sha256; exit 3 when the
         deadline (STAGE8_DEADLINE minus --reserve-min) stops it before the last format (and when the lb file is
         missing: the cal step has not finished)
The formats run in the order --formats gives (default STORY_FORMATS); the records of a file are in STORY_FORMATS order.
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct (or the --model given), FP32, CPU; N9 and C n = 2 each;
frames [] (no frames file); outputs tagged TEST_<key>.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from ckeys import edits, fresh, stage9
from ckeys.clamp import capture_kv, clamp_kv, stack_rows
from ckeys.encoding import LETTERS, WRAPPER_USED
from ckeys.generate import greedy
from ckeys.interventions import blocks
from ckeys.surface import FormSet
from experiments import fresh_factorial as ff
from experiments.fresh_factorial import Invalid, encode_item
from experiments.format_factorial import provenance as ff_provenance
from experiments.ioi_factorial import device_name
from experiments.stage6_heads import write_atomic

TINY = ff.TINY
MAX_NEW = 16
KEYS = ("mistral7", "llama8", "qwen7")
LB_MAX = 0.2
LB_RULE = "l_b = first onset with late AFTER flip <= 0.2 over all C cores"
STAGE8B = "results/gpu_stage8b"


def log(s):
    print(s, flush=True)


def r5(x):
    if isinstance(x, (list, tuple)):
        return [r5(v) for v in x]
    return round(float(x), 5)


# --------------------------------------------------------------------------- rows (SPEC 7.1)
def row_names(nL: int) -> list[str]:
    """The window rows in batch order: none, KV@0, late@l, early@l, K@0, V@0, Klate@l, Vlate@l, none2."""
    on = stage9.story_onsets(nL)
    kv = [l for l in stage9.KV_ONLY_ONSETS if l < nL]
    return (["none", "KV@0"] + [f"late@{l}" for l in on] + [f"early@{l}" for l in on] + ["K@0", "V@0"]
            + [f"Klate@{l}" for l in kv] + [f"Vlate@{l}" for l in kv] + ["none2"])


def donor(row: str, l: int, ch: str) -> str:
    """The run ("S" or "B") whose captured K (ch "k") or V ("v") the row writes at p in layer ``l``."""
    if row in ("none", "none2"):
        return "B"
    kind, _, at = row.partition("@")
    l0 = int(at)
    s = {"KV": True, "late": l >= l0, "early": l < l0, "K": ch == "k", "V": ch == "v",
         "Klate": ch == "k" and l >= l0, "Vlate": ch == "v" and l >= l0}[kind]
    return "S" if s else "B"


# --------------------------------------------------------------------------- form sets, answers
def letter_form_set(tok) -> FormSet:
    return FormSet(tok, LETTERS, {"E": edits.LETTER_FRAMES})


def form_set9(tok, it, fmt: str, frames=()) -> FormSet:
    """SPEC 9.1: the letter set under LETTER, else ckeys.fresh.form_set(tok, it, frames)[0]."""
    return letter_form_set(tok) if fmt == "LETTER" else fresh.form_set(tok, it, frames)[0]


def six(res: dict, name: str, it, fmt: str, i: int) -> list[float]:
    """The six scores of batch row ``i`` in canonical word order: index j is it.words[j]; under LETTER the score of the
    letter that names word j in the item's listing, LETTERS[it.order.index(j)]."""
    if fmt == "LETTER":
        return [float(res[name][LETTERS[it.order.index(j)]][i]) for j in range(6)]
    return [float(res[name][w][i]) for w in it.words]


def answer(text: str, it, fmt: str) -> tuple[str, str]:
    """(answer word or "other", how in {"word", "letter", "none"})."""
    if fmt == "LETTER":
        return stage9.parse_letter(text, it)
    w = fresh.parse(text, it.words)
    return w, ("word" if w != "other" else "none")


def gen_fields(text: str, it, fmt: str) -> dict:
    """The answer fields of a generated row: ans and how (SPEC 9.2); under LETTER also "word" = ckeys.fresh.parse(text)
    (the first candidate word the text names, or "other"), so that a letter answer contradicted by a word in the same
    text ("A drawer" where A is not the drawer) can be counted. The answer is SPEC 9.2's either way."""
    a, how = answer(text, it, fmt)
    out = {"ans": a, "how": how}
    if fmt == "LETTER":
        out["word"] = fresh.parse(text, it.words)
    return out


def stop_of(tok, it, fmt: str):
    return stage9.letter_stop(tok, it) if fmt == "LETTER" else fresh.stop_fn(tok, it.words)


# --------------------------------------------------------------------------- one item in one format
@torch.no_grad()
def run_item9(model, tok, it, fmt: str, fs: FormSet, nL: int, dev, rows=None, gen: bool = True, scorer: str = "trie",
              max_new: int = MAX_NEW) -> dict:
    """The record of SPEC 11.4 for one item and format: per row E, L (canonical word order) and, with ``gen``, the
    generated text, answer, parse route (LETTER: also the word parse, gen_fields) and number of new tokens; the same for
    the clean S run; and E, L of the clean B run of the capture pass ("clean_B", the reference of the BF16 floor).
    Raises Invalid."""
    assert (list(fs.words) == list(LETTERS)) == (fmt == "LETTER"), "LETTER needs the letter form set, the others the words"
    if fmt != "LETTER":
        assert list(fs.words) == list(it.words), "the word form set must list the item's six words in canonical order"
    rows = list(rows or row_names(nL))
    ids, p = encode_item(tok, it, fmt)
    if scorer == "cached" and fs.nodes:
        assert max(len(n) for n in fs.nodes) + 1 < p, "a trie path as long as p"
    sc = ff.SCORERS[scorer]
    with capture_kv(model, [p], range(nL)) as t:
        rc = sc(model, torch.cat([ids[n] for n in "BSX"]).to(dev), fs)
    kv = {n: {k: v[i] for k, v in t.items()} for i, n in enumerate("BSX")}
    tabs = stack_rows(kv, [lambda l, ch, r=r: donor(r, l, ch) for r in rows], range(nL))
    with clamp_kv(model, [p], tabs, range(nL)):
        rr = sc(model, ids["B"].to(dev).expand(len(rows), -1), fs)
    out = {r: {"E": r5(six(rr, "E", it, fmt, i)), "L": r5(six(rr, "L", it, fmt, i))} for i, r in enumerate(rows)}
    clean = {"E": r5(six(rc, "E", it, fmt, 1)), "L": r5(six(rc, "L", it, fmt, 1))}
    clean_b = {"E": r5(six(rc, "E", it, fmt, 0)), "L": r5(six(rc, "L", it, fmt, 0))}
    if gen:
        R = len(rows)
        gids = torch.cat([ids["B"].expand(R, -1), ids["S"]]).to(dev)
        gt = stack_rows(kv, [lambda l, ch, r=r: donor(r, l, ch) for r in rows] + [lambda l, ch: "S"], range(nL))
        sel = torch.tensor([True] * R + [False])
        with clamp_kv(model, [p], gt, range(nL), per_row=sel):
            g = greedy(model, tok, gids, max_new=max_new, stop=stop_of(tok, it, fmt))
        for i, d in enumerate(list(out.values()) + [clean]):
            text = fresh.gen_text(tok, gids[i].tolist(), g[i])
            d.update(gen=text, **gen_fields(text, it, fmt), n_tok=len(g[i]))
    return {**it.meta(), "pop": it.pop, "idx": it.idx, "fmt": fmt, "p": p, "T": int(ids["B"].shape[1]),
            "rows": out, "clean_S": clean, "clean_B": clean_b}


# --------------------------------------------------------------------------- l_b (SPEC 9.6)
def late_flips(items: list[dict], onsets, fmt: str = "AFTER") -> dict:
    """{onset: share of the records of ``fmt`` whose late@onset answer is S} (no competence filter)."""
    recs = [r for r in items if r["fmt"] == fmt]
    if not recs:
        return {}
    return {l: sum(r["rows"][f"late@{l}"]["ans"] == r["track"]["S"] for r in recs) / len(recs) for l in onsets}


def lb_of(flips: dict, onsets) -> int | None:
    """The first onset (ascending) whose late AFTER flip is <= LB_MAX, else None."""
    return next((l for l in sorted(onsets) if flips[l] <= LB_MAX), None)


# --------------------------------------------------------------------------- model, files, provenance
def load_model(a, need_model=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() and not a.test else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if not need_model:
        return None, tok, dev
    kw = dict(dtype=getattr(torch, a.dtype), revision=a.revision, attn_implementation=a.attn)
    if dev == "cuda":
        kw["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if dev == "cpu":
        model = model.to(dev)
    assert model.config._attn_implementation == a.attn
    for prm in model.parameters():
        prm.requires_grad_(False)
    return model, tok, dev


def stage8b_files(a) -> dict:
    """The stage-8b frames and G3 files of the key (pinned sha256 asserted), or the TEST stand-ins (no files)."""
    if a.test:
        return {"frames_file": None, "frames_sha256": None, "frames": [], "g3_file": None, "g3_sha256": None,
                "scorer": "trie"}
    assert a.key in KEYS, f"stage 9 runs {KEYS}, not {a.key}"
    out = {}
    for kind in ("frames", "g3"):
        rel = f"{a.stage8b}/{kind}/{a.key}.json"
        f = Path(rel) if Path(rel).is_absolute() else stage9.ROOT / rel
        h = stage9.file_sha(f)
        pin = stage9.ARTEFACT_SHA256[f"{STAGE8B}/{kind}/{a.key}.json"]
        assert h == pin, f"{rel}: sha256 {h} differs from the pin {pin}"
        out[f"{kind}_file"], out[f"{kind}_sha256"] = rel, h
        out[kind] = json.loads(f.read_text())
    assert out["g3"]["scorer"] == "trie" and out["g3"]["pass"], "the stage-8b G3 file does not choose the trie"
    out["frames"], out["scorer"] = list(out["frames"]["frames"]), "trie"
    del out["g3"]
    return out


def verified_of(model_dir):
    return ff.verified(model_dir) if model_dir and Path(model_dir).is_dir() else None


def prov(a, model, dev, extra=None) -> dict:
    ver = verified_of(a.model)
    p = ff_provenance(a) | stage9.run_meta(a.test) | {
        "stage9": True, "arm": "stories", "stage": a.stage, "model": a.model, "model_key": a.key,
        "revision": a.revision or (ver or {}).get("revision"), "verified": ver, "test_mode": a.test, "max_new": a.max_new,
        "reserve_min": a.reserve_min, "kv_only_onsets": list(stage9.KV_ONLY_ONSETS), "timings": {}}
    if model is not None:
        cfg = model.config
        p |= {"dtype": str(next(model.parameters()).dtype), "attn_implementation": cfg._attn_implementation,
              "device": device_name(dev), "n_layers": len(blocks(model)), "hidden": cfg.hidden_size,
              "n_heads": cfg.num_attention_heads, "n_kv_heads": cfg.num_key_value_heads}
    p["wrapper"] = dict(WRAPPER_USED)
    return p | (extra or {})


def items_of(a, pop: str) -> list:
    items = list(stage9.stories() if pop == "N9" else stage9.calibration())
    n = a.n if pop == "N9" else a.n_cal
    return items[:n] if n else items


def pop_fields(a, pop, items) -> dict:
    return {"population": pop, "population_sha256": stage9.POP_SHA256[pop], "n_items": len(items)}


def time_left(a) -> float:
    """Seconds before STAGE8_DEADLINE minus the reserve of the later steps (inf without a deadline)."""
    d = os.environ.get("STAGE8_DEADLINE")
    return math.inf if not d else float(d) - time.time() - 60.0 * a.reserve_min


def out_file(a, kind) -> Path:
    o = Path(a.out)
    return {"check": o / "check" / f"{a.tag}_stories.json", "lb": o / "lb" / f"{a.tag}.json",
            "lb_partial": o / "lb" / f"{a.tag}.json.partial", "eval": o / "stories" / f"{a.tag}.json"}[kind]


# --------------------------------------------------------------------------- stage check (tokenizer only)
def stage_check(a):
    _, tok, _ = load_model(a, need_model=False)
    t0, fails, skipped, info = time.time(), [], {}, {}
    try:
        files = stage8b_files(a)
    except (AssertionError, OSError, KeyError, ValueError) as ex:
        files = {"frames_file": None, "frames_sha256": None, "frames": [], "g3_file": None, "g3_sha256": None,
                 "scorer": None}
        fails.append(f"stage-8b files: {ex}")
    pops = {}
    for pop in ("N9", "C"):
        try:
            pops[pop] = items_of(a, pop)
        except AssertionError as ex:
            fails.append(f"population {pop}: {ex}")
    nodes, bad_decode, dropped = {}, [], {}
    try:
        lfs = letter_form_set(tok)
        bad_decode += [("LETTER", x) for x in lfs.check_decode()]
        info["letter_nodes"] = len(lfs)
    except ValueError as ex:
        fails.append(f"letter FormSet: {ex}")
    for pop, items in pops.items():
        for it in items:
            try:
                fs, drop = fresh.form_set(tok, it, files["frames"])
                nodes.setdefault(pop, []).append(len(fs))
                bad_decode += [(it.key, x) for x in fs.check_decode()]
                if drop:
                    dropped[it.key] = drop
            except ValueError as ex:
                fails.append(f"{it.key}: word FormSet: {ex}")
            for fmt in stage9.STORY_FORMATS:
                try:
                    encode_item(tok, it, fmt)
                except Invalid as ex:
                    skipped.setdefault(f"{pop}/{fmt}", []).append([it.key, str(ex)])
    if bad_decode:
        fails.append(f"{len(bad_decode)} forms do not decode back: {bad_decode[:5]}")
    if dropped:
        fails.append(f"frames dropped for {len(dropped)} items: {dict(list(dropped.items())[:3])}")
    if skipped:
        fails.append(f"skipped items: { {k: len(v) for k, v in skipped.items()} }")
    for w in fresh.LEX1 + fresh.LEX2:
        if len(tok(" " + w, add_special_tokens=False).input_ids) != 1:
            fails.append(f"' {w}' is not one token")
    P = prov(a, None, None, {"formats": list(stage9.STORY_FORMATS), "populations": {
        k: {"population_sha256": stage9.POP_SHA256[k], "n_items": len(v)} for k, v in pops.items()},
        "skipped_items": [[k, *x] for k, v in skipped.items() for x in v],
        **{k: v for k, v in files.items() if k != "frames"}, "frames": files["frames"],
        "seconds": round(time.time() - t0, 1)})
    rep = {"provenance": P, "pass": not fails, "fails": fails, "skipped": skipped,
           "trie_nodes": {k: [min(v), max(v)] for k, v in nodes.items()}, "frames_dropped": dropped, **info}
    write_atomic(rep, out_file(a, "check"))
    log(f"check {'PASS' if not fails else 'FAIL'}: {len(fails)} failures {fails[:3]}; trie nodes {rep['trie_nodes']}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------- the format loop of cal and eval
def run_formats(a, model, tok, dev, items, path, P, old_ok):
    """Run every format not yet in the file (resume by format), rewriting ``path`` after each; returns (records,
    formats not run). ``old_ok(prov)`` decides whether an existing file is resumed."""
    nL = len(blocks(model))
    rows = row_names(nL)
    fmts = list(a.formats)
    recs, done = [], []
    if path.exists():
        old = json.loads(path.read_text())
        op = old["provenance"]
        if old_ok(op) and op.get("n_items") == len(items) and old.get("rows") == rows:
            done = [f for f in op.get("formats_done", []) if f in fmts]
            recs = [r for r in old["items"] if r["fmt"] in done]
            P["timings"] |= {f: op.get("timings", {}).get(f) for f in done}
            P["skipped_items"] = [s for s in op.get("skipped_items", []) if s["fmt"] in done]
            log(f"resuming {path}: formats {done} kept")
        else:
            log(f"{path} exists from another commit, revision, lb file or population: rerun from the start")
    fsets = {}
    per, not_run = None, []

    def write():
        P["formats_done"] = [f for f in fmts if f in done]
        P["formats_not_run"] = [f for f in fmts if f not in done]
        P["finished_utc"] = stage9.run_meta(a.test)["finished_utc"]
        recs.sort(key=lambda r: (stage9.STORY_FORMATS.index(r["fmt"]), r["idx"]))   # file order: canonical
        write_atomic({"provenance": P, "population": P["population"], "rows": rows, "items": recs}, path)

    for fmt in fmts:
        if fmt in done:
            continue
        left = time_left(a)
        if left <= 0 or (per is not None and left < 1.2 * per):
            not_run.append(fmt)
            log(f"  {fmt} not run: the deadline (minus {a.reserve_min} min reserve) is reached")
            continue
        t0 = time.time()
        for it in items:
            try:
                key = (it.key, fmt == "LETTER")
                if key not in fsets:
                    fsets[key] = form_set9(tok, it, fmt, P["frames"])
                recs.append(run_item9(model, tok, it, fmt, fsets[key], nL, dev, rows=rows, gen=True,
                                      scorer=P["scorer"], max_new=a.max_new))
            except Invalid as ex:
                P["skipped_items"].append({"key": it.key, "fmt": fmt, "reason": str(ex)})
        dt = time.time() - t0
        per = dt if per is None else max(per, dt)
        P["timings"][fmt] = round(dt, 1)
        done.append(fmt)
        write()
        log(f"  {fmt}: {sum(r['fmt'] == fmt for r in recs)} items ({dt:.0f}s)")
    write()
    return recs, not_run


def setup(a):
    model, tok, dev = load_model(a)
    nL = len(blocks(model))
    files = stage8b_files(a)
    log(f"frames file {files['frames_file']} sha256 {files['frames_sha256']}: frames {files['frames']}; "
        f"G3 {files['g3_file']} -> scorer {files['scorer']}")
    return model, tok, dev, nL, files


# --------------------------------------------------------------------------- stage cal (C -> l_b)
@torch.no_grad()
def stage_cal(a):
    if "AFTER" not in a.formats:
        log(f"cal needs AFTER (the l_b rule reads the late AFTER flips); --formats {','.join(a.formats)} (exit 1)")
        return 1
    model, tok, dev, nL, files = setup(a)
    items = items_of(a, "C")
    onsets = stage9.story_onsets(nL)
    P = prov(a, model, dev, {**pop_fields(a, "C", items), "formats": list(a.formats), "onsets": onsets,
                             "skipped_items": [], **files})
    part = out_file(a, "lb_partial")
    recs, not_run = run_formats(a, model, tok, dev, items, part, P,
                                lambda op: op.get("git_commit") == P["git_commit"] and op.get("revision") == P["revision"])
    if not_run:
        log(f"cal partial: formats {not_run} not run; progress in {part} (exit 3, the next session goes on)")
        return 3
    fl = late_flips(recs, onsets)
    lb = lb_of(fl, onsets) if fl else None
    P |= {"formats_not_run": [], "l_b": lb, "finished_utc": stage9.run_meta(a.test)["finished_utc"]}
    rows = row_names(nL)
    res = {"provenance": P, "population": "C", "rows": rows, "l_b": lb,
           "late_after_flip": {str(l): r5(v) for l, v in fl.items()}, "rule": LB_RULE, "items": recs}
    f = out_file(a, "lb")
    write_atomic(res, f)
    part.unlink(missing_ok=True)
    log(f"l_b = {lb} (late AFTER flips {res['late_after_flip']}); wrote {f} sha256 {stage9.file_sha(f)}")
    return 0


# --------------------------------------------------------------------------- stage eval (N9)
def lb_mismatch(lbp: dict, a) -> list[str]:
    """Why an lb file's provenance ``lbp`` cannot serve this eval: another git commit (the scorer wants one commit
    across the results files) or another model revision or key. [] when it can."""
    ver = verified_of(a.model)
    here = {"git_commit": stage9.run_meta(a.test)["git_commit"],
            "revision": a.revision or (ver or {}).get("revision"), "model_key": a.key}
    return [f"{k} {lbp.get(k)} != this run's {v}" for k, v in here.items() if lbp.get(k) != v]


@torch.no_grad()
def stage_eval(a):
    lbf = out_file(a, "lb")
    if not lbf.exists():
        log(f"no lb file {lbf}: the cal step has not finished (exit 3, the next session goes on)")
        return 3
    lb_sha = stage9.file_sha(lbf)
    lbj = json.loads(lbf.read_text())
    log(f"lb file {lbf} sha256 {lb_sha}: l_b = {lbj['l_b']}")
    bad = lb_mismatch(lbj["provenance"], a)
    if bad:
        log(f"the lb file {lbf} was not written by this commit and model: {'; '.join(bad)}. Redo the cal step "
            f"(FORCE_STEPS=cal_{a.key}) before the eval (exit 1)")
        return 1
    model, tok, dev, nL, files = setup(a)
    assert lbj["provenance"]["n_layers"] == nL and lbj["provenance"]["model_key"] == a.key, "the lb file is another model's"
    items = items_of(a, "N9")
    P = prov(a, model, dev, {**pop_fields(a, "N9", items), "formats": list(a.formats),
                             "onsets": stage9.story_onsets(nL), "skipped_items": [], **files,
                             "lb_file": str(lbf), "lb_sha256": lb_sha, "l_b": lbj["l_b"]})
    path = out_file(a, "eval")
    _, not_run = run_formats(a, model, tok, dev, items, path, P,
                             lambda op: (op.get("git_commit"), op.get("revision"), op.get("lb_sha256"))
                             == (P["git_commit"], P["revision"], lb_sha))
    log(f"wrote {path} (formats not run {not_run})")
    return 3 if not_run else 0


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "cal", "eval"])
    ap.add_argument("--model", default=None, help="a local verified directory (s8_fetch) or a Hub id")
    ap.add_argument("--key", default="model", help="the model key of scripts/stage8_models.json (the output tag)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--formats", default=",".join(stage9.STORY_FORMATS))
    ap.add_argument("--n", type=int, default=0, help="N9 items (0 = all 150)")
    ap.add_argument("--n-cal", type=int, default=0, help="C items (0 = all 30)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--stage8b", default=STAGE8B, help="the committed stage-8b results (frames and G3 files)")
    ap.add_argument("--reserve-min", type=float, default=0.0, help="minutes the pipeline needs after this step")
    ap.add_argument("--out", default="results/gpu_stage9")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: Qwen2.5-0.5B FP32 on the CPU, n = 2")
    a = ap.parse_args(argv)
    a.test = a.test or bool(os.environ.get("TEST_MODE"))
    if a.test:   # the pipeline passes the 0.5B hub id (s8_fetch in TEST_MODE); the tests pass a tiny random model
        a.model, a.revision, a.dtype = a.model or TINY, None, "float32"
        a.n, a.n_cal = a.n or 2, a.n_cal or 2
    assert a.model, "--model is required"
    a.formats = [f for f in a.formats.split(",") if f]
    assert all(f in stage9.STORY_FORMATS for f in a.formats), a.formats
    a.tag = ("TEST_" if a.test else "") + a.key
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    return {"check": stage_check, "cal": stage_cal, "eval": stage_eval}[a.stage](a) or 0


if __name__ == "__main__":
    sys.exit(main())   # 3: complete up to the deadline, parts not run; 1: the check failed
