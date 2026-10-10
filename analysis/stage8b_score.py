"""Score part B of preregistration P-2026-10-10-J (docs/PREREGISTRATION.md, GPU stage 8 part B): the fresh-sample
replication in new model families, scored on the forms the models emit. Gates J-B-G0 to J-B-G5 and lines J-B1 to J-B8
with J-B5b, J-B-NULL, J-B-LB and J-B-SMALL, as worded in the entry (line table and per-line logic in
analysis/stage8b_parts/lines.py; the bootstrap and the criteria in analysis/stage8b_parts/stats.py).

Inputs under --results (default results/gpu_stage8b), as written by scripts/gpu_stage8b.sh and
experiments/fresh_factorial.py: tokcheck/<tag>.json (J-B-G0b), frames/<tag>.json (the calibration), g3/<tag>.json
(J-B-G3), eval/<tag>_F.json and eval/<tag>_S0.json (eval/<tag><suffix>_S0.json: the exploratory X1 runs), jb8/<tag>.json
(experiments/paper1_frames.py --score E) and RELEASE.txt, verified/<key>.json (J-B-G1), the pytest log under logs/ (J-B-G0;
outside TEST a missing log fails it), COMMIT.txt, ENV.txt, REVISIONS.txt, SKIPPED.txt, FETCH_FAILED.txt; and the committed
stage-1 and stage-3b result files (results/gpu_stage1, results/gpu_stage3b/format_2x2; J-B-G2).
Output (--out, default {results}/STAGE8B_SCORE.txt), in this order: PROVENANCE, POPULATION, GATES, PREDICTIONS (one line
per confirmatory line: code, class, kind, prior, verdict; then per model the numbers, bounds and evaluability), REPORTED
(per-family verdicts with the pre-written sentence that applies, J-B-G2 detail, the numbers behind the figure),
SUMMARY (per class and kind: lines, MET, NOT MET, NOT EVALUABLE, observed against expected met count, Brier score; the
met rate among R account lines; the Holm sensitivity analysis over the R-class account lines, computed with the
shared analysis/stage8_holm.py), EXPLORATORY. The paper tables tab_fresh.tex,
tab_mass.tex and tab_invariance.tex are written to the results directory.
--test (TEST_MODE; also when every results file is tagged TEST_): size floors and J-B-G1 are waived, J-B-G2 has no
reference, and the verdicts are plumbing checks, not results.
Exit status 1 if a part of the scorer raised (traceback in the report, the line NOT EVALUABLE); 2 if, outside TEST, the
provenance or population check reports MISMATCH (the file is still written).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage8b_parts import lines as ln  # noqa: E402
from stage8b_parts import stats as st  # noqa: E402
from stage8b_parts import tables as tb  # noqa: E402
from stage8b_parts.data import Pop, ff_ident  # noqa: E402

ROOT = HERE.parent
POP_SHA = {"F": "e87047c9c877a21db89bf5081d082de748ea5d77b620e33135d178c3bf24b14f",
           "S0": "48bb0a3ad22463ee1831cabaef714d87b2a7cb7f721e23d877111f19ba0d6900",
           "C": "a625fd13d1dd67bc0c01c3a173807c7b71ee8347451c139d93ffc20f1c6486e9"}
N_POP = {"F": 150, "S0": 150, "C": 30}
G0_FILES = {"tests/test_fresh.py": 24, "tests/test_fresh_factorial.py": 12, "tests/test_stage8b_score.py": 20,
            "tests/test_surface.py": 13, "tests/test_generate.py": 5, "tests/test_clamp_families.py": 5,
            "tests/test_stage8_populations.py": 5, "tests/test_stage8_holm.py": 7}   # every test of each file
G0_SKIP_OK = ("cached_tokenizer", "study_tokenizers")   # tokenizer tests that skip without a cache; JB-G0b repeats them
MODEL_ORDER = ln.P4 + ln.N4 + (ln.FALLBACK, ln.SMALL, ln.JB8_KEY) + ln.X2


def load_json(f):
    try:
        return json.load(open(f))
    except Exception as ex:  # noqa: BLE001  (a truncated file is reported, not fatal)
        return {"error": repr(ex)}


def sha_file(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def keyof(tag):
    return tag[5:] if tag.startswith("TEST_") else tag


def order(keys):
    return sorted(keys, key=lambda k: (MODEL_ORDER.index(k) if k in MODEL_ORDER else 99, k))


def load(root):
    F = {}
    for d in ("tokcheck", "frames", "g3", "eval", "jb8", "verified"):
        for f in sorted((root / d).glob("*.json")) if (root / d).is_dir() else []:
            F[f"{d}/{f.name}"] = load_json(f)
    return F


# --------------------------------------------------------------------------- pytest gate
def pytest_counts(root, files):
    """{file: (passed, failed or error, skipped, skipped ids)} from the last pytest session in the logs that ran them."""
    logs = sorted((root / "logs").glob("*pytest*")) if (root / "logs").is_dir() else []
    best = None
    for f in logs:
        t = f.read_text(errors="replace")
        for sess in re.split(r"=+ test session starts =+", t)[1:]:
            if any(x in sess for x in files):
                best = (f.name, sess)
    if best is None:
        return None, {}
    res = {}
    for x in files:
        stt = {}
        for m in re.finditer(rf"({re.escape(x)}::\S+) (PASSED|FAILED|ERROR|SKIPPED)|(PASSED|FAILED|ERROR|SKIPPED) ({re.escape(x)}::\S+)", best[1]):
            tid, w = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            stt[tid] = w
        sk = [t for t, w in stt.items() if w == "SKIPPED"]
        res[x] = (sum(w == "PASSED" for w in stt.values()), sum(w in ("FAILED", "ERROR") for w in stt.values()), len(sk), sk)
    return best[0], res


def gate_g0(root, out):
    log, res = pytest_counts(root, G0_FILES)
    if log is None:
        out("  J-B-G0  FP32 exactness: no pytest log in the results -> NOT EVALUABLE (not run)")
        return None
    ok = True
    parts = []
    for f, (p, x, s, sk) in res.items():
        bad_skip = [t for t in sk if not any(a in t for a in G0_SKIP_OK)]
        good = p + (s - len(bad_skip)) >= G0_FILES[f] and p > 0 and x == 0 and not bad_skip
        ok &= good
        parts.append(f"{f} {p} passed, {x} failed, {s} skipped" + (f" (not allowed: {bad_skip})" if bad_skip else "")
                     + f" (>= {G0_FILES[f]})")
    out(f"  J-B-G0  FP32 exactness ({log}, its last session that ran them; skips allowed only for the cached-tokenizer "
        f"tests, which J-B-G0b repeats per model): " + "; ".join(parts) + f" -> {st.V(ok)}")
    return ok


# --------------------------------------------------------------------------- provenance and population
def provenance(root, F, out, test):
    out("PROVENANCE (the pipeline's COMMIT.txt, ENV.txt, REVISIONS.txt, SKIPPED.txt, FETCH_FAILED.txt, RELEASE.txt; every results file)")
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt", "SKIPPED.txt", "FETCH_FAILED.txt", "RELEASE.txt"):
        f = root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text().splitlines() if l.strip()) if f.exists() else " absent"))
    ok, commits, bad = True, set(), []
    for lab, J in sorted(F.items()):
        if lab.startswith("verified/"):
            continue
        if "error" in J or "provenance" not in J:
            out(f"  {lab}: " + (f"UNREADABLE {J['error']}" if "error" in J else "NO PROVENANCE") + "  MISMATCH")
            bad.append(lab)
            continue
        p = J["provenance"]
        if lab.startswith("jb8/"):
            commits.add(None)
            out(f"  {lab}: repo {p.get('repo')}, revision {p.get('revision')}, score {p.get('score')}, frames sha256 "
                f"{str(p.get('frames_sha256'))[:12]}, dtype {p.get('dtype')}, attn {p.get('attn_implementation')}, verified "
                f"{(p.get('verified') or {}).get('revision')}")
            continue
        commits.add(p.get("git_commit"))
        tag = Path(lab).stem
        key = keyof(tag.split("_F")[0].split("_S0")[0]) if lab.startswith("eval/") else keyof(tag)
        flag = []
        if lab.startswith("eval/"):
            ff = root / "frames" / f"{('TEST_' if tag.startswith('TEST_') else '')}{key.split('_x')[0]}.json"
            if not ff.exists() or p.get("frames_sha256") != sha_file(ff):
                flag.append("frames file differs from the one this evaluation used (or is absent)")
            g3 = root / "g3" / ff.name
            if not g3.exists() or p.get("g3_sha256") != sha_file(g3):
                flag.append("G3 file differs from the one this evaluation used (or is absent)")
            elif p.get("scorer") != load_json(g3).get("scorer") and "_x" not in tag:
                flag.append(f"scorer {p.get('scorer')} is not the one JB-G3 chose")
        if not test and lab.startswith(("eval/", "g3/", "frames/")) and "_x" not in tag:
            want = "eager" if key.startswith("gemma") else "sdpa"
            if not (str(p.get("dtype", "")).endswith("bfloat16") and p.get("attn_implementation") == want):
                flag.append(f"the entry fixes BF16 and {want} attention")
            if tag.startswith("TEST_") or p.get("test_mode"):
                flag.append("a TEST_MODE file outside TEST")
        ok &= not flag
        ver = p.get("verified") or {}
        out(f"  {lab}: commit {str(p.get('git_commit'))[:10]}, model {p.get('model')} (key {p.get('model_key')}), revision "
            f"{p.get('revision') or ver.get('revision')}, dtype {p.get('dtype', '-')}, attn {p.get('attn_implementation', '-')}, "
            f"device {p.get('device', '-')}, transformers {p.get('transformers')}, scorer {p.get('scorer', '-')}, frames "
            f"{str(p.get('frames_sha256', '-'))[:12]}, wrapper {p.get('wrapper')}" + "".join(f"  MISMATCH: {x}" for x in flag))
    commits.discard(None)
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else "")
        + ("; UNREADABLE or without provenance: " + ", ".join(bad) if bad else ""))
    return ok and len(commits) <= 1 and not bad


def population(F, pops, out, test):
    out("POPULATION (" + ("TEST: sizes not checked; " if test else "") + "F = 150 fresh cores (seed 20261013), S0 = "
        "make_cores(150, Random(0)); items valid in every arm; hashes pinned)")
    ok = True
    for (k, name), p in sorted(pops.items(), key=lambda x: (order([x[0][0]]), x[0][1])):
        P = p.P
        sk = P.get("skipped_items", [])
        shaok = P.get("population_sha256") == POP_SHA.get(P.get("population"))
        full = test or (len(p.keys) + len({s["key"] for s in sk}) == N_POP[P["population"]] and not p.missing)
        good = shaok and full
        ok &= good
        why = {}
        for s in sk:
            why[s["reason"]] = why.get(s["reason"], 0) + 1
        out(f"  {k} {name}: {len(p.keys)} items valid in every arm run ({', '.join(p.arms)}), {len(sk)} skipped "
            f"({'; '.join(f'{r}: {n}' for r, n in why.items()) or 'none'}), arms not run {P.get('arms_not_run') or 'none'}; "
            f"population sha256 {'OK' if shaok else 'MISMATCH'}; valid + skipped = {N_POP[P['population']]} "
            f"{'OK' if full else 'MISMATCH'}; clusters {len(set(p.clusters))}")
    out(f"  population: {'OK' if ok else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------- models and gates
def build_models(root, F, g0, test, out):
    """Model objects with their model-level gates; the N4 slots (yi9 in the slot of a refused or tokcheck-failed model)."""
    keys = sorted({keyof(Path(l).stem).split("_x")[0].rsplit("_", 1)[0] if l.startswith("eval/") else keyof(Path(l).stem)
                   for l in F if l.startswith(("eval/", "tokcheck/", "g3/"))} - {ln.JB8_KEY})   # mistral24: J-B8 only
    pref = "TEST_" if test else ""
    models, pops = {}, {}
    fetch_failed = (root / "FETCH_FAILED.txt").read_text() if (root / "FETCH_FAILED.txt").exists() else ""
    for k in keys:
        why = []
        if g0 is False or (g0 is None and not test):   # outside TEST a missing pytest log fails J-B-G0
            why.append("J-B-G0 failed" if g0 is False else "J-B-G0: no pytest log")
        tc = F.get(f"tokcheck/{pref}{k}.json")
        if tc is None or "error" in tc:
            why.append("J-B-G0b: no tokenizer check")
        elif not tc.get("pass"):
            why.append(f"J-B-G0b failed: {tc.get('fails', [])[:2]}")
        if not test and f"verified/{k}.json" not in F:
            why.append("J-B-G1: no verified file set")
        g3 = F.get(f"g3/{pref}{k}.json")
        if g3 is None or "error" in g3:
            why.append("J-B-G3: no G3 file")
        P = {}
        for name in ("F", "S0"):
            J = F.get(f"eval/{pref}{k}_{name}.json")
            if J and "results" in J and J["results"]:
                P[name] = Pop(J, k)
                pops[(k, name)] = P[name]
                pr = J["provenance"]
                if g3 and "scorer" in g3 and pr.get("scorer") != g3["scorer"]:
                    why.append(f"J-B-G3: {name} scored with {pr.get('scorer')}, G3 chose {g3['scorer']}")
                ff = root / "frames" / f"{pref}{k}.json"
                if not ff.exists() or pr.get("frames_sha256") != sha_file(ff):
                    why.append(f"{name}: the frames file differs from the one used")
        models[k] = ln.Model(k, P.get("F"), P.get("S0"), ok=not why, why="; ".join(why), test=test)
    slots = fallback_slots(root, models, fetch_failed, out)
    return models, pops, slots


def fallback_slots(root, models, fetch_failed, out):
    """The N4 slots. The pipeline records its one fallback decision in COMMIT.txt ("fallback: yi9 replaces <key>"),
    taken when <key>'s files were refused (FETCH_FAILED.txt) or its tokenizer check failed (that check's file is then
    moved aside as a failed step's output, so the record is the only trace), always before any output of <key> existed.
    Without a record, a key refused in FETCH_FAILED.txt (exit 1) is the replaced one. yi9 fills at most one slot, and
    only when it has results."""
    commit = (root / "COMMIT.txt").read_text() if (root / "COMMIT.txt").exists() else ""
    rec = [k for k in re.findall(rf"fallback: {re.escape(ln.FALLBACK)} replaces (\S+)", commit) if k in ln.N4]
    refused = set(re.findall(r"FETCH REFUSED (\S+)", fetch_failed))
    target = rec[0] if rec else next((k for k in ln.N4 if k in refused and (k not in models or models[k].F is None)), None)
    if len(set(rec)) > 1:
        out(f"  note: COMMIT.txt records more than one fallback ({sorted(set(rec))}); the first, {rec[0]}, is used")
    have_fb = ln.FALLBACK in models and models[ln.FALLBACK].F is not None
    slots = {k: (ln.FALLBACK if (k == target and have_fb) else k) for k in ln.N4}
    if target and have_fb and target in models and models[target].F is not None:
        out(f"  note: {target} has F results although {ln.FALLBACK} replaced it; the slot is {ln.FALLBACK}'s (the recorded decision)")
    if ln.FALLBACK in models and ln.FALLBACK not in slots.values():
        out(f"  note: {ln.FALLBACK} results present but no N4 slot was replaced; {ln.FALLBACK} is reported only")
    return slots


@functools.lru_cache(maxsize=None)
def _ref_file(f):
    return json.load(open(f))["results"]


def g2_reference(key, arm):
    """The committed published-path rows of a P4 model and arm (stage 1 for P1, stage 3b's 2x2 otherwise)."""
    name = ln.REF_NAME[key]
    f = ROOT / ("results/gpu_stage1" if arm == "P1" else "results/gpu_stage3b/format_2x2") / f"{name}_s0.json"
    if not f.exists():
        return None
    return [r for r in _ref_file(str(f)) if r["arm"] == arm and r["view"] == "direct"]


def g2_stats(items):
    """s_ID^L and ID_K^L (means over items) from format_factorial-schema items."""
    K = np.array([ff_ident(r, "K") for r in items])
    Vv = np.array([ff_ident(r, "V") for r in items])
    return float(K.mean() / (K.mean() + Vv.mean())), float(K.mean())


def gate_g2(models, out, test):
    out("  J-B-G2  reproduction (S0, P4; the published-path pass against the committed stage-1 / 3b files): per arm "
        "|ds_ID^L| <= 0.03 and |dr^L| <= 0.03 (a failure is flagged; J-B6 is a same-pass comparison and is still scored)")
    res = {}
    for k in ln.P4:
        m = models.get(k)
        if m is None or m.S0 is None:
            out(f"    {k}: no S0 results")
            continue
        if test:
            out(f"    {k}: TEST: no committed reference for Qwen2.5-0.5B -> NOT EVALUABLE")
            continue
        refA = g2_reference(k, "AFTER")
        if refA is None:
            out(f"    {k}: committed reference files absent -> NOT EVALUABLE")
            continue
        new = {a: [r["plain"] for r in m.S0.recs(a) if "plain" in r] for a in ln.ARMS_S0 if m.S0.has(a)}
        if "AFTER" not in new:   # r is scaled by AFTER: a deadline stop before it leaves nothing to compare
            out(f"    {k}: AFTER not run -> NOT EVALUABLE")
            continue
        sA_new, kA_new = g2_stats(new["AFTER"])
        sA_ref, kA_ref = g2_stats(refA)
        parts, okm = [], True
        for a in ln.ARMS_S0:
            if a not in new or not new[a]:
                parts.append(f"{a} not run")
                okm = False
                continue
            ref = g2_reference(k, a)
            s1, k1 = g2_stats(new[a])
            s0, k0 = g2_stats(ref)
            r1, r0 = k1 / kA_new, k0 / kA_ref
            good = abs(s1 - s0) <= 0.03 and abs(r1 - r0) <= 0.03
            okm &= good
            parts.append(f"{a} s {s1:+.3f} vs {s0:+.3f}, r {r1:+.3f} vs {r0:+.3f} {'ok' if good else 'DIFFERS'}")
        res[k] = okm
        out(f"    {k}: " + "; ".join(parts) + f" -> {st.V(okm)}")
    return res


def print_gates(models, slots, out, g0, test):
    out("GATES (J-B-G0 before any model; J-B-G0b tokenizer check, J-B-G1 files, J-B-G3 trie floor per model; J-B-G4 coverage "
        "and floor per cell; J-B-G5 competence per arm and anchor per model; J-B-G2 reproduction, flagged only)")
    out(f"  N4 slots: {slots}")
    for k in order(models):
        m = models[k]
        out(f"  {k:9s} model-level gates (J-B-G0, G0b, G1, G3, frames hash): {'OK' if m.ok else 'FAILED: ' + m.why}")
    return None


def print_cells(models, root, F, out, test):
    pref = "TEST_" if test else ""
    for k in order(models):
        m = models[k]
        g3 = F.get(f"g3/{pref}{k}.json") or {}
        if "arms" in g3:
            out(f"  {k:9s} J-B-G3 (first {next(iter(g3['arms'].values()))['stat']['n']} F cores): " + "; ".join(
                f"{a} |ds| {v['stat']['abs_ds']:.4f} (<= 0.02), mean|dL| {v['stat']['mean_abs_dL']:.4f} (<= {v['stat']['tol_dL']:.3f})"
                for a, v in g3["arms"].items()) + f" -> {st.V(g3.get('pass'))}; scorer used: {g3.get('scorer')}")
        if m.F is None:
            continue
        v = m.F.view()
        for a in m.F.arms:
            try:
                cov, row = m.cov(a)
                fl = m.floor(a)
                out(f"  {k:9s} {a:9s} J-B-G4 coverage {cov:.3f} (min at {row}; >= 0.8 for E) floor {fl[0]:.4f} (<= {fl[1]:.4f}) "
                    f"-> {'E' if cov >= ln.COV_MIN and fl[2] else 'beta counterpart' if cov < ln.COV_MIN else 'NOT EVALUABLE under E'}; "
                    f"J-B-G5 acc_B {m.acc(a):.3f} (>= 0.8), acc_S {float(v.acc(a, 'S').mean()):.3f}, other(B) {float(v.other(a, 'B').mean()):.3f}")
            except Exception as ex:  # noqa: BLE001  (e.g. AFTER not run: no floor scale)
                out(f"  {k:9s} {a:9s} J-B-G4 / J-B-G5 not computable: {type(ex).__name__}: {ex}")
        for sg in ("E", "beta"):
            try:
                an = m.anchor(sg, 0.95)
                out(f"  {k:9s} J-B-G5 anchor under {sg}: " + ("holds" if not an else an[0].ne))
            except Exception as ex:  # noqa: BLE001
                out(f"  {k:9s} J-B-G5 anchor under {sg}: not computable ({ex})")


# --------------------------------------------------------------------------- JB8
def jb8_rows(results, arm, score):
    rows = {}
    LOCS = ["box", "basket", "shelf", "drawer", "cabinet", "closet"]
    for r in results:
        if r["arm"] != arm:
            continue
        c = r["core"]
        if len({c["base"], c["source"], c["target"]}) < 3:
            continue
        alpha = LOCS if arm != "LETTER" else LOCS
        iS, iT = alpha.index(c["source"]), alpha.index(c["target"])
        vec = (lambda v: v["cand"]) if score == "L" else (lambda v: v[score])
        m = {k: vec(v)[iT] - vec(v)[iS] for k, v in r["runs"].items()}
        avg = lambda pre: float(np.mean([m[f"{pre}_{s}"] for s in (101, 102, 103)]))  # noqa: E731
        rows[c["id"]] = {"S": m["S"], "T": m["T"], "M": avg("m3"), "P": avg("pca"), "add": avg("addition"),
                         "addv": avg("addition_v"), "mass": r["runs"]["B"]["mass"]["E"] if "mass" in r["runs"]["B"] else float("nan")}
    return rows


def line_jb8(F, root, out_, test, g0=None):
    """J-B8 at mistral24. Its model-level gates as in build_models: J-B-G0 (outside TEST a missing pytest log fails it),
    J-B-G0b (tokcheck), J-B-G1 (verified file set, waived in TEST) and the frames file's sha256 recorded by the jb8 file;
    any failing makes the line NOT EVALUABLE."""
    J = next((v for k, v in F.items() if k.startswith("jb8/") and "results" in v), None)
    if J is None:
        sk = (root / "SKIPPED.txt").read_text() if (root / "SKIPPED.txt").exists() else ""
        why = next((l for l in sk.splitlines() if "jb8" in l), "no jb8 results")
        return st.Res(None, "", why=f"not run: {why.strip()}"), {}
    if J["provenance"].get("score") != "E":
        return st.Res(None, "", why="the jb8 file was not scored with --score E"), {}
    pref, why = "TEST_" if test else "", []
    if g0 is False or (g0 is None and not test):
        why.append("J-B-G0 failed" if g0 is False else "J-B-G0: no pytest log")
    tc = F.get(f"tokcheck/{pref}{ln.JB8_KEY}.json")
    if tc is None or "error" in tc:
        why.append("J-B-G0b: no tokenizer check")
    elif not tc.get("pass"):
        why.append(f"J-B-G0b failed: {tc.get('fails', [])[:2]}")
    if not test and f"verified/{ln.JB8_KEY}.json" not in F:
        why.append("J-B-G1: no verified file set")
    ff = root / "frames" / f"{pref}{ln.JB8_KEY}.json"
    if not ff.exists() or J["provenance"].get("frames_sha256") != sha_file(ff):
        why.append("the frames file differs from the one jb8 used (or is absent)")
    if why:
        return st.Res(None, "", why="; ".join(why)), {}
    cs, rep = [], {}
    for arm in ("P1", "NONE", "BEFORE", "POST", "LETTER"):
        RL, RE = jb8_rows(J["results"], arm, "L"), jb8_rows(J["results"], arm, "E")
        ids = sorted(set(RL) & set(RE))
        if not ids:
            if arm != "LETTER":
                cs.append(st.Comp(arm, False, ne="format not run"))
            continue
        b = st.boot_of(None, len(ids))
        A = lambda R, f: np.array([f(R[i]) for i in ids])  # noqa: E731
        q = {"phi": (lambda x: x["M"] - x["P"], lambda x: x["T"] - x["S"]),
             "psi_K": (lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"]),
             "psi_V": (lambda x: x["addv"] - x["P"], lambda x: x["M"] - x["P"])}
        mass = float(np.nanmean(A(RE, lambda x: x["mass"])))
        rep[arm] = {"mass_E": mass}
        for name, (num, den) in q.items():
            e = st.est(b, lambda a1, b1, a2, b2: a1 / b1 - a2 / b2, A(RE, num), A(RE, den), A(RL, num), A(RL, den))
            rep[arm][name] = (float(A(RL, num).mean() / A(RL, den).mean()), float(A(RE, num).mean() / A(RE, den).mean()))
            c = st.inside(e, -0.05, 0.05, 0.95, f"{arm}: {name}^E - {name}^L")
            if arm == "LETTER":
                rep[arm][name + "_diff"] = str(c)
            else:
                cs.append(c)
        if arm != "LETTER":
            cs.append(st.Comp(f"{arm}: M^E(natural B) {mass:.3f} >= 0.80 (point)", mass >= 0.8))
    return st.decide(cs, f"{len(ids)} cores"), rep


# --------------------------------------------------------------------------- predictions
def models_for(code, models, slots):
    sset = ln.LINES[code][3]
    if sset in ("P4f", "S0"):
        return {k: models.get(k) for k in ln.P4}
    if sset == "N4":
        return {f"{k}" + (f"->{slots[k]}" if slots[k] != k else ""): models.get(slots[k]) for k in ln.N4}
    if sset == "SMALL":
        return {ln.SMALL: models.get(ln.SMALL)}
    return {}


def score_lines(models, slots, F, root, out, res, test, g0=None):
    out("PREDICTIONS (code [class, kind (A account, V measurement validity), recorded prior P(met)] title -> verdict; then per "
        "model: verdict, numbers and bounds; P4f: 95 % intervals, every evaluable model (NOT MET if one fails, else >= 3 "
        "evaluable); N4: 98.75 %, 3 of 4)")
    extra = {}
    for code in ln.ORDER:
        cls, kind, prior, sset, title = ln.LINES[code]
        per = {}
        if code == "J-B6c":
            r = ln.line_b6c(models)
            per = {"P4": r}
            comb = r.ok
        elif code == "J-B8":
            r, rep = line_jb8(F, root, out, test, g0)
            extra["jb8"] = rep
            per = {ln.JB8_KEY: r}
            comb = r.ok
        elif code == "J-B7":
            p4 = {k: _safe(ln.line_b7, models.get(k), None) for k in ln.P4}
            n4 = {k + (f"->{slots[k]}" if slots[k] != k else ""): _safe(ln.line_b7, models.get(slots[k]), None) for k in ln.N4}
            per = {**{"P4 " + k: v for k, v in p4.items()}, **{"N4 " + k: v for k, v in n4.items()}}
            a, b = st.comb_every({k: v.ok for k, v in p4.items()}), st.comb_k_of({k: v.ok for k, v in n4.items()})
            comb = False if (a is False or b is False) else None if (a is None or b is None) else True
        else:
            lv = ln.LV_K4 if sset == "N4" else ln.LV_EVERY
            fn = {"J-B6a": ln.line_b6a, "J-B6b": ln.line_b6b}.get(code) or ln.LINE_FNS[ln.base_code(code)]
            per = {k: _safe(fn, m, lv) for k, m in models_for(code, models, slots).items()}
            okd = {k: r.ok for k, r in per.items()}
            comb = st.comb_k_of(okd) if sset == "N4" else st.comb_single(okd) if sset == "SMALL" else st.comb_every(okd)
        res[code] = (comb, per)
        n_met = sum(r.ok is True for r in per.values())
        n_ev = sum(r.ok is not None for r in per.values())
        out(f"  {code:13s} [{cls}, {kind}, prior {prior:.2f}] {title} -> {st.V(comb)} ({n_met} of {n_ev} evaluable met; "
            f"{len(per)} slots)")
        for k, r in per.items():
            out(f"      {k}: {st.V(r.ok)}  {r.txt}" + (f"  [not evaluable: {r.why}]" if r.ok is None and r.why else ""))
            for c in r.comps:
                out(f"          {c}")
    return extra


def _safe(fn, m, lv):
    if m is None:
        return st.Res(None, "", why="no results for this model")
    try:
        return fn(m, lv)
    except Exception as ex:  # noqa: BLE001  (a missing record makes this model's line not evaluable)
        return st.Res(None, "", why=f"not computable: {type(ex).__name__}: {ex}")


# --------------------------------------------------------------------------- summary
def summary(res, out):
    out("\nSUMMARY (G4: per class and kind; expected = the sum of the recorded priors of the lines with a verdict; Brier = "
        "mean (prior - 1[MET])^2 over the same lines; NOT EVALUABLE lines are left out of both and listed)")
    for kind, kname in (("A", "account lines"), ("V", "measurement-validity lines")):
        for cls in ("L", "M", "R"):
            codes = [c for c in res if ln.LINES[c][0] == cls and ln.LINES[c][1] == kind]
            if not codes:
                continue
            ev = [c for c in codes if res[c][0] is not None]
            met = sum(res[c][0] is True for c in ev)
            exp = sum(ln.LINES[c][2] for c in ev)
            brier = float(np.mean([(ln.LINES[c][2] - (res[c][0] is True)) ** 2 for c in ev])) if ev else float("nan")
            ne = [c for c in codes if res[c][0] is None]
            out(f"  {kname}, class {cls}: {len(codes)} lines: {met} MET, {sum(res[c][0] is False for c in ev)} NOT MET, "
                f"0 MET IN PART, {len(ne)} NOT EVALUABLE; observed {met} against expected {exp:.2f}; Brier {brier:.3f}"
                + (f"; not evaluable: {', '.join(ne)}" if ne else ""))
    R = [c for c in res if ln.LINES[c][0] == "R" and ln.LINES[c][1] == "A" and res[c][0] is not None]
    out(f"  met rate among R account lines with a verdict: {sum(res[c][0] is True for c in R)} of {len(R)} "
        f"(expected {sum(ln.LINES[c][2] for c in R):.2f})")
    holm_sensitivity(res, out)


def holm_family(res):
    """The Holm family (common part): the one-sided interval components of this part's R-class account lines, in every
    model where the line is evaluable; [(code, model, Comp, component dict of analysis/stage8_holm.py)]."""
    fam = []
    for code, (_, per) in res.items():
        if ln.LINES[code][0] != "R" or ln.LINES[code][1] != "A":
            continue
        for k, r in per.items():
            if r.ok is None:
                continue
            for c in r.comps:
                if c.ne is None and c.holm is not None:
                    h = c.holm
                    fam.append((code, k, c, {"line": code, "name": f"{k}: H1 {h['name']} {h['direction']} {h['bound']:g}",
                                             "est": h["est"], "se": h["se"], "bound": h["bound"], "direction": h["direction"]}))
    return fam


def holm_sensitivity(res, out):
    """Holm over the family above with the shared helper; per line, the components whose decision changes and the
    verdict the line would get with Holm's decisions in place of the interval decisions (reported, no verdict uses it)."""
    fam = holm_family(res)
    try:
        from stage8_holm import holm as holm_shared   # analysis/stage8_holm.py, identical in every part (D2)
        got = holm_shared([d for *_, d in fam]) if fam else []
    except Exception as ex:  # noqa: BLE001  (reported; no verdict depends on it)
        out(f"  Holm sensitivity NOT COMPUTED: analysis/stage8_holm.py failed ({type(ex).__name__}: {ex})")
        return None
    by = {(d.get("line"), d.get("name")): d for d in got}
    rej = {}
    for i, (code, k, c, d) in enumerate(fam):
        g = by.get((code, d["name"]), got[i] if i < len(got) else {})
        rej[id(c)] = bool(g.get("reject", g.get("rejected", False)))
    out(f"  Holm sensitivity (analysis/stage8_holm.py: one-sided p from the bootstrap SE, step-down at familywise 0.025 over "
        f"the {len(fam)} interval components of the R-class account lines, in the models where the line is evaluable; "
        f"reported, no verdict uses it):")
    changed = {}
    for code, (v, per) in res.items():
        if ln.LINES[code][0] != "R" or ln.LINES[code][1] != "A":
            continue
        flips = [f"{k}: {c.label.split(' rejected at')[0]} {'rejected -> not rejected' if c.passed else 'not rejected -> rejected'}"
                 for cc, k, c, _ in fam if cc == code and rej[id(c)] != c.passed]
        new = {k: (None if r.ok is None else all(rej.get(id(c), c.passed) for c in r.comps)) for k, r in per.items()}
        sset = ln.LINES[code][3]
        nv = st.comb_k_of(new) if sset == "N4" else st.comb_single(new) if sset == "SMALL" else st.comb_every(new)
        changed[code] = (bool(flips), nv != v)
        out(f"    {code}: {sum(cc == code for cc, *_ in fam)} components; decisions changed by Holm: "
            + ("none" if not flips else "; ".join(flips)) + f"; verdict {st.V(v)}" + (f" -> {st.V(nv)} under Holm" if nv != v else
                                                                                     " (unchanged under Holm)"))
    return changed


def per_family(res, out):
    out("  Per-family verdicts of the N4 lines (B-12; the entry's pre-written sentence for the number of new families not meeting):")
    for code, (v, per) in res.items():
        if ln.LINES[code][3] != "N4":
            continue
        fail = [k for k, r in per.items() if r.ok is not True]
        nev = [k for k, r in per.items() if r.ok is None]
        sent = {0: "none", 1: "exactly one", 2: "exactly two"}.get(len(fail), "three or more")
        out(f"    {code}: " + ", ".join(f"{ln.FAMILY.get(k.split('->')[-1], k)} {st.V(r.ok)}" for k, r in per.items())
            + f"; not meeting: {sent} ({', '.join(fail) or '-'})" + (f"; not evaluable: {', '.join(nev)}" if nev else ""))


# --------------------------------------------------------------------------- reported and exploratory
def reported(models, slots, out, F, test, extra):
    out("\nREPORTED (no verdicts)")
    out("  Numbers per model on F (E unless a cell's coverage is below 0.8; 95 % cluster intervals): r(f), s_ID(f), beta_K(f), "
        "beta_V(f) -- the behavioural crossover figure")
    for k in order(models):
        m = models[k]
        if m.F is None:
            continue
        v = m.F.view()
        for a in m.F.arms:
            try:
                sg = "E" if m.cov(a)[0] >= ln.COV_MIN and m.cov("AFTER")[0] >= ln.COV_MIN else "beta"
                bk, bv = st.est(v.boot, st.mean, v.k(a, "beta")), st.est(v.boot, st.mean, v.v(a, "beta"))
                out(f"    {k:9s} {a:9s} [{sg}] r {ln.r_est(v, a, sg).txt(0.95)}  s_ID {ln.s_est(v, a, sg).txt(0.95)}  "
                    f"beta_K {bk.txt(0.95)}  beta_V {bv.txt(0.95)}  ID_K^E {v.k(a, 'E').mean():+.2f}  ID_V^E {v.v(a, 'E').mean():+.2f}  "
                    f"flip {ln.flip_rate(v, a):.3f}")
            except Exception as ex:  # noqa: BLE001
                out(f"    {k:9s} {a:9s} not computable: {ex}")
    out("  S0 relative changes of ID_K and ID_V, E against L (tolerance 0.15; J-B6, reported), and the list cells:")
    for k in ln.P4:
        m = models.get(k)
        if m is None or m.S0 is None:
            continue
        v = m.S0.view()
        for a in m.S0.arms:
            rk = v.k(a, "E").mean() / v.k(a, "L").mean() - 1
            rv = v.v(a, "E").mean() / v.v(a, "L").mean() - 1
            out(f"    {k:9s} {a:7s} M^L(B) {ln.ml_mass(v, a):.3f}  ID_K {v.k(a, 'L').mean():+.2f} -> {v.k(a, 'E').mean():+.2f} "
                f"({rk:+.1%}, {'within' if abs(rk) <= 0.15 else 'beyond'} 15%)  ID_V {v.v(a, 'L').mean():+.2f} -> "
                f"{v.v(a, 'E').mean():+.2f} ({rv:+.1%}, {'within' if abs(rv) <= 0.15 else 'beyond'} 15%)"
                + ("  [list cell]" if a in ("P1", "AFTER", "BEFORE") else ""))
    if extra.get("jb8"):
        out("  J-B8 detail (L -> E): " + "; ".join(f"{a}: " + ", ".join(f"{n} {x[0]:+.3f} -> {x[1]:+.3f}" for n, x in d.items()
                                                                          if isinstance(x, tuple)) + f", M^E {d['mass_E']:.3f}"
                                                  for a, d in extra["jb8"].items()))


def exploratory(models, F, out, test):
    out("\n######## EXPLORATORY (not scored)")
    for k in order(models):
        m = models[k]
        for name, p in (("F", m.F), ("S0", m.S0)):
            if p is None:
                continue
            v = p.view()
            out(f"  [{k} {name}] Sigma and L versions; competent-only; onsets; uncovered first-token mass of clean B")
            for a in p.arms:
                try:
                    ss = {sg: ln.s_est(v, a, sg).pt for sg in ("L", "sigma", "E")}
                    rr = {sg: ln.r_est(v, a, sg).pt for sg in ("L", "sigma", "E")}
                    keep = v.comp(a) > 0
                    vc = p.view(keep)
                    sc = ln.s_est(vc, a, "E").pt if vc.n >= 2 else float("nan")
                    nL = p.recs(a)[0]["n_layers"]
                    l0s = sorted({int(r.split("@")[1]) for r in p.recs(a)[0]["rows"] if r.startswith("K_S@")})
                    on = []
                    for l0 in l0s:
                        dk, dv = v.d(a, f"K_S@{l0}", "E").mean(), v.d(a, f"V_S@{l0}", "E").mean()
                        on.append(f"l0={l0}/{nL} s_K {dk / (dk + dv):+.3f}")
                    mass = {sg: float(v.mass(a, "B", sg).mean()) for sg in ("L", "sigma", "E")}
                    out(f"    {a:9s} s_ID L/Sigma/E {ss['L']:+.3f}/{ss['sigma']:+.3f}/{ss['E']:+.3f}  r {rr['L']:+.3f}/{rr['sigma']:+.3f}/"
                        f"{rr['E']:+.3f}  mass(B) {mass['L']:.3f}/{mass['sigma']:.3f}/{mass['E']:.3f}  gap(B) {v.gap(a, 'B').mean():.3f}  "
                        f"competent n={vc.n} s_ID^E {sc:+.3f}  {'; '.join(on)}")
                except Exception as ex:  # noqa: BLE001
                    out(f"    {a:9s} not computable: {ex}")
            if name == "F" and p.has("POST") and p.has("AFTER"):
                try:
                    for lab, sel in [(f"lexicon {x}", lambda r, x=x: r["lex"] == x) for x in (1, 2)] + \
                                    [(f"sentence {j}", lambda r, j=j: r["sent"] == j) for j in range(8)]:
                        keep = np.array([sel(r) for r in p.recs("POST")])
                        if keep.sum() < 2:
                            continue
                        vs = p.view(keep)
                        out(f"    per level {lab:11s} (n={vs.n}): r^E(POST) {ln.r_est(vs, 'POST', 'E').pt:+.3f}  s_ID^E(POST) "
                            f"{ln.s_est(vs, 'POST', 'E').pt:+.3f}  s_ID^E(AFTER) {ln.s_est(vs, 'AFTER', 'E').pt:+.3f}")
                except Exception as ex:  # noqa: BLE001
                    out(f"    per-level estimates not computable: {ex}")
            if name == "F" and p.has("BEFORE") and p.has("NONE"):
                out(f"    LIST-BEFORE sign: ID_K^E(BEFORE) - ID_K^E(NONE) {v.k('BEFORE', 'E').mean() - v.k('NONE', 'E').mean():+.3f}; "
                    f"b_ID(POST) {ln.s_est(v, 'POST', 'beta').pt:+.3f}" if p.has("POST") else "")
    for lab, J in sorted(F.items()):
        if lab.startswith("frames/") and "counts" in J:
            out(f"  frame census {lab}: admitted {J.get('frames')}; " + "; ".join(
                f"{a}: " + ", ".join(f"{f!r} {n}" for f, n in sorted(c.items(), key=lambda x: -x[1])[:4]) for a, c in J["counts"].items()))
    for lab, J in sorted(F.items()):
        if lab.startswith("eval/") and "_x" in lab and "results" in J:
            try:
                p = Pop(J, lab)
                v = p.view()
                out(f"  {lab} (exploratory X1/X2): " + "; ".join(f"{a} s_ID^E {ln.s_est(v, a, 'E').pt:+.3f} s_ID^L "
                                                                    f"{ln.s_est(v, a, 'L').pt:+.3f}" for a in p.arms))
            except Exception as ex:  # noqa: BLE001
                out(f"  {lab}: not computable: {ex}")
    for k in ln.P4:
        m = models.get(k)
        if m is None or m.S0 is None:
            continue
        try:
            d = [abs(r["ff"]["m"][row]["lp"][t] - r["plain"]["m"][row]["lp"][t]) for a in m.S0.arms for r in m.S0.recs(a)
                 if "plain" in r for row in r["ff"]["m"] for t in ("S", "B", "X")]
            out(f"  [{k} S0] trie L against the published path (all items, rows, S/B/X): mean |dL| {np.mean(d):.4f}, max {np.max(d):.4f}")
        except Exception as ex:  # noqa: BLE001
            out(f"  [{k} S0] trie vs published path not computable: {ex}")


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", "--root", dest="results", default="results/gpu_stage8b")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: size floors and J-B-G1 waived; verdicts are plumbing checks")
    ap.add_argument("--out", default=None, help="default {results}/STAGE8B_SCORE.txt")
    a = ap.parse_args(argv)
    root = Path(a.results)
    F = load(root)
    tags = [Path(k).stem for k in F if k.startswith(("eval/", "g3/"))]
    test = a.test or (bool(tags) and all(t.startswith("TEST_") for t in tags))
    lines, errors = [], []
    out = lines.append
    out(f"Stage 8 part B scoring, preregistration P-2026-10-10-J part B; results {root}")
    if test:
        out("TEST MODE: the size floors and J-B-G1 are waived, J-B-G2 has no reference, and the verdict lines below are "
            "plumbing checks, not results")
    out("")
    prov_ok = provenance(root, F, out, test)
    out("")
    gl = []
    g0 = gate_g0(root, gl.append)
    models, pops, slots = build_models(root, F, g0, test, gl.append)
    pop_ok = population(F, pops, out, test)
    out("")
    print_gates(models, slots, out, g0, test)
    for x in gl:
        out(x)
    if test and g0 is None:
        out("  (TEST: no pytest log; J-B-G0 treated as passed for the plumbing check)")
    elif g0 is None:
        out("  (no pytest log: outside TEST J-B-G0 fails, so every line is NOT EVALUABLE)")
    for lab, J in sorted(F.items()):
        if lab.startswith("tokcheck/"):
            out(f"  {keyof(Path(lab).stem):9s} J-B-G0b tokenizer check: {st.V(J.get('pass'))}"
                + (f" ({J.get('fails')})" if not J.get("pass") else f"; ROOM {J.get('room_tokens')} tokens, sentences "
                   f"{ {k: v for k, v in (J.get('sentence_tokens') or {}).items()} }, trie nodes {J.get('trie_nodes')}"))
        if lab.startswith("verified/"):
            out(f"  {Path(lab).stem:9s} J-B-G1 files verified: {J.get('repo')} at {J.get('revision')}, {len(J.get('files', {}))} files")
    try:
        print_cells(models, root, F, out, test)
        gate_g2(models, out, test)
    except Exception:  # noqa: BLE001
        errors.append("gates")
        out("  GATE REPORT ERROR\n" + traceback.format_exc())
    out("")
    res, extra = {}, {}
    try:
        extra = score_lines(models, slots, F, root, out, res, test, g0)
    except Exception:  # noqa: BLE001
        errors.append("lines")
        out("  SCORER ERROR; the lines are NOT EVALUABLE\n" + traceback.format_exc())
    try:
        reported(models, slots, out, F, test, extra)
        if res:
            per_family(res, out)
    except Exception:  # noqa: BLE001
        errors.append("reported")
        out("  REPORTED ERROR\n" + traceback.format_exc())
    if res:
        summary(res, out)
        n = lambda v: sum(x[0] is v for x in res.values())  # noqa: E731
        out(f"  overall: {n(True)} MET, {n(False)} NOT MET, {n(None)} NOT EVALUABLE of {len(res)} lines; provenance "
            f"{'OK' if prov_ok else 'MISMATCH'}; population {'OK' if pop_ok else 'MISMATCH'}" + (f"; SCORER ERROR in {errors}" if errors else ""))
    try:
        exploratory(models, F, out, test)
    except Exception:  # noqa: BLE001
        errors.append("exploratory")
        out("  EXPLORATORY ERROR\n" + traceback.format_exc())
    try:
        tb.write_tables(root, models, slots)
        out(f"\n  paper tables written: {', '.join(tb.NAMES)} in {root}")
    except Exception:  # noqa: BLE001
        errors.append("tables")
        out("  TABLES ERROR\n" + traceback.format_exc())
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE8B_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return 1 if errors else 2 if not test and not (prov_ok and pop_ok) else 0


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
