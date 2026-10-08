"""Score preregistration P-2026-10-08-I (docs/PREREGISTRATION.md, GPU stage 7): blocking the reader heads while applying
the released remap of our predecessor, Anonymous (2026), at Mistral-Small-24B. Gates I-G0 to I-G4, predictions I1-I7,
the reported lines and the exploratory report, as worded in the entry (part scorers in analysis/stage7_parts).

Inputs under --root (default results/gpu_stage7), as written by scripts/gpu_stage7.sh: preflight.json, heads/rank.json,
heads/gate.json, heads/remaprank.json (exploratory, optional), link/<tag>.json (experiments/stage7_link.py),
frames/<tag>.json (experiments/paper1_frames.py, the family run), log_pytest_stage7.txt (Gate I-G0: its last pytest run),
COMMIT.txt, ENV.txt, REVISIONS.txt; and --ref, the committed stage-3b frames file (results/gpu_stage3b/paper1_frames_v/
mistral.json), the reference of I-G1 (a). The scorer needs no copy of the predecessor's release.
Output, also written to --out (default {root}/STAGE7_SCORE.txt), in this order: PROVENANCE (the pipeline's files; per
results file the commit, model, revision, dtype, attention, device, versions; the sets and MU hashes; the release, bases
and stories-file hashes against the pins of experiments/stage7_link.py; MISMATCH when the files hold more than one
commit, another revision, a hash differs, or outside a TEST_ tag a dtype/attention other than the entry's), POPULATION
(96 E cores per format and condition in one order, 60 R cores = make_cores(60, Random(0)), R and E disjoint, no skipped
items, 600 family results), GATES, VERDICTS (one line per prediction I1-I7 with its per-format lines), REPORTED (not
scored), SUMMARY, then the EXPLORATORY report.
Evaluability (the entry's): I-G0 or I-G1 (d) failing -> I1-I7 NOT EVALUABLE; I1 in f needs I-G1 (a, b) and I-G3 (1) in
f; I2, I3 also I-G2 (a-d) and I-G3 (2); I4-I7 need I-G1 (a, b), I-G2 and I-G3 (1) in f; I4 and I5 also NO-MENTION's
I-G1 (a, b) and I-G3 (1), and I4 under OPTIONS-AFTER kappa(OPTIONS-AFTER) and kappa(NO-MENTION) defined by the kappa
rule. In an evaluable format an undefined kappa after blocking, or more than 5 % of resamples dropped, is NOT MET with
the reason printed. A prediction over OPTIONS-AFTER and LETTERS-AFTER is MET if met in both, NOT MET if not met in an
evaluable format, else NOT EVALUABLE; I5 and I6 are scored under OPTIONS-AFTER alone.
--tag TEST_<model> (TEST_MODE): sizes are not checked, I-G1 (a) has no reference (NOT EVALUABLE) and the verdict lines
are plumbing checks, not results.
Exit status 1 if a part raised (traceback in the report, its lines NOT EVALUABLE); 2 if, outside a TEST_ tag, the
provenance or population check reports MISMATCH, or link results exist without a passing Gate I-G0 (the file is still
written).
"""
import argparse
import hashlib
import json
import random
import re
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage7_parts import heads as hd  # noqa: E402
from stage7_parts import link as lk  # noqa: E402
from stage7_parts.common import NAMES, V, comb, est, f3  # noqa: E402

from ckeys.story import make_cores  # noqa: E402

REV = "9527884be6e5616bdd54de542f9ae13384489724"
RELEASE_SHA = "2dea297d508e51f07b927e9eb0571f40e99d25d996ccd7ad3342cb46942d0416"
N_E, N_R, N_FRAMES = 96, 60, 600
IG0_TESTS = 16   # the test functions of tests/test_stage7_link.py
TITLE = {"I1": "the remap's key is read at the option words", "I2": "through the natural readers",
         "I3": "specificity of the route (random and active sets)", "I4": "the key share falls toward NO-MENTION under A(H*)",
         "I5": "the value channel carries more of the remap", "I6": "the behaviour is kept", "I7": "random sets change nothing"}
STORY_FIELDS = ("agent", "other", "object", "distractor", "initial", "distractor_location", "base", "source")
CONFIRM = {"0", "A:H", "A:rand0", "A:rand1", "A:rand2", "A:active", "A+:H", "N:H", "N:null"}


def pins():
    """The pins of experiments/stage7_link.py (imported lazily: it imports torch and transformers)."""
    from experiments.stage7_link import BASES_SHA, NATIVE_SHA
    return BASES_SHA, NATIVE_SHA


def load_json(f):
    try:
        return json.load(open(f))
    except Exception as ex:  # a truncated file is reported, not fatal
        return {"error": repr(ex)}


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- provenance
def provenance(root, F, tag, out, test):
    out("PROVENANCE (the pipeline's COMMIT.txt, ENV.txt, REVISIONS.txt, every line; every results file)")
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt"):
        f = root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text().splitlines() if l.strip()) if f.exists() else " absent"))
    ok, commits, bad = True, set(), []
    want_attn = {"heads/rank.json": "eager", "heads/gate.json": "sdpa", f"link/{tag}.json": "sdpa", "heads/remaprank.json": "eager"}
    for lab, J in F.items():
        if lab == "frames" or J is None:
            continue
        if "error" in J or "provenance" not in J:
            out(f"  {lab}: " + (f"UNREADABLE {J['error']}" if "error" in J else "NO PROVENANCE") + "  MISMATCH")
            bad.append(lab)
            continue
        p = J["provenance"]
        if p.get("skipped"):
            out(f"  {lab}: skipped ({p['skipped']})")
            continue
        commits.add(p.get("git_commit"))
        flag = []
        if not test:
            if p.get("revision") != REV:
                flag.append(f"revision {p.get('revision')} != {REV}")
            if lab in want_attn and not (str(p.get("dtype", "")).endswith("bfloat16") and p.get("attn_implementation") == want_attn[lab]):
                flag.append(f"the entry fixes BF16 and {want_attn[lab]} attention")
        ok &= not flag
        out(f"  {lab}: commit {str(p.get('git_commit'))[:10]}, model {p.get('model')} rev {p.get('revision')}, dtype {p.get('dtype', '-')}, "
            f"attn {p.get('attn_implementation', '-')}, device {p.get('device', '-')}, transformers {p.get('transformers')}, torch {p.get('torch')}, "
            f"test_mode {p.get('test_mode')}" + ("".join(f"  MISMATCH: {x}" for x in flag)))
    R = F.get("heads/rank.json")
    if R and "sets" in R:
        sh = R.get("sets_sha256")
        good = canon(R["sets"]) == sh
        same = all((F[k] or {}).get("provenance", {}).get("sets_sha256", sh) == sh for k in ("heads/gate.json", f"link/{tag}.json", "heads/remaprank.json") if F.get(k))
        mu = all((F[k] or {}).get("provenance", {}).get("mu_sha256", R.get("mu_sha256")) == R.get("mu_sha256") for k in ("heads/gate.json", f"link/{tag}.json") if F.get(k))
        ok &= good and same and mu
        out(f"  head sets sha256 {str(sh)[:16]}: " + ("recomputed OK" if good else "MISMATCH (sets changed after hashing)")
            + "; every later file records it: " + ("OK" if same else "MISMATCH") + f"; MU sha256 {str(R.get('mu_sha256'))[:16]}: " + ("OK" if mu else "MISMATCH"))
    BASES_SHA, NATIVE_SHA = pins()
    for lab in ("preflight.json", f"link/{tag}.json"):
        J = F.get(lab)
        rel = (J or {}).get("release") or (J or {}).get("provenance", {}).get("release")
        if not rel:
            continue
        good = rel.get("release_sha256") == RELEASE_SHA and rel.get("bases_sha256") == BASES_SHA and rel.get("native_sha256") == NATIVE_SHA
        ok &= good
        out(f"  the predecessor's release ({lab}): RELEASE.json {str(rel.get('release_sha256'))[:12]}, bases and stories file "
            + ("equal the pins (stage-4 provenance; release manifest)" if good else "MISMATCH against the pins") + f"; bases used: {rel.get('bases_used')}")
    fr = F.get("frames")
    if fr is not None:
        p = fr.get("provenance", {}) if "error" not in fr else {}
        fl = []
        if not test:
            if p.get("revision") != REV:
                fl.append(f"revision {p.get('revision')}")
            if {k: v.get("sha256") for k, v in p.get("bases", {}).items() if "/" not in k} != BASES_SHA:
                fl.append("bases sha256")
        ok &= not fl and "error" not in fr
        out(f"  frames/{tag}.json: {p.get('repo')} rev {p.get('revision')}, device {p.get('device')} x{p.get('n_gpus')}, transformers {p.get('transformers')}"
            + "".join(f"  MISMATCH: {x}" for x in fl) + ("  UNREADABLE MISMATCH" if "error" in fr else ""))
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else "")
        + ("; UNREADABLE or without provenance: " + ", ".join(bad) if bad else ""))
    return ok and len(commits) <= 1 and not bad


# --------------------------------------------------------------------------- population
def population(F, tag, L, out, test):
    out("POPULATION (" + ("TEST: sizes not checked; " if test else "") + "E: 96 native cores per format and condition, one order; R: make_cores(60, Random(0)))")
    ok = True
    R = F.get("heads/rank.json")
    E_ids = None
    if L is not None:
        E_ids = [r["core"]["id"] for r in L.R("P1")] if L.R("P1") else None
        sk = L.P.get("skipped_items", {})
        line = []
        for arm in lk.ARMS:
            n = len(L.R(arm))
            miss = [] if not n else sorted(c for c in (CONFIRM if arm in ("P1", "LETTER", "POST") else {"0"} | ({"N:H_P1"} if arm == "BEFORE" else set()))
                                        if not L.has(arm, c))
            good = (test or n == N_E) and not miss and sk.get(arm, 0) == 0
            ok &= good
            line.append(f"{arm} n={n}" + ("" if test or n == N_E else " MISMATCH") + (f" missing {miss} MISMATCH" if miss else "") + (f" skipped {sk.get(arm)} MISMATCH" if sk.get(arm) else ""))
        ok &= L.same_cores
        out(f"  link: " + "; ".join(line) + f"; same cores in the same order in every format: {'OK' if L.same_cores else 'MISMATCH'}")
    G = F.get("heads/gate.json")
    if G and "arms" in G:
        ns = {a: len(v.get("eval", [])) for a, v in G["arms"].items()} | {"NONE clamp": len(G.get("none_clamp", {}).get("eval", []))}
        order = all([e["core"]["id"] for e in v.get("eval", [])] == E_ids for v in list(G["arms"].values()) + [G.get("none_clamp", {})]) if E_ids else True
        good = (test or all(n == N_E for n in ns.values())) and order
        ok &= good
        out(f"  gate: " + ", ".join(f"{a} n={n}" for a, n in ns.items()) + f"; cores as the link run: {'OK' if order else 'MISMATCH'}" + ("" if good else "  MISMATCH"))
    if R and "arms" in R:
        line = []
        for arm, A in R["arms"].items():
            cores = [r["core"] for r in A["rank"]]
            same = cores == make_cores(len(cores), random.Random(0))
            n_ok = test or len(cores) == N_R
            ok &= same and n_ok
            line.append(f"{arm} n={len(cores)}" + ("" if n_ok else " MISMATCH") + f" make_cores(n, Random(0)) {'OK' if same else 'MISMATCH'}")
        key = lambda c: tuple(c[f] for f in STORY_FIELDS)  # noqa: E731
        Ecores = [r["core"] for r in L.R("P1")] if L is not None else []
        ov = len({key(r["core"]) for A in R["arms"].values() for r in A["rank"]} & {key(c) for c in Ecores})
        ok &= ov == 0
        out(f"  rank: " + "; ".join(line) + f"; R and E share {ov} stories" + ("" if ov == 0 else "  MISMATCH"))
    fr = F.get("frames")
    if fr is not None and "results" in fr:
        n = len(fr["results"])
        good = test or n == N_FRAMES
        ok &= good
        out(f"  frames: {n} results" + ("" if good else f"  MISMATCH ({N_FRAMES})"))
    out(f"  population: {'OK' if ok else 'MISMATCH (a file does not hold the preregistered n, cores or conditions)'}")
    return ok


# --------------------------------------------------------------------------- gates
def gate_i0(root, out):
    f = root / "log_pytest_stage7.txt"
    if not f.exists():
        out("  I-G0  FP32 exactness (tests/test_stage7_link.py): no pytest log in the root -> NOT EVALUABLE (not run)")
        return None
    t = f.read_text()
    hd_ = list(re.finditer(r"^==== (\S+) .*-m pytest.*$", t, re.M))
    ts = hd_[-1].group(1) if hd_ else None
    t = t[hd_[-1].end():] if hd_ else t
    p, x, s = (len(re.findall(rf"tests/test_stage7_link\.py::\S+ {w}", t)) for w in ("PASSED", "(?:FAILED|ERROR)", "SKIPPED"))
    ok = p >= IG0_TESTS and x == 0 and s == 0
    out(f"  I-G0  FP32 exactness, tests/test_stage7_link.py in log_pytest_stage7.txt (last of {len(hd_)} run(s), {ts or 'undated'}): "
        f"{p} passed, {x} failed, {s} skipped (>= {IG0_TESTS} tests, none failing or skipped) -> {V(ok)}")
    return ok


def score(root, tag, ref, out, test):
    """Gates, evaluability, verdicts and reported lines. Returns the result dict."""
    F = {lab: (load_json(root / lab) if (root / lab).exists() else None)
         for lab in ("preflight.json", "heads/rank.json", "heads/gate.json", "heads/remaprank.json", f"link/{tag}.json")}
    F["frames"] = load_json(root / "frames" / f"{tag}.json") if (root / "frames" / f"{tag}.json").exists() else None
    LJ = F[f"link/{tag}.json"]
    L = lk.Link(LJ) if LJ and "arms" in LJ else None
    Hh = hd.Heads(F["heads/rank.json"], F["heads/gate.json"]) if F["heads/rank.json"] and F["heads/gate.json"] and "sets" in F["heads/rank.json"] else None
    frames = F["frames"] if F["frames"] and "results" in F["frames"] else None
    return F, L, Hh, frames


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/gpu_stage7")
    ap.add_argument("--tag", default="mistral", help="TEST_<model> in TEST_MODE: sizes not checked, verdicts are not results")
    ap.add_argument("--ref", default=None, help="the stage-3b frames file (default results/gpu_stage3b/paper1_frames_v/mistral.json; none under a TEST_ tag)")
    ap.add_argument("--out", default=None, help="default {root}/STAGE7_SCORE.txt")
    a = ap.parse_args(argv)
    root, test = Path(a.root), a.tag.startswith("TEST_")
    ref_f = Path(a.ref) if a.ref else (None if test else HERE.parent / "results/gpu_stage3b/paper1_frames_v/mistral.json")
    lines, report, errors = [], [], []
    out = lines.append
    out(f"Stage 7 scoring, preregistration P-2026-10-08-I; root {root}; tag {a.tag}; reference {ref_f}")
    if test:
        out("TEST MODE: sizes are not checked and the verdict lines below are plumbing checks, not results")
    F, L, Hh, frames = score(root, a.tag, ref_f, out, test)
    ref = load_json(ref_f) if ref_f and ref_f.exists() else None
    ref = ref if ref and "results" in ref else None
    out("")
    prov_ok = provenance(root, F, a.tag, out, test)
    out("")
    pop_ok = population(F, a.tag, L, out, test)
    out("")
    res = {"gates": {}, "verdicts": {}}
    try:
        R = run_gates_and_verdicts(root, L, Hh, frames, ref, out, report, test, res)
    except Exception:
        errors.append("link")
        out("  SCORER ERROR; the predictions are NOT EVALUABLE\n" + traceback.format_exc())
        R = {f"I{i}": None for i in range(1, 8)}
        res["verdicts"] = R
    i0 = res["gates"].get("I-G0")
    i0_bad = not test and L is not None and not i0
    n = lambda v: sum(x is v for x in R.values())  # noqa: E731
    out(f"\nSUMMARY: {n(True)} MET, {n(False)} NOT MET, {n(None)} NOT EVALUABLE of 7; provenance {'OK' if prov_ok else 'MISMATCH'}; "
        f"population {'OK' if pop_ok else 'MISMATCH'}" + (f"; SCORER ERROR in {errors}" if errors else "")
        + ("; Gate I-G0 not passed with link results (exit 2)" if i0_bad else ""))
    out("\n######## EXPLORATORY (not scored)")
    try:
        exploratory(F, L, Hh, out)
    except Exception:
        out("  exploratory report failed:\n" + traceback.format_exc())
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE7_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return 1 if errors else 2 if not test and not (prov_ok and pop_ok and not i0_bad) else 0


def run_gates_and_verdicts(root, L, Hh, frames, ref, out, report, test, res):
    G = res["gates"]
    out("GATES (I-G0 before any model; I-G1 reproduction and floors; I-G2 the readers at 24B; I-G3 the remap acts; I-G4 not a verdict gate)")
    G["I-G0"] = gate_i0(root, out)
    g1a, g1b, g1c, g2, g3 = {}, {}, {}, {}, {}
    for arm in lk.ARMS:
        ok, txt, _ = lk.gate_g1a(frames, ref, arm, test)
        g1a[arm] = ok
        out(f"  I-G1a {NAMES[arm]:15s} {txt} -> {V(ok)}")
    for arm in lk.ARMS:
        ok, txt, _ = lk.gate_g1b(L, frames, arm) if L is not None else (None, "no link results", {})
        g1b[arm] = ok
        out(f"  I-G1b {NAMES[arm]:15s} {txt} -> {V(ok)}")
    for arm in hd.ARMS:
        ok, txt = lk.gate_g1c(L, arm) if L is not None else (None, "no link results")
        g1c[arm] = ok
        out(f"  I-G1c {NAMES[arm]:15s} (gates the N lines only) {txt} -> {V(ok)}")
    ok, txt, _ = lk.gate_g1d(L) if L is not None else (None, "no link results", None)
    G["I-G1d"] = ok
    out(f"  I-G1d LIST-BEFORE       {txt} -> {V(ok)}")
    for arm in hd.ARMS:
        ok, txt, _ = Hh.gate_i2(arm) if Hh is not None else (None, "no head results", {})
        g2[arm] = ok
        out(f"  I-G2  {NAMES[arm]:15s} {txt} -> {V(ok)}")
    for arm in hd.ARMS:
        out(f"        H1/H2 pattern at 24B ({NAMES[arm]}): " + ("replicated" if g2[arm] else "not replicated" if g2[arm] is False else "not evaluable"))
    for arm in lk.ARMS[:4]:
        c1, c2, txt = lk.gate_g3(L, arm) if L is not None else (None, None, "no link results")
        g3[arm] = (c1, c2)
        out(f"  I-G3  {NAMES[arm]:15s} {txt}")
    g4, txt = lk.gate_g4(L) if L is not None else (None, "no link results")
    G["I-G4"] = g4
    out(f"  I-G4  OPTIONS-AFTER     {txt} -> I5/I6 readable as value takeover: " + ("yes" if g4 else "no" if g4 is False else "not evaluable") + " (not a verdict gate)")
    G |= dict(g1a=g1a, g1b=g1b, g1c=g1c, g2=g2, g3=g3)
    base = bool(G["I-G0"]) and bool(G["I-G1d"])
    if not base:
        out("  I-G0 or I-G1 (d) " + ("failed" if G["I-G0"] is False or G["I-G1d"] is False else "not passed") + ": I1-I7 are NOT EVALUABLE in every format")

    def ev(pred, arm):
        """Evaluability of prediction ``pred`` in format ``arm`` (the entry's split), with the reason when not."""
        if not base:
            return False, "I-G0 / I-G1 (d)"
        need = [("I-G1a", g1a.get(arm)), ("I-G1b", g1b.get(arm)), ("I-G3 (1)", g3.get(arm, (None, None))[0])]
        if pred in ("I2", "I3"):
            need += [("I-G2", g2.get(arm)), ("I-G3 (2)", g3.get(arm, (None, None))[1])]
        if pred in ("I4", "I5", "I6", "I7"):
            need += [("I-G2", g2.get(arm))]
        if pred in ("I4", "I5"):
            need += [("NO-MENTION I-G1a", g1a.get("NONE")), ("NO-MENTION I-G1b", g1b.get("NONE")), ("NO-MENTION I-G3 (1)", g3.get("NONE", (None, None))[0])]
        if pred == "I4" and arm == "P1" and L is not None and L.has("P1", "0") and L.has("NONE", "0"):
            S = lk.cond_stats(L, "P1", "0")
            need += [("kappa(OPTIONS-AFTER) defined", not np.isnan(S["kappa0"][0])), ("kappa(NO-MENTION) defined", not np.isnan(S["kappaN"][0]))]
        miss = [n for n, ok in need if not ok]
        return not miss, ", ".join(miss)

    psiK_none = float("nan")
    KN = None
    if L is not None and L.has("NONE", "0"):
        fN = L.fam("NONE", "0")
        psiK_none, KN = fN["K"].mean() / fN["D"].mean(), fN["K"]
    fns = {"I1": lambda arm: lk.i1(L, arm, KN), "I2": lambda arm: lk.i2(L, arm, psiK_none), "I3": lambda arm: lk.i3(L, arm),
           "I4": lambda arm: lk.i4(L, arm), "I5": lambda arm: lk.i5(L, arm), "I6": lambda arm: lk.i6(L, arm), "I7": lambda arm: lk.i7(L, arm)}
    out("\nVERDICTS (I1-I7; confirmatory formats OPTIONS-AFTER and LETTERS-AFTER: MET if met in both, NOT MET if not met in an evaluable "
        "format, else NOT EVALUABLE; I5 and I6 under OPTIONS-AFTER alone)")
    F = {}
    for p, fn in fns.items():
        arms = ("P1",) if p in ("I5", "I6") else lk.CONF
        per, sub = {}, []
        for arm in arms:
            ok_ev, why = ev(p, arm)
            try:
                ok, txt, *_ = fn(arm) if L is not None else (None, "no link results")
            except Exception as e:  # noqa: BLE001  (a missing record makes this line not evaluable)
                ok, txt = None, f"not computable: {type(e).__name__}: {e}"
            per[arm] = ok if ok_ev else None
            sub.append(f"         {NAMES[arm]}: {txt} -> {V(per[arm])}" + ("" if ok_ev else f"  [computed {V(ok)}; not evaluable: {why}]"))
        F[p] = comb(per.values())
        out(f"  {p:3s} {TITLE[p]:52s} -> {V(F[p])}")
        for s in sub:
            out(s)
    res["verdicts"] = F
    out("\nREPORTED (not scored: SENTENCE-AFTER against the same thresholds, LETTERS-AFTER I5/I6 two-sided, the knockout contrast N(H*))")
    if L is not None:
        for p in ("I1", "I2", "I3", "I4", "I5", "I6", "I7"):
            try:
                ok, txt, *_ = fns[p]("POST")
                out(f"  {p} SENTENCE-AFTER: {txt} -> {V(ok)} (reported)")
            except Exception as e:  # noqa: BLE001
                out(f"  {p} SENTENCE-AFTER: not computable: {type(e).__name__}: {e}")
        for p in ("I5", "I6"):
            try:
                ok, txt, *_ = fns[p]("LETTER")
                out(f"  {p} LETTERS-AFTER (two-sided, no verdict): {txt}")
            except Exception as e:  # noqa: BLE001
                out(f"  {p} LETTERS-AFTER: not computable: {type(e).__name__}: {e}")
        for arm in hd.ARMS:
            if not L.has(arm, "N:H"):
                continue
            S = lk.cond_stats(L, arm, "N:H")
            note = "" if g1c.get(arm) else "  [I-G1 (c) not passed: the N lines are not interpretable]"
            out(f"  N(H*) {NAMES[arm]}: {lk.stats_txt(S)}{note}")
            out(f"        argmax rates {lk.rates_txt(L, arm)}")
            if L.has(arm, "A:H"):
                SA, tN = lk.cond_stats(L, arm, "A:H"), S["t"][0]
                out(f"        t under A {SA['t'][0]:+.3f} vs under N {tN:+.3f}" + ("; t falls under N while it holds under A: the stage-5 / H3 difference recurs at the head level"
                                                                                  if SA["t"][0] >= 0.75 and tN < 0.6 else ""))
        for arm in hd.ARMS:
            if L.has(arm, "A:H"):
                out(f"  A(H*) {NAMES[arm]}: {lk.stats_txt(lk.cond_stats(L, arm, 'A:H'))}")
            if L.has(arm, "A+:H"):
                out(f"  A+(H*) {NAMES[arm]}: {lk.stats_txt(lk.cond_stats(L, arm, 'A+:H'))}")
    return F


# --------------------------------------------------------------------------- exploratory
def exploratory(F, L, Hh, out):
    if Hh is not None:
        out("-- head sets and the gate curves (step 1)")
        Hh.report(out)
    if L is None:
        out("  no link results")
        return
    out("-- exploratory conditions (step 2)")
    for arm in hd.ARMS:
        if not L.R(arm):
            continue
        out(f"   {NAMES[arm]}:")
        for c in ("N:allG", "A:L4", "N:rand0", "A:H_P1"):
            if L.has(arm, c):
                out(f"      {c:8s} {lk.stats_txt(lk.cond_stats(L, arm, c))}")
        if L.R(arm) and "x/rem_H_101" in L.R(arm)[0]["runs"]:
            rho = est(lambda m, r, p: (m - r) / (m - p), L.sm(arm, "x", "M"), L.sm(arm, "x", "rem_H"), L.sm(arm, "x", "P"))
            rx = est(lambda s, g: s / g, L.a(arm, "s_H"), L.a(arm, "s_G"))
            out(f"      removal through the readers rho_K^x(H*) {f3(rho)} (frames rho_K 0.762 / 0.986); sufficiency R_x(H*) {f3(rx)}")
        R0 = L.R(arm)[0]["runs"]
        ks = sorted(int(k.split("_")[1][1:]) for k in R0 if k.startswith("curve/x_k") and k.endswith("_101"))
        if ks:
            out("      KO_x(k) " + " ".join(f"{k}:{lk.ko_x(L, arm, f'x_k{k}', 'curve')[0]:+.2f}" for k in ks))
        if "curve/x_HP1_101" in R0:
            out(f"      KO_x(H*_OPTIONS-AFTER) under LETTERS-AFTER {f3(lk.ko_x(L, arm, 'x_HP1', 'curve'))}")
        for s in lk.SEEDS:
            aa = np.array([lk.m_of(r["runs"][f"x/x_all_{s}"], r) - lk.m_of(r["runs"][f"x/P_{s}"], r) for r in L.R(arm)])
            an = np.array([lk.m_of(r["runs"][f"x/x_notG_{s}"], r) - lk.m_of(r["runs"][f"x/P_{s}"], r) for r in L.R(arm)])
            aH = np.array([lk.m_of(r["runs"][f"x/x_H_{s}"], r) - lk.m_of(r["runs"][f"x/P_{s}"], r) for r in L.R(arm)])
            out(f"      seed {s}: g_K {1 - an.mean() / aa.mean():+.3f}, KO_x(H*) {(aa.mean() - aH.mean()) / (aa.mean() - an.mean()):+.3f}")
        dist = np.array([r["core"]["distractor_location"] in (r["core"]["base"], r["core"]["source"]) for r in L.R(arm)])
        aa, an, aH = L.a(arm, "x_all"), L.a(arm, "x_notG"), L.a(arm, "x_H")
        for lab, sel in (("distractor at B or S", dist), ("the rest", ~dist)):
            if sel.any():
                out(f"      {lab} (n={int(sel.sum())}): g_K {1 - an[sel].mean() / aa[sel].mean():+.3f}, "
                    f"KO_x(H*) {(aa[sel].mean() - aH[sel].mean()) / (aa[sel].mean() - an[sel].mean()):+.3f}")
    RR = F.get("heads/remaprank.json")
    if RR and RR.get("arms") and Hh is not None:
        from scipy.stats import hypergeom
        out("-- remap ranking (a3 of P vs P + K_M at the T and S rows, eager, in-sample on E)")
        for arm, A in RR["arms"].items():
            Hr, Hs = {tuple(c) for c in A["H_rem"]}, set(Hh.sets[arm]["H"])
            ov = len(Hr & Hs)
            jac = ov / max(1, len(Hr | Hs))
            p = hypergeom.sf(ov - 1, Hh.n_elig, len(Hs), len(Hr))
            ko = A["ko"]
            m = lambda r, n, s: r["runs"][f"{n}_{s}"]["cand"][r["iT"]] - r["runs"][f"{n}_{s}"]["cand"][r["iS"]]  # noqa: E731
            sm = lambda n: np.array([np.mean([m(r, n, s) for s in lk.SEEDS]) for r in ko])  # noqa: E731
            aa, an, ah = sm("x_all") - sm("P"), sm("x_notG") - sm("P"), sm("x_Hrem") - sm("P")
            k = est(lambda a, n, h: (a - h) / (a - n), aa, an, ah)
            out(f"   {NAMES[arm]}: |H_rem & H*| = {ov} of {len(Hs)}, Jaccard {jac:.3f}, hypergeometric P {p:.1e}; KO_x(H_rem) {f3(k)} (in-sample)")
    elif RR is not None:
        out("-- remap ranking: " + str((RR.get("provenance") or {}).get("skipped", "no results")))
    sk = L.P.get("explo_skipped") or []
    if sk:
        out(f"-- exploratory conditions skipped at the deadline: {len(sk)} (first {sk[:5]})")


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
