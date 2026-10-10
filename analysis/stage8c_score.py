"""Score part C of preregistration P-2026-10-10-J (docs/PREREGISTRATION.md, GPU stage 8): independently obtained identity
edits at the writing token and the channel-ratio law. Gates J-C-G0 to J-C-G8, the confirmatory lines J-C1 ... J-C-WIN,
the reported lines (the pi(t) diagnostic, the depth-tracking consistency check JC4, the lexical-code reading, kappa
against sigma), the summary by risk class and the exploratory report (part scorers in analysis/stage8c_parts).

Inputs under --results (default results/gpu_stage8c), as written by scripts/gpu_stage8c.sh: preflight/, calib/, das/,
eval/, readers/, explore/ (experiments/stage8_edits.py), overlap/<tag>/ (experiments/stage8_overlap.py), jc6/<tag>/
(experiments/prakash_caa.py), logs/pytest.log (J-C-G0: its last pytest run), COMMIT.txt, ENV.txt, REVISIONS.txt,
SKIPPED.txt. Tags: qwen7, mistral7, llama8, qwen14 (TEST_<key> with --test).
Output STAGE8C_SCORE.txt (also --out) with the sections PROVENANCE, POPULATION, GATES, PREDICTIONS, REPORTED, SUMMARY,
EXPLORATORY. Exit status 1 if a part raised (its lines NOT EVALUABLE, traceback printed); 2 if, outside --test, the
provenance or population check reports MISMATCH, or results exist without a passing J-C-G0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage8c_parts import law as LW  # noqa: E402
from stage8c_parts import prakash as PK  # noqa: E402
from stage8c_parts import readers as RD  # noqa: E402
from stage8c_parts.stats import LOG05, LOG08, LOG125, Q, V, holm, mean_q  # noqa: E402

MODELS = ("qwen7", "mistral7", "llama8")
FAMILIES = {"qwen7": ("E1", "E2", "E3", "E4", "E5"), "mistral7": ("E1", "E2", "E4", "E5"), "llama8": ("E1", "E2", "E5")}
E5_SUB = ("E5FR", "E5DE", "E5SYN", "E5NL")
MIN_STORIES = 60
G0_FILES = {"tests/test_stage8_edits.py": 14, "tests/test_sae.py": 7, "tests/test_stage8c_score.py": 12}
POP_E = "a23465a211577f4c6e9efe78d4c7a588b598c05437406001145a420fec9d8c1b"
# code -> (class, kind, prior, title); the priors were recorded in the entry before any stage-8 output
LINES = {
    "J-C1": ("L", "A", 0.80, "the law in Tier 0: E1 at every depth, E2 at l in {3, 7}; Qwen2.5-7B, Mistral-7B"),
    "J-C2": ("M", "A", 0.80, "the law in Tier 1: E2 at l in {11, 15} (Qwen2.5-7B, Mistral-7B); E1, E2 at Llama-3.1-8B"),
    "J-C3": ("R", "A", 0.55, "the law for E3, third-party SAE features (Qwen2.5-7B)"),
    "J-C4": ("R", "A", 0.35, "the law for E4, the DAS remap at p (Qwen2.5-7B, Mistral-7B; seeds as a level)"),
    "J-C5": ("R", "A", 0.35, "the law on the identity-carrying PERP / NONLEX components"),
    "J-C-BOUND": ("R", "A", 0.50, "the lexical boundary: a non-lexical identity edit is under-read by keys (lambda <= log 0.5)"),
    "J-C-READa": ("M", "A", 0.85, "the same reader heads read E1 and E2 (P1, l = 7)"),
    "J-C-READb": ("R", "A", 0.60, "the same reader heads read E3 and E4 (P1, l = 7)"),
    "J-C6": ("M", "A", 0.85, "a CAA identity edit at block 28 is not key-flat (Qwen2.5-14B, Prakash et al.'s material)"),
    "J-C-SCREEN": ("R", "A", 0.50, "no overlap window: the binding swap acts only after the identity key route closes"),
    "J-C-WIN": ("R", "A", 0.30, "inside an overlap window the binding swap follows the natural read (H7 and H9)"),
}


PATTERNS = {}     # line code -> {combo: pattern of the law (equivalent, R1, R3, graded departure)}
COMPONENTS = []   # (line code, label, bootstrap p, rejected by the line's own interval rule): the Holm sensitivity analysis
CARRIERS = []     # (model, component, depth) of every identity-carrying PERP / NONLEX component of an effective edit (J-C5)


def comp_tost(code, lab, q):
    """The two one-sided components of a TOST statistic (own rule: the 90 % interval, 5 % per side)."""
    lo, hi = q.ci(0.90)
    COMPONENTS.append((code, f"{lab} H0: lambda <= -log 1.25", q.p_le(-LOG125), bool(lo > -LOG125)))
    COMPONENTS.append((code, f"{lab} H0: lambda >= log 1.25", q.p_ge(LOG125), bool(hi < LOG125)))


def load_json(f):
    try:
        return json.load(open(f))
    except Exception as ex:  # a truncated file is reported, not fatal
        return {"error": repr(ex)}


def tagof(key, test):
    return ("TEST_" if test else "") + key


# --------------------------------------------------------------------------- inputs
class Inputs:
    def __init__(self, root: Path, test: bool):
        self.root, self.test = root, test
        self.F = {}
        # rule G2: yi9 takes llama8's slot when llama8's files failed verification before any output of it existed
        self.slot = {k: k for k in MODELS}
        if not (root / "eval" / f"{tagof('llama8', test)}.json").exists() and (root / "eval" / f"{tagof('yi9', test)}.json").exists():
            self.slot["llama8"] = "yi9"
        for sub in ("preflight", "calib", "das", "eval", "readers", "explore"):
            for k in MODELS:
                f = root / sub / f"{tagof(self.slot[k], test)}.json"
                self.F[(sub, k)] = load_json(f) if f.exists() else None
        for k in ("qwen7", "llama8"):
            for fn in ("screen", "window"):
                f = root / "overlap" / tagof(k, test) / f"{fn}.json"
                self.F[(fn, k)] = load_json(f) if f.exists() else None
        f = root / "jc6" / tagof("qwen14", test) / "exchange.json"
        self.F[("jc6", "qwen14")] = load_json(f) if f.exists() else None
        self.models = {}
        for k in MODELS:
            J = self.F[("eval", k)]
            if J and "stories" in J and J["stories"]:
                self.models[k] = LW.Model(J, test)

    def get(self, sub, k):
        J = self.F.get((sub, k))
        return J if J and "error" not in J else None


# --------------------------------------------------------------------------- provenance and population
def manifest_revisions():
    try:
        J = json.load(open(HERE.parent / "scripts" / "stage8_models.json"))
        return {k: v["revision"] for k, v in J["models"].items()}
    except Exception:  # noqa: BLE001
        return {}


def provenance(I: Inputs, out):
    out("PROVENANCE (the pipeline's COMMIT.txt, ENV.txt, REVISIONS.txt, SKIPPED.txt; every results file)")
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt", "SKIPPED.txt"):
        f = I.root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text().splitlines()[-40:] if l.strip()) if f.exists() else " absent"))
    revs = manifest_revisions()
    ok, commits, cal = True, set(), {}
    for (sub, k), J in sorted(I.F.items()):
        if J is None:
            continue
        if "error" in J or "provenance" not in J:
            out(f"  {sub}/{k}: " + (f"UNREADABLE {J.get('error')}" if "error" in J else "NO PROVENANCE") + "  MISMATCH")
            ok = False
            continue
        p = J["provenance"]
        if p.get("skipped"):
            out(f"  {sub}/{k}: skipped ({p['skipped']})")
            continue
        commits.add(p.get("git_commit"))
        flag = []
        ver = p.get("verified") or {}
        if not I.test and sub not in ("preflight",):
            if ver and revs.get(I.slot.get(k, k)) and ver.get("revision") != revs[I.slot.get(k, k)]:
                flag.append(f"revision {ver.get('revision')} != the manifest's {revs[I.slot.get(k, k)]}")
            if "dtype" in p and not str(p.get("dtype")).endswith("bfloat16"):
                flag.append("dtype is not BF16")
            if "attn_implementation" in p and p.get("attn_implementation") != "sdpa":
                flag.append("attention is not sdpa")
        if sub in ("eval", "readers", "explore") and p.get("calib_sha256"):
            cal.setdefault(k, set()).add(p["calib_sha256"])
        ok &= not flag
        out(f"  {sub}/{k}: commit {str(p.get('git_commit'))[:10]}, model {p.get('model') or (p.get('args') or {}).get('model')} ({(ver or {}).get('repo')} @ {str((ver or {}).get('revision'))[:10]}), "
            f"dtype {p.get('dtype', '-')}, attn {p.get('attn_implementation', '-')}, device {p.get('device', '-')}, transformers {p.get('transformers')}, "
            f"test_mode {p.get('test_mode')}" + "".join(f"  MISMATCH: {x}" for x in flag))
    for k in MODELS:
        C = I.get("calib", k)
        if C and k in cal:
            same = cal[k] == {C.get("calib_pt_sha256")}
            ok &= same
            out(f"  calibration file of {k}: sha256 {str(C.get('calib_pt_sha256'))[:16]}; every later file records it: {'OK' if same else 'MISMATCH'}")
    if I.slot["llama8"] != "llama8":
        out("  rule G2: yi9 (01-ai/Yi-1.5-9B-Chat) holds llama8's slot (its files failed verification; FETCH_FAILED.txt)")
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else ""))
    return ok and len(commits) <= 1


def population(I: Inputs, out):
    out("POPULATION (E: the first 80 cores of the Random(8101) stream with pi(S) != B, pi(X) != B, disjoint from U; "
        + ("TEST: sizes not checked)" if I.test else f"a model needs >= {MIN_STORIES} evaluated stories)"))
    ok = True
    for k in MODELS:
        pf = I.get("preflight", k)
        if pf:
            good = pf.get("hashes", {}).get("E") == POP_E or I.test
            ok &= good and not pf.get("skipped_items")
            out(f"  preflight {k}: sizes {pf.get('sizes')}, E hash {str(pf.get('hashes', {}).get('E'))[:16]} {'OK' if good else 'MISMATCH'}, "
                f"overlap with U {pf.get('overlap', {}).get('U')}, skipped {len(pf.get('skipped_items', []))}")
        m = I.models.get(k)
        if m is not None:
            idx = [s["index"] for s in m.S]
            seq = idx == list(range(len(idx)))
            p_ok = m.P.get("populations", {}).get("E") == POP_E
            ok &= seq and p_ok
            out(f"  eval {k}: {m.n} stories (indices 0..{m.n - 1} in order: {'OK' if seq else 'MISMATCH'}); population hash recorded: "
                f"{'OK' if p_ok else 'MISMATCH'}" + ("" if I.test or m.n >= MIN_STORIES else f"; fewer than {MIN_STORIES}: the model is not evaluable"))
    out(f"  population: {'OK' if ok else 'MISMATCH'}")
    return ok


def gate_g0(root, out):
    f = root / "logs" / "pytest.log"
    if not f.exists():
        out("  J-C-G0  FP32 unit tests: no logs/pytest.log -> NOT EVALUABLE (not run)")
        return None
    t = f.read_text()
    hd = list(re.finditer(r"^==== (\S+) .*-m pytest.*$", t, re.M))
    t = t[hd[-1].end():] if hd else t
    res, ok = [], True
    for fn, need in G0_FILES.items():
        p, x, s = (len(re.findall(rf"^{re.escape(fn)}::\S+ {w}", t, re.M)) for w in ("PASSED", "(?:FAILED|ERROR)", "SKIPPED"))
        good = p >= need and x == 0 and s == 0
        ok &= good
        res.append(f"{fn} {p} passed, {x} failed, {s} skipped")
    other = len(re.findall(r"^tests/\S+::\S+ (?:FAILED|ERROR)", t, re.M))
    ok &= other == 0
    out(f"  J-C-G0  FP32 unit tests (last of {len(hd)} run(s)): {'; '.join(res)}; failures in the run {other} -> {V(ok)}")
    return ok


# --------------------------------------------------------------------------- the per-model gates
class Gates:
    def __init__(self, I: Inputs, out, g0):
        self.I, self.g0 = I, g0
        self.model_ok, self.g8, self.g3, self.why = {}, {}, {}, {}
        for k in MODELS:
            m = I.models.get(k)
            if m is None:
                self.model_ok[k], self.why[k] = None, "no eval results"
                out(f"  {k}: no eval results -> its lines NOT EVALUABLE")
                continue
            if not I.test and m.n < MIN_STORIES:
                self.model_ok[k], self.why[k] = None, f"{m.n} stories < {MIN_STORIES}"
                out(f"  {k}: {m.n} stories < {MIN_STORIES} -> its lines NOT EVALUABLE")
                continue
            g1, t1 = m.g1()
            g2, t2 = m.g2()
            out(f"  J-C-G1 {k:8s} {t1} -> {V(g1)}")
            out(f"  J-C-G2 {k:8s} {t2} -> {V(g2)}")
            ok8, l8 = LW.g8(m)
            out(f"  J-C-G8 {k:8s} sensitivity -> {V(ok8)}")
            for x in l8:
                out(f"           {x}")
            self.g8[k] = ok8
            self.model_ok[k] = bool(g1 and g2)
            self.why[k] = "" if g1 and g2 else "J-C-G1 or J-C-G2 failed"
            for l in m.depths:
                cells = " ".join(f"{f}:{'ok' if m.cell_ok(f, l)[0] else 'x(' + m.cell_ok(f, l)[1] + ')'}" for f in LW.FORMATS)
                out(f"  J-C-G4 {k:8s} l={l:2d} {cells}")
            for l, (ok6, t6) in m.g6().items():
                out(f"  J-C-G6 {k:8s} l={l:2d} {t6} -> {'passed' if ok6 else 'FLAGGED' if ok6 is False else 'not evaluable'}")
            for Z in self.instances(k):
                for l in m.depths:
                    eff, t5 = m.effective(l, Z, "carry" if Z in E5_SUB else "edit")
                    out(f"  J-C-G5 {k:8s} {Z:6s} l={l:2d} {t5} -> {'effective' if eff else 'ineffective' if eff is False else 'not evaluable'}")
        C = I.get("calib", "qwen7")
        for l, g in ((C or {}).get("sae") or {}).items():
            f = g["fve"]
            pub = g.get("published_fve")
            ok = bool(pub is not None and f.get("0", -1) >= pub - 0.05 and f.get("0", -1) > f.get("-1", 9) and f.get("0", -1) > f.get("1", 9))
            self.g3[int(l)] = ok
            out(f"  J-C-G3 qwen7    l={l:>2s} FVE {f.get('-1', float('nan')):.3f} / {f.get('0', float('nan')):.3f} / {f.get('1', float('nan')):.3f} at l-1 / l / l+1 "
                f"(published {pub}; >= published - 0.05 and above both neighbours); FVE at p {g.get('fve_at_p', float('nan')):.3f}; "
                f"k_F {g['kF']} beta {g['beta']}" + (" (ineffective at calibration)" if g.get("ineffective_at_calibration") else "") + f" -> {V(ok)}")

    def instances(self, k):
        f = FAMILIES[k]
        z = [x for x in ("E1", "E2", "E3") if x in f] + (["E4"] if "E4" in f else []) + (list(E5_SUB) if "E5" in f else [])
        m = self.I.models.get(k)
        return [x for x in z if m is None or any(m.has("NONE", l, LW.probe(x) + "|KV|S") for l in m.depths)]

    def ok(self, k, law=True):
        base = self.g0 is not False and self.model_ok.get(k) is True
        return base and (self.g8.get(k) is True if law else True)


# --------------------------------------------------------------------------- lines
def combine(per, need):
    """per: {label: True/False/None}; MET iff >= need evaluable and every evaluable met; NOT MET if an evaluable one fails
    (and >= need evaluable); else NOT EVALUABLE."""
    ev = [v for v in per.values() if v is not None]
    if len(ev) < need:
        return None
    return all(ev)


def law_line(G: Gates, I: Inputs, combos, need, out_lines, code=""):
    per = {}
    for k, Z, depths, gate in combos:
        lab = f"{k} {Z} l in {list(depths)}"
        if not G.ok(k):
            per[lab] = None
            out_lines.append(f"    {lab}: not evaluable ({G.why.get(k) or ('J-C-G8 failed' if G.g8.get(k) is False else 'gates')})")
            continue
        m = I.models[k]
        depths = [l for l in depths if l in m.depths]
        met, txt, st, pat = LW.law(m, Z, depths, gate)
        per[lab] = met
        out_lines.append(f"    {lab}: {txt} -> {V(met)}")
        if met is not None:
            PATTERNS.setdefault(code, {})[lab] = pat
            for x in st:
                comp_tost(code, f"{lab} {x['code']}", x["q"])
    return combine(per, need), per


def lines_law(G: Gates, I: Inputs):
    R = {}
    t1 = []
    combos = [(k, "E1", (3, 7, 11, 15), "edit") for k in ("qwen7", "mistral7")] + [(k, "E2", (3, 7), "edit") for k in ("qwen7", "mistral7")]
    R["J-C1"] = (*law_line(G, I, combos, 1 if I.test else 3, t1, "J-C1"), t1)
    t2 = []
    combos = [(k, "E2", (11, 15), "edit") for k in ("qwen7", "mistral7")] + [("llama8", z, (3, 7, 11, 15), "edit") for z in ("E1", "E2")]
    R["J-C2"] = (*law_line(G, I, combos, 1 if I.test else 2, t2, "J-C2"), t2)
    t3 = []
    m = I.models.get("qwen7")
    dep = [l for l in (3, 7, 11, 15) if G.g3.get(l)]
    if len(dep) < 4:
        t3.append(f"    E3 depths passing J-C-G3: {dep}")
    R["J-C3"] = (*law_line(G, I, [("qwen7", "E3", dep, "edit")], 1, t3, "J-C3"), t3)
    t4 = []
    R["J-C4"] = (*law_line(G, I, [(k, "E4", (3, 7, 11, 15), "edit") for k in ("qwen7", "mistral7")], 1, t4, "J-C4"), t4)
    for k in ("qwen7", "mistral7"):
        if G.ok(k) and k in I.models:
            for z in ("E4a", "E4b"):
                met, txt, _, _ = LW.law(I.models[k], z, I.models[k].depths, "edit")
                t4.append(f"    (seed-level, reported) {k} {z}: {txt} -> {V(met)}")
    R["J-C5"] = j_c5(G, I)
    R["J-C-BOUND"] = j_bound(G, I)
    return R


def j_c5(G: Gates, I: Inputs):
    lines, stats = [], []
    for k in MODELS:
        if not G.ok(k):
            continue
        m = I.models[k]
        for Z in ("E1", "E2", "E3", "E4"):
            if Z not in FAMILIES[k] or (Z == "E3" and not G.g3):
                continue
            for comp in ("PERP", "NONLEX"):
                inst = f"{comp}:{Z}"
                if not any(m.has("NONE", l, f"{LW.probe(inst)}|KV|S") for l in m.depths):
                    continue
                for l in m.depths:
                    if Z == "E3" and not G.g3.get(l):
                        continue
                    eff_z, _ = m.effective(l, Z, "edit")
                    car, ctxt = m.effective(l, inst, "carry")
                    if not (eff_z and car):
                        continue
                    CARRIERS.append((k, inst, l))
                    st, skip = m.stats_of(inst, [l], gate=None, formats=LW.COMP_FORMATS)
                    stats += st
                    for x in st:
                        comp_tost("J-C5", f"{k} {inst} {x['code']}", x["q"])
                    lines.append(f"    {k} {inst} l={l} carries identity ({ctxt}): " + (", ".join(
                        f"{s['code']} {s['q'].pt:+.2f} [{s['q'].lower(0.9):+.2f},{s['q'].upper(0.9):+.2f}]" + ("" if s["equivalent"] else "*") for s in st) or "no statistic: " + "; ".join(skip)))
    if not stats:
        return None, {}, lines + ["    no identity-carrying component with a lambda statistic"]
    return all(s["equivalent"] for s in stats), {}, lines


def j_bound(G: Gates, I: Inputs):
    lines, per = [], {}
    for k in MODELS:
        if not G.ok(k) or "E5" not in FAMILIES[k]:
            continue
        m = I.models[k]
        for sub in E5_SUB:
            if not any(m.has("NONE", l, f"{sub}|KV|S") for l in m.depths):
                continue
            st, skip = m.stats_of(sub, m.depths, gate="carry")
            if not st:
                lines.append(f"    {k} {sub}: no lambda statistic at a gated-in depth ({'; '.join(skip[:4])})")
                continue
            codes = {(s["code"], s["l"]) for s in st}
            st2, _ = m.stats_of("E2", sorted({s["l"] for s in st}), gate="edit")
            st2 = [s for s in st2 if (s["code"], s["l"]) in codes]
            lab = f"{k} {sub}"
            if not st2:
                per[lab] = None
                lines.append(f"    {lab}: E2 has no lambda statistic in the same cells -> not evaluable")
                continue
            p5 = mean_q([s["q"] for s in st])
            p2 = mean_q([s["q"] for s in st2])
            ok = bool(p5.pt <= LOG05 and p5.upper() < LOG08 and p2.pt >= LOG08)
            per[lab] = ok
            COMPONENTS.append(("J-C-BOUND", f"{lab} H0: lambda_E5 >= log 0.8", p5.p_ge(LOG08), bool(p5.upper() < LOG08)))
            lines.append(f"    {lab}: pooled lambda_E5 {p5.txt()} (<= log 0.5 = {LOG05:+.3f}; H0: >= log 0.8 rejected, upper < {LOG08:+.3f}) over "
                         f"{len(st)} statistics; E2 at the same cells {p2.txt()} (>= log 0.8) -> {'met' if ok else 'not met'}")
    return combine(per, 1), per, lines


def j_readers(G: Gates, I: Inputs):
    out = {"a": {}, "b": {}}
    lines = []
    for k in ("qwen7", "mistral7"):
        J = I.get("readers", k)
        if J is None or not G.ok(k, law=False):
            lines.append(f"    {k}: " + ("no readers results" if J is None else "model not evaluable"))
            continue
        Rr = RD.Readers(J)
        g, gt = Rr.gate()
        lines.append(f"    {k}: gate {gt} -> {V(g)}")
        m = I.models[k]
        for D in J["donors"]:
            if D == "nat":
                continue
            fam = "E4" if D == "E4a" else D
            eff, _ = m.effective(7, fam, "edit")
            grp = "a" if D in ("E1", "E2") else "b"
            if not g:
                out[grp][f"{k} {D}"] = None
                continue
            if not eff:
                out[grp][f"{k} {D}"] = None
                lines.append(f"    {k} {D}: not effective at l = 7 (J-C-G5): not counted")
                continue
            ok, t = Rr.family(D)
            out[grp][f"{k} {D}"] = ok
            if ok is not None:
                kq = Rr.ko(D, "H")
                COMPONENTS.append((f"J-C-READ{grp}", f"{k} {D} H0: KO <= 0.5", kq.p_le(RD.KO_LO), bool(kq.lower() > RD.KO_LO)))
            lines.append(f"    {k} {D}: {t} -> {V(ok)}")
    return (combine(out["a"], 1), out["a"], lines), (combine(out["b"], 1), out["b"], [])


def j_prakash(I: Inputs):
    R = {}
    J = I.get("jc6", "qwen14")
    if J is None:
        R["J-C6"] = (None, {}, ["    not run (deadline or no results)"])
    else:
        v, g7, l6 = PK.jc6(J, I.test, COMPONENTS)
        R["J-C6"] = (v, {"J-C-G7": g7}, ["    " + x for x in l6])
    sl, wl, evs, wins, wres = [], [], {}, {}, {}
    for k in ("qwen7", "llama8"):
        S = I.get("screen", k)
        if S is None:
            sl.append(f"    {k}: no screen results")
            continue
        ev, W, lw, ls, same = PK.screen(S, I.test)
        evs[k], wins[k] = ev, bool(W)
        sl += [f"    {k}: " + ls[0]] + ["      " + x for x in ls[1:]]
        Wj = I.get("window", k)
        if W:
            if Wj is None:
                wres[k] = None
                wl.append(f"    {k}: window {W} but no window results")
            else:
                v, lines = PK.window(Wj, I.test, COMPONENTS)
                wres[k] = v
                wl += [f"    {k} (l_w = {Wj['l_w']}):"] + ["    " + x for x in lines]
    ev_models = [k for k, e in evs.items() if e]
    if not ev_models:
        R["J-C-SCREEN"] = (None, {}, sl + ["    no screened model is evaluable"])
    else:
        R["J-C-SCREEN"] = (not any(wins[k] for k in ev_models), {}, sl)
    win = {k: wres.get(k) for k in ev_models if wins.get(k)}
    R["J-C-WIN"] = (combine(win, 1), win, wl or ["    no window in any evaluable screened model"])
    return R


# --------------------------------------------------------------------------- reported and exploratory
def reported(G: Gates, I: Inputs, out):
    """The reported lines; returns whether every effective (model, family, depth) acts through the natural lexical code
    (None when there is none), for outcome (b) of the headline."""
    out("\nREPORTED (no verdict counted)")
    codes = []
    for k in ("qwen7", "mistral7"):
        m = I.models.get(k)
        if m is None or "E4" not in FAMILIES[k]:
            continue
        for l in m.depths:
            if not m.has("LETTER", l, "E4a|K|S"):
                continue
            d = []
            for s in range(m.n):
                st = m.S[s]
                for z in ("E4a", "E4b"):
                    for t, it, ip in (("S", st["iS"], st["iPiS"]), ("X", st["iX"], st["iPiX"])):
                        r = m.row(s, "LETTER", l, f"{z}|K|{t}")
                        d.append(r[ip] - r[it])
            arr = np.array(d).reshape(m.n, -1).mean(1)
            q = Q(*m.boot.mean(arr))
            read = "the key raises pi(t)'s letter more than t's: a lexical-key departure" if q.lower() > 0 else \
                "the key raises t's letter more than pi(t)'s" if q.upper() < 0 else "no difference resolved"
            out(f"  pi(t) diagnostic {k} l={l} (LETTER, K-only E4 rows): mean d_pi(t) - d_t {q.txt()} -> {read}")
    for k, m in I.models.items():
        for Z in ("E1", "E2", "E3", "E4"):
            if Z not in FAMILIES[k]:
                continue
            for l in m.depths:
                eff, _ = m.effective(l, Z, "edit")
                if not eff:
                    continue
                phiZ = m.psi("NONE", l, Z, "KV").pt
                par = m.psi("NONE", l, f"PAR:{Z}", "KV").pt if m.has("NONE", l, f"{LW.probe('PAR:' + Z)}|KV|S") else float("nan")
                lex = m.psi("NONE", l, f"LEX:{Z}", "KV").pt if m.has("NONE", l, f"{LW.probe('LEX:' + Z)}|KV|S") else float("nan")
                code = bool((par / phiZ >= 0.8) or (lex / phiZ >= 0.8))
                codes.append(code)
                out(f"  lexical-code reading {k} {Z} l={l}: phi {phiZ:+.3f}, PAR {par:+.3f}, LEX {lex:+.3f} -> "
                    + ("acts through the natural lexical code (PAR or LEX >= 0.8 of the edit)" if code else "not carried by PAR or LEX alone"))
    out("  JC4 depth tracking (consistency check; delta_sigma = sigma(f, 3) - sigma(f, 15) >= 0.25 with both cells evaluable):")
    qual = []
    for k, m in I.models.items():
        if 3 not in m.depths or 15 not in m.depths:
            continue
        for Z in ("E1", "E2", "E3", "E4"):
            if Z not in FAMILIES[k] or not (m.effective(3, Z)[0] and m.effective(15, Z)[0]):
                continue
            for f in LW.FORMATS:
                if not (m.cell_ok(f, 3)[0] and m.cell_ok(f, 15)[0]):
                    continue
                ds = m.sigma(f, 3).pt - m.sigma(f, 15).pt
                if ds < 0.25:
                    continue
                dk = m.kappa(f, 3, Z).pt - m.kappa(f, 15, Z).pt
                ok = bool(dk >= 0.5 * ds and abs(dk - ds) <= 0.2)
                qual.append(ok)
                out(f"    {k} {Z} {f}: delta_sigma {ds:+.3f} delta_kappa {dk:+.3f} -> {'holds' if ok else 'does not hold'}")
    out(f"    JC4: {sum(qual)} of {len(qual)} qualifying pairs hold" + (" (>= 80 % of >= 3: consistent)" if len(qual) >= 3 and np.mean(qual) >= 0.8 else
                                                                           " (fewer than 3 pairs: not evaluable)" if len(qual) < 3 else " (inconsistent)"))
    return (all(codes) if codes else None)


def exploratory(G: Gates, I: Inputs, out):
    out("\n######## EXPLORATORY (not scored)")
    for k, m in I.models.items():
        out(f"-- {k}: kappa against sigma per cell (descriptive, Fig. 4), nu, K/V cosines and the residual cosine (means over stories and targets)")
        for l in m.depths:
            for f in LW.FORMATS:
                if not m.cell_ok(f, l)[0]:
                    continue
                s = m.sigma(f, l).pt
                parts = []
                for Z in ("E1", "E2", "E3", "E4a", "E4b") + E5_SUB:
                    if m.has(f, l, f"{Z}|K|S"):
                        parts.append(f"{Z} {m.kappa(f, l, Z).pt:+.2f}")
                out(f"   {f}@{l}: sigma {s:+.3f}; kappa " + ", ".join(parts))
            st = []
            for Z in ("E1", "E2", "E3", "E4a", "E4b", "PERP:E1", "NONLEX:E1"):
                nu = np.nanmean([m.stat(l, Z, t, "nu") for t in ("S", "X")])
                if np.isfinite(nu):
                    st.append(f"{Z} nu {nu:.2f} cosK {np.nanmean([m.stat(l, Z, t, 'cos_k') for t in 'SX']):.2f} "
                              f"cosV {np.nanmean([m.stat(l, Z, t, 'cos_v') for t in 'SX']):.2f} a {np.nanmean([m.stat(l, Z, t, 'cos_resid') for t in 'SX']):.2f}")
            out(f"   l={l}: " + "; ".join(st))
        out(f"-- {k}: story-level distance D = mean |ID_K^Z - ID_K^nat| + |ID_V^Z - ID_V^nat| over mean ID_KV^nat (evaluable cells)")
        for l in m.depths:
            parts = []
            for f in LW.FORMATS:
                if not m.cell_ok(f, l)[0]:
                    continue
                den = np.nanmean(m.ID(f, l, "nat", "KV"))
                for Z in ("T", "E1", "E2", "E3", "E4a"):
                    if m.has(f, l, f"{Z}|K|S"):
                        Dz = np.nanmean(np.abs(m.ID(f, l, Z, "K") - m.ID(f, l, "nat", "K")) + np.abs(m.ID(f, l, Z, "V") - m.ID(f, l, "nat", "V"))) / den
                        parts.append(f"{f}:{Z} {Dz:.3f}")
            out(f"   l={l}: " + ", ".join(parts))
        out(f"-- {k}: O1, first-order prediction of psi_K from the edit's key displacement projected on the natural one (readers' KV groups where known)")
        for l in m.depths:
            for Z in ("E1", "E2", "E3", "E4a"):
                if not m.has("P1", l, f"{Z}|K|S"):
                    continue
                pred = np.nanmean([m.stat(l, Z, t, "proj_k_readers") for t in "SX"])
                pred_all = np.nanmean([m.stat(l, Z, t, "proj_k") for t in "SX"])
                obs = m.psi("P1", l, Z, "K").pt
                out(f"   {Z} l={l}: predicted {pred:+.3f} (readers) / {pred_all:+.3f} (all heads); observed psi_K(P1) {obs:+.3f}")
        out(f"-- {k}: L-scored lambda pooled over the law statistics")
        for Z in ("E1", "E2", "E3", "E4"):
            if Z in FAMILIES[k]:
                try:
                    st, _ = m.stats_of(Z, m.depths)
                    if st:
                        lam = []
                        for s in st:
                            code, l = s["code"], s["l"]
                            fk, fv = {"W": (code[2:-1].split(",")[0],) * 2, "A_L": ("LETTER", "NONE"), "A_P": ("P1", "NONE")}[code.split("(")[0]]
                            from stage8c_parts.stats import log_ratio
                            lam.append(log_ratio(m.psi(fk, l, Z, "K", "L"), m.psi(fv, l, Z, "V", "L")).pt)
                        out(f"   {Z}: mean lambda under L {np.mean(lam):+.3f} vs E {np.mean([s['q'].pt for s in st]):+.3f}")
                except Exception as ex:  # noqa: BLE001
                    out(f"   {Z}: L lambda failed: {ex}")
    for k in MODELS:
        X = I.get("explore", k)
        if not X or "stories" not in X:
            continue
        out(f"-- {k}: greedy-generation flip rates of the KV rows at l = 7 (O4)")
        for f in ("NONE", "P1"):
            parts = []
            for inst in ["nat"] + X["instances"]:
                h = []
                for s in X["stories"]:
                    a = s["answers"][f]
                    h += [a.get(f"{inst}|S") == s["S"], a.get(f"{inst}|X") == s["X"]]
                parts.append(f"{inst} {np.mean(h):.2f}")
            out(f"   {f}: " + ", ".join(parts))
    for k in MODELS:
        C = I.get("calib", k)
        if C:
            out(f"-- {k}: frames admitted {C.get('frames')}, dropped {len(C.get('frames_dropped', []))}; E5 alpha {C.get('e5_alpha')}")
        D = I.get("das", k)
        if D:
            out(f"-- {k}: E4 fits " + "; ".join(f"{x} {v['init']} loss {v['loss_first50']:.2f}->{v['loss_last50']:.2f} THOLD flip {v['flip_THOLD']:.2f}"
                                               for x, v in D.get("fits", {}).items()))


# --------------------------------------------------------------------------- main
def headline(R, lexical_all=None):
    """The pre-committed headline reading (C-6) and the Section-5 outcome of the entry (C-12), checked in this order:
    (a) the law met on Tier 2 and J-C-BOUND met; (a') Tier 2 met, J-C-BOUND not met; (c) J-C-BOUND met without Tier 2;
    (b) every effective family acts through the natural lexical code and no PERP / NONLEX component carries identity;
    (e) J-C1 NOT MET; (d) a Tier-2 combo NOT MET (its pattern printed); (f) otherwise (Tier 0 at most)."""
    v = {k: x[0] for k, x in R.items()}
    tier2 = (v.get("J-C3") is True and v.get("J-C4") is True) or v.get("J-C5") is True
    bound = v.get("J-C-BOUND") is True
    if tier2 and bound:
        return "outcome (a): SUPPORTED OUT OF SAMPLE (the law met in E3 and E4, or in identity-carrying PERP / NONLEX, and J-C-BOUND met)"
    if tier2:
        return "outcome (a'): LAW MET ON TIER 2, BOUNDARY NOT MET (non-lexical edits are not under-read by keys)"
    if bound:
        return "outcome (c): LEXICAL BOUNDARY MET without Tier-2 support: the key reads only the token-form part of an edit"
    if lexical_all is True and not CARRIERS and v.get("J-C5") is not True:
        return "outcome (b): THE EDITS ACT ONLY THROUGH THE NATURAL LEXICAL CODE (not evidence that attribution is edit-independent)"
    if v.get("J-C1") is False:
        return "outcome (e): FAILED (even near-natural steering vectors depart from the natural channel ratio)"
    pats = {f"{c} {lab}": p for c in ("J-C3", "J-C4") for lab, p in PATTERNS.get(c, {}).items() if not p.startswith("equivalent")}
    if pats:
        return "outcome (d): DEPARTURE on Tier 2: " + "; ".join(f"{k}: {p}" for k, p in pats.items())
    return "outcome (f): " + ("CONSISTENT FOR NEAR-NATURAL STEERING VECTORS ONLY (Tier 0 met; no Tier-2 verdict)" if v.get("J-C1") is True
                              else "NOT SUPPORTED OR NOT EVALUABLE (no tier of the law met)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/gpu_stage8c")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true", help="TEST_ tags: sizes not checked, verdicts are plumbing checks")
    a = ap.parse_args(argv)
    root = Path(a.results)
    COMPONENTS.clear()
    PATTERNS.clear()
    CARRIERS.clear()
    lines, errors = [], []
    out = lines.append
    out(f"Stage 8 part C scoring, preregistration P-2026-10-10-J; results {root}" + ("; TEST MODE (verdicts are plumbing checks)" if a.test else ""))
    I = Inputs(root, a.test)
    out("")
    prov_ok = provenance(I, out)
    out("")
    pop_ok = population(I, out)
    out("")
    out("GATES (J-C-G0 before any model; per model J-C-G1 equivalence control, J-C-G2 format effect, J-C-G8 sensitivity; "
        "J-C-G3 SAE alignment; J-C-G4 cells; J-C-G5 efficacy; J-C-G6 specificity)")
    g0 = gate_g0(root, out)
    R = {}
    try:
        G = Gates(I, out, g0)
    except Exception:
        errors.append("gates")
        out("  SCORER ERROR in the gates; every line NOT EVALUABLE\n" + traceback.format_exc())
        G = None
    for name, fn in (("law", lambda: lines_law(G, I)), ("readers", lambda: j_readers(G, I)), ("prakash", lambda: j_prakash(I))):
        try:
            if G is None and name != "prakash":
                raise RuntimeError("gates failed")
            r = fn()
            if name == "readers":
                R["J-C-READa"], R["J-C-READb"] = r
            else:
                R |= r
        except Exception:
            errors.append(name)
            out(f"  SCORER ERROR ({name})\n" + traceback.format_exc())
    if g0 is not True:
        R = {k: (None, v[1], v[2] + ["    J-C-G0 not passed: NOT EVALUABLE"]) for k, v in R.items()}
    out("\nPREDICTIONS (code, class, kind, prior, verdict; the numbers it was decided on)")
    comps = []
    for code, (cls, kind, prior, title) in LINES.items():
        v, per, det = R.get(code, (None, {}, ["    not computed"]))
        out(f"  {code:11s} {cls} {kind} prior {prior:.2f}  {V(v):13s} {title}")
        for x in det:
            out(x)
    rep, lex_all = [], None
    try:
        lex_all = reported(G, I, rep.append)
    except Exception:
        rep.append("  reported lines failed:\n" + traceback.format_exc())
    out("\nHEADLINE (C-6, pre-committed; the Section-5 outcome of the entry): " + headline(R, lex_all))
    for x in rep:
        out(x)
    out("\nSUMMARY")
    for cls in ("L", "M", "R"):
        codes = [c for c, x in LINES.items() if x[0] == cls]
        vs = [R.get(c, (None,))[0] for c in codes]
        ev = [(c, v) for c, v in zip(codes, vs) if v is not None]
        exp = sum(LINES[c][2] for c, _ in ev)
        met = sum(bool(v) for _, v in ev)
        brier = np.mean([(LINES[c][2] - bool(v)) ** 2 for c, v in ev]) if ev else float("nan")
        ne = [c for c, v in zip(codes, vs) if v is None]
        out(f"  class {cls} (account lines): {len(codes)} lines; MET {met}, NOT MET {len(ev) - met}, NOT EVALUABLE {len(ne)}"
            f"{' (' + ', '.join(ne) + ')' if ne else ''}; observed {met} vs expected {exp:.2f} (sum of priors of the lines with a verdict); Brier {brier:.3f}")
    out("  measurement-validity lines: none in part C (the T control and the sensitivity rows are gates)")
    ch = holm([(f"{c}: {lab}", p, own) for c, lab, p, own in COMPONENTS], alpha=0.025)
    out(f"  Holm (sensitivity, no verdict uses it): {len(COMPONENTS)} interval components at familywise one-sided 0.025; "
        f"{len(ch)} decisions change" + "".join(f"\n    {lab}: p {p:.4f}, own rule {'rejects' if own else 'does not reject'}, Holm "
                                                 f"{'rejects' if rej else 'does not reject'}" for lab, p, own, rej in ch[:40]))
    if COMPONENTS and 1.0 / 10001 > 0.025 / len(COMPONENTS):
        out(f"    note: the smallest bootstrap p (1 / 10,001) exceeds the first Holm threshold 0.025 / {len(COMPONENTS)}, so Holm over "
            "all of the part's components cannot reject any; the list above is then uninformative by construction")
    flip = sorted({lab.split(":")[0] for lab, p, own, rej in ch if own and not rej and R.get(lab.split(":")[0], (None,))[0] is True})
    out("    verdicts that would change under Holm (MET lines with a component no longer rejected): " + (", ".join(flip) or "none"))
    out(f"  provenance {'OK' if prov_ok else 'MISMATCH'}; population {'OK' if pop_ok else 'MISMATCH'}; J-C-G0 {V(g0)}"
        + (f"; SCORER ERROR in {errors}" if errors else ""))
    try:
        exploratory(G, I, out)
    except Exception:
        out("  exploratory report failed:\n" + traceback.format_exc())
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE8C_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    has_results = bool(I.models)
    return 1 if errors else 2 if not a.test and not (prov_ok and pop_ok and (g0 or not has_results)) else 0


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
