"""Score part A of preregistration P-2026-10-10-J (docs/PREREGISTRATION.md, GPU stage 8 part A): the key/value format law
on counterfactual SQuAD passages. Gates J-A-G0 to J-A-G4 and J-A-HA-G0 to J-A-HA-G2, lines J-A1 to J-A8d and J-A-HA1 to
J-A-HA3b, as worded in the entry (part scorers in analysis/stage8a_parts).

Inputs under --results (default results/gpu_stage8a), as written by scripts/gpu_stage8a.sh: preflight.json,
frames/<tag>.json, factorial/<tag>.json and explore/<tag>.json (experiments/natural_factorial.py), heads/<tag>.json and
heads/mu_<tag>.pt (experiments/natural_heads.py), the pytest log(s) under logs/ (J-A-G0: tests/test_natural_clamp.py
and tests/test_kvquant.py; J-A-HA-G0: tests/test_natural_heads.py; the last pytest session in the log that ran them),
COMMIT.txt, ENV.txt, REVISIONS.txt, SKIPPED.txt.
Output, also written to --out (default {results}/STAGE8A_SCORE.txt), in this order: PROVENANCE (every line of the
pipeline's files; per results file commit, model, revision, dtype, attention, device, versions; the frames, items,
SQuAD, stage-6 and mu hashes; MISMATCH when the files hold more than one commit, a hash differs, or outside TEST a dtype
or attention other than the entry's), POPULATION, GATES, PREDICTIONS (one line per line: code, class, prior, verdict,
then each model's numbers, bounds and evaluability), REPORTED, SUMMARY (per class: lines, MET, NOT MET, NOT
EVALUABLE with reasons, the observed met count against the sum of the priors of the lines with a verdict, the Brier
score; the Holm sensitivity analysis of the R-class lines by analysis/stage8_holm.py, per line), EXPLORATORY.
Combination (the entry's common rule): a J-A line is NOT MET as soon as one evaluable model does not meet it, whatever
the number of evaluable models; else MET if >= 3 models are evaluable including >= 1 fresh family (Llama, Gemma, or the
Yi fallback), else NOT EVALUABLE; a head line is NOT MET if not met in Qwen2.5-7B or in Mistral-7B, MET if met in both,
else NOT EVALUABLE. J-A3 is derived and not counted.
--test (TEST_MODE; also when every results file is tagged TEST_): the size floors are waived and the verdicts are
plumbing checks, not results.
Exit status 1 if a part raised (traceback in the report, its lines NOT EVALUABLE); 2 if, outside TEST, the provenance or
population check reports MISMATCH (the file is still written).
"""
import argparse
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage8a_parts import common as cm  # noqa: E402
from stage8a_parts import factorial as fa  # noqa: E402
from stage8a_parts import heads as hd  # noqa: E402
from stage8_holm import holm as holm_j  # noqa: E402  (entry J's shared Holm sensitivity helper, decision D2)

ITEMS = HERE.parent / "data" / "stage8a_items.json"
SQUAD_SHA = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
STAGE6_SHA = {"Qwen2.5-7B-Instruct.json": "ed828a9b701d60f552eb8dc10247d85de44364b75f6086e5abe4eedb637353be",
              "Mistral-7B-Instruct-v0.3.json": "88ababd91bdb3155ef8e4aa691eff30d9edf88a13260148c145e654ed37e69af"}
A_G0 = {"tests/test_natural_clamp.py": 14, "tests/test_kvquant.py": 4}
HA_G0 = {"tests/test_natural_heads.py": 5}
MODEL_ORDER = ("llama8", "gemma9", "yi9", "qwen7", "mistral7")


def load_json(f):
    try:
        return json.load(open(f))
    except Exception as ex:  # a truncated file is reported, not fatal
        return {"error": repr(ex)}


def sha_file(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def keyof(tag):
    return tag[5:] if tag.startswith("TEST_") else tag


def order(keys):
    return sorted(keys, key=lambda k: (MODEL_ORDER.index(k) if k in MODEL_ORDER else 99, k))


# --------------------------------------------------------------------------- pytest gates
def pytest_counts(root, files):
    """{file: (passed, failed or error, skipped)} from the last pytest session (in any log under root) that ran these
    files; verbose lines "path::test PASSED" and -rA lines "PASSED path::test" are both read."""
    logs = sorted(list((root / "logs").glob("*pytest*")) + list(root.glob("log_pytest*.txt"))) if root.exists() else []
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
        st = {}
        for m in re.finditer(rf"({re.escape(x)}::\S+) (PASSED|FAILED|ERROR|SKIPPED)|(PASSED|FAILED|ERROR|SKIPPED) ({re.escape(x)}::\S+)", best[1]):
            tid, w = (m.group(1), m.group(2)) if m.group(1) else (m.group(4), m.group(3))
            st[tid] = w
        res[x] = (sum(w == "PASSED" for w in st.values()), sum(w in ("FAILED", "ERROR") for w in st.values()), sum(w == "SKIPPED" for w in st.values()))
    return best[0], res


def gate_tests(root, files, name, out):
    log, res = pytest_counts(root, files)
    if log is None:
        out(f"  {name}  FP32 exactness ({', '.join(files)}): no pytest log in the results -> NOT EVALUABLE (not run)")
        return None
    ok = all(res[f][0] >= n and res[f][1] == 0 and res[f][2] == 0 for f, n in files.items())
    out(f"  {name}  FP32 exactness ({log}, its last session that ran them): " + "; ".join(
        f"{f} {p} passed, {x} failed, {s} skipped (>= {files[f]}, none failing or skipped)" for f, (p, x, s) in res.items()) + f" -> {cm.V(ok)}")
    return ok


# --------------------------------------------------------------------------- provenance and population
def provenance(root, F, out, test):
    out("PROVENANCE (the pipeline's COMMIT.txt, ENV.txt, REVISIONS.txt, SKIPPED.txt, every line; every results file)")
    for name in ("COMMIT.txt", "ENV.txt", "REVISIONS.txt", "SKIPPED.txt", "FETCH_FAILED.txt"):
        f = root / name
        out(f"  {name}:" + ("".join(f"\n    {l.rstrip()}" for l in f.read_text().splitlines() if l.strip()) if f.exists() else " absent"))
    ok, commits, bad = True, set(), []
    items_sha = sha_file(ITEMS)
    pre = F.get("preflight.json")
    if pre is None:
        out("  preflight.json: absent" + ("" if test else "  MISMATCH (the item rebuild is part of the entry)"))
        ok &= test
    elif "error" in pre:
        out(f"  preflight.json: UNREADABLE {pre['error']}  MISMATCH")
        bad.append("preflight.json")
    else:
        good = pre.get("squad_sha256") == SQUAD_SHA and pre.get("rebuild_identical") and pre.get("items_sha256") == items_sha
        ok &= bool(good)
        commits.add(pre.get("provenance", {}).get("git_commit"))
        out(f"  preflight.json: SQuAD sha256 {str(pre.get('squad_sha256'))[:12]}, rebuild identical {pre.get('rebuild_identical')}, items sha256 "
            f"{str(pre.get('items_sha256'))[:12]} (committed {items_sha[:12]})" + ("" if good else "  MISMATCH"))
    frames = {lab.split("/")[1][:-5]: (lab, J) for lab, J in F.items() if lab.startswith("frames/")}
    for lab, J in sorted(F.items()):
        if lab == "preflight.json" or J is None:
            continue
        if "error" in J or "provenance" not in J:
            out(f"  {lab}: " + (f"UNREADABLE {J['error']}" if "error" in J else "NO PROVENANCE") + "  MISMATCH")
            bad.append(lab)
            continue
        p = J["provenance"]
        commits.add(p.get("git_commit"))
        tag = Path(lab).stem
        key = keyof(tag)
        flag = []
        if p.get("items_sha256") not in (None, items_sha):
            flag.append("items sha256 differs from data/stage8a_items.json")
        if not lab.startswith("frames/") and tag in frames and "frames_sha256" in p:
            if p["frames_sha256"] != sha_file(root / frames[tag][0]):
                flag.append("frames file differs from the one this file used")
        if not test:
            want = "eager" if lab.startswith("heads/") or key.startswith("gemma") else "sdpa"
            if not (str(p.get("dtype", "")).endswith("bfloat16") and p.get("attn_implementation") == want):
                flag.append(f"the entry fixes BF16 and {want} attention")
            if str(tag).startswith("TEST_") or p.get("test_mode"):
                flag.append("a TEST_MODE file outside TEST")
        if lab.startswith("heads/"):
            st = p.get("stage6", {})
            if STAGE6_SHA.get(st.get("file")) != st.get("sha256"):
                flag.append("stage-6 template file hash")
            if "sets" in J and canon(J["sets"]) != J.get("sets_sha256"):
                flag.append("head sets changed after hashing")
            mu = root / "heads" / J.get("rank", {}).get("mu_file", "")
            if J.get("rank") and (not mu.is_file() or sha_file(mu) != J["rank"].get("mu_sha256")):
                flag.append("mu file hash")
        ok &= not flag
        ver = p.get("verified") or {}
        out(f"  {lab}: commit {str(p.get('git_commit'))[:10]}, model {p.get('model')} (key {p.get('model_key')}), revision "
            f"{p.get('revision') or ver.get('revision')}, dtype {p.get('dtype', '-')}, attn {p.get('attn_implementation', '-')}, "
            f"device {p.get('device', '-')}, transformers {p.get('transformers')}, torch {p.get('torch')}, frame {p.get('frame', J.get('frame'))!r}, "
            f"wrapper {p.get('wrapper')}, test_mode {p.get('test_mode')}" + "".join(f"  MISMATCH: {x}" for x in flag))
    out(f"  commits: {sorted(str(c)[:10] for c in commits)}" + ("  MISMATCH: not one commit" if len(commits) > 1 else "")
        + ("; UNREADABLE or without provenance: " + ", ".join(bad) if bad else ""))
    return ok and len(commits) <= 1 and not bad


def population(F, models, heads, out, test):
    out("POPULATION (" + ("TEST: sizes not checked; " if test else "") + "E items of data/stage8a_items.json valid for each model's "
        "tokenizer and frame, the same items in every format; heads: the first min(60, n) of the n valid R items and the "
        "first min(80, n) of the n valid E items, valid in NOM and OPTA under the model's frame, n from provenance n_valid)")
    ok = True
    items = json.load(open(ITEMS))
    E = {i["id"] for i in items if i["split"] == "E"}
    Rr = [i["id"] for i in sorted(items, key=lambda i: i["rank"]) if i["split"] == "R"]
    for k, m in models.items():
        p = m.P
        valid = sorted(next(iter(m.recs.values()), {}))
        sk = p.get("skipped_items", [])
        same = all(sorted(m.recs[f]) == valid for f in m.recs)
        inE = set(valid) <= E
        full = test or len(valid) + len(sk) == len(E)
        shaok = p.get("population_sha256") == hashlib.sha256("\n".join(sorted(valid)).encode()).hexdigest()
        good = same and inE and full and shaok
        ok &= good
        why = {}
        for s in sk:
            why[s["reason"]] = why.get(s["reason"], 0) + 1
        out(f"  {k}: {len(valid)} valid E items, {len(sk)} skipped (" + "; ".join(f"{r}: {n}" for r, n in why.items()) + ")"
            f"; formats {sorted(m.recs)}; same items in every format {'OK' if same else 'MISMATCH'}; all from E {'OK' if inE else 'MISMATCH'}; "
            f"valid + skipped = |E| {'OK' if full else 'MISMATCH'}; population sha256 {'OK' if shaok else 'MISMATCH'}"
            + (f"; reduced at the deadline: {p.get('reduced')}" if p.get("reduced") else ""))
    for k, h in heads.items():
        rk = [r["id"] for r in h.H.get("rank", {}).get("items", [])]
        inR = set(rk) <= set(Rr) and [i for i in Rr if i in set(rk)] == rk
        nv = h.P.get("n_valid", {})   # without the record: exactly 60 and 80
        want = (min(60, nv.get("R", 60)), min(80, nv.get("E", 80)))
        n_ok = test or (len(rk), len(h.E)) == want
        good = inR and n_ok and all(e["id"] in E for e in h.E)
        ok &= good
        out(f"  heads {k}: ranking items {len(rk)} (R, in rank order: {'OK' if inR else 'MISMATCH'}), evaluation items {len(h.E)} "
            f"(expected {want[0]} and {want[1]}; " + (f"valid R {nv.get('R')}, valid E {nv.get('E')}" if nv else "no n_valid record")
            + f"), ablation items OPTA {len(h.abl['OPTA'])} NOM {len(h.abl['NOM'])}" + ("" if good else "  MISMATCH"))
    out(f"  population: {'OK' if ok else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------- main
def load(root):
    F = {}
    if (root / "preflight.json").exists():
        F["preflight.json"] = load_json(root / "preflight.json")
    for d in ("frames", "factorial", "explore", "heads"):
        for f in sorted((root / d).glob("*.json")) if (root / d).is_dir() else []:
            F[f"{d}/{f.name}"] = load_json(f)
    return F


def score_lines(models, heads, out, res):
    """PREDICTIONS: every line in the entry's order; res[code] = (combined verdict, {model: Res})."""
    out("PREDICTIONS (code [class, recorded prior P(met)] title -> verdict; then per model: numbers, bounds, verdict)")
    for code in cm.ORDER:
        cls, prior, title = cm.LINES[code]
        if code.startswith("J-A-HA"):
            per = {}
            for k in order(heads):
                try:
                    per[k] = hd.HEAD_FNS[code](heads[k])
                except Exception as ex:  # noqa: BLE001  (a missing record makes this model's line not evaluable)
                    per[k] = fa.Res(None, f"not computable: {type(ex).__name__}: {ex}", [], "error")
            comb = cm.comb_both({k: r.ok for k, r in per.items()})
            n_ev = sum(r.ok is not None for r in per.values())
        else:
            per = {}
            for k in order(models):
                try:
                    per[k] = fa.LINE_FNS[code](models[k])
                except Exception as ex:  # noqa: BLE001
                    per[k] = fa.Res(None, f"not computable: {type(ex).__name__}: {ex}", [], "error")
            comb = cm.comb_models({k: r.ok for k, r in per.items()})
            n_ev = sum(r.ok is not None for r in per.values())
        res[code] = (comb, per)
        n_met = sum(r.ok is True for r in per.values())
        tag = " (derived; not counted)" if cls == "D" else ""
        out(f"  {code:9s} [{cls}, prior {prior:.2f}] {title} -> {cm.V(comb)}{tag} ({n_met} of {n_ev} evaluable models met; {len(per)} run)")
        for k, r in per.items():
            out(f"      {k}: {r.txt} -> {cm.V(r.ok)}" + (f"  [not evaluable: {r.why}]" if r.ok is None and r.why else ""))
            if r.ok is not None:
                out("          " + "; ".join(str(c) for c in r.comps))


def holm_family(res):
    """The Holm family of entry J (D2): every interval component (one bound of one statistic in one model) of Part A's
    R-class account lines, in every model where the line has a verdict. Returns the helper's components and, for each,
    (line, model, decision under the interval rule)."""
    comps, where = [], []
    for code in cm.ORDER:
        if code not in res or cm.LINES[code][0] != "R":
            continue
        for k, r in res[code][1].items():
            if r.ok is None:
                continue
            for c in r.comps:
                for name, passed, t in c.tests:
                    # a degenerate bootstrap (every resample equal, se 0) is passed as se 1e-12: p is then 0 or 1 by the
                    # side of the bound the estimate lies on (0.5 on the bound), the normal approximation's limit
                    se = t["se"] if not (t["se"] == 0) else 1e-12
                    comps.append(dict(t, se=se, line=code, name=f"{k}: {name}"))
                    where.append((code, k, passed))
    return comps, where


def summary(res, out):
    out("\nSUMMARY (G4: per class; all part-A lines are account lines, none is a measurement-validity line; J-A3 derived and "
        "not counted; expected = the sum of the recorded priors of the lines with a verdict; Brier = mean (prior - 1[MET])^2 "
        "over the same lines; NOT EVALUABLE lines are left out of both and listed)")
    for cls in ("L", "M", "R"):
        codes = [c for c in cm.ORDER if c in res and cm.LINES[c][0] == cls]
        ev = [c for c in codes if res[c][0] is not None]
        met = sum(res[c][0] is True for c in ev)
        exp = sum(cm.LINES[c][1] for c in ev)
        brier = float(np.mean([(cm.LINES[c][1] - (res[c][0] is True)) ** 2 for c in ev])) if ev else float("nan")
        ne = [c for c in codes if res[c][0] is None]
        why = ["{} ({})".format(c, "; ".join(f"{k}: {r.why}" for k, r in res[c][1].items() if r.ok is None) or "no model run")
               for c in ne]
        out(f"  account lines, class {cls}: {len(codes)} lines: {met} MET, {sum(res[c][0] is False for c in ev)} NOT MET, "
            f"0 MET IN PART, {len(ne)} NOT EVALUABLE; observed {met} against expected {exp:.2f}; Brier {brier:.3f}"
            + (f"; not evaluable: {' | '.join(why)}" if ne else ""))
    R = [c for c in cm.ORDER if c in res and cm.LINES[c][0] == "R" and res[c][0] is not None]
    out(f"  met rate among R account lines with a verdict: {sum(res[c][0] is True for c in R)} of {len(R)} "
        f"(expected {sum(cm.LINES[c][1] for c in R):.2f})")
    comps, where = holm_family(res)
    try:
        hres = holm_j(comps) if comps else []
    except Exception:  # noqa: BLE001  (the sensitivity analysis never stops the report)
        out("  Holm sensitivity: the helper failed\n" + traceback.format_exc())
        return
    hdec = {}
    changed = {}
    for (code, k, passed), h in zip(where, hres):
        hdec.setdefault((code, k), []).append(bool(h["reject"]))
        if bool(h["reject"]) != passed:
            changed.setdefault(code, []).append(f"{h['name']} (p {h['p']:.2g}, Holm threshold {h['threshold']:.2g}; interval "
                                                f"rule {'rejects' if passed else 'does not reject'}, Holm "
                                                f"{'rejects' if h['reject'] else 'does not reject'})")
    out(f"  Holm sensitivity (analysis/stage8_holm.py; reported, no verdict uses it): step-down at one-sided 0.025 over the "
        f"{len(comps)} interval components of the R-class account lines in the models where the line has a verdict; "
        f"p = Phi(-(est - bound)/se) for '>' and Phi((est - bound)/se) for '<', se the bootstrap standard deviation")
    for code in cm.ORDER:
        if code not in res or cm.LINES[code][0] != "R":
            continue
        v, per = res[code]
        new = {}
        for k, r in per.items():
            if r.ok is None:
                new[k] = None
            elif r.undef:
                new[k] = False
            else:
                pts = all(c.passed for c in r.comps if not c.tests)
                new[k] = pts and all(hdec.get((code, k), [True]))
        nv = cm.comb_both(new) if code.startswith("J-A-HA") else cm.comb_models(new)
        ch = changed.get(code, [])
        out(f"    {code}: " + (f"{len(ch)} component decision(s) change under Holm: " + "; ".join(ch) if ch
                               else "no component decision changes under Holm")
            + (f"; verdict {cm.V(v)} -> {cm.V(nv)} under Holm" if nv != v else f"; verdict unchanged ({cm.V(v)})"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", "--root", dest="results", default="results/gpu_stage8a")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: size floors waived; verdicts are plumbing checks")
    ap.add_argument("--out", default=None, help="default {results}/STAGE8A_SCORE.txt")
    a = ap.parse_args(argv)
    root = Path(a.results)
    F = load(root)
    tags = [Path(k).stem for k in F if k.startswith(("factorial/", "heads/"))]
    test = a.test or (bool(tags) and all(t.startswith("TEST_") for t in tags))
    lines, errors = [], []
    out = lines.append
    out(f"Stage 8 part A scoring, preregistration P-2026-10-10-J part A; results {root}")
    if test:
        out("TEST MODE: the size floors are waived and the verdict lines below are plumbing checks, not results")
    out("")
    prov_ok = provenance(root, F, out, test)
    out("")
    gl = []
    a_g0 = gate_tests(root, A_G0, "J-A-G0", gl.append)
    ha_g0 = gate_tests(root, HA_G0, "J-A-HA-G0", gl.append)
    models, heads = {}, {}
    for lab, J in F.items():
        if lab.startswith("factorial/") and J and "formats" in J:
            k = keyof(Path(lab).stem)
            X = F.get(f"explore/{Path(lab).name}")
            models[k] = fa.Model(k, J, X if X and "parts" in X else None, test, bool(a_g0) or test and a_g0 is None)
    for lab, J in F.items():
        if lab.startswith("heads/") and J and "eval" in J:
            k = keyof(Path(lab).stem)
            heads[k] = hd.Heads(k, J, models.get(k), test, bool(ha_g0) or test and ha_g0 is None)
    pop_ok = population(F, models, heads, out, test)
    out("")
    out("GATES (J-A-G0 and J-A-HA-G0 before any model; J-A-G1 BF16 floor, J-A-G2 competence, J-A-G3 emitted form per model and "
        "format; J-A-G4 per line and cell, printed with the line; J-A-HA-G1 floor, J-A-HA-G2 option rows are the readers)")
    for x in gl:
        out(x)
    if test and a_g0 is None:
        out("  (TEST: no pytest log; J-A-G0 and J-A-HA-G0 treated as passed for the plumbing check)")
    for k in order(models):
        m = models[k]
        for f in fa.FORMATS:
            if f in m.gates:
                g = m.gates[f]
                out(f"  {k:8s} {f:4s} J-A-G1 {g['txt']['G1']} -> {cm.V(g['G1'])}")
                out(f"  {k:8s} {f:4s} J-A-G2 {g['txt']['G2']} -> {cm.V(g['G2'])}")
                out(f"  {k:8s} {f:4s} J-A-G3 {g['txt']['G3']} -> {cm.V(g['G3'])}")
            else:
                out(f"  {k:8s} {f:4s} not run" + (f" ({m.P.get('reduced', {}).get(f)})" if m.P.get("reduced", {}).get(f) else ""))
    for k in order(heads):
        h = heads[k]
        for name, fn in (("J-A-HA-G1", h.gate1), ("J-A-HA-G2", h.gate2)):
            try:
                okg, txt = fn()
            except Exception as ex:  # noqa: BLE001
                okg, txt = None, f"not computable: {ex}"
            out(f"  {k:8s} {name} {txt} -> {cm.V(okg)}")
    out("")
    res = {}
    try:
        score_lines(models, heads, out, res)
    except Exception:
        errors.append("lines")
        out("  SCORER ERROR; the lines are NOT EVALUABLE\n" + traceback.format_exc())
    out("\nREPORTED (no verdicts)")
    for k in order(models):
        try:
            s_opta = None
            if models[k].has("OPTA"):
                ids = models[k].C("OPTA")
                kk, vv = models[k].idkv("OPTA", ids) if ids else (np.zeros(0), np.zeros(0))
                s_opta = cm.est(models[k].arts(ids), cm.sid, kk, vv) if ids else None
            fa.reported(models[k], out, s_opta)
        except Exception:  # noqa: BLE001
            out(f"  [{k}] reported lines failed:\n" + traceback.format_exc())
    for k in order(heads):
        try:
            hd.reported(heads[k], out)
        except Exception:  # noqa: BLE001
            out(f"  [{k}] head reported lines failed:\n" + traceback.format_exc())
    if res:
        summary(res, out)
    n = lambda v: sum(x[0] is v for c, x in res.items() if cm.LINES[c][0] != "D")  # noqa: E731
    out(f"  overall: {n(True)} MET, {n(False)} NOT MET, {n(None)} NOT EVALUABLE of {sum(cm.LINES[c][0] != 'D' for c in res)} counted lines; "
        f"provenance {'OK' if prov_ok else 'MISMATCH'}; population {'OK' if pop_ok else 'MISMATCH'}" + (f"; SCORER ERROR in {errors}" if errors else ""))
    out("\n######## EXPLORATORY (not scored)")
    for k in order(models):
        try:
            fa.exploratory(models[k], out)
        except Exception:  # noqa: BLE001
            out(f"  [{k}] exploratory report failed:\n" + traceback.format_exc())
    for k in order(heads):
        try:
            out(f"  heads [{k}]")
            hd.exploratory(heads[k], out)
        except Exception:  # noqa: BLE001
            out(f"  [{k}] head exploratory report failed:\n" + traceback.format_exc())
    sOP = {k: (lambda m: (lambda ids: cm.sid(*[x.mean() for x in m.idkv("OPTA", ids)]) if ids else float("nan"))(m.C("OPTA")))(models[k])
           for k in order(models) if models[k].has("OPTA")}
    if sOP:
        out("  s_ID(OPTA) per model: " + ", ".join(f"{k} {v:+.3f}" for k, v in sOP.items()) + f"; unweighted mean {np.nanmean(list(sOP.values())):+.3f}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE8A_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return 1 if errors else 2 if not test and not (prov_ok and pop_ok) else 0


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
