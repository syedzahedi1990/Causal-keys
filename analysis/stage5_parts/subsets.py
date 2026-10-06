"""Part (c) of P-2026-10-05-G (docs/PREREGISTRATION.md): the shared Gate 0 and predictions G9-G12 on the stage-5 factorial
({root}/factorial/{model}_s{seed}.json from experiments/format_factorial.py with --arm-modules ckeys.subsets) and the
row-restricted splice ({root}/row_restricted/{model}_direct.json), exactly as drafted. Called by analysis/stage5_score.py
(``score(root)``) or standalone; ``gate0`` is shared with part (d).

Per core (analysis/stage1_prereg.per_core): ID_K, ID_V, d_C against the batched self-clamp row, keys/values clamped
from layer 0. Arms: sentence family S2 {S,X}, S3 {B,S,X}, S3out {B,o1,o2}, S3half, S4, S4out, S6 (= POST byte for
byte); list family L2, L3, L3out, L4, L4out, L6 (= AFTER); NONE. S6/L6 and POST/AFTER stand in for each other when only
one of the pair was run. Means over cores; 95 % percentile CIs from 10,000 core-bootstrap resamples with one fixed
index set per n, so paired differences and ratios of means are recomputed within the same resamples.
Gate 0 (per model): skipped_items = 0; |ID_K - stage 3b| <= max(0.5, 0.10 x |ref|) for POST, NONE and AFTER; lower CI
  bounds of ID_K(POST) and ID_K(AFTER) > 0. A model without a reference is "no reference" (passes only in --test).
  A model failing Gate 0 is not evaluable in (c) and (d); not evaluable counts as not met in every k/k line.
G9  per family F in {S, L}: (a) paired ID_K(F3) - ID_K(NONE) > 0 (CI excl. 0) and ID_K(F3)/ID_K(F6) >= 0.5;
    (b) paired ID_K(F3) - ID_K(F3out) > 0 (CI excl. 0) and the upper bound of paired ID_K(F3out) - ID_K(NONE) <= 1.0.
G10 per family: lower bound of r_2 = ID_K(F2)/ID_K(F6) > 1/3 and of r_3 > 1/2; r_2, r_3, r_4 reported against 0.5.
G11 paired ID_V(L3out) - ID_V(L3) > 0 (CI excl. 0) and R_V = [ID_V(L3out) - ID_V(L3)] / [ID_V(NONE) - ID_V(L3)] >= 0.5;
    evaluable only if mean ID_V(NONE) - mean ID_V(L3) >= 2.0 nats.
G12 splice (n = 60): f_words = mean d(mention_words)/mean d(all) >= 0.5 with lower bound > 0.25 in S2, S3, L2, L3,
    12/12 arm-model cells; validity per cell |mean d(none)| <= 0.01 and mean d(all) >= 2.0 nats.
Each of G9-G12 needs 3/3 primary models (Qwen2.5-7B, 14B, Mistral-7B); seed 1 is reported as "replicated in k/3".
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_prereg import per_core  # noqa: E402

from ckeys.story import pick_x  # noqa: E402
from ckeys.subsets import named_set  # noqa: E402

FACTORIAL, ROWS = "factorial", "row_restricted"
PRIMARY = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct", "Mistral-7B-Instruct-v0.3"]
EXPLORATORY = ["Qwen3-8B", "OLMo-2-1124-7B-Instruct"]
# stage 3b (results/gpu_stage3b/format_2x2, seed 0, n = 150): ID_K under POST, NONE, AFTER
REF3B = {"Qwen2.5-7B-Instruct": {"POST": 5.52, "NONE": 1.04, "AFTER": 20.60},
         "Qwen2.5-14B-Instruct": {"POST": 12.08, "NONE": 1.05, "AFTER": 35.81},
         "Mistral-7B-Instruct-v0.3": {"POST": 7.87, "NONE": 0.55, "AFTER": 15.55},
         "OLMo-2-1124-7B-Instruct": {"POST": 2.01, "NONE": 0.38, "AFTER": 8.65}}
SENT = ["S2", "S3", "S3out", "S3half", "S4", "S4out", "S6"]
LIST = ["L2", "L3", "L3out", "L4", "L4out", "L6"]
ALIAS = {"S6": "POST", "L6": "AFTER", "POST": "S6", "AFTER": "L6"}
SPLICE_ARMS, SPLICE_EXTRA = ["S2", "S3", "L2", "L3"], ["S6", "L6", "S3out", "L3out"]
SEED, B = 20261005, 10000
_idx = {}
NAN = (float("nan"),) * 3


def IDX(n):
    if n not in _idx:
        _idx[n] = np.random.default_rng(SEED).integers(0, n, (B, n))
    return _idx[n]


def boot(x):
    x = np.asarray(x, float)
    if not len(x):
        return NAN
    bs = x[IDX(len(x))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ratio(a, b):
    """Ratio of means over the same cores, recomputed in each resample."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return NAN
    idx = IDX(len(a))
    r = a[idx].mean(1) / b[idx].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def fmt(t, d=2):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f},{t[2]:+.{d}f}]"


def verdict(ok):
    return "MET" if ok else "NOT MET"


def vec(A, ids, key="idK"):
    return np.array([A[i][key] for i in ids])


def load_factorial(root, model, seed=0):
    """(provenance, {arm: per_core}) with S6/L6 and POST/AFTER aliased to each other when only one was run."""
    f = Path(root) / FACTORIAL / f"{model}_s{seed}.json"
    if not f.exists():
        return None, {}
    d = json.load(open(f))
    arms = {a: per_core(d["results"], a) for a in sorted({r["arm"] for r in d["results"]})}
    arms = {a: v for a, v in arms.items() if v}
    for a, b in ALIAS.items():
        if a not in arms and b in arms:
            arms[a] = arms[b]
    return d.get("provenance", {}), arms


def gate0(prov, arms, model, out=print, test=False):
    """The shared reproduction gate; returns True / False / None (no reference, passes only in --test)."""
    skipped = prov.get("skipped_items", "?")
    ref = REF3B.get(model)
    checks = [("skipped_items == 0", skipped == 0, f"skipped_items {skipped}")]
    for a in ("POST", "NONE", "AFTER"):
        if a not in arms:
            checks.append((f"{a} present", False, f"{a} missing"))
            continue
        t = boot([v["idK"] for v in arms[a].values()])
        if ref:
            tol = max(0.5, 0.10 * abs(ref[a]))
            checks.append((f"|ID_K({a}) - 3b| <= {tol:.2f}", abs(t[0] - ref[a]) <= tol, f"ID_K({a}) {fmt(t)} vs stage 3b {ref[a]:+.2f}"))
        if a != "NONE":
            checks.append((f"lower CI of ID_K({a}) > 0", t[1] > 0, f"ID_K({a}) {fmt(t)}"))
    for name, ok, txt in checks:
        out(f"  gate 0 [{name:28s}] {txt} -> {'pass' if ok else 'FAIL'}")
    passed = all(ok for _, ok, _ in checks)
    if ref is None:
        out(f"  gate 0: no stage-3b reference for {model}" + (" (TEST MODE: reproduction part skipped)" if test else " -> FAIL"))
        return (passed or None) if test else False
    out(f"  Gate 0 -> {'passed' if passed else 'FAILED (not evaluable in parts c and d; counts as not met)'}")
    return passed


def paired(A, Bm, ids, key="idK"):
    return boot(vec(A, ids, key) - vec(Bm, ids, key))


def table(arms, ids, out):
    out(f"  {'arm':7s} {'ID_K':>22s} {'ID_V':>22s} {'d_K':>7s} {'d_V':>7s} {'s_K':>5s} {'s_ID':>5s}")
    for a in ["NONE"] + SENT + LIST:
        if a not in arms:
            continue
        k, v = boot(vec(arms[a], ids)), boot(vec(arms[a], ids, "idV"))
        dK, dV = np.mean([arms[a][i]["d"]["K_S@0"] for i in ids]), np.mean([arms[a][i]["d"]["V_S@0"] for i in ids])
        out(f"  {a:7s} {fmt(k):>22s} {fmt(v):>22s} {dK:+7.2f} {dV:+7.2f} {dK / (dK + dV):5.2f} {k[0] / (k[0] + v[0]):5.2f}")


def predictions(arms, ids, out, label=""):
    """G9-G11 on one factorial file; returns {name: bool | None (not evaluable)}."""
    res = {}
    for F in ("S", "L"):
        k2, k3, k3o, k4, k6 = (F + s for s in ("2", "3", "3out", "4", "6"))
        if any(a not in arms for a in (k2, k3, k3o, k4, k6, "NONE")):
            out(f"  G9/G10 {F}: arm missing (not met)")
            res[f"G9 {F}"] = res[f"G10 {F}"] = False
            continue
        d_in, r3 = paired(arms[k3], arms["NONE"], ids), ratio(vec(arms[k3], ids), vec(arms[k6], ids))
        d_io, d_on = paired(arms[k3], arms[k3o], ids), paired(arms[k3o], arms["NONE"], ids)
        a_ok, b_ok = bool(d_in[1] > 0 and r3[0] >= 0.5), bool(d_io[1] > 0 and d_on[2] <= 1.0)
        res[f"G9 {F}"] = a_ok and b_ok
        out(f"  G9{label} {F} membership: (a) {k3}-NONE {fmt(d_in)} > 0, {k3}/{k6} {fmt(r3)} >= 0.5 -> {a_ok}; "
            f"(b) {k3}-{k3o} {fmt(d_io)} > 0, {k3o}-NONE {fmt(d_on)} upper <= 1.0 -> {b_ok}  => {verdict(res[f'G9 {F}'])}")
        rec = ratio(vec(arms[k3o], ids) - vec(arms["NONE"], ids), vec(arms[k3], ids) - vec(arms["NONE"], ids))
        out(f"      exploratory (no preregistered criterion): ID_K({k3o}) {fmt(boot(vec(arms[k3o], ids)))}; recovered fraction "
            f"[{k3o}-NONE]/[{k3}-NONE] {fmt(rec)}")
        rs = {k: ratio(vec(arms[k], ids), vec(arms[k6], ids)) for k in (k2, k3, k4)}
        res[f"G10 {F}"] = bool(rs[k2][1] > 1 / 3 and rs[k3][1] > 1 / 2)
        out(f"  G10{label} {F} proportionality refuted: r_2 {fmt(rs[k2])} lower > 1/3 -> {rs[k2][1] > 1 / 3}; r_3 {fmt(rs[k3])} lower > 1/2 -> "
            f"{rs[k3][1] > 1 / 2}  => {verdict(res[f'G10 {F}'])};  ladder r_2 {rs[k2][0]:.2f} r_3 {rs[k3][0]:.2f} r_4 {rs[k4][0]:.2f} (reference 0.5)")
        out(f"      steps (paired ID_K): {k3}-{k2} {fmt(paired(arms[k3], arms[k2], ids))}  {k4}-{k3} {fmt(paired(arms[k4], arms[k3], ids))}  "
            f"{k6}-{k4} {fmt(paired(arms[k6], arms[k4], ids))}")
    if all(a in arms for a in ("L3", "L3out", "NONE")):
        gap = np.mean(vec(arms["NONE"], ids, "idV")) - np.mean(vec(arms["L3"], ids, "idV"))
        d = paired(arms["L3out"], arms["L3"], ids, "idV")
        RV = ratio(vec(arms["L3out"], ids, "idV") - vec(arms["L3"], ids, "idV"), vec(arms["NONE"], ids, "idV") - vec(arms["L3"], ids, "idV"))
        ok = bool(d[1] > 0 and RV[0] >= 0.5)
        res["G11"] = ok if gap >= 2.0 else None
        out(f"  G11{label} copy carries the identity again: ID_V(L3out)-ID_V(L3) {fmt(d)} > 0 -> {d[1] > 0}; R_V {fmt(RV)} >= 0.5 -> {RV[0] >= 0.5}; "
            f"gap ID_V(NONE)-ID_V(L3) {gap:+.2f} >= 2.0 -> {gap >= 2.0}  => {verdict(ok) if gap >= 2.0 else 'NOT EVALUABLE (gap < 2.0; counts as not met)'}")
    else:
        res["G11"] = False
        out(f"  G11{label}: arm missing (not met)")
    return res


def exploratory(arms, ids, out):
    out("  exploratory:")
    if "S3half" in arms and "S3" in arms:
        out(f"    S3half/S3 {fmt(ratio(vec(arms['S3half'], ids), vec(arms['S3'], ids)))} (heuristic expectation about 0.5)")
    for a, b in (("S4", "S4out"), ("L4", "L4out")):
        if a in arms and b in arms:
            dd = boot([arms[a][i]["d"]["K_S@0"] - arms[b][i]["d"]["K_S@0"] for i in ids])
            out(f"    B named vs not at k = 4: ID_K({a})-ID_K({b}) {fmt(paired(arms[a], arms[b], ids))};  d_K difference {fmt(dd)}")
    for a in ("S3out", "L3out"):
        if a in arms and "NONE" in arms:
            d = vec(arms[a], ids) - vec(arms["NONE"], ids)
            cores = [json.loads(i) for i in ids]
            strata = {}
            for name, key in (("initial", "initial"), ("distractor loc", "distractor_location")):
                present = np.array([c[key] in named_set(a, c, pick_x(c)) for c in cores])
                strata[name] = (boot(d[present]), boot(d[~present]), int(present.sum()))
            out(f"    {a}-NONE (paired ID_K) {fmt(boot(d))};  " + "; ".join(
                f"{{o1,o2}} has {k}: {fmt(v[0])} (n={v[2]}) vs not {fmt(v[1])}" for k, v in strata.items()))
    if "S3out" in arms and "S3" in arms:
        out(f"    sentence value complement ID_V(S3out)-ID_V(S3) {fmt(paired(arms['S3out'], arms['S3'], ids, 'idV'))}")
    for F, k6 in (("S", "S6"), ("L", "L6")):
        if k6 in arms and ALIAS[k6] in arms and arms[k6] is not arms[ALIAS[k6]]:
            out(f"    within-run reproduction {k6} vs {ALIAS[k6]} (identical prompts): paired ID_K {fmt(paired(arms[k6], arms[ALIAS[k6]], ids))}")


def diagnostics(res, arms, ids, out):
    out("  diagnostics (clean B run): " + "  ".join(
        f"{a} mass {np.mean([r['clean']['B']['mass'] for r in res if r['arm'] == a]):.2f} argmax-B "
        f"{np.mean([r['clean']['B']['argmax_cand'] == r['core']['base'] for r in res if r['arm'] == a]):.2f}"
        for a in ["NONE"] + SENT + LIST if a in arms and any(r["arm"] == a for r in res)))


def splice(root, model, arms, out):
    """G12 cells of one model: {arm: bool | None}; S6/L6 accept the legacy POST/AFTER groups."""
    f = Path(root) / ROWS / f"{model}_direct.json"
    cells = {}
    if not f.exists():
        out(f"  G12 splice: {f.name} MISSING (12 cells not evaluable)")
        return {a: None for a in SPLICE_ARMS}
    d = json.load(open(f))
    res, prov = (d["results"], d.get("provenance", {})) if isinstance(d, dict) else (d, {})  # the pre-stage-5 files are bare lists
    out(f"    splice provenance: skipped_items {prov.get('skipped_items', '-')}, commit {str(prov.get('git_commit'))[:10]}, dtype {prov.get('dtype', '-')}")
    for arm in SPLICE_ARMS + SPLICE_EXTRA:
        R = [r for r in res if r["arm"] == arm] or [r for r in res if r["arm"] == ALIAS.get(arm)]
        if not R:
            if arm in SPLICE_ARMS:
                cells[arm] = None
                out(f"    {arm}: MISSING (not evaluable)")
            continue
        words = "mention_words" if "mention_words" in R[0]["m"] else "remention_words" if "remention_words" in R[0]["m"] else "choice_words"
        span = "mention" if "mention" in R[0]["m"] else "remention" if "remention" in R[0]["m"] else "choices"
        d = {g: np.array([r["m"][g] - r["m_B"] for r in R]) for g in ("all", "none", words, span, "rest_after_p")}
        valid = bool(abs(d["none"].mean()) <= 0.01 and d["all"].mean() >= 2.0)
        fw = ratio(d[words], d["all"])
        ok = bool(fw[0] >= 0.5 and fw[1] > 0.25)
        cells[arm] = (ok if valid else None) if arm in SPLICE_ARMS else None
        tag = (verdict(ok) if valid else "NOT EVALUABLE (validity; counts as not met)") if arm in SPLICE_ARMS else "exploratory"
        extra = ""
        if arms and arm in arms:
            same = [i for i in (json.dumps(r["core"], sort_keys=True) for r in R) if i in arms[arm]]
            extra = f"; factorial d_K same cores {np.mean([arms[arm][i]['d']['K_S@0'] for i in same]):+.2f} (n={len(same)})"
        out(f"    {arm:5s} n={len(R)} d(all) {d['all'].mean():+.2f} d(none) {d['none'].mean():+.3f} valid {valid}; f_words {fmt(fw)} "
            f"(>= 0.5, lower > 0.25); f_{span} {ratio(d[span], d['all'])[0]:.2f} remainder {ratio(d[span] - d[words], d['all'])[0]:.2f} "
            f"rest_after_p {ratio(d['rest_after_p'], d['all'])[0]:.2f} -> {tag}{extra}")
    return cells


def score(root, models=None, exploratory_models=None, test=False, out=print):
    root = Path(root)
    if test:
        out("TEST MODE: verdict lines are not preregistered results; every model with a factorial file is treated as primary")
        models = sorted(f.stem[:-3] for f in (root / FACTORIAL).glob("*_s0.json")) or []
        exploratory_models = []
    models = PRIMARY if models is None else models
    exploratory_models = EXPLORATORY if exploratory_models is None else exploratory_models
    out("== Part (c): membership, dose and reader rows (sentence family S*, list family L*; S6 = POST, L6 = AFTER)")
    g0, pred, cells, rep = {}, {}, {}, {}
    for m in models + exploratory_models:
        prov, arms = load_factorial(root, m)
        primary = m in models
        out(f"\n## {m} ({'primary' if primary else 'exploratory'})")
        if not arms:
            out("  factorial MISSING")
            if primary:
                g0[m], pred[m], cells[m] = False, {}, {a: None for a in SPLICE_ARMS}
            continue
        ids = sorted(set.intersection(*(set(arms[a]) for a in arms)))
        out(f"  n = {len(ids)} cores in every arm; arms {sorted(arms)}")
        table(arms, ids, out)
        res = json.load(open(root / FACTORIAL / f"{m}_s0.json"))["results"]
        diagnostics(res, arms, ids, out)
        g = gate0(prov, arms, m, out, test)
        p = predictions(arms, ids, out)
        if g is not True:
            out("  G9-G11 above are NOT EVALUABLE for this model (Gate 0 not passed) and count as not met")
            p = {k: None for k in p}
        out("  G12 reader rows (splice, n = 60; f_words = d(mention_words)/d(all)):")
        c = splice(root, m, arms, out)
        if g is not True:
            c = {k: None for k in c}
        exploratory(arms, ids, out)
        if primary:
            g0[m], pred[m], cells[m] = g, p, c
        # seed-1 replication
        prov1, arms1 = load_factorial(root, m, 1)
        if arms1:
            ids1 = sorted(set.intersection(*(set(arms1[a]) for a in arms1)))
            lo = {F: boot(vec(arms1[F + "6"], ids1))[1] if F + "6" in arms1 else float("nan") for F in ("S", "L")}
            ok1 = prov1.get("skipped_items") == 0 and all(v > 0 for v in lo.values())
            out(f"  seed-1 replication: n = {len(ids1)}, skipped_items {prov1.get('skipped_items', '?')}, lower bounds ID_K(S6) {lo['S']:+.2f} "
                f"ID_K(L6) {lo['L']:+.2f} -> {'evaluable' if ok1 else 'NOT EVALUABLE'}")
            table(arms1, ids1, out)
            p1 = predictions(arms1, ids1, out, " (seed 1)")
            if primary:
                rep[m] = p1 if ok1 else {k: None for k in p1}
        elif primary:
            rep[m] = {}
    n = len(models)
    out(f"\n== Verdicts (G9-G12; {n}/{n} primary models {models}; not evaluable counts as not met)")
    out(f"  Gate 0 passed in {sum(g0.get(m) is True for m in models)}/{n}: " + ", ".join(f"{m} {g0.get(m)}" for m in models))
    names = {"G9 S": "membership, sentence family", "G9 L": "membership, list family", "G10 S": "proportionality refuted, sentence",
             "G10 L": "proportionality refuted, list", "G11": "copy carries the identity again"}
    final = {}
    for k, name in names.items():
        ok = [pred.get(m, {}).get(k) is True for m in models]
        final[k] = sum(ok) == n
        out(f"  {k:6s} {name:36s} {sum(ok)}/{n} -> {verdict(final[k])}" + (f"   (seed 1: replicated in {sum(rep.get(m, {}).get(k) is True for m in models)}/{n})"))
    cell_ok = [cells.get(m, {}).get(a) is True for m in models for a in SPLICE_ARMS]
    final["G12"] = all(cell_ok) and len(cell_ok) == 4 * n
    out(f"  G12    reader rows at k = 2 and 3                 {sum(cell_ok)}/{4 * n} cells -> {verdict(final['G12'])}")
    return {"gate0": g0, "pred": pred, "cells": cells, "final": final}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/gpu_stage5")
    ap.add_argument("--models", default=",".join(PRIMARY))
    ap.add_argument("--exploratory", default=",".join(EXPLORATORY))
    ap.add_argument("--test", action="store_true", help="treat every <model>_s0.json as primary; banner")
    a = ap.parse_args(argv)
    score(a.root, list(filter(None, a.models.split(","))), list(filter(None, a.exploratory.split(","))), a.test)


if __name__ == "__main__":
    main()
