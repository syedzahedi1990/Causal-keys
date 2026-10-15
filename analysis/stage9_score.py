"""Score stage 9 (docs/STAGE9_PREDICTIONS.md: predictions, thresholds, falsifiers and the primary analysis;
docs/stage9/SPEC.md section 14: what is computed). Stage 9 is a dated prediction note, not a preregistration entry: no
class or prior per line, no Holm family; every number is printed.

Inputs under --results (default results/gpu_stage9), as written by scripts/gpu_stage9.sh and experiments/stage9_*.py:
check/<tag>_{stories,edits,passages}.json, lb/<tag>.json, stories/<tag>.json, edits/<tag>.json, passages/<tag>.json and
passages/<tag>_LETA.json, verified/<key>.json, logs/pytest.log (the FP32 gate), COMMIT.txt (the guard's freeze record),
ENV.txt, REVISIONS.txt, SKIPPED.txt, FETCH_FAILED.txt; and, read-only, scripts/stage8_models.json (revisions), the
stage-8a factorial files (the passage E set) and the stage-8a explore files (one exploratory reproduction).
Output: STAGE9_SCORE.txt (sections PROVENANCE, POPULATION, CHECKS, CONTROLS, PREDICTIONS, FALSIFIERS, DIRECTION B,
REPORTED, EXPLORATORY, SUMMARY; each header on its own line starting with "#### "), stage9_summary.json (every number
printed, keyed "<arm>/<model>/<statistic>" with pt, lo, hi, n; the verdicts keyed "verdict/<model or ALL>/<id>" and the
falsifiers "falsifier/<model or ALL>/<id>"), fig_window.csv, fig_edits.csv, fig_dose.csv, all in the results directory.
--test (TEST_MODE; also when every results file is tagged TEST_): the population sizes and pins, the stage-8a E-set
comparison, BF16, sdpa, the freeze record and the pytest-log gate are waived; the verdicts are plumbing, not results.
Exit status 1 if a part of the scorer raised (traceback in the report); 2 if, outside TEST, a check printed MISMATCH (every
output is still written); else 0. A deadline-partial arm (stories or passages formats not run, edits cores not run) is
not a MISMATCH: its missing parts (edits: the whole arm of that model) are NOT EVALUABLE and the CHECKS section lists it
as PARTIAL.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from stage9_parts import checks as CK  # noqa: E402
from stage9_parts import preds as PR  # noqa: E402
from stage9_parts import report as RP  # noqa: E402
from stage9_parts.data import EditRun, PassageRun, StoryRun  # noqa: E402
from stage9_parts import stats as ST  # noqa: E402
from stage9_parts.stats import NE, NOT, PT, SUP  # noqa: E402

REPO = HERE.parent
SECTIONS = ("PROVENANCE", "POPULATION", "CHECKS", "CONTROLS", "PREDICTIONS", "FALSIFIERS", "DIRECTION B", "REPORTED",
            "EXPLORATORY", "SUMMARY")


def load_json(f):
    try:
        return json.loads(Path(f).read_text())
    except Exception as ex:  # noqa: BLE001  (a truncated file is reported, not fatal)
        return {"error": repr(ex)}


def load(root: Path, test: bool) -> dict:
    """{"<dir>/<file>": json} of the run's results files (TEST: only TEST_ files; outside TEST every file, so that a
    TEST_ file in a real results directory is caught)."""
    F = {}
    for d in ("check", "lb", "stories", "edits", "passages", "verified"):
        for f in sorted((root / d).glob("*.json")) if (root / d).is_dir() else []:
            if test and d != "verified" and not f.name.startswith("TEST_"):
                continue
            F[f"{d}/{f.name}"] = load_json(f)
    return F


def pins():
    """(artefact sha256 pins, population pins) from ckeys/stage9.py (the single source of the pins)."""
    from ckeys import stage9
    return dict(stage9.ARTEFACT_SHA256), dict(stage9.POP_SHA256)


def stage8a_populations(repo: Path) -> dict:
    out = {}
    for k in CK.KEYS:
        f = repo / "results" / "gpu_stage8a" / "factorial" / f"{k}.json"
        if f.exists():
            try:
                out[k] = json.loads(f.read_text())["provenance"].get("population_sha256")
            except Exception:  # noqa: BLE001
                out[k] = None
    return out


def build_models(F: dict, test: bool) -> dict:
    """{key: preds.Model} for every key with any results file (the three study keys first)."""
    pre = "TEST_" if test else ""
    keys = sorted({CK.keyof(rel) for rel in F if not rel.startswith(("verified/", "check/"))},
                  key=lambda k: (CK.KEYS.index(k) if k in CK.KEYS else 9, k))
    models = {}
    for k in keys:
        def get(rel, cls):
            J = F.get(rel)
            return cls(J, rel) if J and "error" not in J and "provenance" in J else None
        models[k] = PR.Model(k, st=get(f"stories/{pre}{k}.json", StoryRun), lb=get(f"lb/{pre}{k}.json", StoryRun),
                             ed=get(f"edits/{pre}{k}.json", EditRun), ps=get(f"passages/{pre}{k}.json", PassageRun),
                             leta=get(f"passages/{pre}{k}_LETA.json", PassageRun))
    return models


def summary(ctx, out, log, gate, test):
    word = {True: "TRIGGERED", False: "not triggered", None: "NOT EVALUABLE"}
    out("  Predictions (combined over the models; per model in PREDICTIONS):")
    for code in ("P1", "P1s", "P1p", "P2", "P2s", "P2p", "P3", "P3s", "P3p", "P4", "P5a", "P5b", "POST", "B1", "B2", "B3"):
        if code in ctx.res:
            lv, per = ctx.res[code]
            out(f"    {code:5s} {lv:24s} " + ", ".join(f"{k} {v.level}" for k, v in per.items()))
    out("    P5c   " + ", ".join(f"{k} {'TRANSFERS' if t else 'DOES NOT TRANSFER' if t is False else 'NOT EVALUABLE'}"
                                 for k, (t, _) in ctx.p5c.items()))
    if "P2" in ctx.res:
        out(f"    (P2 {PR.P2_PER_MODEL})")
    out("  Falsifiers (at the point, any model):")
    for code in ("F1", "F2s", "F2p", "F2", "F3", "B-F1", "B-F2"):
        if code in ctx.fal:
            t, per = ctx.fal[code]
            out(f"    {code:5s} {word[t]:14s} " + ", ".join(f"{k} {word[x]}" for k, (x, _) in per.items()))
    n = {lv: sum(ctx.res[c][0] == lv for c in ("P1", "P2", "P3", "P4", "P5a", "P5b", "POST", "B1", "B2", "B3") if c in ctx.res)
         for lv in (SUP, PT, NOT, NE)}
    out(f"  Totals over P1-P4, P5a, P5b, POST, B1-B3: {n[SUP]} SUPPORTED, {n[PT]} SUPPORTED AT THE POINT, {n[NOT]} NOT "
        f"SUPPORTED, {n[NE]} NOT EVALUABLE")
    out("  Pre-written readings (note section 6) that the outcome selects:")
    for r in PR.readings(ctx):
        out("    - " + r)
    out(f"  Checks: {'MISMATCH' if log.any() else 'OK'}" + (" (TEST: not a result)" if test else "")
        + f"; FP32 gate: {gate}" + ("" if gate is True else " (TEST: waived)" if test else " -> every verdict NOT EVALUABLE"))
    if log.part:
        out("  Partial at the deadline (NOT EVALUABLE, not a MISMATCH; a later session resumes it): "
            + ", ".join(f"{k} {arm}" for k, arm in sorted(log.part)))
    if test:
        out("  TEST MODE: the verdicts above are plumbing checks, not results.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", "--root", dest="results", default="results/gpu_stage9")
    ap.add_argument("--test", action="store_true", help="TEST_MODE: sizes, pins, BF16/sdpa, freeze and pytest gate waived")
    ap.add_argument("--out", default=None, help="default {results}/STAGE9_SCORE.txt")
    ap.add_argument("--repo", default=str(REPO), help="the repository whose committed files are read (manifest, stage 8a)")
    a = ap.parse_args(argv)
    root, repo = Path(a.results), Path(a.repo)
    tags = [f.name for d in ("lb", "stories", "edits", "passages", "check") if (root / d).is_dir() for f in (root / d).glob("*.json")]
    test = a.test or (bool(tags) and all(t.startswith("TEST_") for t in tags))
    F = load(root, test)
    lines, errors = [], []
    out = lines.append
    sec = lambda name, note="": out(f"\n#### {name}" + (f" ({note})" if note else ""))  # noqa: E731
    out(f"Stage 9 scoring: the listed candidates set the edit window (docs/STAGE9_PREDICTIONS.md; docs/stage9/SPEC.md); "
        f"results {root}")
    if test:
        out("TEST MODE: the population sizes and pins, the stage-8a E-set comparison, BF16, sdpa, the freeze record and the "
            "pytest-log gate are waived; every verdict below is a plumbing check, not a result")
    log = CK.Log()
    try:
        art_pins, pop_pins = pins()
    except Exception as ex:  # noqa: BLE001
        art_pins, pop_pins = {}, {}
        errors.append("pins")
        out(f"PINS ERROR (ckeys.stage9 not importable: {type(ex).__name__}: {ex})")
    sec("PROVENANCE", "COMMIT.txt and the other pipeline records; every results file")
    try:
        CK.provenance(root, repo, F, test, log, out, art_pins)
    except Exception:  # noqa: BLE001
        errors.append("provenance")
        out("  PROVENANCE ERROR\n" + traceback.format_exc())
    sec("POPULATION", "N9 150, C 30, E9 80 (pins in ckeys/stage9.py); passages E 185 = stage 8a's set" + ("; TEST: waived" if test else ""))
    try:
        CK.population(F, test, log, out, pop_pins, stage8a_populations(repo))
    except Exception:  # noqa: BLE001
        errors.append("population")
        out("  POPULATION ERROR\n" + traceback.format_exc())
    models = build_models(F, test)
    sec("CHECKS", "the FP32 gate; l_b recomputed and read in order; T_check; every MISMATCH")
    gate, gtxt = CK.pytest_gate(root)
    out(f"  FP32 gate: {gtxt} -> " + ("PASSED" if gate else "NOT PASSED") + ("" if gate or not test else " (TEST: waived)"))
    if gate is not True and not test:
        log.run.append("FP32 tests not passed" + (": no outcome" if gate is None else ""))
    pre = "TEST_" if test else ""
    try:
        for k, m in models.items():
            m.l_b = CK.lb_checks(root, k, pre + k, m.lb, m.st, test, log, out)
            CK.t_check(m.ed, k, test, log, out)
    except Exception:  # noqa: BLE001
        errors.append("checks")
        out("  CHECKS ERROR\n" + traceback.format_exc())
    for k in [k for k in CK.KEYS if k in models]:   # a failed check was moved aside by the pipeline (S8_OUTPUTS)
        for arm in ("stories", "edits", "passages"):
            rel = f"check/{pre}{k}_{arm}.json"
            if rel not in F:
                aside = sorted(p.name for p in (root / "check").glob(f"{pre}{k}_{arm}.json.failed.*")) if (root / "check").is_dir() else []
                out(f"  {k} {arm}: no check file {rel}" + (f" (the failed check's file was moved aside: {', '.join(aside)})"
                                                         if aside else "") + " -> the arm's results, if any, are reported as found")
    for (k, arm), why in sorted(log.bad.items()):
        out(f"  MISMATCH {k} {arm}: " + "; ".join(why) + " -> its predictions are NOT EVALUABLE")
    for w in log.run:
        out(f"  MISMATCH (run): {w} -> every verdict is NOT EVALUABLE")
    for w in log.warn:
        out(f"  MISMATCH: {w}")
    for (k, arm), why in sorted(log.part.items()):
        out(f"  PARTIAL {k} {arm}: " + "; ".join(why) + " -> its predictions are NOT EVALUABLE (not a MISMATCH)")
    out(f"  checks: {'MISMATCH' if log.any() else 'OK'}" + (" (TEST: the scorer does not exit 2)" if test and log.any() else ""))
    ff = (root / "FETCH_FAILED.txt").read_text(errors="replace") if (root / "FETCH_FAILED.txt").exists() else ""
    ctx = PR.Ctx(models, log, gate, test, ff)
    for name, fn_, note in (("CONTROLS", RP.controls, "per model and format; no verdicts"),
                            ("PREDICTIONS", PR.predictions, "generated answers, competent items, containment; 95 % "
                             f"two-stage cluster bootstrap, {ST.NB:,} resamples, seed {ST.SEED}"),
                            ("FALSIFIERS", PR.falsifiers, "at the point in each model; triggered in any model"),
                            ("DIRECTION B", PR.direction_b, "blocks 3 and 7; r_K = proj_k"),
                            ("REPORTED", RP.reported, "no verdicts"),
                            ("EXPLORATORY", None, "not scored")):
        sec(name, note)
        try:
            if name == "PREDICTIONS":
                fn_(ctx, out)
                PR.p5c_lines(ctx, out)
            elif name == "EXPLORATORY":
                RP.exploratory(ctx, out, repo)
            else:
                fn_(ctx, out)
        except Exception:  # noqa: BLE001
            errors.append(name.lower())
            out(f"  {name} ERROR\n" + traceback.format_exc())
    sec("SUMMARY")
    try:
        summary(ctx, out, log, gate, test)
    except Exception:  # noqa: BLE001
        errors.append("summary")
        out("  SUMMARY ERROR\n" + traceback.format_exc())
    try:
        RP.write_csvs(ctx, root)
        out(f"  figure tables written: fig_window.csv, fig_edits.csv, fig_dose.csv in {root}")
    except Exception:  # noqa: BLE001
        errors.append("csv")
        out("  CSV ERROR\n" + traceback.format_exc())
    if errors:
        out(f"  SCORER ERROR in {errors}")
    nums = dict(ctx.nums)
    for code, (lv, per) in ctx.res.items():
        nums[f"verdict/ALL/{code}"] = {"verdict": lv}
        for k, v in per.items():
            nums[f"verdict/{k}/{code}"] = {"verdict": v.level, "why": v.why or None}
            if code == "P2":
                nums[f"verdict/{k}/{code}"]["note"] = f"{PR.P2_PER_MODEL}; {v.label}"
    for code, (t, per) in ctx.fal.items():
        nums[f"falsifier/ALL/{code}"] = {"triggered": t}
        for k, (x, txt) in per.items():
            nums[f"falsifier/{k}/{code}"] = {"triggered": x, "text": txt}
    for k, (t, txt) in ctx.p5c.items():
        nums[f"verdict/{k}/P5c"] = {"transfers": t, "text": txt}
    nums["meta/run/checks"] = {"mismatch": log.any(), "test": test, "fp32_gate": gate, "errors": errors,
                               "partial": [f"{k}/{arm}" for k, arm in sorted(log.part)],
                               "readings": PR.readings(ctx) if ctx.res else []}
    text = "\n".join(lines) + "\n"
    print(text, end="")
    f = Path(a.out) if a.out else root / "STAGE9_SCORE.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(f)
    js = root / "stage9_summary.json"
    tmp = js.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(nums, indent=1, sort_keys=True, default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else str(x)))
    tmp.replace(js)
    return 1 if errors else 2 if (not test and log.any()) else 0


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        sys.exit(main())
