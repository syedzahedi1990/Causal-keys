"""CONTROLS, REPORTED and EXPLORATORY sections of the stage-9 scorer and its figure tables (docs/stage9/SPEC.md sections
11.7 and 14; docs/STAGE9_PREDICTIONS.md sections 4 and 6). Nothing here decides a prediction."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from . import preds as PR
from .data import EDIT_FORMATS, PASSAGE_FORMATS, STORY_FORMATS, nearest_onset
from .stats import Q, missing, rate

KEYS = PR.KEYS
ALPHAS = {"J": (0.25, 0.5, 0.65, 0.8, 0.9, 1.25), "K": (0.5, 0.8), "V": (0.5, 0.8)}


def _f(x, d=3):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def _share(a, b):
    return f"{a / b:.3f} ({a}/{b})" if b else "n/a"


def _nanmean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _models(ctx):
    return [(k, ctx.models[k]) for k in list(KEYS) + sorted(set(ctx.models) - set(KEYS)) if k in ctx.models]


# --------------------------------------------------------------------------- CONTROLS
def _e(x) -> str:
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.1e}"


def _num(ctx, arm, key, stat, pt, n, **extra):
    """A printed number without an interval into stage9_summary.json (lo and hi null)."""
    f = lambda x: None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 6)  # noqa: E731
    ctx.nums[f"{arm}/{key}/{stat}"] = {"pt": f(pt), "lo": None, "hi": None, "n": int(n)} | extra


def _conflicts(c, nl, nr, per) -> str:
    if not nr:
        return "no row stores the word parse"
    top = ", ".join(f"{k} {v}" for k, v in sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))[:8])
    return (f"{_share(c, nl)} of the letter answers name another candidate word in the same text ({nr} LETTER rows in "
            f"all)" + (f"; by row: {top}" + (" ..." if len(per) > 8 else "") if per else ""))


def controls(ctx, out):
    out("  The window ends (late@0 = early@L = KV@0 = the clean S run; early@0 = late@L = none = the clean B run) and their "
        "complementarity (early on [0, l) and late on [l, L) together = KV@0) are FP32-tested (tests/test_stage9_stories.py, "
        "test_stage9_passages.py); no GPU check.")
    out("  BF16 floor: the none row (edits: self) against the clean B run of the same item in another batch shape (stories: "
        "clean_B of the capture pass; passages: clean_B of the capture batch; edits: the CAA batch's self row, unclamped): "
        "the mean absolute difference of E (passages: of the decision log-probabilities lp) and the share of items whose "
        "argmax answer differs (E-argmax; passages: the decision-token argmax; edits also the generated answers). In-batch "
        "determinism: none against none2 (edits: self against self2), the same ids and tables in one forward (0 expected, "
        "also in BF16). LETTER: the letter answers whose text names another candidate word (\"A drawer\" read as option "
        "A), a sensitivity count; the answer stays SPEC 9.2's.")
    for k, m in _models(ctx):
        if m.st is not None:
            S = m.st
            for f in S.fmts:
                pres, comp = S.present(f).sum(), S.comp(f).sum()
                kv = ctx.sflip(m, f, "KV@0", sub="all")
                fl, dis, nf = S.e_floor(f)
                mx, gd, nd = S.determinism(f)
                a, b = S.agreement(f)
                _num(ctx, "stories", k, f"{f}:bf16_floor_mean_abs_E", fl, nf, argmax_differs=None if nf == 0 else round(dis, 6))
                _num(ctx, "stories", k, f"{f}:in_batch_max_abs_E", mx, nd, answers_differ=None if nd == 0 else round(gd, 6))
                out(f"  {k:8s} stories {f:9s} competent {_share(int(comp), int(pres))}; flip(KV@0) all items {kv.txt()}; "
                    f"BF16 floor mean|E(none) - E(clean B)| {_f(fl, 4)}, E-argmax differs {_f(dis)} (n={nf}); in-batch "
                    f"determinism max|E(none) - E(none2)| {_e(mx)}, answers differ {_f(gd)}; E-argmax = "
                    f"generation {_share(a, b)}")
            if S.has("LETTER"):
                c, nl, nr, per = S.letter_conflicts("LETTER")
                _num(ctx, "stories", k, "LETTER:letter_word_conflicts", c / nl if nl else None, nl, count=c, rows=nr)
                out(f"  {k:8s} stories LETTER    letter/word conflicts: {_conflicts(c, nl, nr, per)}")
        if m.ps is not None:
            for name, sp in m.ps.splits.items():
                for f in sp.fmts:
                    pres, comp = sp.present(f).sum(), sp.comp(f).sum()
                    kv = ctx.pflip(m, name, f, "KV@0", sub="all")
                    fl, dis, nf = sp.floor(f)
                    mx, gd, nd = sp.determinism(f)
                    a, b = sp.agreement(f)
                    _num(ctx, "passages", k, f"{name}:{f}:bf16_floor_mean_abs_lp", fl, nf,
                         argmax_differs=None if nf == 0 else round(dis, 6))
                    _num(ctx, "passages", k, f"{name}:{f}:in_batch_max_abs_lp", mx, nd,
                         answers_differ=None if nd == 0 else round(gd, 6))
                    out(f"  {k:8s} passages {name}/{f:5s} competent (containment) {_share(int(comp), int(pres))}, strict "
                        f"{_share(int(sp.comp(f, 'strict').sum()), int(pres))}; flip(KV@0) all items {kv.txt()}; BF16 floor "
                        f"mean|lp(none) - lp(clean B)| {_f(fl, 4)}, decision argmax differs {_f(dis)} (n={nf}); in-batch "
                        f"determinism max|lp(none) - lp(none2)| {_e(mx)}, answers differ {_f(gd)}; lp-argmax = containment "
                        f"answer {_share(a, b)}")
        if m.ed is not None:
            E = m.ed
            for f in EDIT_FORMATS:
                parts = []
                for l in E.blocks:
                    if not E.has_cell(f, l):
                        continue
                    sB = E.self_hit(f, l, "B")
                    fl, dis, gdis, nf = E.floor(f, l)
                    mx, gd, nd = E.determinism(f, l)
                    a, b = E.agreement(f, l)
                    _num(ctx, "edits", k, f"{f}@{l}:bf16_floor_mean_abs_E", fl, nf,
                         argmax_differs=None if nf == 0 else round(dis, 6), answers_differ=None if nf == 0 else round(gdis, 6))
                    _num(ctx, "edits", k, f"{f}@{l}:in_batch_max_abs_E", mx, nd, answers_differ=None if nd == 0 else round(gd, 6))
                    parts.append(f"@{l} self = B {sB.mean():.3f}, floor mean|E(self) - E(clean B)| {_f(fl, 4)} (E-argmax differs "
                                 f"{_f(dis)}, answers differ {_f(gdis)}), in-batch max|E(self) - E(self2)| {_e(mx)} (answers "
                                 f"differ {_f(gd)}), E-argmax = generation {_share(a, b)}")
                if parts:
                    out(f"  {k:8s} edits {f:6s} " + "; ".join(parts))
            for l in E.blocks:
                if E.has_cell("LETTER", l):
                    c, nl, nr, per = E.letter_conflicts(l)
                    _num(ctx, "edits", k, f"LETTER@{l}:letter_word_conflicts", c / nl if nl else None, nl, count=c, rows=nr)
                    out(f"  {k:8s} edits LETTER@{l} letter/word conflicts: {_conflicts(c, nl, nr, per)}")
            if E.has_row("NONE", 3, "nat"):
                out(f"  {k:8s} edits competence: nat|t at NONE@3 answers t: {ctx.eflip(m, 'NONE', 3, 'nat').txt()} (over (core, t); "
                    "the edit arm's flips use all cores)")


# --------------------------------------------------------------------------- REPORTED
def reported(ctx, out):
    for k, m in _models(ctx):
        if m.st is not None:
            S = m.st
            out(f"  [{k} stories, N9, L = {S.L}; competent items; generated answers] late and early flip curves, l_c, K/V-only")
            for f in S.fmts:
                for w in ("late", "early"):
                    out(f"    {f:9s} {w:5s} " + "; ".join(f"{l}: {ctx.sflip(m, f, f'{w}@{l}').txt(2)}" for l in S.onsets))
                q, l = ctx.lc(m, f)
                out(f"    {f:9s} l_c = {l if l is not None else 'n/a'}" + (" (censored: > last onset)" if l == S.L else "")
                    + f", l_c/L {q.txt()}")
                out(f"    {f:9s} K@0 {ctx.sflip(m, f, 'K@0').txt(2)}; V@0 {ctx.sflip(m, f, 'V@0').txt(2)}; "
                    + "; ".join(f"Klate@{l} {ctx.sflip(m, f, f'Klate@{l}').txt(2)}" for l in S.kv_onsets) + "; "
                    + "; ".join(f"Vlate@{l} {ctx.sflip(m, f, f'Vlate@{l}').txt(2)}" for l in S.kv_onsets))
            out(f"    l_b = {m.l_b}")
            at = sorted({x for x in (m.l_b, nearest_onset(S.onsets, 0.3 * S.L)) if x is not None})
            for l in at:
                for f in S.fmts:
                    a, b, c = ctx.silent(m, f, f"late@{l}")
                    out(f"    silent failure {f:9s} late@{l}: share B among non-S {a.txt()}; median absolute p(B) of B answers "
                        f"{b.txt()}; median six-way p(B) {c.txt()}")
            if S.has("NONE") and S.has("AFTER"):
                msk = S.comp("NONE") * S.comp("AFTER")
                parts = []
                for l in S.onsets:
                    x = S.is_("NONE", f"late@{l}", "S") * S.is_("AFTER", f"late@{l}", "B")
                    q = ctx.rec("stories", k, f"inconsistency:late@{l}", rate(S.B, x, msk)) if msk.sum() else missing("no items")
                    parts.append(f"{l}: {q.txt(2)}")
                out("    inconsistency (NONE late answer S while AFTER late answer B; items competent in both) " + "; ".join(parts))
        if m.ps is not None:
            for name, sp in m.ps.splits.items():
                out(f"  [{k} passages {name}, L = {m.ps.L}; competent items; containment] curves")
                for f in sp.fmts:
                    for w in ("late", "early", "Klate", "Vlate"):
                        out(f"    {f:5s} {w:5s} " + "; ".join(f"{l}: {ctx.pflip(m, name, f, f'{w}@{l}').txt(2)}" for l in m.ps.onsets))
                    out(f"    {f:5s} KV@0 {ctx.pflip(m, name, f, 'KV@0').txt(2)}; K@0 {ctx.pflip(m, name, f, 'K@0').txt(2)}; "
                        f"V@0 {ctx.pflip(m, name, f, 'V@0').txt(2)}")
        if m.leta is not None:
            for name, sp in m.leta.splits.items():
                out(f"  [{k} passages {name} LETA (exploratory, deadline-gated)]")
                for f in sp.fmts:
                    for w in ("late", "early"):
                        out(f"    {f:5s} {w:5s} " + "; ".join(f"{l}: {ctx.pflip(m, name, f, f'{w}@{l}', run=m.leta).txt(2)}"
                                                         for l in m.leta.onsets))
        if m.ed is not None:
            E = m.ed
            out(f"  [{k} edits, E9, {E.n} cores; generated answers over (core, t)] flip, stale, tau, r_K per row")
            for l in E.blocks:
                for Z in E.kinds(l):
                    rk = _nanmean(E.rk(l, Z))
                    cells = [f"{f} {ctx.eflip(m, f, l, Z).txt(2)} stale {_f(ctx.eflip(m, f, l, Z, who='B').pt, 2)}"
                             for f in EDIT_FORMATS if E.has_row(f, l, Z)]
                    tau = ctx.tau(m, l, Z)
                    out(f"    @{l} {Z:7s} r_K {_f(rk)}  " + "; ".join(cells) + f"; tau {tau.txt(2)}")
            for sg in ("E", "logit"):
                for l in E.blocks:
                    if l not in PR.DOSE_BLOCKS:
                        continue
                    for f in EDIT_FORMATS:
                        parts = []
                        for kind, al in ALPHAS.items():
                            for a in al:
                                q = ctx.psi(m, f, l, kind, a, sg)
                                if q.why and not math.isfinite(q.pt):
                                    continue
                                parts.append(f"psi_{kind}({a}) {q.txt(2)}")
                        if parts:
                            out(f"    psi ({'nats' if sg == 'E' else 'logits'}) {f}@{l}: " + "; ".join(parts))
    out("  Passage predictions on the held-out R items (decide nothing) and in strict match (sensitivity):")
    for lab, kw in (("R, containment", {"split": "R"}), ("E, strict", {"rule": "strict"}), ("R, strict", {"split": "R", "rule": "strict"})):
        for code, f, rule in (("P1p", PR.p1p, "every"), ("P2p", PR.p2p, 2), ("P3p", PR.p3p, "every")):
            PR.run_line(ctx, f"{code} [{lab}]", f, KEYS, rule, out, arm="passages", quiet=True, **kw)


# --------------------------------------------------------------------------- EXPLORATORY
def exploratory(ctx, out, repo: Path):
    out("  E-argmax versions (the six-way E argmax of the trie scores instead of the generated answer):")
    for code, f in (("P1s", PR.p1s), ("P2s", PR.p2s), ("P3s", PR.p3s), ("POST", PR.post)):
        PR.run_line(ctx, f"{code} [E-argmax]", f, KEYS, "every", out, arm="stories", quiet=True, how="E")
    out("  All items instead of the competent items:")
    for code, f in (("P1s", PR.p1s), ("P2s", PR.p2s), ("P3s", PR.p3s), ("POST", PR.post)):
        PR.run_line(ctx, f"{code} [all items]", f, KEYS, "every", out, arm="stories", quiet=True, sub="all")
    for code, f in (("P1p", PR.p1p), ("P3p", PR.p3p)):
        PR.run_line(ctx, f"{code} [all items]", f, KEYS, "every", out, arm="passages", quiet=True, sub="all")
    PR.run_line(ctx, "P2p [all items]", PR.p2p, KEYS, 2, out, arm="passages", quiet=True, sub="all")
    out("  Per lexicon and per list order (B listed before S, or S before B) at l_b (late and early AFTER / NONE flips):")
    for k, m in _models(ctx):
        S = m.st
        if S is None or m.l_b is None:
            continue
        lex, bs = S.lex(), S.b_before_s()
        groups = [(f"lexicon {x}", lex == x) for x in (1, 2)] + [("B before S", bs > 0), ("S before B", bs == 0)]
        for lab, g in groups:
            parts = []
            for f in ("AFTER", "LETTER", "NONE", "BEFORE"):
                if not S.has(f):
                    continue
                for w in ("late", "early"):
                    x, msk = S.is_(f, f"{w}@{m.l_b}", "S"), S.comp(f) * g
                    q = ctx.rec("stories", k, f"{lab}:{f}:{w}@{m.l_b}:flip", rate(S.B, x, msk)) if msk.sum() else missing("no items")
                    parts.append(f"{f} {w} {q.txt(2)}")
            out(f"    {k:8s} {lab:11s}: " + "; ".join(parts))
    out("  Stage 8a's OPTA K_S@0.3L decision-token share reproduced by Klate@round(0.3 L) (E items, all items, argmax over "
        "the decision tokens; stage 8a: results/gpu_stage8a/explore/<key>.json, part onset_piece, row K_S@on):")
    for k, m in _models(ctx):
        if m.ps is None or m.ps.split("E") is None:
            continue
        l = m.ps.onsets[1]
        q = ctx.pflip(m, "E", "OPTA", f"Klate@{l}", sub="all", rule="lp")
        ref = "absent"
        f = repo / "results" / "gpu_stage8a" / "explore" / f"{k}.json"
        if f.exists() and not ctx.test:
            try:
                R = json.loads(f.read_text())["parts"]["onset_piece"]["OPTA"]
                ref = f"{np.mean([max(r['rows']['K_S@on']['lp'], key=r['rows']['K_S@on']['lp'].get) == 'S' for r in R]):.3f} (n={len(R)})"
            except Exception as ex:  # noqa: BLE001
                ref = f"not readable ({type(ex).__name__})"
        elif ctx.test:
            ref = "TEST: no stage-8a reference"
        out(f"    {k:8s} stage 9 OPTA Klate@{l} {q.txt()} against stage 8a K_S@on {ref}")


# --------------------------------------------------------------------------- figure tables
def write_csvs(ctx, root: Path):
    rows = []
    for k, m in _models(ctx):
        if m.st is not None:
            S = m.st
            for f in S.fmts:
                for w, on in (("late", [0] + S.onsets), ("early", [0] + S.onsets), ("Klate", [0] + S.kv_onsets),
                              ("Vlate", [0] + S.kv_onsets)):
                    for l in on:
                        row = {("late", 0): "KV@0", ("early", 0): "none", ("Klate", 0): "K@0", ("Vlate", 0): "V@0"}.get((w, l), f"{w}@{l}")
                        q, s = ctx.sflip(m, f, row), ctx.sflip(m, f, row, who="B")
                        e = ctx.sflip(m, f, row, how="E")
                        rows.append([k, "stories", "N9", f, w, l, l / S.L if S.L else "", q.n, q.pt, q.lo, q.hi, s.pt, s.lo, s.hi, e.pt])
        for run, arm in ((m.ps, "passages"), (m.leta, "passages")):
            if run is None:
                continue
            for name, sp in run.splits.items():
                for f in sp.fmts:
                    for w in ("late", "early", "Klate", "Vlate"):
                        for l in [0] + run.onsets:
                            row = {("late", 0): "KV@0", ("early", 0): "none", ("Klate", 0): "K@0", ("Vlate", 0): "V@0"}.get((w, l), f"{w}@{l}")
                            q = ctx.pflip(m, name, f, row, run=run)
                            s = ctx.pflip(m, name, f, row, Y="B", run=run)
                            e = ctx.pflip(m, name, f, row, rule="lp", run=run)
                            rows.append([k, arm, name, f, w, l, l / run.L if run.L else "", q.n, q.pt, q.lo, q.hi, s.pt, s.lo, s.hi, e.pt])
    _csv(root / "fig_window.csv", ["model", "arm", "split", "format", "window", "onset", "onset_frac", "n", "flip", "flip_lo",
                                   "flip_hi", "stale", "stale_lo", "stale_hi", "eflip"], rows)
    rows = []
    for k, m in _models(ctx):
        E = m.ed
        if E is None:
            continue
        for f in EDIT_FORMATS:
            for l in E.blocks:
                for Z in E.kinds(l):
                    if not E.has_row(f, l, Z):
                        continue
                    q, s = ctx.eflip(m, f, l, Z), ctx.eflip(m, f, l, Z, who="B")
                    rk = _nanmean(E.rk(l, Z))
                    rows.append([k, f, l, Z, q.n, q.pt, q.lo, q.hi, s.pt, rk])
    _csv(root / "fig_edits.csv", ["model", "format", "block", "row", "n", "flip", "lo", "hi", "stale", "r_K"], rows)
    rows = []
    for k, m in _models(ctx):
        E = m.ed
        if E is None:
            continue
        for f in EDIT_FORMATS:
            for l in E.blocks:
                if l not in PR.DOSE_BLOCKS:
                    continue
                for kind, al in ALPHAS.items():
                    for a in al:
                        qn, ql = ctx.psi(m, f, l, kind, a, "E"), ctx.psi(m, f, l, kind, a, "logit")
                        if qn.why and not math.isfinite(qn.pt):
                            continue
                        rows.append([k, f, l, kind, a, qn.pt, qn.lo, qn.hi, ql.pt, ql.lo, ql.hi])
    _csv(root / "fig_dose.csv", ["model", "format", "block", "kind", "alpha", "psi_nats", "psi_nats_lo", "psi_nats_hi",
                                 "psi_logit", "psi_logit_lo", "psi_logit_hi"], rows)


def _csv(path: Path, head, rows):
    def cell(x):
        if isinstance(x, float):
            return "" if not math.isfinite(x) else f"{x:.6g}"
        return x
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for r in rows:
            w.writerow([cell(x) for x in r])
    tmp.replace(path)
