"""Score preregistration P-2026-10-05-F (docs/PREREGISTRATION.md, GPU stage 4): gates G1-G3, predictions F1-F4, and
the exploratory list, exactly as written there.

Inputs (defaults under --root results/gpu_stage4):
  frames/{tag}.json            experiments/paper1_frames.py --refit none=... --refit p1=..., "Answer:" prefill (primary)
  frames_noprefill/{tag}.json  the same with prefill "" (exploratory)
  fit_none/run, fit_p1/run     experiments/refit_remap.py runs: bases (G1, principal cosines) and updates.jsonl (loss)
  {p1_root}/gpu/component_data/bases/mistral_original_1000_{obj}_{seed}.npz   Paper 1's released bases
  results/gpu_stage2/format_factorial/Mistral-Small-24B-Instruct-2501_s0.json  s_ID = ID_K / (ID_K + ID_V)
Population: native cores with B, S, T distinct (n = 96, checked in every family x format cell); fits averaged within
core (stage2_score.per_core); 95% core bootstrap, 10,000 resamples, fixed seed, ratio of means. Every statistic over the
same cores uses the same resample indices, so F3's difference is paired over cores.
  phi   = [m(M) - m(P)] / [m(T) - m(S)]
  psi_K = [m(P+K_M) - m(P)] / [m(M) - m(P)]     psi_V = [m(P+V_M) - m(P)] / [m(M) - m(P)]
  rho_K = [m(M+K_P) - m(M)] / [m(P) - m(M)]     rho_V = [m(M+V_P) - m(M)] / [m(P) - m(M)]
Families: "none/" = fit_none, "p1/" = fit_p1 (each m3 paired with its own run's P), "" = released bases (anchor).
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage1_prereg as s1  # noqa: E402
import stage2_score as s2  # noqa: E402

ARMS = ("P1", "NONE", "BEFORE", "POST", "LETTER")
FITS = {"none/": "fit_none", "p1/": "fit_p1"}
FRAME = {"none/": "NONE", "p1/": "P1"}
LABEL = {"none/": "fit_none", "p1/": "fit_p1", "": "released"}
SEEDS = (101, 102, 103)
OBJ = ("pca", "m3")  # the basis files read by the scorer
SEED, B = 20261005, 10000
STATS = {"phi": (lambda x: x["M"] - x["P"], lambda x: x["T"] - x["S"]),
         "psiK": (lambda x: x["add"] - x["P"], lambda x: x["M"] - x["P"]),
         "psiV": (lambda x: x["addv"] - x["P"], lambda x: x["M"] - x["P"]),
         "rhoK": (lambda x: x["rem"] - x["M"], lambda x: x["P"] - x["M"]),
         "rhoV": (lambda x: x["remv"] - x["M"], lambda x: x["P"] - x["M"])}
SKIP = "SKIPPED"


def stat(R, ids, num, den):
    a = np.array([num(R[i]) for i in ids]); b = np.array([den(R[i]) for i in ids])
    r = a[IDX(len(ids))].mean(1) / b[IDX(len(ids))].mean(1)
    return a.mean() / b.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5), r


_idx = {}
def IDX(n):  # fixed-seed core resamples, shared by every statistic over n cores
    if n not in _idx:
        _idx[n] = np.random.default_rng(SEED).integers(0, n, (B, n))
    return _idx[n]


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def num(x):
    return "n/a" if x is None else f"{x:+.3f}"


def verdict(ok):
    return SKIP if ok is None else ("MET" if ok else "NOT MET")


def load_frames(path):
    if not path.exists():
        return None, {}
    d = json.load(open(path))
    res, rows = d["results"], {}
    for fam in ("", *FITS):
        if not any(f"{fam}m3_101" in r["runs"] for r in res):
            continue
        for arm in ARMS:
            R = s2.per_core(res, arm, fam)
            if R:
                rows[fam, arm] = R
    return d, rows


def table(rows, title):
    est = {}
    print(f"\n{title}")
    print("  (rho_K, rho_V and the f_star column are exploratory)")
    for fam in ("none/", "p1/", ""):
        for arm in ARMS:
            if (fam, arm) not in rows:
                continue
            R = rows[fam, arm]
            ids = sorted(R)
            for k, (num, den) in STATS.items():
                est[fam, arm, k] = stat(R, ids, num, den)
            line = "  ".join(f"{k} {fmt(est[fam, arm, k])}" for k in STATS)
            fs = ""
            if not math.isnan(R[ids[0]]["F"]):
                est[fam, arm, "phiF"] = stat(R, ids, lambda x: x["F"] - x["P"], lambda x: x["T"] - x["S"])
                fs = f"  f_star (F-P)/(T-S) {fmt(est[fam, arm, 'phiF'])}"
            print(f"  {LABEL[fam]:9s} {arm:6s} n={len(R):3d}  {line}{fs}")
    if not est:
        print(f"  {SKIP}: no stage-4 frames results")
    return est


def did(rows):
    """D = [psi_K(LETTER) - psi_K(NONE)]_fit_none - [same]_fit_p1, paired over cores."""
    keys = [(f, a) for f in FITS for a in ("LETTER", "NONE")]
    if not all(k in rows for k in keys):
        return None
    ids = sorted(set.intersection(*(set(rows[k]) for k in keys)))
    p = {k: stat(rows[k], ids, *STATS["psiK"]) for k in keys}
    pt = (p["none/", "LETTER"][0] - p["none/", "NONE"][0]) - (p["p1/", "LETTER"][0] - p["p1/", "NONE"][0])
    bs = (p["none/", "LETTER"][3] - p["none/", "NONE"][3]) - (p["p1/", "LETTER"][3] - p["p1/", "NONE"][3])
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(ids)


def load_basis(f):
    with np.load(f, allow_pickle=False) as z:
        return z["rank_16"].astype(np.float64)


def msq_cos(A, Bm):
    """Mean squared principal cosine between the row spaces of two orthonormal (16, d) bases."""
    return float(np.mean(np.linalg.svd(A @ Bm.T, compute_uv=False) ** 2))


def released(p1_root, obj, s):
    return Path(p1_root) / "gpu/component_data/bases" / f"mistral_original_1000_{obj}_{s}.npz" if p1_root else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(fits, files, p1_root):
    """The scored families must be the gated fit runs, and the basis files read here (G1, cosines) the ones evaluated:
    each fit directory's FRAME.json (frame, status COMPLETE, complete_sha256 = sha256 of COMPLETE.json) and every basis
    file against COMPLETE.json's inventory; in each frames file the family's frame, the same COMPLETE.json sha256, the
    prefill, and the sha256 of every refit basis and of every released basis it records."""
    bad, on_disk = [], {}
    for fam, run in fits.items():
        fr, cp = run.parent / "FRAME.json", run / "COMPLETE.json"
        if not fr.exists() or not cp.exists():
            bad.append(f"{run}: missing {fr.name if not fr.exists() else cp.name}")
            continue
        info, c = json.load(open(fr)), sha(cp)
        if info.get("frame") != FRAME[fam] or info.get("status") != "COMPLETE" or info.get("complete_sha256") != c:
            bad.append(f"{fr}: frame {info.get('frame')} status {info.get('status')} complete_sha256 "
                       f"{str(info.get('complete_sha256'))[:12]}, expected {FRAME[fam]} COMPLETE {c[:12]}")
        inventory = json.load(open(cp))["artifacts"]
        for obj in OBJ:
            for s in SEEDS:
                f = run / "bases" / f"{obj}_ts{s}.npz"
                on_disk[f"{fam}{obj}_{s}"] = h = sha(f) if f.exists() else None
                if h is None or h != inventory.get(f"bases/{f.name}", {}).get("sha256"):
                    bad.append(f"{f}: sha256 {h and h[:12]} is not COMPLETE.json's inventory entry")
    for obj in OBJ:
        for s in SEEDS:
            g = released(p1_root, obj, s)
            if g is not None and g.exists():
                on_disk[f"{obj}_{s}"] = sha(g)
    for name, (d, prefill) in files.items():
        if d is None:
            continue
        p = d["provenance"]
        if p.get("prefill") != prefill:
            bad.append(f"{name}: prefill {p.get('prefill')!r}, expected {prefill!r}")
        for fam, run in fits.items():
            r = p.get("refits", {}).get(fam[:-1])
            c = sha(run / "COMPLETE.json") if (run / "COMPLETE.json").exists() else None
            if r is None or r.get("frame") != FRAME[fam] or r.get("complete_sha256") != c:
                bad.append(f"{name}: family {fam} is {r and (r.get('run'), r.get('frame'))}, expected {run} "
                           f"({FRAME[fam]}, COMPLETE.json sha256 {c and c[:12]})")
        for key, h in on_disk.items():
            rec = p.get("bases", {}).get(key, {})
            if ("/" in key or "sha256" in rec) and rec.get("sha256") != h:  # released: absent if random (TEST_MODE)
                bad.append(f"{name}: basis {key} sha256 {str(rec.get('sha256'))[:12]}, file read here {h and h[:12]}")
    ok = "OK, scored families = gated fit runs, basis files as evaluated, prefills as named"
    print(f"  provenance: {ok if not bad else 'MISMATCH'}")
    for b in bad:
        print(f"    {b}")
    return bad


def population(files, test, n=96):
    """The preregistered population: every (family, format) cell of both frames files holds the same cores, the 96
    native cores with B, S and T distinct, from a run over all cores (--n 0) and the five formats; TEST_ files: the
    same cores in every cell only."""
    bad, cells = [], {}
    for name, (d, rows) in files.items():
        if d is None:
            continue
        args = d["provenance"].get("args", {})
        if not test and (args.get("n") != 0 or args.get("arms") != ",".join(ARMS)):
            bad.append(f"{name}: args n={args.get('n')!r} arms={args.get('arms')!r}, expected n=0 arms={','.join(ARMS)!r}")
        for fam in ("", *FITS):
            for arm in ARMS:
                cells[f"{name} {LABEL[fam]} {arm}"] = set(rows.get((fam, arm), {}))
    if not cells:
        return []
    ref = next(iter(cells.values()))
    bad += [f"{c}: n={len(v)}, cores differ from the first cell's" for c, v in cells.items() if v != ref]
    if not test and len(ref) != n:
        bad.append(f"cells hold n={len(ref)} cores, expected {n}")
    print(f"  population: {'OK, ' + str(len(ref)) + ' cores in every family x format cell' if not bad else 'MISMATCH'}")
    for b in bad:
        print(f"    {b}")
    return bad


def gate_g1(root, fits, p1_root):
    print("\nG1 same activations: each refit P vs the released P of the same seed, mean squared principal cosine >= 0.99")
    oks = []
    for fam, run in fits.items():
        for s in SEEDS:
            f, g = Path(run) / "bases" / f"pca_ts{s}.npz", released(p1_root, "pca", s)
            if not f.exists() or g is None or not g.exists():
                why = f"missing {f}" if not f.exists() else ("no --p1-root" if g is None else f"missing {g}")
                print(f"  {LABEL[fam]} seed {s}: {SKIP} ({why})")
                oks.append(None)
                continue
            A, Bm = load_basis(f), load_basis(g)
            if A.shape != Bm.shape:
                print(f"  {LABEL[fam]} seed {s}: {SKIP} (width {A.shape[1]} vs released {Bm.shape[1]}: test model)")
                oks.append(None)
                continue
            v = msq_cos(A, Bm)
            oks.append(v >= 0.99)
            print(f"  {LABEL[fam]} seed {s}: {v:.4f}")
    ok = None if None in oks else all(oks)
    print(f"  G1 -> {verdict(ok)}")
    return ok


def s_id(path):
    if not Path(path).exists():
        return None
    res = json.load(open(path))["results"]
    out = {}
    for arm in ARMS:
        R = s1.per_core(res, arm)
        if R:
            k, v = np.mean([x["idK"] for x in R.values()]), np.mean([x["idV"] for x in R.values()])
            out[arm] = k / (k + v)
    return out


def loss_curves(fits):
    print("\nExploratory: loss curves (updates.jsonl; six-way conditional CE, chance ln 6 = 1.792); mean loss by decile of steps")
    for fam, run in fits.items():
        f = Path(run) / "updates.jsonl"
        if not f.exists():
            print(f"  {LABEL[fam]}: {SKIP} (no {f})")
            continue
        by = {}
        for line in open(f):
            r = json.loads(line)
            by.setdefault(r["fit"], []).append(r["loss"])
        for fit, L in sorted(by.items()):
            L = np.array(L)
            dec = " ".join(f"{c.mean():.3f}" for c in np.array_split(L, min(10, len(L))))
            k = min(100, len(L))
            print(f"  {LABEL[fam]:8s} {fit:13s} steps {len(L):4d}  first {k} {L[:k].mean():.3f}  last {k} {L[-k:].mean():.3f}  deciles {dec}")


def cosines(fits, p1_root):
    print("\nExploratory: mean squared principal cosines of M (m3) between fits, against the between-seed baseline")
    B_ = {}
    for fam, run in fits.items():
        for s in SEEDS:
            f = Path(run) / "bases" / f"m3_ts{s}.npz"
            if f.exists():
                B_[fam, s] = load_basis(f)
    for s in SEEDS:
        g = released(p1_root, "m3", s)
        if g is not None and g.exists():
            B_["", s] = load_basis(g)
    pairs = [(("none/", s), ("", s)) for s in SEEDS] + [(("p1/", s), ("", s)) for s in SEEDS] + \
            [(("none/", s), ("p1/", s)) for s in SEEDS] + \
            [((f, a), (f, b)) for f in ("", "none/", "p1/") for a, b in ((101, 102), (101, 103), (102, 103))]
    for x, y in pairs:
        name = f"{LABEL[x[0]]} m3_{x[1]} vs {LABEL[y[0]]} m3_{y[1]}"
        if x not in B_ or y not in B_:
            print(f"  {name}: {SKIP} (missing)")
        elif B_[x].shape != B_[y].shape:
            print(f"  {name}: {SKIP} (width {B_[x].shape[1]} vs {B_[y].shape[1]})")
        else:
            print(f"  {name}: {msq_cos(B_[x], B_[y]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/gpu_stage4")
    ap.add_argument("--tag", default="mistral", help="frames file name (TEST_MODE: TEST_<model>)")
    ap.add_argument("--p1-root", default=None, help="default: the frames file's provenance args.p1_root")
    ap.add_argument("--fit-none", default=None, help="default ROOT/fit_none/run")
    ap.add_argument("--fit-p1", default=None, help="default ROOT/fit_p1/run")
    ap.add_argument("--factorial", default="results/gpu_stage2/format_factorial/Mistral-Small-24B-Instruct-2501_s0.json")
    a = ap.parse_args()
    root = Path(a.root)
    fits = {"none/": Path(a.fit_none or root / "fit_none/run"), "p1/": Path(a.fit_p1 or root / "fit_p1/run")}
    print("Stage 4 scoring, preregistration P-2026-10-05-F")
    for fam, run in fits.items():
        fr = run.parent / "FRAME.json"
        info = json.load(open(fr)) if fr.exists() else {}
        print(f"  {LABEL[fam]}: {run}  frame {info.get('frame', '?')}  status {info.get('status', SKIP)}  "
              f"repo {info.get('repo', '?')}  test_mode {info.get('test_mode', '?')}")
    d, rows = load_frames(root / "frames" / f"{a.tag}.json")
    d0, rows0 = load_frames(root / "frames_noprefill" / f"{a.tag}.json")
    if d is not None:
        p = d["provenance"]
        a.p1_root = a.p1_root or p.get("args", {}).get("p1_root")
        print(f"  frames: {p['repo']} @ {p['revision']}, {p['device']} x{p['n_gpus']}, transformers {p['transformers']}, "
              f"prefill {p.get('prefill')!r}, tokenizer check {p.get('tokenizer_check')}")
    print(f"  p1_root: {a.p1_root}")
    print(f"  factorial (s_ID): {a.factorial} sha256 {sha(a.factorial)[:16] if Path(a.factorial).exists() else SKIP}")
    bad = provenance(fits, {"frames": (d, "Answer:"), "frames_noprefill": (d0, "")}, a.p1_root)
    bad += population({"frames": (d, rows), "frames_noprefill": (d0, rows0)}, a.tag.startswith("TEST_"))
    if bad:
        sys.exit("PROVENANCE MISMATCH: fit runs, basis files, frames files or the population are not the ones gated "
                 "and preregistered here; not scored")
    est = table(rows, "Primary evaluation ('Answer:' prefill): family x format")

    g = lambda fam, arm, k: est[fam, arm, k][0] if (fam, arm, k) in est else None
    both = lambda *v: None if any(x is None for x in v) else all(v)
    ge = lambda x, t=0.5: None if x is None else x >= t
    print("\nGATES")
    G1 = gate_g1(root, fits, a.p1_root)
    G2 = both(ge(g("none/", "NONE", "phi")), ge(g("p1/", "P1", "phi")))
    print(f"G2 the fits work (primary evaluation, fitting prompt): phi_none(NONE) {num(g('none/', 'NONE', 'phi'))}, phi_p1(P1) {num(g('p1/', 'P1', 'phi'))}, both >= 0.5 "
          f"-> {verdict(G2)}")
    G3 = both(ge(g("p1/", "LETTER", "psiK")), ge(g("p1/", "NONE", "psiV")))
    print(f"G3 the control reproduces the released crossover: fit_p1 psi_K(LETTER) {num(g('p1/', 'LETTER', 'psiK'))}, "
          f"psi_V(NONE) {num(g('p1/', 'NONE', 'psiV'))}, both >= 0.5 -> {verdict(G3)}")
    gates = {"G1": G1, "G2": G2, "G3": G3}
    failed = [k for k, v in gates.items() if v is False]
    skipped = [k for k, v in gates.items() if v is None]
    if failed:
        print(f"GATES FAILED: {', '.join(failed)} (report as a failed fit or a failed control; predictions are not interpreted)")
    elif skipped:
        print(f"GATES NOT EVALUATED: {', '.join(skipped)} {SKIP} (predictions are not interpreted)")
    else:
        print("ALL GATES MET: the predictions are interpreted")

    print("\nPREDICTIONS (H_read)")
    F1 = both(ge(g("none/", "LETTER", "psiK")), ge(g("none/", "NONE", "psiV")))
    print(f"F1 crossover without later mentions at fitting: fit_none psi_K(LETTER) {num(g('none/', 'LETTER', 'psiK'))}, "
          f"psi_V(NONE) {num(g('none/', 'NONE', 'psiV'))}, both >= 0.5 -> {verdict(F1)}")
    F2 = ge(g("none/", "LETTER", "phi"))
    print(f"F2 transfer to lettered options: fit_none phi(LETTER) {num(g('none/', 'LETTER', 'phi'))} >= 0.5 -> {verdict(F2)}")
    D = did(rows)
    F3 = None if D is None else (D[1] >= -0.25 and D[2] <= 0.25)
    print(f"F3 no fit-format effect on the crossover: D = {'n/a' if D is None else fmt(D) + f' (n={D[3]})'}, "
          f"95% CI within [-0.25, +0.25] -> {verdict(F3)}")
    sid = s_id(a.factorial)
    print(f"F4 the channel follows the natural read, for each of fit_none and fit_p1 (fits averaged within core): s_ID = mean ID_K / (mean ID_K + mean ID_V) over the factorial items "
          f"(not stage 2's share@0) by format "
          f"{SKIP if sid is None else {k: round(float(v), 3) for k, v in sid.items()}}")
    f4 = []
    for fam in FITS:
        sh = {arm: g(fam, arm, "psiK") / (g(fam, arm, "psiK") + g(fam, arm, "psiV"))
              for arm in ARMS if g(fam, arm, "psiK") is not None and g(fam, arm, "psiV") is not None}
        if sid is None or any(arm not in sh or arm not in sid for arm in ARMS):
            print(f"  {LABEL[fam]}: {SKIP} (needs all five formats in the frames and in the factorial)")
            f4.append(None)
            continue
        small = [x for x in ARMS if g(fam, x, "psiK") + g(fam, x, "psiV") < 0.2]
        if small:
            print(f"  {LABEL[fam]}: NOT EVALUABLE, psi_K + psi_V < 0.2 in {small} (counts as NOT MET)")
            f4.append(False)
            continue
        r = float(np.corrcoef([sh[x] for x in ARMS], [sid[x] for x in ARMS])[0, 1])
        f4.append(r >= 0.9)
        print(f"  {LABEL[fam]}: key share psi_K/(psi_K+psi_V) {', '.join(f'{x} {sh[x]:+.3f}' for x in ARMS)}; "
              f"Pearson r = {r:+.3f} (>= 0.9: {verdict(r >= 0.9)})")
    F4 = None if None in f4 else all(f4)
    print(f"F4 -> {verdict(F4)}")
    hfit = both(None if F1 is None else not F1, ge(g("none/", "LETTER", "psiV")), None if F2 is None else not F2,
                None if D is None else D[0] <= -0.4)
    print(f"H_fit pattern as stated: F1 fails, fit_none psi_V(LETTER) {num(g('none/', 'LETTER', 'psiV'))} >= 0.5, F2 fails, "
          f"D <= -0.4 -> {verdict(hfit)}")
    if failed or skipped:
        print("(predictions above are not interpreted: " + ("GATES FAILED" if failed else "gates not evaluated") + ")")

    print("\n######## EXPLORATORY")
    table(rows0, "Exploratory: evaluation without the 'Answer:' prefill: family x format")
    D0 = did(rows0)
    print(f"  D without prefill = {'n/a (' + SKIP + ')' if D0 is None else fmt(D0) + f' (n={D0[3]})'}")
    print("\nExploratory: rho_K and rho_V are the rhoK/rhoV columns above; f_star fits are the f_star column above")
    cosines(fits, a.p1_root)
    loss_curves(fits)


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        main()
