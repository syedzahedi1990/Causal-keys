"""Zero-GPU pre-analyses of stage 9 (docs/stage9/SPEC.md section 15; group S9P of paper/make_figures.py).

Post hoc re-analyses of committed stage-5 to stage-8 results, run before the stage-9 GPU run. They re-derive every number
that docs/STAGE9_PREDICTIONS.md section 1 ("Seen before this note") quotes, and the in-hand analogues of its section 2,
from the results files alone. Every function is pure: it reads committed results files under ``root`` and returns
numbers (no file is written, no model is loaded). paper/make_figures.py registers the numbers as macros of the group S9P
(``rmac``: "[recomputed; label exploratory/post hoc]"; printed score-file strings with ``smac``) and draws
paper/figures/fig_window_pre.pdf from them; tests/test_stage9_pre.py checks them against the note at its printed precision.

An estimate is a tuple (point, lower, upper) with a 95 % percentile interval (numpy linear interpolation) from the
bootstrap the source stage used:
  item 1 (stage 8b, fresh stories): two-stage cluster bootstrap over (lexicon, base, source), seed 20261013
         (analysis/stage8b_parts/stats.Boot through data.Pop, one index set per model shared by every arm);
  items 2, 4, 6 (stage 8c, E stories): two-stage cluster bootstrap over (base, source), seed 20261012
         (analysis/stage8c_parts/stats.Boot; the eval file's index set is law.Model's own);
  items 3, 8 (stage 8a, passages): two-stage article bootstrap, seed 20261010 (analysis/stage8a_parts/common.est; the
         article label is the title, as the part-A scorer's Model.arts uses it);
  item 7 (stage 5 knockout): item bootstrap, 10,000 resamples, seed 20261015 (stats.Boot of stage 8b without clusters).

Items (the design review's section 7, SPEC 15.2; item 8 is the review's "one-token copy" sentence):
  1 format_gap         the behavioural format gap on the 8 fresh-story models (E-argmax at the KV patch from round(0.3 L))
  2 onset_curves       the natural KV / K / V patch from blocks 4, 8, 12, 16 by format (E-argmax), l_c brackets
  3 passage_onsets     natural passages: OPTA key and NOM value read from layer 0 and from round(0.3 L)
  4 edit_transfer      the edits of stage 8c in generation (block 7, 40 stories), failures that stay at B, kappa/sigma
  5 binding_swap       Prakash et al.'s binding swap by format (Qwen2.5-14B), parsed from STAGE6_SCORE.txt
  6 dose, law_diagnostics   the SK/SV interpolation rows (psi_K, psi_V, behavioural flips) and the post hoc lambda law
  7 knockout           the answer's attention to the writing token cut (M3) and every non-candidate later row's (M7)
  8 one_token_copy     NOM multi-token answers: completion after the first entity token under V_S and KV_S
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from stage8a_parts import common as CA  # noqa: E402
from stage8b_parts.data import Pop  # noqa: E402
from stage8b_parts.stats import Boot as BootB  # noqa: E402
from stage8c_parts import law as LW  # noqa: E402
from stage8c_parts.stats import boot_of as boot_c  # noqa: E402

ROOT = HERE.parent
NAN = float("nan")

FRESH8 = ("mistral7", "llama8", "phi4", "olmo7", "falcon7", "gemma9", "qwen7", "qwen14")
ARMS_B = ("AFTER", "POST", "POST-NULL", "NONE", "PRE", "BEFORE")
FREE_ARMS = ("NONE", "BEFORE", "POST-NULL", "PRE")       # no later mention, or the list before the story
MODELS_C = ("qwen7", "mistral7", "llama8")
FORMATS_C = ("NONE", "POST", "P1", "LETTER")
CHANNELS = ("KV", "K", "V")
MODELS_A = ("qwen7", "mistral7", "llama8", "gemma9")
KO_FILES = {"qwen7": "Qwen2.5-7B-Instruct", "mistral7": "Mistral-7B-Instruct-v0.3", "qwen14": "Qwen2.5-14B-Instruct"}
KO_SEED, KO_NB = 20261015, 10000
DOSE_BLOCKS = (3, 7)
# the lambda-law instances, in the order the vetting script enumerated them (it fixes the unit order of the bootstrap)
LAW_INSTS = (("E1", "E2", "E3", "E4a", "E4b", "E5FR", "E5DE", "E5SYN", "E5NL")
             + tuple(f"{c}:{z}" for z in ("E1", "E2", "E3", "E4a", "E4b") for c in ("PERP", "NONLEX")) + ("NONLEX:nat",))
LAW_CODES = (("W(P1)", "P1", "P1"), ("W(POST)", "POST", "POST"), ("A_L", "LETTER", "NONE"), ("A_P", "P1", "NONE"))
LAW_PHI, LAW_FLOOR, LAW_NB, LAW_SEED = 0.3, 0.01, 2000, 1
KAPPA_INSTS = ("E1", "E2", "E3", "E4a", "E4b", "E5FR", "E5DE", "E5SYN", "E5NL")
KAPPA_MODELS = ("qwen7", "mistral7")


def _ci(pt, bs):
    """(point, 2.5th, 97.5th percentile of the defined resamples)."""
    b = np.asarray(bs, float)
    b = b[~np.isnan(b)]
    if not b.size:
        return (float(pt), NAN, NAN)
    return (float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- item 1: the behavioural format gap (stage 8b, F)
def _onsets_b(rec):
    return sorted({int(k.split("@")[1]) for k in rec["rows"] if k.startswith("KV_S@")})


def _argword(rec, row):
    return rec["words"][int(np.argmax(rec["rows"][row]["E"]))]


def format_gap(root=ROOT, models=FRESH8):
    """Per model: L, l0 = the third onset of the KV_S rows (= round(0.3 L)), n; per arm the E-argmax shares of the
    KV_S@l0 row that name B ("stale") and S ("flip"), and of KV_S@0 that name S ("flip0"); under AFTER also: the share B
    among the non-flipped items, the stale count, the median absolute p(B) = exp(E_B) and the median six-way p(B) of the
    stale items, and the generated V_S@0 answer's flip ("gen_v0")."""
    root = Path(root)
    out = {}
    for k in models:
        J = _load(root / f"results/gpu_stage8b/eval/{k}_F.json")
        P = Pop(J, k)
        L = int(J["provenance"]["n_layers"])
        ons = _onsets_b(P.recs("AFTER")[0])
        l0 = ons[2]
        assert l0 == round(0.3 * L), (k, ons, L)
        row = f"KV_S@{l0}"
        res = {"L": L, "l0": l0, "n": len(P.keys), "arms": {}}
        for arm in ARMS_B:
            R = P.recs(arm)
            a = [_argword(r, row) for r in R]
            st = np.array([x == r["track"]["B"] for x, r in zip(a, R)], float)
            fl = np.array([x == r["track"]["S"] for x, r in zip(a, R)], float)
            f0 = np.array([_argword(r, "KV_S@0") == r["track"]["S"] for r in R], float)
            res["arms"][arm] = {q: _ci(x.mean(), P.boot.means(x)) for q, x in (("stale", st), ("flip", fl), ("flip0", f0))}
            if arm == "AFTER":
                stl = [r for x, r in zip(a, R) if x == r["track"]["B"]]
                pabs, psix = [], []
                for r in stl:
                    E = np.asarray(r["rows"][row]["E"], float)
                    iB = r["words"].index(r["track"]["B"])
                    pabs.append(math.exp(E[iB]))
                    p = np.exp(E - E.max())
                    psix.append(float(p[iB] / p.sum()))
                g = np.array([r["ans"]["V_S@0"] == r["track"]["S"] for r in R], float)
                res["after"] = {"nonflip_b": float(st.sum() / (1 - fl).sum()), "n_stale": len(stl), "n_nonflip": int((1 - fl).sum()),
                                "med_abs_pb": float(np.median(pabs)), "med_six_pb": float(np.median(psix)),
                                "gen_v0": _ci(g.mean(), P.boot.means(g))}
        out[k] = res
    return out


# ---------------------------------------------------------------- item 2: onset curves by format (stage 8c eval)
def load_c(root=ROOT, models=MODELS_C):
    """law.Model per model from results/gpu_stage8c/eval/<k>.json."""
    return {k: LW.Model(_load(Path(root) / f"results/gpu_stage8c/eval/{k}.json")) for k in models}


def flip_c(M, f, l, Z, C):
    """Per-story E-argmax flip of the rows Z|C|t (t in S, X): the mean over t of 1[argmax = t] (law.Model.flip, any C)."""
    out = np.full(M.n, NAN)
    for s in range(M.n):
        a, b = M.row(s, f, l, f"{Z}|{C}|S"), M.row(s, f, l, f"{Z}|{C}|X")
        if a is None or b is None:
            continue
        out[s] = 0.5 * (float(int(a[13]) == int(M.iS[s])) + float(int(b[13]) == int(M.iX[s])))
    return out


def bracket(onsets, flips, L):
    """l_c bracket from a late-patch curve: (last onset with flip >= 0.5 before the first one below, first onset with flip
    < 0.5); the lower end is 0 when the first onset is already below; (last onset, L) and censored when none is below."""
    lo = 0
    for o, f in zip(onsets, flips):
        if f < 0.5:
            return {"lo": lo, "hi": o, "censored": False}
        lo = o
    return {"lo": lo, "hi": L, "censored": True}


def onset_curves(root=ROOT, models=None):
    """Per model: L, the depths (blocks l) and onsets l + 1; flips[C][f] = [(pt, lo, hi) per depth] of the natural rows
    nat|C|t (E-argmax, over stories and t); the l_c bracket of the KV curve per format (in onsets); the E1 rows' KV flips
    and the largest |flip(E1) - flip(nat)| over the format x block cells; tau(3) of E4b = flip(P1, 3) / flip(NONE, 3) where
    E4b exists (both seeds' rows are stored as E4a and E4b; tau uses E4b, seed 102)."""
    models = models or load_c(root)
    out = {}
    for k, M in models.items():
        L = int(M.P["n_layers"])
        res = {"L": L, "depths": list(M.depths), "onsets": [l + 1 for l in M.depths], "n": M.n, "flips": {}, "lc": {}, "eone": {}}
        for C in CHANNELS:
            res["flips"][C] = {f: [_ci(*M.boot.mean(flip_c(M, f, l, "nat", C))) for l in M.depths] for f in FORMATS_C}
        for f in FORMATS_C:
            res["lc"][f] = bracket(res["onsets"], [x[0] for x in res["flips"]["KV"][f]], L)
        gap = (-1.0, None)
        for f in FORMATS_C:
            res["eone"][f] = [_ci(*M.boot.mean(flip_c(M, f, l, "E1", "KV"))) for l in M.depths]
            for j, l in enumerate(M.depths):
                g = abs(res["eone"][f][j][0] - res["flips"]["KV"][f][j][0])
                if g > gap[0]:
                    gap = (g, (f, l))
        res["eone_gap"] = {"max": gap[0], "cell": gap[1]}
        if M.has("NONE", 3, "E4b|KV|S"):
            fp, fn = np.nanmean(flip_c(M, "P1", 3, "E4b", "KV")), np.nanmean(flip_c(M, "NONE", 3, "E4b", "KV"))
            res["tau_e4b_3"] = {"P1": float(fp), "NONE": float(fn), "tau": float(fp / fn)}
        out[k] = res
    return out


# ---------------------------------------------------------------- item 3: passages (stage 8a explore, onset_piece)
DEC = ("B", "S", "X", "Z", "D")
PSG_ROWS = {"NOM": ("V_S", "V_S@on"), "OPTA": ("K_S", "K_S@on")}


def _argdec(lp):
    assert set(lp) == set(DEC), sorted(lp)
    return max(DEC, key=lambda Y: lp[Y])


def passage_onsets(root=ROOT, models=MODELS_A):
    """Per model: L, onset (= round(0.3 L)), n; share[f][row] = the share of the items whose argmax over the five decision
    tokens is S (NOM: V_S from layer 0 and V_S@on from the onset; OPTA: K_S and K_S@on); ratio[f] = the ratio of the
    mean margin shifts d(row) = m(row) - m(ID), m = lp_S - lp_B, from the onset to from layer 0 (the scorer's printed
    'fractions K_S@on / V_S@on'); every estimate with the article bootstrap."""
    out = {}
    for k in models:
        J = _load(Path(root) / f"results/gpu_stage8a/explore/{k}.json")
        L, on = int(J["provenance"]["n_layers"]), int(J["provenance"]["onset"])
        assert on == round(0.3 * L), (k, on, L)
        res = {"L": L, "onset": on, "share": {}, "ratio": {}, "n": {}}
        for f, rows in PSG_ROWS.items():
            its = J["parts"]["onset_piece"][f]
            arts = [it["title"] for it in its]
            res["n"][f] = len(its)
            res["share"][f] = {}
            for r in rows:
                x = np.array([_argdec(it["rows"][r]["lp"]) == "S" for it in its], float)
                e = CA.est(arts, CA.mean, x)
                res["share"][f][r] = (e.pt, e.lo, e.hi)
            m = lambda it, r: it["rows"][r]["lp"]["S"] - it["rows"][r]["lp"]["B"]  # noqa: E731
            d0 = np.array([m(it, rows[0]) - m(it, "ID") for it in its])
            d1 = np.array([m(it, rows[1]) - m(it, "ID") for it in its])
            e = CA.est(arts, CA.ratio, d1, d0)
            res["ratio"][f] = (e.pt, e.lo, e.hi)
        out[k] = res
    return out


# ---------------------------------------------------------------- item 4: edit transfer in generation (stage 8c explore)
def kappa_line(score_lines, k):
    """(line index, sigma, {instance: kappa}) of the printed 'P1@7: sigma ...; kappa ...' line of model k in
    STAGE8C_SCORE.txt, located by content: the first such line after the header '-- <k>: kappa against sigma'."""
    hd = [i for i, x in enumerate(score_lines) if x.startswith(f"-- {k}: kappa against sigma per cell")]
    assert len(hd) == 1, (k, hd)
    for i in range(hd[0] + 1, len(score_lines)):
        x = score_lines[i]
        if x.startswith("-- "):
            break
        mm = re.match(r"^\s+P1@7: sigma ([+-]\d\.\d+); kappa (.*)$", x)
        if mm:
            kap = {}
            for part in mm[2].split(", "):
                z, v = part.split()
                kap[z] = v
            return i, mm[1], kap
    raise AssertionError(f"no P1@7 kappa line for {k}")


def edit_transfer(root=ROOT, models=MODELS_C):
    """Per model (block 7, the first 40 E stories): flip[Z][f] = mean over stories of the mean over t in (S, X) of
    1[generated answer = t], Z in nat and the file's instances, f in NONE, P1 (two-stage (base, source) bootstrap of the 40
    stories, seed 20261012); fail_b[Z] = among the (story, t) units whose P1 answer is not t, the share answering B (and their
    number). pairs: (model, Z) of qwen7 and mistral7 among the identity edits with a printed kappa at P1@7 and NONE flip >=
    0.5: kappa/sigma from the printed line and the transfer ratio (flip(P1) / flip_nat(P1)) / flip(NONE); rho = their
    Spearman correlation."""
    root = Path(root)
    S8C = (root / "results/gpu_stage8c/STAGE8C_SCORE.txt").read_text().splitlines()
    out = {}
    for k in models:
        J = _load(root / f"results/gpu_stage8c/explore/{k}.json")
        St = J["stories"]
        B = boot_c([(s["core"]["base"], s["core"]["source"]) for s in St])
        res = {"layer": int(J["provenance"]["layer"]), "n": len(St), "insts": ["nat"] + list(J["instances"]), "flip": {}, "fail_b": {}}
        for z in res["insts"]:
            res["flip"][z] = {}
            for f in ("NONE", "P1"):
                x = np.array([np.mean([s["answers"][f][f"{z}|{t}"] == s[t] for t in ("S", "X")]) for s in St], float)
                res["flip"][z][f] = _ci(*B.mean(x))
            fails = [s["answers"]["P1"][f"{z}|{t}"] == s["B"] for s in St for t in ("S", "X") if s["answers"]["P1"][f"{z}|{t}"] != s[t]]
            res["fail_b"][z] = (float(np.mean(fails)) if fails else NAN, len(fails))
        out[k] = res
    pairs = []
    for k in KAPPA_MODELS:
        if k not in out:
            continue
        i, sig, kap = kappa_line(S8C, k)
        out[k]["kappa_line"] = (i, sig, kap)
        natp = out[k]["flip"]["nat"]["P1"][0]
        for z in KAPPA_INSTS:
            if z not in kap or z not in out[k]["flip"]:
                continue
            fn, fp = out[k]["flip"][z]["NONE"][0], out[k]["flip"][z]["P1"][0]
            if fn >= 0.5:
                pairs.append({"k": k, "z": z, "kappa": kap[z], "sigma": sig, "rel": float(kap[z]) / float(sig),
                              "transfer": (fp / natp) / fn, "none": fn, "p1": fp, "nat_p1": natp})
    rho = float(spearmanr([p["rel"] for p in pairs], [p["transfer"] for p in pairs])[0]) if len(pairs) > 2 else NAN
    return {"models": out, "pairs": pairs, "rho": rho}


# ---------------------------------------------------------------- item 5: the binding swap by format (stage 6 score file)
def sweeps(score_lines):
    """{(arm, format): (line index, {block: (IIA string, Phi string)})} of the printed sweep lines of STAGE6_SCORE.txt."""
    out = {}
    for i, x in enumerate(score_lines):
        mm = re.match(r"\s+sweep (BIND|ID)\s+(\S+)\s+IIA/Phi: (.*)$", x)
        if mm:
            out[mm[1], mm[2]] = (i, {int(b): tuple(v.split("/")) for b, v in (p.split(":") for p in mm[3].split())})
    return out


def lstar_line(score_lines):
    """(line index, l*) of the printed 'l* = <n> (lstar.json <n>)' line of STAGE6_SCORE.txt (the preregistered BIND layer
    of stage 6, re-derived there from the sweep files; located by content, the two printed values equal)."""
    hits = [(i, mm) for i, x in enumerate(score_lines) if (mm := re.match(r"\s+l\* = (\d+) \(lstar\.json (\d+)\)", x))]
    assert len(hits) == 1, [i for i, _ in hits]
    i, mm = hits[0]
    assert mm[1] == mm[2], score_lines[i]
    return i, int(mm[1])


def binding_swap(root=ROOT, blocks=(18, 40)):
    """BIND IIA by block (Qwen2.5-14B): the largest printed IIA over blocks[0]..blocks[1] in each named format (QNAMES,
    OPTIONS-AFTER, LETTERS-AFTER) with its line; under NO-MENTION the printed IIA at the preregistered l* (the anchor,
    'none_lstar': sweep line, IIA string, l*, l* line) and, as a secondary best-block value, the largest over all blocks
    ('none'); the ID span patch: the last block of the run of 1.00 from block 0 under NO-MENTION, and the first block from
    which every later block is 0.00 under OPTIONS-AFTER."""
    S6 = (Path(root) / "results/gpu_stage6/STAGE6_SCORE.txt").read_text().splitlines()
    sw = sweeps(S6)
    out = {"named": {}, "blocks": blocks}
    for f in ("QNAMES", "OPTIONS-AFTER", "LETTERS-AFTER"):
        i, d = sw["BIND", f]
        assert all(b in d for b in range(blocks[0], blocks[1] + 1)), f
        best = max(range(blocks[0], blocks[1] + 1), key=lambda b: float(d[b][0]))
        out["named"][f] = (i, d[best][0], best)
    i, d = sw["BIND", "NO-MENTION"]
    best = max(d, key=lambda b: float(d[b][0]))
    out["none"] = (i, d[best][0], best)
    il, ls = lstar_line(S6)
    out["none_lstar"] = (i, d[ls][0], ls, il)
    i, d = sw["ID", "NO-MENTION"]
    last = -1
    for b in sorted(d):
        if d[b][0] != "1.00":
            break
        last = b
    out["id_none_one_to"] = (i, last)
    i, d = sw["ID", "OPTIONS-AFTER"]
    bs = sorted(d)
    first = next(b for b in bs if all(d[c][0] == "0.00" for c in bs if c >= b))
    out["id_opt_zero_from"] = (i, first)
    return out


# ---------------------------------------------------------------- item 6: dose curves and the post hoc lambda law (stage 8c)
def dose(root=ROOT, models=None):
    """Per model, at blocks 3 and 7 and every format: psi_K(f, l, a) = mean ID_K^{SK a} / mean ID_K^nat (law.Model.psi with
    the K rows of SK50, SK80), psi_V likewise from the V rows of SV50, SV80; 'used' = the channel's natural read is >= 2
    nats and >= 0.1 of ID_KV (law.Model.used); flip[row][f, l] = the E-argmax flip of the row's KV rows (SK50, SK80, SV50,
    SV80, nat) over stories and t. At qwen7 also the per-story ratio ID(P1, 3, SK50, K) / ID(P1, 3, nat, K) over the stories
    with natural ID_K > 2 nats: its 10th / 50th / 90th percentiles and the SK80 median."""
    models = models or load_c(root)
    out = {}
    for k, M in models.items():
        res = {"psi": {}, "used": {}, "flip": {}}
        for l in DOSE_BLOCKS:
            for f in FORMATS_C:
                for C in ("K", "V"):
                    res["used"][f, l, C] = bool(M.used(f, l, C))
                    for a in (50, 80):
                        q = M.psi(f, l, f"S{C}{a}", C)
                        res["psi"][C, f, l, a] = (q.pt, *q.ci())
                for Z in ("SK50", "SK80", "SV50", "SV80", "nat"):
                    res["flip"][Z, f, l] = _ci(*M.boot.mean(flip_c(M, f, l, Z, "KV")))
        if k == "qwen7":
            n_ = M.ID("P1", 3, "nat", "K")
            sel = n_ > 2
            r50 = M.ID("P1", 3, "SK50", "K")[sel] / n_[sel]
            r80 = M.ID("P1", 3, "SK80", "K")[sel] / n_[sel]
            res["story"] = {"n": int(sel.sum()), "p10": float(np.percentile(r50, 10)), "p50": float(np.percentile(r50, 50)),
                            "p90": float(np.percentile(r50, 90)), "med80": float(np.median(r80)), "above_half": float(np.mean(r50 > 0.5))}
        out[k] = res
    return out


def curve(a, c50, c80):
    """The calibrated dose curve: 0 for a <= 0; linear through (0, 0), (0.5, c50), (0.8, c80), (1, 1) for a <= 1; slope
    (1 - c80) / 0.2 above 1."""
    if a <= 0:
        return 0.0
    if a <= 1:
        return float(np.interp(a, [0, 0.5, 0.8, 1.0], [0, c50, c80, 1.0]))
    return 1 + (a - 1) * (1 - c80) / 0.2


def lg(x):
    return float(np.log(max(x, LAW_FLOOR)))


def law_stats(models):
    """The post hoc lambda statistics (SPEC 15.2 item 6): per (model, instance Z, depth l) with phi(NONE) >= 0.3 and per
    code (f_K, f_V) whose rows exist and whose natural channels are used: lambda = Lg(psi_K) - Lg(psi_V), the curve
    predictor Lg(curve_K(p_k)) - Lg(curve_V(p_v)), O1 = Lg(p_k) - Lg(p_v) and the baseline Lg(p_k)."""
    out = []
    for k, M in models.items():
        mean = lambda f, l, Z, C: float(np.mean(M.ID(f, l, Z, C)))  # noqa: E731  (every story has the row or none does)
        for z in LAW_INSTS:
            for l in M.depths:
                if not M.has("NONE", l, f"{z}|KV|S"):
                    continue
                if not mean("NONE", l, z, "KV") / mean("NONE", l, "nat", "KV") >= LAW_PHI:
                    continue
                st = [M.S[s]["stats"][str(l)][f"{z}|{t}"] for s in range(M.n) for t in ("S", "X")]
                pk, pv = float(np.mean([x["proj_k"] for x in st])), float(np.mean([x["proj_v"] for x in st]))
                for code, fk, fv in LAW_CODES:
                    if not (M.has(fk, l, f"{z}|K|S") and M.has(fv, l, f"{z}|V|S")):
                        continue
                    if not (M.used(fk, l, "K") and M.used(fv, l, "V")):
                        continue
                    nk, nv = mean(fk, l, "nat", "K"), mean(fv, l, "nat", "V")
                    psk, psv = mean(fk, l, z, "K") / nk, mean(fv, l, z, "V") / nv
                    ck = curve(pk, mean(fk, l, "SK50", "K") / nk, mean(fk, l, "SK80", "K") / nk)
                    cv = curve(pv, mean(fv, l, "SV50", "V") / nv, mean(fv, l, "SV80", "V") / nv)
                    out.append({"k": k, "z": z, "l": l, "code": code, "lam": lg(psk) - lg(psv), "cur": lg(ck) - lg(cv),
                                "o1": lg(pk) - lg(pv), "lpk": lg(pk), "pk": pk, "pv": pv})
    return out


def law_diagnostics(models=None, root=ROOT):
    """The rank and calibration diagnostics of the lambda law over law_stats: Spearman of the curve predictor and of log
    proj_k with lambda, their difference and its 95 % interval from 2,000 resamples of the units (model, Z, depth) drawn
    with numpy.random.default_rng(1) (units in first-seen order), the resample mean, MAE of the curve and of O1, the partial
    rank correlation of the curve with lambda given log proj_k (ranks residualised on [1, rank(log p_k)]) and
    Spearman(p_k, p_v)."""
    models = models or load_c(root)
    st = law_stats(models)
    a = lambda key, S=st: np.array([o[key] for o in S])  # noqa: E731
    lam, cur, lpk, o1 = a("lam"), a("cur"), a("lpk"), a("o1")
    units = {}
    for o in st:
        units.setdefault((o["k"], o["z"], o["l"]), []).append(o)
    keys = list(units)
    rng = np.random.default_rng(LAW_SEED)
    d = []
    for _ in range(LAW_NB):
        S = [o for j in rng.integers(0, len(keys), len(keys)) for o in units[keys[j]]]
        lm = a("lam", S)
        d.append(spearmanr(a("cur", S), lm)[0] - spearmanr(a("lpk", S), lm)[0])
    r = rankdata
    X = np.c_[np.ones(len(st)), r(lpk)]
    res_l = r(lam) - X @ np.linalg.lstsq(X, r(lam), rcond=None)[0]
    res_c = r(cur) - X @ np.linalg.lstsq(X, r(cur), rcond=None)[0]
    rc, rl = float(spearmanr(cur, lam)[0]), float(spearmanr(lpk, lam)[0])
    return {"n_stats": len(st), "n_units": len(keys), "rho_curve": rc, "rho_logpk": rl, "diff": rc - rl,
            "diff_lo": float(np.percentile(d, 2.5)), "diff_hi": float(np.percentile(d, 97.5)), "diff_mean": float(np.mean(d)),
            "mae_curve": float(np.mean(np.abs(lam - cur))), "mae_o1": float(np.mean(np.abs(lam - o1))),
            "partial": float(np.corrcoef(res_l, res_c)[0, 1]), "rho_pk_pv": float(spearmanr(a("pk"), a("pv"))[0])}


# ---------------------------------------------------------------- item 7: knockout (stage 5)
def knockout(root=ROOT, models=tuple(KO_FILES)):
    """Per model: keep_b[f, M] = the share of items whose answer (argmax over the candidates of the masked self-clamp row
    ID, as experiments/attention_knockout.per_core's acc_B) stays at the base location, f in AFTER, P1 and M in M0, M3, M7;
    under NONE and M3 the share sent to the initial or the distractor location ("init") and the share kept at B; item
    bootstrap (10,000, seed 20261015) per (format, mask) population."""
    out = {}
    for k in models:
        J = _load(Path(root) / f"results/gpu_stage5/knockout/{KO_FILES[k]}_s0.json")
        res = {"keep_b": {}, "n": {}}

        def est(x):
            x = np.asarray(x, float)
            return _ci(x.mean(), BootB(None, n=len(x), nb=KO_NB, seed=KO_SEED).means(x))

        for f, masks in (("AFTER", ("M0", "M3", "M7")), ("P1", ("M0", "M3", "M7")), ("NONE", ("M0", "M3"))):
            for mk in masks:
                R = [r for r in J["results"] if r["arm"] == f and mk in r["m"]]
                a = [r["m"][mk]["ID"]["argmax_cand"] for r in R]
                res["n"][f, mk] = len(R)
                res["keep_b"][f, mk] = est([x == r["core"]["base"] for x, r in zip(a, R)])
                if f == "NONE" and mk == "M3":
                    res["init"] = est([x in (r["core"]["initial"], r["core"]["distractor_location"]) for x, r in zip(a, R)])
        out[k] = res
    return out


# ---------------------------------------------------------------- item 8: one-token copy (stage 8a factorial, NOM)
COPY_ROWS = (("V_S", "S"), ("KV_S", "S"))


def one_token_copy(root=ROOT, models=MODELS_A):
    """Per model and row in (V_S, KV_S): among the NOM items of type PERSON or PLACE whose S entity has at least two tokens
    from the decision token on (len(c_S) - j >= 2) and whose generated first entity token is S's decision token, the share
    whose generated answer is S (the whole entity); article bootstrap. The value-only clamp copies the first token; the
    continuation needs the key (SPEC: the design review's 'one-token copy')."""
    out = {}
    for k in models:
        J = _load(Path(root) / f"results/gpu_stage8a/factorial/{k}.json")
        its = J["formats"]["NOM"]["items"]
        res = {}
        for row, Y in COPY_ROWS:
            sel = [it for it in its if it["sub"] in ("PERSON", "PLACE") and len(it["c"][Y]) - it["j"] >= 2
                   and it["gen"][row]["g1"] == it["dec"][Y]]
            x = np.array([it["gen"][row]["who"] == Y for it in sel], float)
            e = CA.est([it["title"] for it in sel], CA.mean, x)
            res[row] = {"est": (e.pt, e.lo, e.hi), "n": len(sel), "arts": len({it["title"] for it in sel})}
        out[k] = res
    return out


def all_items(root=ROOT):
    """Every item at once (the stage-8c eval files are read once)."""
    mc = load_c(root)
    return {"format_gap": format_gap(root), "onset_curves": onset_curves(root, mc), "passage_onsets": passage_onsets(root),
            "edit_transfer": edit_transfer(root), "binding_swap": binding_swap(root), "dose": dose(root, mc),
            "law": law_diagnostics(mc), "knockout": knockout(root), "one_token_copy": one_token_copy(root)}
