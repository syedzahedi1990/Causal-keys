"""Part (a) of P-2026-10-05-H (docs/PREREGISTRATION.md): Gates a2, a3 and predictions H1-H6 on the outputs of
experiments/stage6_heads.py ({root}/{model}.json), as worded in the entry. Called by analysis/stage6_score.py
(``score(root)``) or standalone (``python analysis/stage6_parts/heads.py --root DIR``).

Statistics: means over the evaluation stories E; 95 % percentile intervals from 10,000 story-bootstrap resamples with
one fixed index set per n (seed 20261005), every ratio of means and paired difference recomputed within each resample;
verdicts on the point estimates and the bounds the entry names. Every batched quantity is differenced against its
in-batch reference row (none / all_G / ID); the single passes (clean, full clamp) enter only Gates a2 and a3.
  R(k) = mean[m(top-k) - m(none)] / mean d_G, d_G = m(all_G) - m(none) (sufficiency batch of that head set);
  KO(k) = 1 - mean[m(all but top-k) - m(none_KO)] / mean[m(all_G,KO) - m(none_KO)] (knockout batch);
  rho_K(c) = mean ID_K(c) / mean ID_K(none); dV(c) = paired ID_V(c) - ID_V(none);
  r_X = 1 - mean[m(row X) - m(ID)] / mean[m(K_S) - m(ID)] (second-hop batch rows, HOP_ROWS of the experiment).
Gate a2 (per model and format): mean |m(none) - m(clean pass)| and mean |m(all_T) - m(full clamp pass)| <= max(0.5,
  0.02 x mean d_full), all_T = the in-batch row where every head sees K_S in every row (the full key clamp; the all_G
  row equals the option-row splice, not the full clamp, and its gap is printed as a raw floor); every story's hop
  exactness row within 0.1 nats of its K_S row.
Gate a3 (per model): mean d_full >= 10 under OPTIONS-AFTER and > 0 under SENTENCE-AFTER; mean d_G / mean d_full >= 0.8
  under OPTIONS-AFTER (d_G of the a3 sufficiency batch).
A model's OPTIONS-AFTER predictions are NOT EVALUABLE when its Gate a3 or its OPTIONS-AFTER Gate a2 fails (the computed
verdict is still printed); the SENTENCE-AFTER lines and H6 also need the SENTENCE-AFTER Gate a2. A prediction over both
models is MET when it is met in both, NOT MET when it is not met in at least one evaluable model, else NOT EVALUABLE (a
missing model or one that fails its gates is not evaluable); the same rule combines the parts of H2 and H3 within a model.
The required models are the preregistered two when either file is present, else every model found (TEST_MODE).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, spearmanr

np.seterr(all="ignore")
MODELS = ["Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"]
DV_FLOOR = {"Qwen2.5-7B-Instruct": 2.7, "Mistral-7B-Instruct-v0.3": 2.8}   # 0.25 x [ID_V(NO-MENTION) - ID_V(OPTIONS-AFTER)], stage 1
N_PREREG = 60
HOP = ["ID", "K_S", "ans_K", "ans_V", "ans_KV", "all_KV", "other_KV", "exact"]
SEED, NB = 20261005, 10000
NAN = (float("nan"),) * 3
_W = {}


def W(n):
    """[NB, n] resampling weights (count / n) of one fixed index set per n: W @ x = the resampled means of x."""
    if n not in _W:
        idx = np.random.default_rng(SEED).integers(0, n, (NB, n))
        w = np.zeros((NB, n))
        np.add.at(w, (np.arange(NB)[:, None], idx), 1.0 / n)
        _W[n] = w
    return _W[n]


def est(fn, *xs):
    """(point, lo, hi) of fn(mean x1, mean x2, ...) over stories (rows of each x)."""
    xs = [np.asarray(x, float) for x in xs]
    n = len(xs[0])
    if not n:
        return NAN
    pt = fn(*[x.mean(0) for x in xs])
    bs = np.asarray(fn(*[W(n) @ x for x in xs]), float)
    if np.all(np.isnan(bs)):
        return float(pt), float("nan"), float("nan")
    return float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def f3(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def V(ok):
    return "NOT EVALUABLE" if ok is None else "MET" if ok else "NOT MET"


def comb(subs):
    """MET when every part is met, NOT MET when an evaluable part is not met, else NOT EVALUABLE (also with no part)."""
    subs = list(subs)
    return False if any(s is not None and not s for s in subs) else None if not subs or any(s is None for s in subs) else True


def ratio(a, b):
    return a / b


def one_minus(a, b):
    return 1 - a / b


class Model:
    """One results file: per-story vectors over E (``ev``), the provenance, the head grids."""

    def __init__(self, name, J):
        self.name, self.J, self.P = name, J, J["provenance"]
        self.KS, self.kstar, self.n_rand = self.P["KS_eff"], self.P["kstar"], self.P["n_rand"]
        self.n_heads, self.ik = self.P["n_heads"], self.P["KS_eff"].index(self.P["kstar"])
        self.dup = {k: np.array(v) for k, v in J["dup"].items()}

    def arms(self):
        return list(self.J["arms"])

    def E(self, arm):
        return self.J["arms"][arm]["eval"] if arm in self.J["arms"] else []

    def v(self, arm, f):
        return np.array([f(e) for e in self.E(arm)], float)

    def dfull(self, arm):
        return self.v(arm, lambda e: e["mF"] - e["mB"])

    def dG(self, arm, s="a3"):
        return self.v(arm, lambda e: e["curves"][s]["allG"] - e["curves"][s]["none"])

    def has(self, arm, s="a3"):
        E = self.E(arm)
        return bool(E) and s in E[0]["curves"]

    def R(self, arm, s, i):
        return est(ratio, self.v(arm, lambda e: e["curves"][s]["suff"][i] - e["curves"][s]["none"]), self.dG(arm, s))

    def KO(self, arm, s, i):
        c = lambda e: e["curves"][s]  # noqa: E731
        return est(one_minus, self.v(arm, lambda e: c(e)["ko"][i] - c(e)["ko_none"]), self.v(arm, lambda e: c(e)["ko_allG"] - c(e)["ko_none"]))

    def rand_mean(self, arm, i, ko=False):
        """Mean over the random draws of R(k) or KO(k), recomputed within each resample."""
        vs = []
        for r in range(self.n_rand):
            c = lambda e, r=r: e["curves"][f"rand{r}"]  # noqa: E731
            if ko:
                vs += [self.v(arm, lambda e, c=c: c(e)["ko"][i] - c(e)["ko_none"]), self.v(arm, lambda e, c=c: c(e)["ko_allG"] - c(e)["ko_none"])]
            else:
                vs += [self.v(arm, lambda e, c=c: c(e)["suff"][i] - c(e)["none"]), self.v(arm, lambda e, c=c: c(e)["allG"] - c(e)["none"])]
        if not vs:
            return NAN
        f = (lambda *x: np.mean([1 - x[2 * j] / x[2 * j + 1] for j in range(len(x) // 2)], 0)) if ko else \
            (lambda *x: np.mean([x[2 * j] / x[2 * j + 1] for j in range(len(x) // 2)], 0))
        return est(f, *vs)

    def k80(self, arm, s):
        for i, k in enumerate(self.KS):
            if self.R(arm, s, i)[0] >= 0.8:
                return k
        return None

    def ab(self, arm, cond, key):
        return self.v(arm, lambda e: float(e["ablation"][cond][key]))

    def conds(self, arm):
        E = self.E(arm)
        return list(E[0]["ablation"]) if E else []

    def hop(self, arm):
        return np.array([e["hop"]["m"] for e in self.E(arm)], float).reshape(-1, len(HOP))

    def rank(self, arm, key="a3"):
        return [tuple(c) for c in self.J["arms"][arm]["rankings"][key]]

    def grid(self, arm, key, part="rank"):
        X = np.array([r[key] for r in self.J["arms"][arm][part]], float)
        return np.nanmean(X, 0)


# ---------------------------------------------------------------- gates
def gate_a2(M, arm):
    """(verdict, text) for one model and format."""
    if not M.has(arm):
        return None, "no results"
    E, df = M.E(arm), M.dfull(arm)
    bound = max(0.5, 0.02 * df.mean())
    nf = np.mean([abs(e["curves"]["a3"]["none"] - e["mB"]) for e in E])
    af = np.mean([abs(e["curves"]["a3"]["allT"] - e["mF"]) for e in E])
    gf = np.mean([abs(e["curves"]["a3"]["allG"] - e["mF"]) for e in E])
    H = M.hop(arm)
    gap = np.abs(H[:, 7] - H[:, 1])
    ok = nf <= bound and af <= bound and gap.max() <= 0.1
    return ok, (f"|none - clean| {nf:.3f}, |all_T - full| {af:.3f} <= {bound:.3f}; hop |exact - K_S| max {gap.max():.4f} "
                f"(mean {gap.mean():.4f}) <= 0.1  [raw: |all_G - full| {gf:.3f}]")


def gate_a3(M):
    if not (M.has("P1") and M.has("POST")):
        return None, "needs both formats", {}
    d1, d2 = est(lambda a: a, M.dfull("P1")), est(lambda a: a, M.dfull("POST"))
    g = est(ratio, M.dG("P1"), M.dfull("P1"))
    subs = [d1[0] >= 10, d2[0] > 0, g[0] >= 0.8]
    return all(subs), (f"d_full OPTIONS-AFTER {f3(d1)} >= 10: {V(subs[0])}; SENTENCE-AFTER {f3(d2)} > 0: {V(subs[1])}; "
                       f"d_G/d_full OPTIONS-AFTER {f3(g)} >= 0.8: {V(subs[2])}"), dict(dfull_P1=d1, dfull_POST=d2, dG_ratio=g)


# ---------------------------------------------------------------- predictions (one model, OPTIONS-AFTER unless stated)
def h1(M, arm="P1"):
    r = M.R(arm, "a3", M.ik)
    k80s = {s: M.k80(arm, s) for s in ("a3", "fplus", "dminus") if M.has(arm, s)}
    return r[0] >= 0.8 and r[1] >= 0.7, f"R({M.kstar}) {f3(r)} (>= 0.8, lower >= 0.7); k80 " + ", ".join(f"{s} {k}" for s, k in k80s.items()), r


def h2(M, arm="P1"):
    ko, rr, kr = M.KO(arm, "a3", M.ik), M.rand_mean(arm, M.ik), M.rand_mean(arm, M.ik, ko=True)
    subs = [ko[0] >= 0.8, None if math.isnan(rr[0]) else rr[0] <= 0.25, None if math.isnan(kr[0]) else kr[0] <= 0.25]
    return comb(subs), (f"KO({M.kstar}) {f3(ko)} >= 0.8: {V(subs[0])}; R_rand {f3(rr)} <= 0.25: {V(subs[1])}; "
                        f"KO_rand {f3(kr)} <= 0.25: {V(subs[2])} (means over {M.n_rand} draws)"), dict(KO=ko, R_rand=rr, KO_rand=kr)


def h3(M, arm="P1"):
    C = M.conds(arm)
    if "top_kstar_mean" not in C:
        return None, "no ablation records", {}
    k0, v0 = M.ab(arm, "none", "idK"), M.ab(arm, "none", "idV")
    rho = est(ratio, M.ab(arm, "top_kstar_mean", "idK"), k0)
    a = rho[0] <= 0.5 and rho[2] <= 0.6
    mass = M.ab(arm, "top_kstar_mean", "mass") >= 0.9
    fm = est(lambda x: x, mass.astype(float))
    b = fm[0] >= 0.8
    ctrl = [c for c in C if c.startswith("rand") and c.endswith("_kstar_mean")] + ["active_kstar_mean"]
    rc = {c: est(ratio, M.ab(arm, c, "idK"), k0) for c in ctrl if c in C}
    c_ = all(t[0] >= 0.75 for t in rc.values()) if len(rc) == len(ctrl) and rc else None
    dv = est(lambda x: x, M.ab(arm, "top_kstar_mean", "idV") - v0)
    floor = DV_FLOOR.get(M.name)
    d = None if floor is None else dv[1] > 0 and dv[0] >= floor
    base = est(lambda x: x, M.ab(arm, "top_kstar_mean", "base_ok"))
    e = (base[0] >= 0.8) if d else None
    txt = (f"(a) rho_K {f3(rho)} <= 0.5, upper <= 0.6: {V(a)}; (b) clean-B mass >= 0.9 in {fm[0]:.2f} of stories (>= 0.8): {V(b)}; "
           f"(c) rho_K " + ", ".join(f"{c.replace('_kstar_mean', '')} {t[0]:+.3f}" for c, t in rc.items()) + f" each >= 0.75: {V(c_)}; "
           f"(d) dV {f3(dv)} > 0 (CI excl. 0) and >= floor {'n/a' if floor is None else floor}: {V(d)}; "
           f"base-argmax rate {base[0]:.2f}" + (f" >= 0.8 (predicted as (d) is met): {V(e)}" if d else " (predicted >= 0.8 only if (d) is met: not applicable)"))
    return comb([a, b, c_, d] + ([e] if d else [])), txt, dict(rho=rho, rho_ctrl=rc, mass=fm, dV=dv, base=base)


def causal_set(M):
    k80 = M.k80("P1", "a3")
    kc = M.kstar if k80 is None else min(k80, M.kstar)
    return kc, M.rank("P1")[:kc]


def med_over(M, key, C):
    """median over the heads C of a per-sequence score (D, I), with a sequence bootstrap."""
    X = M.dup[f"{key}_seq"][:, [c[0] for c in C], [c[1] for c in C]]
    return est(lambda x: np.median(x, -1), X)


def tdup_over(M, arm, C, key="tdup", n="n_dup"):
    R = M.J["arms"][arm]["rank"]
    S = np.array([np.array(r[key])[[c[0] for c in C], [c[1] for c in C]] for r in R], float)
    N = np.array([r[n] for r in R], float)
    return est(lambda s, k: np.median(s / np.expand_dims(k, -1), -1), S, N)


def tdup_grid(M, arm, key="tdup", n="n_dup"):
    R = M.J["arms"][arm]["rank"]
    return np.sum([r[key] for r in R], 0) / max(1, sum(r[n] for r in R))


def h4(M):
    kc, C = causal_set(M)
    D, I, T = med_over(M, "D", C), med_over(M, "I", C), tdup_over(M, "P1", C)
    subs = [D[0] >= 0.2, I[0] <= 0.1, T[0] >= 0.2]
    A3 = M.grid("P1", "a3")
    top_a3 = set(M.rank("P1")[:10])
    order = np.argsort(-M.dup["D"].ravel(), kind="stable")[:10]
    top_d = {tuple(map(int, np.unravel_index(i, A3.shape))) for i in order}
    ov = len(top_a3 & top_d)
    p = hypergeom.sf(ov - 1, M.n_heads, 10, 10)
    TD = tdup_grid(M, "P1")
    rd, rt = spearmanr(A3.ravel(), M.dup["D"].ravel())[0], spearmanr(A3.ravel(), TD.ravel())[0]
    txt = (f"C = top-{kc} by a3: median D {f3(D)} >= 0.2: {V(subs[0])}; median I {f3(I)} <= 0.1: {V(subs[1])}; "
           f"median T_dup {f3(T)} >= 0.2: {V(subs[2])}; secondary: |top10(a3) & top10(D)| = {ov} (>= 3: {V(ov >= 3)}, "
           f"P = {p:.1e}), Spearman(a3, D) {rd:+.3f}, Spearman(a3, T_dup) {rt:+.3f}")
    return all(subs), txt, dict(kC=kc, D=D, I=I, T_dup=T, overlap=ov, P=p, rho_D=rd, rho_T=rt)


def hop_r(M, arm):
    H = M.hop(arm)
    F = H[:, 1] - H[:, 0]
    r = {n: est(one_minus, H[:, i] - H[:, 0], F) for i, n in enumerate(HOP) if i >= 2}
    return H, F, r


def h5(M, arm="P1"):
    if not M.has(arm):
        return None, "no results", {}
    H, F, r = hop_r(M, arm)
    diff = est(lambda a, o, f: (o - a) / f, H[:, 4] - H[:, 0], H[:, 6] - H[:, 0], F)
    kv = est(lambda k, v, f: (v - k) / f, H[:, 2] - H[:, 0], H[:, 3] - H[:, 0], F)
    ra, rl = r["ans_KV"], r["all_KV"]
    if arm == "P1":
        subs = [ra[0] >= 0.5 and ra[1] >= 0.4, diff[1] > 0, rl[0] >= 0.8]
        txt = (f"r_ans(KV) {f3(ra)} >= 0.5, lower >= 0.4: {V(subs[0])}; r_ans - r_other {f3(diff)} > 0 (CI excl. 0): {V(subs[1])} "
               f"[r_other {f3(r['other_KV'])}]; r_all(KV) {f3(rl)} >= 0.8: {V(subs[2])}; secondary r_ans(K) - r_ans(V) {f3(kv)} > 0: "
               f"{V(kv[0] > 0)} [r_ans(K) {r['ans_K'][0]:+.3f}, r_ans(V) {r['ans_V'][0]:+.3f}]; strong r_ans(KV) >= 0.7: {V(ra[0] >= 0.7)}")
    else:
        subs = [ra[0] >= 0.35, rl[0] >= 0.5]
        txt = f"r_ans(KV) {f3(ra)} >= 0.35: {V(subs[0])}; r_all(KV) {f3(rl)} >= 0.5: {V(subs[1])}; r_other {f3(r['other_KV'])}"
    return all(subs), txt, dict(r=r, diff=diff, KminusV=kv)


def h6(M):
    if not (M.has("P1") and M.has("POST")):
        return None, "needs both formats", {}
    ov = len(set(M.rank("P1")[:20]) & set(M.rank("POST")[:20]))
    p = hypergeom.sf(ov - 1, M.n_heads, 20, 20)
    return ov >= 10, f"|top-20(a3, OPTIONS-AFTER) & top-20(a3, SENTENCE-AFTER)| = {ov} >= 10 (null P = {p:.1e})", dict(overlap=ov, P=p)


TITLE = {"H1": "sparsity: top-k* by a3 suffice", "H2": "necessity and specificity (knockout, random sets)",
         "H3": "ablation: ID_K drop, specificity, copy route", "H4": "canonical duplicate-token heads",
         "H5": "the second hop at the answer", "H5s": "the second hop, SENTENCE-AFTER (reported)",
         "H6": "the same readers for list and sentence"}


# ---------------------------------------------------------------- report
def provenance(M, out, test):
    P = M.P
    keys = ("git_commit", "model", "revision", "dtype", "attn_implementation", "device", "transformers", "torch", "test_mode")
    out(f"   {M.name}: " + ", ".join(f"{k}={P.get(k)}" for k in keys))
    out(f"      n_heads={M.n_heads} ({P['n_layers']}x{P['heads_per_layer']}, {P['n_kv_heads']} KV), k*={M.kstar}, KS_eff={M.KS}, "
        f"n_rand={M.n_rand}, loo={P['loo']}, grid layers {'all' if len(P['grid_layers']) == P['n_layers'] else P['grid_layers']}, "
        f"dup {P['dup']}")
    bad = []
    for arm in M.arms():
        nr, ne = len(M.J["arms"][arm]["rank"]), len(M.E(arm))
        flag = "" if (nr, ne) == (N_PREREG, N_PREREG) else (" (TEST)" if test else " MISMATCH")
        bad += [arm] if flag == " MISMATCH" else []
        out(f"      {arm}: n_rank={nr}, n_eval={ne}{flag}")
    if not test and (P.get("attn_implementation") != "eager" or not str(P.get("dtype", "")).endswith("bfloat16")):
        bad.append("attn/dtype")
        out("      MISMATCH: the entry fixes BF16 and eager attention")
    return bad


def exploratory(M, out):
    for arm in M.arms():
        if not M.has(arm):
            continue
        out(f"   -- {M.name} {arm}")
        for s in [x for x in M.E(arm)[0]["curves"]]:
            Rs = [M.R(arm, s, i)[0] for i in range(len(M.KS))]
            Ks = [M.KO(arm, s, i)[0] for i in range(len(M.KS))]
            out(f"      {s:7s} R  " + " ".join(f"{k}:{r:+.2f}" for k, r in zip(M.KS, Rs)))
            out(f"      {s:7s} KO " + " ".join(f"{k}:{r:+.2f}" for k, r in zip(M.KS, Ks)))
        L = np.array([e["layer"]["m"] for e in M.E(arm)]) - np.array([[e["layer"]["none"]] for e in M.E(arm)])
        dg = np.mean([e["layer"]["allG"] - e["layer"]["none"] for e in M.E(arm)])
        out("      layer profile (fraction of d_G): " + " ".join(f"{x / dg:+.2f}" for x in L.mean(0)))
        aR, aE = M.grid(arm, "a3"), M.grid(arm, "a3", "eval")
        out(f"      split-half a3 (R vs E) Spearman {spearmanr(aR.ravel(), aE.ravel())[0]:+.3f}; top-10 a3 (R): "
            + ", ".join(f"{c}:{aR[c]:.2f}" for c in M.rank(arm)[:10]))
        k0, v0 = M.ab(arm, "none", "idK"), M.ab(arm, "none", "idV")
        out(f"      ablation: ID_K(none) {f3(est(lambda x: x, k0))}, ID_V(none) {f3(est(lambda x: x, v0))}")
        for c in M.conds(arm):
            out(f"        {c:20s} rho_K {f3(est(ratio, M.ab(arm, c, 'idK'), k0))}  dV {f3(est(lambda x: x, M.ab(arm, c, 'idV') - v0))}"
                f"  mass>=0.9 {np.mean(M.ab(arm, c, 'mass') >= 0.9):.2f}  base-argmax {np.mean(M.ab(arm, c, 'base_ok')):.2f}")
        H, F, r = hop_r(M, arm)
        out("      hop: " + ", ".join(f"r_{n} {r[n][0]:+.3f}" for n in r) + f"; r_ans(KV) + r_other {r['ans_KV'][0] + r['other_KV'][0]:+.3f}"
            f" vs r_all {r['all_KV'][0]:+.3f}; |row0 - clean| {np.mean(np.abs(H[:, 0] - [e['mB'] for e in M.E(arm)])):.3f}")
        for g, x in M.E(arm)[0]["hop"]["explo"].items():
            m = np.array([e["hop"]["explo"][g]["m"] for e in M.E(arm)])
            Fx = m[:, 1] - m[:, 0]
            out(f"        rows {g:4s}: " + ", ".join(f"r_{n} {est(one_minus, m[:, i] - m[:, 0], Fx)[0]:+.3f}" for i, n in enumerate(x["names"]) if i >= 2))
        kc, C = causal_set(M)
        T = tdup_over(M, arm, C)
        Tc = tdup_over(M, arm, C, "tctrl", "n_ctrl")
        Pm = np.median([M.dup["P"][c] for c in C])
        out(f"      over C (top-{kc}, P1 ranking): median T_dup {T[0]:+.3f}, T_ctrl {Tc[0]:+.3f}, previous-token P {Pm:+.3f}")
        if arm != "P1":
            Ca = M.rank(arm)[:kc]
            out(f"      over the top-{kc} by this format's a3: median D {med_over(M, 'D', Ca)[0]:+.3f}, I {med_over(M, 'I', Ca)[0]:+.3f}, "
                f"T_dup {tdup_over(M, arm, Ca)[0]:+.3f}")


def score(root, models=None, test=None, out=print):
    root = Path(root)
    files = {f.stem: f for f in sorted(root.glob("*.json"))}
    Ms = {m: Model(m, json.load(open(f))) for m, f in files.items() if not f.name.endswith(".tmp")}
    test = (not any(m in Ms for m in MODELS)) if test is None else test
    req = models or (MODELS if any(m in Ms for m in MODELS) else list(Ms))
    out(f"== Part (a) reader heads and second hop (P-2026-10-05-H, Gates a2-a3, H1-H6): root {root}")
    out(f"   required models: {req}; found: {list(Ms)}" + ("  [TEST: the verdict lines are not results]" if test else ""))
    out("   Gate a1 (FP32 exactness, tests/test_head_splice.py) runs in scripts/gpu_stage6.sh before any model; not scored here")
    bad = []
    for M in Ms.values():
        bad += provenance(M, out, test)
    res = {"gates": {}, "verdicts": {}, "per_model": {}, "mismatch": bad}
    out("-- gates")
    ok1, ok2 = {}, {}
    for m in req:
        M = Ms.get(m)
        if M is None:
            out(f"  Gate a2  {m:28s} missing -> NOT EVALUABLE")
            out(f"  Gate a3  {m:28s} missing -> NOT EVALUABLE")
            ok1[m] = ok2[m] = False
            continue
        g2 = {arm: gate_a2(M, arm) for arm in ("P1", "POST")}
        for arm, (ok, txt) in g2.items():
            out(f"  Gate a2  {m:28s} {arm:4s} {txt} -> {V(ok)}")
        g3, txt, _ = gate_a3(M)
        out(f"  Gate a3  {m:28s} {txt} -> {V(g3)}")
        res["gates"][m] = dict(a2_P1=g2["P1"][0], a2_POST=g2["POST"][0], a3=g3)
        ok1[m], ok2[m] = bool(g3 and g2["P1"][0]), bool(g3 and g2["P1"][0] and g2["POST"][0])
    out("-- predictions (OPTIONS-AFTER unless stated; per model, then over the required models)")
    fns = {"H1": (h1, ok1), "H2": (h2, ok1), "H3": (h3, ok1), "H4": (lambda M: h4(M), ok1), "H5": (h5, ok1),
           "H5s": (lambda M: h5(M, "POST"), ok2), "H6": (h6, ok2)}
    for H, (fn, gate) in fns.items():
        per = {}
        for m in req:
            M = Ms.get(m)
            if M is None:
                per[m] = None
                out(f"  {H:4s} {m:28s} missing -> NOT EVALUABLE")
                continue
            try:
                ok, txt, *_ = fn(M)
            except Exception as e:  # noqa: BLE001  (a missing record makes this model's line not evaluable)
                ok, txt = None, f"not computable: {type(e).__name__}: {e}"
            per[m] = ok if gate[m] else None
            note = "" if gate[m] else f"; computed {V(ok)}, gate failed"
            out(f"  {H:4s} {m:28s} {txt} -> {V(per[m])}{note}")
        res["per_model"][H] = per
        res["verdicts"][H] = comb(per.values())
        out(f"  {H:4s} {TITLE[H]:50s} over {len(req)} model(s) -> {V(res['verdicts'][H])}" + ("  [reported, not confirmatory]" if H == "H5s" else ""))
    out("-- exploratory")
    for M in Ms.values():
        try:
            exploratory(M, out)
        except Exception as e:  # noqa: BLE001
            out(f"   exploratory block of {M.name} failed: {type(e).__name__}: {e}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/gpu_stage6/heads")
    ap.add_argument("--models", default=None)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    r = score(a.root, a.models.split(",") if a.models else None, True if a.test else None)
    sys.exit(0)
