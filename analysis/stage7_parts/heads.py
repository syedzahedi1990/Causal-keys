"""Step 1 of P-2026-10-08-I (docs/PREREGISTRATION.md): the reader heads at Mistral-Small-24B, Gate I-G2 on
heads/gate.json and heads/rank.json of experiments/stage7_link.py (stages rank and gate), as worded in the entry.

Per format f (on E, point estimates; intervals from the fixed core bootstrap of common.W):
  d_full = m(full K_S clamp from layer 5) - m(clean); d_G = m(all_G) - m(none) of H*'s sufficiency batch;
  R(k) = mean[m(top-k) - m(none)] / mean d_G; KO(k) = 1 - mean[m(all but top-k) - m(ko_none)] / mean[m(ko_all_G) - m(ko_none)];
  random-set means of R(k*) and KO(k*) over the three draws, each ratio recomputed within each resample;
  Gate a2 (I-G2 (d)): mean |m(none) - m(clean)| and mean |m(all_T) - m(full)| <= max(0.5, 0.02 x mean d_full).
I-G2: (a) d_full >= 3 with CI excluding 0 (SENTENCE-AFTER: > 0 with CI excluding 0); (b) d_G / d_full >= 0.7 (SENTENCE-AFTER
0.5), d_G / [d_full - d_full(NONE)] printed; (c) R(k*) >= 0.7, KO(k*) >= 0.8, R_rand(k*) <= 0.25, KO_rand(k*) <= 0.25; (d) Gate a2.
"""
import numpy as np
from scipy.stats import hypergeom

from .common import NAMES, V, est, f3

ARMS = ("P1", "LETTER", "POST")
B_THR = {"P1": 0.7, "LETTER": 0.7, "POST": 0.5}


def ratio(a, b):
    return a / b


def one_minus(a, b):
    return 1 - a / b


class Heads:
    def __init__(self, rank, gate):
        self.rank, self.gate = rank, gate
        S = rank["sets"]
        self.kstar = S["kstar"]
        self.KS = rank["provenance"]["KS"]
        self.ik = self.KS.index(self.kstar)
        self.sets = {arm: {k: [tuple(c) for c in v] for k, v in S["arms"][arm].items()} for arm in S["arms"]}
        self.rand = [[tuple(c) for c in r] for r in S["rand"]]
        self.nL, self.H = rank["provenance"]["n_layers"], rank["provenance"]["heads_per_layer"]
        self.n_elig = rank["provenance"]["n_eligible"]

    def E(self, arm):
        return self.gate["arms"].get(arm, {}).get("eval", [])

    def v(self, arm, f):
        return np.array([f(e) for e in self.E(arm)], float)

    def dfull(self, arm):
        return self.v(arm, lambda e: e["mF"] - e["mB"])

    def dfull_none(self):
        return np.array([e["mF"] - e["mB"] for e in self.gate.get("none_clamp", {}).get("eval", [])], float)

    def c(self, arm, s, key, i=None):
        return self.v(arm, (lambda e: e["curves"][s][key]) if i is None else (lambda e: e["curves"][s][key][i]))

    def dG(self, arm, s="H"):
        return self.c(arm, s, "allG") - self.c(arm, s, "none")

    def R(self, arm, s, i):
        return est(ratio, self.c(arm, s, "suff", i) - self.c(arm, s, "none"), self.dG(arm, s))

    def KO(self, arm, s, i):
        return est(one_minus, self.c(arm, s, "ko", i) - self.c(arm, s, "ko_none"), self.c(arm, s, "ko_allG") - self.c(arm, s, "ko_none"))

    def rand_mean(self, arm, ko=False):
        vs = []
        for r in range(len(self.rand)):
            s = f"rand{r}"
            if ko:
                vs += [self.c(arm, s, "ko", 0) - self.c(arm, s, "ko_none"), self.c(arm, s, "ko_allG") - self.c(arm, s, "ko_none")]
            else:
                vs += [self.c(arm, s, "suff", 0) - self.c(arm, s, "none"), self.dG(arm, s)]
        f = (lambda *x: np.mean([1 - x[2 * j] / x[2 * j + 1] for j in range(len(x) // 2)], 0)) if ko else \
            (lambda *x: np.mean([x[2 * j] / x[2 * j + 1] for j in range(len(x) // 2)], 0))
        return est(f, *vs)

    def k80(self, arm):
        for i, k in enumerate(self.KS):
            if self.R(arm, "H", i)[0] >= 0.8:
                return k
        return None

    def gate_i2(self, arm):
        """(verdict, text, numbers) of I-G2 (a-d) in one format."""
        if not self.E(arm):
            return None, "no gate results", {}
        df = self.dfull(arm)
        a_ = est(lambda x: x, df)
        a_ok = (a_[0] > 0 if arm == "POST" else a_[0] >= 3) and a_[1] > 0
        b_ = est(ratio, self.dG(arm), df)
        dn = self.dfull_none()
        ms = est(lambda g, f, n: g / (f - n), self.dG(arm), df, dn) if len(dn) == len(df) else (float("nan"),) * 4
        b_ok = b_[0] >= B_THR[arm]
        R, KO, Rr, KOr = self.R(arm, "H", self.ik), self.KO(arm, "H", self.ik), self.rand_mean(arm), self.rand_mean(arm, True)
        c_ok = R[0] >= 0.7 and KO[0] >= 0.8 and Rr[0] <= 0.25 and KOr[0] <= 0.25
        E = self.E(arm)
        bound = max(0.5, 0.02 * df.mean())
        nf = np.mean([abs(e["curves"]["H"]["none"] - e["mB"]) for e in E])
        af = np.mean([abs(e["curves"]["H"]["allT"] - e["mF"]) for e in E])
        d_ok = nf <= bound and af <= bound
        ok = a_ok and b_ok and c_ok and d_ok
        txt = (f"(a) d_full {f3(a_)} {'> 0' if arm == 'POST' else '>= 3'}, CI excl. 0: {V(a_ok)}; "
               f"(b) d_G/d_full {f3(b_)} >= {B_THR[arm]}: {V(b_ok)} [mention-specific d_G/(d_full - d_full(NONE)) {f3(ms)}]; "
               f"(c) R({self.kstar}) {f3(R)} >= 0.7, KO({self.kstar}) {f3(KO)} >= 0.8, R_rand {f3(Rr)} <= 0.25, KO_rand {f3(KOr)} <= 0.25: "
               f"{V(c_ok)} [k80 {self.k80(arm)}]; (d) Gate a2 |none - clean| {nf:.3f}, |all_T - full| {af:.3f} <= {bound:.3f}: {V(d_ok)}")
        return ok, txt, dict(dfull=a_, ratio=b_, R=R, KO=KO, R_rand=Rr, KO_rand=KOr, a2=(nf, af, bound))

    # ---- report
    def report(self, out, dup=True):
        A = self.rank["arms"]
        for arm in ARMS:
            if arm not in self.sets:
                continue
            Hs = self.sets[arm]["H"]
            hist = np.bincount([l for l, _ in Hs], minlength=self.nL)
            out(f"   {NAMES[arm]}: H* = top-{self.kstar} by a3 over R; layer histogram " + " ".join(f"{l}:{n}" for l, n in enumerate(hist) if n))
            l0 = [tuple(c) for c in A[arm]["ranking_l0"]][:self.kstar]
            ov0 = len(set(l0) & set(Hs))
            out(f"      overlap with the top-{self.kstar} of the layer-0 ranking (H's): {ov0}; of those layer-0 heads {sum(l < 5 for l, _ in l0)} lie in layers 0-4")
            if self.E(arm):
                Rs = [self.R(arm, "H", i)[0] for i in range(len(self.KS))]
                Ks = [self.KO(arm, "H", i)[0] for i in range(len(self.KS))]
                out("      R(k)  " + " ".join(f"{k}:{r:+.2f}" for k, r in zip(self.KS, Rs)))
                out("      KO(k) " + " ".join(f"{k}:{r:+.2f}" for k, r in zip(self.KS, Ks)))
                for s in [f"rand{i}" for i in range(len(self.rand))] + ["active"]:
                    if s in self.E(arm)[0]["curves"]:
                        out(f"      {s:7s} R({self.kstar}) {self.R(arm, s, 0)[0]:+.3f}  KO({self.kstar}) {self.KO(arm, s, 0)[0]:+.3f}")
            if dup and "dup" in self.rank:
                D, I = np.array(self.rank["dup"]["D"]), np.array(self.rank["dup"]["I"])
                out(f"      duplicate / induction scores over H*: median D {np.median([D[c] for c in Hs]):+.3f}, median I {np.median([I[c] for c in Hs]):+.3f}")
        arms = [a for a in ARMS if a in self.sets]
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                ov = len(set(self.sets[a]["H"]) & set(self.sets[b]["H"]))
                p = hypergeom.sf(ov - 1, self.n_elig, self.kstar, self.kstar)
                out(f"   overlap of H* across formats: {NAMES[a]} & {NAMES[b]} = {ov} of {self.kstar} (hypergeometric P {p:.1e})")
