"""The confirmatory lines of preregistration J part B, each line's class, kind and recorded prior, the model sets, the
per-cell gates (coverage, floor, competence, anchor) and one function per line and model.

Model sets: P4 = qwen7, qwen14, mistral7, olmo7 (the four models of stages 1 and 3b); N4 = llama8, gemma9, phi4,
falcon7 (new families; yi9 takes the slot of a model whose files failed verification or whose tokenizer check failed,
before any output of it existed); SMALL = gemma2b. P4f lines use 95 % intervals and need every evaluable P4 model (at
least 3 evaluable); N4 lines use 98.75 % intervals and need 3 of the 4 N4 models (a model not evaluable counts as not
meeting; fewer than 3 evaluable: NOT EVALUABLE).
Scoring of a statistic: E when every cell (model x arm) it uses has coverage >= 0.8 (minimum over the cell's 16 rows of
the mean E mass); otherwise its behavioural counterpart (beta_K, beta_V in place of ID_K, ID_V in the same formula; the
same thresholds), except for lines without one (J-B-LB), which are then NOT EVALUABLE. Under E, a cell whose BF16 floor
exceeds 0.05 x ID_K^E(AFTER) makes the statistic NOT EVALUABLE.
Definedness (relative, B-11): with D(f) = ID_K(f) + ID_V(f) and D_A = D(AFTER) under the same scoring,
s_ID(f) = ID_K(f) / D(f) is defined when D(f) > 0, D(f) >= 0.2 max(D_A, 0) and ID_V(f) >= -0.05 max(D_A, 0) (own arm);
r(f) = ID_K(f) / ID_K(AFTER) and delta(a, b) = [ID_K(a) - ID_K(b)] / ID_K(AFTER) are defined when ID_K(AFTER) > 0 and
ID_K(AFTER) >= 0.1 D_A (the anchor). The JB-G5 anchor gate: at the point ID_K(AFTER) >= 0.1 D_A, and its lower bound at the
line's level > 0; otherwise the r- and delta-based components are NOT EVALUABLE.
Competence (JB-G5): generated acc_B >= 0.8 under NONE and AFTER for J-B1 to J-B5 and J-B-LB; also under POST for
J-B3 and J-B-SMALL, POST and POST-NULL for J-B-NULL, POST and PRE for J-B5b, BEFORE for J-B-LB; otherwise the line is
NOT EVALUABLE in that model (waived in TEST_MODE, where the verdicts are plumbing checks).
"""
from __future__ import annotations

import math

import numpy as np

from .stats import (Comp, Res, V, comb_every, comb_k_of, decide, est, inside, lower, mean, point, upper)

P4 = ("qwen7", "qwen14", "mistral7", "olmo7")
N4 = ("llama8", "gemma9", "phi4", "falcon7")
FALLBACK, SMALL, JB8_KEY, X2 = "yi9", "gemma2b", "mistral24", ("qwen1.5", "qwen3b")
FAMILY = {"qwen7": "Qwen2.5-7B", "qwen14": "Qwen2.5-14B", "mistral7": "Mistral-7B", "olmo7": "OLMo-2-7B",
          "llama8": "Llama-3.1-8B", "gemma9": "Gemma-2-9B", "phi4": "Phi-4", "falcon7": "Falcon3-7B", "yi9": "Yi-1.5-9B",
          "gemma2b": "Gemma-2-2B", "mistral24": "Mistral-Small-24B", "qwen1.5": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B"}
REF_NAME = {"qwen7": "Qwen2.5-7B-Instruct", "qwen14": "Qwen2.5-14B-Instruct", "mistral7": "Mistral-7B-Instruct-v0.3",
            "olmo7": "OLMo-2-1124-7B-Instruct"}
ARMS_F = ("AFTER", "BEFORE", "NONE", "POST", "PRE", "POST-NULL")
ARMS_S0 = ("P1", "AFTER", "BEFORE", "POST", "PRE", "NONE")
LV_EVERY, LV_K4 = 0.95, 0.9875
COV_MIN, FLOOR_FRAC, ANCHOR_FRAC, COMP_MIN, COMP_N_MIN = 0.8, 0.05, 0.1, 0.8, 30
S_DEN, S_VNEG = 0.2, 0.05

# code -> (class, kind, recorded prior P(met), set, title). kind: "A" account line, "V" measurement-validity line.
LINES = {
    "J-B1-P4f": ("L", "A", 0.90, "P4f", "the list after the story opens a key read that carries most of the identity"),
    "J-B1-N4": ("M", "A", 0.85, "N4", "the list after the story opens a key read that carries most of the identity"),
    "J-B2-P4f": ("L", "A", 0.90, "P4f", "the same list or sentence before the writing token adds no key read over no mention"),
    "J-B2-N4": ("M", "A", 0.85, "N4", "the same list or sentence before the writing token adds no key read over no mention"),
    "J-B3-P4f": ("L", "A", 0.80, "P4f", "a neutral sentence after the story opens an intermediate key read"),
    "J-B3-N4": ("R", "A", 0.50, "N4", "a neutral sentence after the story opens an intermediate key read"),
    "J-B4-P4f": ("L", "A", 0.90, "P4f", "without a later mention the identity is copied through the value"),
    "J-B4-N4": ("M", "A", 0.85, "N4", "without a later mention the identity is copied through the value"),
    "J-B5-P4f": ("L", "A", 0.85, "P4f", "the channel that decides the generated answer flips with a later list"),
    "J-B5-N4": ("M", "A", 0.80, "N4", "the channel that decides the generated answer flips with a later list"),
    "J-B5b-P4f": ("R", "A", 0.45, "P4f", "the sentence read changes generated answers: beta_K(POST) > beta_K(PRE)"),
    "J-B5b-N4": ("R", "A", 0.40, "N4", "the sentence read changes generated answers: beta_K(POST) > beta_K(PRE)"),
    "J-B-NULL-P4f": ("M", "A", 0.80, "P4f", "a sentence naming no candidate opens no key read; the re-mention does"),
    "J-B-NULL-N4": ("R", "A", 0.55, "N4", "a sentence naming no candidate opens no key read; the re-mention does"),
    "J-B-LB-N4": ("R", "A", 0.65, "N4", "the list before the story gives a key read below no mention"),
    "J-B-SMALL": ("R", "A", 0.40, "SMALL", "no sentence read at 2B in a new family (Gemma-2-2B)"),
    "J-B6a": ("R", "V", 0.50, "S0", "S0: the emitted forms cover >= 0.8 of the mass in every row of every cell"),
    "J-B6b": ("R", "V", 0.45, "S0", "S0: s_ID and r unchanged by the scoring (E vs L within +-0.05) in the low-L-mass cells"),
    "J-B6c": ("R", "V", 0.75, "S0", "S0: the published verdicts (E1a-c, B2, C4's BEFORE bound) unchanged under E (primary JB6)"),
    "J-B7": ("R", "V", 0.35, "P4+N4", "the emitted-form score agrees with generation and covers the mass on F"),
    "J-B8": ("R", "V", 0.55, "JB8", "the 24B intervention frames are invariant to the scoring"),
}
ORDER = list(LINES)


# --------------------------------------------------------------------------- statistics on a view
def _s(k, v, ka, va):
    D, DA = k + v, np.maximum(ka + va, 0)
    return k / D, (D > 0) & (D >= S_DEN * DA) & (v >= -S_VNEG * DA), True


def _anchor(ka, va):
    return (ka > 0) & (ka >= ANCHOR_FRAC * (ka + va))


def s_est(v, f, sg):
    return est(v.boot, _s, v.k(f, sg), v.v(f, sg), v.k("AFTER", sg), v.v("AFTER", sg))


def r_est(v, f, sg):
    return est(v.boot, lambda k, ka, va: (k / ka, True, _anchor(ka, va)), v.k(f, sg), v.k("AFTER", sg), v.v("AFTER", sg))


def delta_est(v, a, b, sg):
    return est(v.boot, lambda x, y, ka, va: ((x - y) / ka, True, _anchor(ka, va)),
               v.k(a, sg), v.k(b, sg), v.k("AFTER", sg), v.v("AFTER", sg))


def ds_est(v, a, b, sg):
    def fn(ka_, va_, kb, vb, ka, va):
        x, o1, _ = _s(ka_, va_, ka, va)
        y, o2, _ = _s(kb, vb, ka, va)
        return x - y, o1 & o2, True
    return est(v.boot, fn, v.k(a, sg), v.v(a, sg), v.k(b, sg), v.v(b, sg), v.k("AFTER", sg), v.v("AFTER", sg))


def nm(sg, what):
    """The statistic's name under a scoring (beta: the behavioural counterpart)."""
    if sg != "beta":
        return what.replace("^", f"^{sg}") if "^" in what else f"{what}^{sg}"
    return {"s_ID": "b_ID", "ID_K": "beta_K", "ID_V": "beta_V", "r": "r^beta", "delta": "delta^beta"}.get(
        what.split("(")[0], what) + ("(" + what.split("(", 1)[1] if "(" in what else "")


# --------------------------------------------------------------------------- one model
class Model:
    """A model's part-B results and its gates: F and S0 populations, tokcheck, G3, frames; ``ok`` False (with ``why``)
    if a model-level gate fails, which makes every line NOT EVALUABLE in this model."""

    def __init__(self, key, F=None, S0=None, ok=True, why="", test=False):
        self.key, self.F, self.S0, self.ok, self.why, self.test = key, F, S0, ok, why, test
        self._cov, self._floor = {}, {}

    # --- per-cell gates (on F unless pop given)
    def cov(self, arm, pop=None):
        p = pop or self.F
        k = (id(p), arm)
        if k not in self._cov:
            self._cov[k] = p.view().coverage(arm, "E")
        return self._cov[k]

    def floor(self, arm):
        if arm not in self._floor:
            v = self.F.view()
            f = float(v.floor(arm).mean())
            thr = FLOOR_FRAC * float(v.k("AFTER", "E").mean())
            self._floor[arm] = (f, thr, f <= thr)
        return self._floor[arm]

    def acc(self, arm):
        return float(self.F.view().acc(arm, "B").mean())

    def gate(self, arms, uses=()):
        """Res(None, ...) if the model fails, if an arm the line uses was not run, or if its competence in ``arms``
        (plus NONE and AFTER) fails; else None."""
        if not self.ok:
            return Res(None, "", why=self.why)
        if self.F is None:
            return Res(None, "", why="no F results")
        missing = [a for a in set(arms) | set(uses) | {"NONE", "AFTER"} if not self.F.has(a)]
        if missing:
            return Res(None, "", why=f"arms not run (deadline): {sorted(missing)}")
        bad = [f"{a} {self.acc(a):.2f}" for a in sorted(set(arms) | {"NONE", "AFTER"}) if not self.acc(a) >= COMP_MIN]
        if bad and not self.test:   # TEST_MODE (0.5B, n = 2): the plumbing check runs the lines without the gate
            return Res(None, "", why=f"competence: generated acc_B < {COMP_MIN} under " + ", ".join(bad))
        return None

    def sigma_for(self, cells, allow_beta=True):
        """('E' | 'beta' | None, note) for a statistic over these F cells (B-8 coverage, JB-G4 floor)."""
        low = [f"{a} {self.cov(a)[0]:.2f}" for a in cells if not self.cov(a)[0] >= COV_MIN]
        if low:
            if not allow_beta:
                return None, "coverage below 0.8 (" + ", ".join(low) + "); no behavioural counterpart"
            return "beta", "behavioural counterpart: coverage below 0.8 (" + ", ".join(low) + ")"
        hi = [f"{a} {self.floor(a)[0]:.3f} > {self.floor(a)[1]:.3f}" for a in cells if not self.floor(a)[2]]
        if hi:
            return None, "BF16 batch floor above 0.05 x ID_K^E(AFTER) (" + ", ".join(hi) + ")"
        return "E", ""

    def anchor(self, sg, level):
        """JB-G5 anchor under scoring sg: [] if it holds, else a NOT EVALUABLE component."""
        v = self.F.view()
        ka, va = v.k("AFTER", sg), v.v("AFTER", sg)
        K = est(v.boot, mean, ka)
        ok = K.pt >= ANCHOR_FRAC * (ka.mean() + va.mean()) and K.bound(level, "lo") > 0
        if ok:
            return []
        return [Comp("anchor", False, ne=f"JB-G5 anchor: {nm(sg, 'ID_K')}(AFTER) {K.txt(level)} against "
                                         f"{ANCHOR_FRAC} x D_A = {ANCHOR_FRAC * (ka.mean() + va.mean()):+.3f}")]


def stat_comps(m, level, cells, build, allow_beta=True, needs_anchor=False):
    """Components of one statistic: choose the scoring for ``cells``, then build(sg) -> list of Comp."""
    sg, note = m.sigma_for(cells, allow_beta)
    if sg is None:
        return [Comp(f"cells {cells}", False, ne=note)], None
    pre = m.anchor(sg, level) if needs_anchor else []
    if pre:
        return pre, sg
    cs = build(sg)
    if note:
        for c in cs:
            c.label += f" [{note}]"
    return cs, sg


# --------------------------------------------------------------------------- F lines
def line_b1(m, lv):
    g = m.gate([])
    if g:
        return g
    v = m.F.view()

    def b(sg):
        s, K = s_est(v, "AFTER", sg), est(v.boot, mean, v.k("AFTER", sg))
        return [point(f"{nm(sg, 's_ID')}(AFTER) {s.pt:+.3f} >= 0.5", s.pt >= 0.5, s),
                lower(s, 0.40, lv, f"{nm(sg, 's_ID')}(AFTER)"), lower(K, 0.0, lv, f"{nm(sg, 'ID_K')}(AFTER)")]
    cs, sg = stat_comps(m, lv, ["AFTER"], b)
    return decide(cs, f"n={v.n} scoring {sg}")


def line_b2(m, lv):
    g = m.gate([], uses=("BEFORE", "PRE"))
    if g:
        return g
    v, cs = m.F.view(), []
    for a in ("BEFORE", "PRE"):
        cs += stat_comps(m, lv, [a, "NONE", "AFTER"], lambda sg, a=a: [upper(delta_est(v, a, "NONE", sg), 0.05, lv,
                                                                       f"{nm(sg, 'r')}({a}) - {nm(sg, 'r')}(NONE)")],
                         needs_anchor=True)[0]
    for a in ("BEFORE", "PRE"):
        cs += stat_comps(m, lv, [a, "AFTER"], lambda sg, a=a: [upper(s_est(v, a, sg), 0.10, lv, f"{nm(sg, 's_ID')}({a})")])[0]
    return decide(cs, f"n={v.n}")


def competent_direction(m, sg):
    """J-B3: on the items competent in POST, NONE, PRE and AFTER, the point estimates of delta(POST - PRE),
    delta(POST - NONE) and s_ID(POST) - s_ID(NONE) are > 0 (needs >= 30 such items outside TEST)."""
    pop = m.F
    keep = np.ones(len(pop.keys), bool)
    for a in ("POST", "NONE", "PRE", "AFTER"):
        keep &= pop.view().comp(a) > 0
    v = pop.view(keep)
    need = 2 if m.test else COMP_N_MIN
    if v.n < need:
        return [Comp("competent-only direction", False, ne=f"{v.n} competent items (< {need})")]
    d1, d2 = delta_est(v, "POST", "PRE", sg), delta_est(v, "POST", "NONE", sg)
    d3 = ds_est(v, "POST", "NONE", sg)
    ok = d1.pt > 0 and d2.pt > 0 and d3.pt > 0
    return [Comp(f"competent-only direction (n={v.n}): {nm(sg, 'delta')}(POST-PRE) {d1.pt:+.3f}, {nm(sg, 'delta')}(POST-NONE) "
                 f"{d2.pt:+.3f}, {nm(sg, 's_ID')}(POST)-(NONE) {d3.pt:+.3f} all > 0 (point)", ok)]


def line_b3(m, lv):
    g = m.gate(["POST"], uses=("PRE",))
    if g:
        return g
    v, cs = m.F.view(), []

    def rb(sg):
        r = r_est(v, "POST", sg)
        return [point(f"{nm(sg, 'r')}(POST) {r.pt:+.3f} >= 0.15", r.pt >= 0.15, r), lower(r, 0.10, lv, f"{nm(sg, 'r')}(POST)"),
                point(f"{nm(sg, 'r')}(POST) <= 0.75", r.pt <= 0.75, r), upper(r, 1.0, lv, f"{nm(sg, 'r')}(POST)")]
    cs += stat_comps(m, lv, ["POST", "AFTER"], rb, needs_anchor=True)[0]
    for b in ("PRE", "NONE"):
        cs += stat_comps(m, lv, ["POST", b, "AFTER"], lambda sg, b=b: [lower(delta_est(v, "POST", b, sg), 0.0, lv,
                                                                       f"{nm(sg, 'delta')}(POST - {b})")], needs_anchor=True)[0]
    cs += stat_comps(m, lv, ["POST", "AFTER"], lambda sg: [lower(s_est(v, "POST", sg), 0.10, lv, f"{nm(sg, 's_ID')}(POST)")])[0]
    c, sg = stat_comps(m, lv, ["POST", "NONE", "AFTER"], lambda sg: [lower(ds_est(v, "POST", "NONE", sg), 0.0, lv,
                                                                     f"{nm(sg, 's_ID')}(POST) - {nm(sg, 's_ID')}(NONE)")])
    cs += c
    if sg is not None:
        cs += competent_direction(m, sg)
    return decide(cs, f"n={v.n}")


def line_b4(m, lv):
    g = m.gate([])
    if g:
        return g
    v, cs = m.F.view(), []

    def sb(sg):
        s = s_est(v, "NONE", sg)
        return [point(f"{nm(sg, 's_ID')}(NONE) {s.pt:+.3f} <= 0.10", s.pt <= 0.10, s), upper(s, 0.15, lv, f"{nm(sg, 's_ID')}(NONE)")]
    cs += stat_comps(m, lv, ["NONE", "AFTER"], sb)[0]
    cs += stat_comps(m, lv, ["NONE"], lambda sg: [lower(est(v.boot, mean, v.v("NONE", sg)), 0.0, lv, f"{nm(sg, 'ID_V')}(NONE)")])[0]
    return decide(cs, f"n={v.n}")


def line_b5(m, lv):
    g = m.gate([], uses=("BEFORE", "PRE"))
    if g:
        return g
    v = m.F.view()
    E = lambda x: est(v.boot, mean, x)  # noqa: E731
    kA, vN = E(v.k("AFTER", "beta")), E(v.v("NONE", "beta"))
    cs = [point(f"beta_K(AFTER) {kA.pt:+.3f} >= 0.5", kA.pt >= 0.5), lower(kA, 0.40, lv, "beta_K(AFTER)"),
          point(f"beta_V(NONE) {vN.pt:+.3f} >= 0.5", vN.pt >= 0.5), lower(vN, 0.40, lv, "beta_V(NONE)")]
    for a in ("NONE", "BEFORE", "PRE"):
        cs.append(upper(E(v.k(a, "beta")), 0.10, lv, f"beta_K({a})"))
    cs.append(lower(E(v.k("AFTER", "beta") - v.v("AFTER", "beta")), 0.0, lv, "beta_K(AFTER) - beta_V(AFTER)"))
    return decide(cs, f"n={v.n}")


def flip_rate(v, arm):
    """The share of items whose generated answer moves to the clamped key's location: 1/2 [P(a=S|K_S) + P(a=X|K_X)]."""
    R = v.pop.recs(arm)
    vals = [0.5 * ((R[i]["ans"]["K_S@0"] == R[i]["track"]["S"]) + (R[i]["ans"]["K_X@0"] == R[i]["track"]["X"])) for i in v.idx]
    return float(np.mean(vals)) if vals else float("nan")


def line_b5b(m, lv):
    g = m.gate(["POST", "PRE"])
    if g:
        return g
    v = m.F.view()
    d = est(v.boot, mean, v.k("POST", "beta") - v.k("PRE", "beta"))
    return decide([lower(d, 0.0, lv, "beta_K(POST) - beta_K(PRE)")],
                  f"n={v.n}; flip rate POST {flip_rate(v, 'POST'):.3f}, PRE {flip_rate(v, 'PRE'):.3f}")


def line_null(m, lv):
    g = m.gate(["POST", "POST-NULL"])
    if g:
        return g
    v, cs = m.F.view(), []
    cs += stat_comps(m, lv, ["POST-NULL", "AFTER"], lambda sg: [upper(r_est(v, "POST-NULL", sg), 0.10, lv, f"{nm(sg, 'r')}(POST-NULL)")],
                     needs_anchor=True)[0]
    cs += stat_comps(m, lv, ["POST-NULL", "AFTER"], lambda sg: [upper(s_est(v, "POST-NULL", sg), 0.10, lv, f"{nm(sg, 's_ID')}(POST-NULL)")])[0]
    cs += stat_comps(m, lv, ["POST", "POST-NULL", "AFTER"], lambda sg: [lower(delta_est(v, "POST", "POST-NULL", sg), 0.0, lv,
                                                                        f"{nm(sg, 'delta')}(POST - POST-NULL)")], needs_anchor=True)[0]
    return decide(cs, f"n={v.n}")


def line_lb(m, lv):
    g = m.gate(["BEFORE"])
    if g:
        return g
    v = m.F.view()
    cs = stat_comps(m, lv, ["BEFORE", "NONE"], lambda sg: [upper(est(v.boot, mean, v.k("BEFORE", sg) - v.k("NONE", sg)), 0.0, lv,
                                                                 f"{nm(sg, 'ID_K')}(BEFORE) - {nm(sg, 'ID_K')}(NONE)")], allow_beta=False)[0]
    return decide(cs, f"n={v.n}")


def line_small(m, lv):
    g = m.gate(["POST"])
    if g:
        return g
    v = m.F.view()
    cs = stat_comps(m, lv, ["POST", "AFTER"], lambda sg: [upper(r_est(v, "POST", sg), 0.10, lv, f"{nm(sg, 'r')}(POST)")],
                    needs_anchor=True)[0]
    return decide(cs, f"n={v.n}")


def line_b7(m, lv):
    """Per model: in every F arm, A >= 0.95, clean-B other rate <= 0.10, coverage >= 0.80 (points)."""
    if not m.ok:
        return Res(None, "", why=m.why)
    if m.F is None:
        return Res(None, "", why="no F results")
    cs, v = [], m.F.view()
    for a in ARMS_F:
        if not m.F.has(a):
            cs.append(Comp(a, False, ne="arm not run"))
            continue
        num, den = m.F.agreement(a)
        A = num / den if den else float("nan")
        oth = float(v.other(a, "B").mean())
        cov, row = m.cov(a)
        cs.append(Comp(f"{a}: A {A:.3f} >= 0.95 (n={den}), other(B) {oth:.3f} <= 0.10, coverage {cov:.3f} >= 0.80 "
                       f"(min at {row}) (points)", den > 0 and A >= 0.95 and oth <= 0.10 and cov >= COV_MIN))
    return decide(cs, f"n={v.n}")


# --------------------------------------------------------------------------- S0 lines (P4)
def s0_gate(m):
    if not m.ok:
        return Res(None, "", why=m.why)
    if m.S0 is None:
        return Res(None, "", why="no S0 results")
    missing = [a for a in ARMS_S0 if not m.S0.has(a)]
    if missing:
        return Res(None, "", why=f"S0 arms not run: {missing}")
    return None


def line_b6a(m, lv):
    g = s0_gate(m)
    if g:
        return g
    cs = []
    for a in ARMS_S0:
        cov, row = m.cov(a, m.S0)
        cs.append(Comp(f"{a}: coverage {cov:.3f} >= 0.80 (min at {row}) (point)", cov >= COV_MIN))
    return decide(cs, f"n={m.S0.view().n}")


def ml_mass(v, a):
    return float(v.mass(a, "B", "L").mean())


def eq_est(v, a, kind):
    """s^E - s^L (kind s) or r^E - r^L (kind r) of S0 cell a, paired in every resample."""
    xs = [v.k(a, "E"), v.v(a, "E"), v.k("AFTER", "E"), v.v("AFTER", "E"),
          v.k(a, "L"), v.v(a, "L"), v.k("AFTER", "L"), v.v("AFTER", "L")]
    if kind == "s":
        def fn(k1, v1, ka1, va1, k2, v2, ka2, va2):
            x, o1, _ = _s(k1, v1, ka1, va1)
            y, o2, _ = _s(k2, v2, ka2, va2)
            return x - y, o1 & o2, True
    else:
        def fn(k1, v1, ka1, va1, k2, v2, ka2, va2):
            return k1 / ka1 - k2 / ka2, True, _anchor(ka1, va1) & _anchor(ka2, va2)
    return est(v.boot, fn, *xs)


def line_b6b(m, lv):
    g = s0_gate(m)
    if g:
        return g
    v = m.S0.view()
    cs, low = [], [a for a in ARMS_S0 if ml_mass(v, a) < 0.5]
    if not low:
        return Res(None, "", why="no S0 cell with M^L(clean B) < 0.5")
    for a in low:
        cs.append(inside(eq_est(v, a, "s"), -0.05, 0.05, lv, f"s_ID^E({a}) - s_ID^L({a}) [M^L {ml_mass(v, a):.3f}]"))
        if a != "AFTER":
            cs.append(inside(eq_est(v, a, "r"), -0.05, 0.05, lv, f"r^E({a}) - r^L({a})"))
    return decide(cs, f"n={v.n}; low-L-mass cells {low}")


def original_verdicts(models, sg):
    """The stage-1/3b/2 verdicts on S0 under scoring sg (95 % core bootstrap, the original rules), per model and line."""
    per = {}
    for k in P4:
        m = models[k]
        v = m.S0.view()
        cb = v.cboot
        E = lambda x: est(cb, mean, x)  # noqa: E731
        K = {a: v.k(a, sg) for a in ARMS_S0}
        lo = lambda e: e.bound(0.95, "lo")  # noqa: E731
        per[k] = {"E1a": lo(E(K["AFTER"])) > 0,
                  "E1b/BEFORE": float(K["BEFORE"].mean()) <= 0.5, "E1b/PRE": float(K["PRE"].mean()) <= 0.5,
                  "E1c/AFTER-BEFORE": lo(E(K["AFTER"] - K["BEFORE"])) > 0, "E1c/POST-PRE": lo(E(K["POST"] - K["PRE"])) > 0,
                  "B2": lo(E(K["P1"] - K["NONE"])) > 0, "C4/BEFORE": float(K["BEFORE"].mean()) <= 0.5}
    cnt = lambda c: sum(per[k][c] for k in P4)  # noqa: E731
    lines = {"E1a": cnt("E1a") == 4, "E1b": cnt("E1b/BEFORE") >= 3 and cnt("E1b/PRE") >= 3,
             "E1c": cnt("E1c/AFTER-BEFORE") == 4 and cnt("E1c/POST-PRE") >= 3, "B2": cnt("B2") >= 3,
             "C4-BEFORE": cnt("C4/BEFORE") == 4}
    return lines, per


def line_b6c(models):
    """MET iff each of E1a, E1b, E1c, B2 and C4's BEFORE bound has the same verdict under E as under L (same pass)."""
    bad = [k for k in P4 if k not in models or s0_gate(models[k]) is not None]
    if bad:
        return Res(None, "", why="needs S0 results of all four P4 models: " + ", ".join(
            f"{k} ({s0_gate(models[k]).why if k in models else 'absent'})" for k in bad))
    lE, pE = original_verdicts(models, "E")
    lL, pL = original_verdicts(models, "L")
    cs = [Comp(f"{ln}: under L {V(lL[ln])}, under E {V(lE[ln])}", lL[ln] == lE[ln]) for ln in lL]
    flips = [f"{k} {c}" for k in P4 for c in pE[k] if pE[k][c] != pL[k][c]]
    return decide(cs, "per-model components changed by E: " + (", ".join(flips) if flips else "none"))


LINE_FNS = {"J-B1": line_b1, "J-B2": line_b2, "J-B3": line_b3, "J-B4": line_b4, "J-B5": line_b5, "J-B5b": line_b5b,
            "J-B-NULL": line_null, "J-B-LB": line_lb, "J-B-SMALL": line_small}


def base_code(code):
    return code.rsplit("-", 1)[0] if code.endswith(("-P4f", "-N4")) else code


def nan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))
