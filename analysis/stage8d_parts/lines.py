"""Per-model statistics of the stage-8 part-D scorer (P-2026-10-10-J part D), read from the files of
experiments/stage8_flag.py. Every function returns plain values and Q statistics (analysis/stage8d_parts/stats.py);
gates and verdicts are applied in analysis/stage8d_score.py.

Scores. Six-way renormalised log-probabilities l_w = lp(w) - logsumexp over the six candidates (list formats, lowercase
candidates; the case-marginalised scores of the diss stage under POST and P1); four-way over the names IO_B, IO_S, IO_X,
S in IOI (the stored order). Delta l_w(row) = l_w(row) - l_w(none) within a batch. c = l_S - l_X (the normalisation
cancels).
"""
from __future__ import annotations

import numpy as np

from .stats import NAN, boot, mean_q, one_minus, ratio

IOI4 = {"B": 0, "S": 1, "X": 2, "Subj": 3}


def lse(a):
    a = np.asarray(a, float)
    m = a.max(-1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(-1, keepdims=True)))[..., 0]


def renorm(a):
    a = np.asarray(a, float)
    return a - lse(a)[..., None]


def rows_arr(stories, key, name, field="rows"):
    """[n, k] renormalised scores of row ``name`` of every story (``key`` None: the story itself holds ``field``)."""
    return renorm(np.array([(s if key is None else s[key])[field][name] for s in stories], float))


def pick(arr, idx):
    """arr [n, k], idx [n] -> [n]."""
    return arr[np.arange(len(idx)), np.asarray(idx)]


def argmax_rate(arr, idx):
    return float(np.mean(np.argmax(arr, 1) == np.asarray(idx)))


# --------------------------------------------------------------------------- inject (E8, P1)
class Inject:
    def __init__(self, J):
        self.S = J["stories"]
        self.n = len(self.S)
        self.b = boot(self.n)
        self.ix = {k: np.array([s["ix"][k] for s in self.S]) for k in ("B", "S", "X", "I", "D")}
        self.R = {nm: rows_arr(self.S, None, nm) for nm in self.S[0]["rows"]}
        self.RT = {nm: rows_arr(self.S, None, nm, "route") for nm in self.S[0]["route"]}
        self.raw = {nm: np.array([s["rows"][nm] for s in self.S], float) for nm in self.S[0]["rows"]}

    def l(self, name, w, src=None):
        return pick((src or self.R)[name], self.ix[w])

    def dl(self, name, w, src=None):
        return self.l(name, w, src) - self.l("none", w, src)

    def c(self, name):
        return self.l(name, "S") - self.l(name, "X")

    def idk(self):
        return 0.5 * (self.c("K_S") - self.c("K_X"))

    def idinj(self, s_row, x_row):
        return 0.5 * (self.c(s_row) - self.c(x_row))

    def NX(self):
        return self.b.mean(self.dl("K_X", "X"))

    def iota(self, name):
        return ratio(self.b.mean(self.dl(name, "X")), self.NX())

    def iota_m(self, name):
        """iota on the raw margin m_X = lp(X) - lp(B) (full vocabulary, not renormalised; exploratory E6)."""
        def m(nm):
            r = self.raw[nm]
            return pick(r, self.ix["X"]) - pick(r, self.ix["B"])
        return ratio(self.b.mean(m(name) - m("none")), self.b.mean(m("K_X") - m("none")))

    def pi_x(self, name):
        return argmax_rate(self.R[name], self.ix["X"])

    def floor(self):
        """Gate J-D-G2: mean over stories of the mean |lp(none2) - lp(none)| over the six candidates (full vocabulary)."""
        return float(np.mean(np.abs(self.raw["none2"] - self.raw["none"])))

    def d1(self):
        return dict(ratio=ratio(self.b.mean(self.idinj("looS", "looX")), self.b.mean(self.idk())), pi_x=self.pi_x("looX"),
                    idk=self.b.mean(self.idk()), idinj_full=ratio(self.b.mean(self.idinj("moveS", "move")), self.b.mean(self.idk())),
                    NX=self.NX(), iota={nm: self.iota(nm) for nm in ("add0.5", "add1", "add2", "move", "subB", "looX")},
                    pi_move=self.pi_x("move"))

    def d2(self):
        out = {}
        im = self.iota("move")
        for c in ("orth", "meanH", "active"):
            ic = self.iota(c)
            out[c] = dict(iota=ic, diff=im - ic, frac=ratio(ic, im))
        sanity = {nm: self.iota(nm) for nm in ("iso0", "iso1", "iso2", "hspan0", "hspan1", "hspan2", "perm", "choices", "question")}
        piX0 = self.pi_x("none")
        return dict(move=im, controls=out, sanity=sanity, capture=ratio(im, self.iota("own")),
                    pi_shift={nm: self.pi_x(nm) - piX0 for nm in ("choices", "question")})

    def d4(self):
        dmove = self.b.mean(self.dl("move", "X", self.RT))
        r = {C: one_minus(ratio(self.b.mean(self.dl(f"ans_{C}", "X", self.RT)), dmove)) for C in ("K", "V", "KV")}
        r["other"] = one_minus(ratio(self.b.mean(self.dl("other_KV", "X", self.RT)), dmove))
        return dict(denom=dmove, r=r, KmV=r["K"] - r["V"])

    def addr(self):
        return ratio(self.b.mean(self.idinj("kvS", "kvX")), self.b.mean(self.idinj("moveS", "move")))

    def kn(self):
        dN = self.R["K_N"] - self.R["none"]
        dB = self.R["subB"] - self.R["none"]
        rs = []
        for a, b in zip(dN, dB):
            sa, sb = a.std(), b.std()
            rs.append(float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else NAN)
        rs = np.array(rs)
        beta = ratio(self.b.mean(self.dl("K_N", "B")), self.b.mean(self.dl("subB", "B")))
        spec = self.b.mean(self.dl("K_N", "S") - self.dl("K_N", "X"))
        return dict(r=self.b.mean(np.nan_to_num(rs)), r_nan=int(np.isnan(rs).sum()), beta=beta, spec=spec, idk=self.b.mean(self.idk()))


# --------------------------------------------------------------------------- ablate (A, P1)
class Ablate:
    def __init__(self, J):
        self.S = J["stories"]
        self.n = len(self.S)
        self.b = boot(self.n)
        self.conds = list(self.S[0]["res"])

    def v(self, cond, k):
        return np.array([s["res"][cond][k] for s in self.S], float)

    def rho(self, cond, k="idK"):
        return ratio(self.b.mean(self.v(cond, k)), self.b.mean(self.v("none", k)))

    def rates(self, cond):
        return float(np.mean(self.v(cond, "base_ok"))), float(np.mean(self.v(cond, "init_ans")))


# --------------------------------------------------------------------------- bind (BIND, P1, three queries)
class Bind:
    def __init__(self, J):
        self.S = J["stories"]
        self.n = len(self.S)
        self.b = boot(self.n)
        self.ix = {k: np.array([s["ix"][k] for s in self.S]) for k in ("B", "S", "X", "I", "D")}
        self.strat = np.array([s["stratum"] == "object_first" for s in self.S])

    def arr(self, view, row):
        return renorm(np.array([s["views"][view][row] for s in self.S], float))

    def dl(self, view, row):
        return pick(self.arr(view, row), self.ix["X"]) - pick(self.arr(view, "none"), self.ix["X"])

    def acc(self, view):
        tgt = {"direct": "B", "other_agent": "I", "irrelevant_object": "D"}[view]
        return argmax_rate(self.arr(view, "none"), self.ix[tgt])

    def psi(self):
        x = (self.dl("other_agent", "init") - self.dl("irrelevant_object", "init")) - \
            (self.dl("other_agent", "dloc") - self.dl("irrelevant_object", "dloc"))
        pooled = self.b.mean(x)
        a, c = self.b.mean(x, self.strat), self.b.mean(x, ~self.strat)
        return dict(pooled=pooled, obj_first=a, dis_first=c, diff=a - c, n_obj=int(self.strat.sum()), n_dis=int((~self.strat).sum()))

    def psi_ev(self):
        x = (self.dl("direct", "ev") - self.dl("other_agent", "ev")) - (self.dl("direct", "init") - self.dl("other_agent", "init"))
        return self.b.mean(x)


# --------------------------------------------------------------------------- sign (Q cells, IOI cells)
class Cell:
    """One arm of the sign stage: the item measures, the transfer batch, the injection rows, the route, the hop-2 rows."""

    def __init__(self, recs, arm):
        self.S, self.arm = recs, arm
        self.n = len(recs)
        self.b = boot(self.n)
        self.ioi = arm in ("INLINE", "INLINE_CHAT", "AFTER")

    def item(self, k):
        return np.array([s["item"][k] for s in self.S], float)

    def competence(self):
        if self.ioi:
            two, four, LD = self.item("two_B"), self.item("four_B"), self.b.mean(self.item("LD_B"))
            ok = bool(two.mean() >= 0.75 and four.mean() >= 0.5 and LD.pt > 0 and LD.lo() > 0)
            return ok, f"two-way {two.mean():.2f} (>= 0.75), four-way {four.mean():.2f} (>= 0.50), LD {LD.txt()} (> 0, CI excl. 0)"
        words = [s["item"]["argmax"] for s in self.S]
        ment = [w in (s["core"]["base"], s["core"]["initial"], s["core"]["distractor_location"]) for w, s in zip(words, self.S)]
        mass = float(np.mean(self.item("mass")))
        rate = float(np.mean(ment)) if self.arm == "Q_IN" else float(1 - np.mean(ment))
        ok = bool(rate >= 0.9 and mass >= 0.5)
        what = "mentioned" if self.arm == "Q_IN" else "unmentioned"
        return ok, f"argmax {what} {rate:.2f} (>= 0.90), candidate mass {mass:.2f} (>= 0.50)"

    def ids(self):
        return self.b.mean(self.item("idK")), self.b.mean(self.item("idV"))

    def tr(self, name):
        cs = np.array([s["transfer"]["S"][name] - s["transfer"]["S"]["none"] for s in self.S], float)
        cx = np.array([s["transfer"]["X"][name] - s["transfer"]["X"]["none"] for s in self.S], float)
        return 0.5 * (cs - cx)

    def transfer(self):
        dG = self.b.mean(self.tr("allG"))
        idT = self.b.mean(self.tr("allT"))
        R = ratio(self.b.mean(self.tr("H")), dG)
        KO = one_minus(ratio(self.b.mean(self.tr("koH")), dG))
        Rr = [ratio(self.b.mean(self.tr(f"rand{i}")), dG) for i in range(3)]
        KOr = [one_minus(ratio(self.b.mean(self.tr(f"korand{i}")), dG)) for i in range(3)]
        return dict(dG=dG, idT=idT, R=R, KO=KO, Rrand=mean_q(Rr), KOrand=mean_q(KOr), dGc=ratio(self.b.mean(self.tr("allGc")), idT),
                    dG_over_idT=ratio(dG, idT))

    def scores(self, field, name):
        k = 4 if self.ioi else 6
        a = np.array([s[field][name] for s in self.S], float)
        assert a.shape[1] == k
        return renorm(a)

    def xi(self):
        return np.array([IOI4["X"]] * self.n) if self.ioi else np.array([s["ix"]["X"] for s in self.S])

    def dlx(self, field, name):
        return pick(self.scores(field, name), self.xi()) - pick(self.scores(field, "none"), self.xi())

    def hop2(self):
        dinj = self.b.mean(self.dlx("hop2", "inj"))
        carry = {h: one_minus(ratio(self.b.mean(self.dlx("hop2", h)), dinj)) for h in ("top10", "rand10", "all")}
        return dict(dinj=dinj, carry=carry)

    def route(self):
        """IOI INLINE: m = l_S - l_B; r(C) = 1 - mean dm(K_S + ans_C) / mean dm(K_S); the same for the injected flag."""
        def m(name):
            a = self.scores("route", name)
            return a[:, IOI4["S"]] - a[:, IOI4["B"]]
        out = {}
        for src in ("K_S", "inj"):
            den = self.b.mean(m(src) - m("none"))
            r = {C: one_minus(ratio(self.b.mean(m(f"{src}+ans_{C}") - m("none")), den)) for C in ("K", "V", "KV")}
            r["other"] = one_minus(ratio(self.b.mean(m(f"{src}+other_KV") - m("none")), den))
            out[src] = dict(denom=den, r=r, KmV=r["K"] - r["V"])
        return out

    def floor(self):
        a = np.array([s["rows"]["none2"] for s in self.S], float)
        b = np.array([s["rows"]["none"] for s in self.S], float)
        return float(np.mean(np.abs(a - b)))

    def behaviour(self):
        """Q_OUT headline (exploratory): the fraction of stories whose answer is the story's own word B under K_S, V_S."""
        bi = np.array([s["ix"]["B"] for s in self.S])
        return {nm: argmax_rate(self.scores("rows", nm), bi) for nm in ("none", "K_S", "V_S")}


# --------------------------------------------------------------------------- diss (D6)
class Diss:
    def __init__(self, J):
        self.J = J
        self.S = J["stories"]
        self.n = len(self.S)
        self.ix = {k: np.array([s["ix"][k] for s in self.S]) for k in ("B", "S", "X")}
        self.comp = np.ones(self.n, bool)
        for f in ("P1", "POST"):
            cb = np.argmax(np.array([s["fmt"][f]["cleanB"] for s in self.S]), 1)
            cs = np.argmax(np.array([s["fmt"][f]["cleanS"] for s in self.S]), 1)
            self.comp &= (cb == self.ix["B"]) & (cs == self.ix["S"])
        self.ci = np.flatnonzero(self.comp)
        self.bc = boot(len(self.ci))
        self.ba = boot(self.n)

    def dlx(self, f, name, field="rows", sel=None):
        sel = self.ci if sel is None else sel
        a = renorm(np.array([self.S[i]["fmt"][f][field][name] for i in sel], float))
        z = renorm(np.array([self.S[i]["fmt"][f][field]["none"] for i in sel], float))
        xi = self.ix["X"][sel]
        return pick(a, xi) - pick(z, xi)

    def rho(self):
        inj = {f: self.bc.mean(self.dlx(f, "injP1")) for f in ("P1", "POST")}
        nx = {f: self.bc.mean(self.dlx(f, "K_X")) for f in ("P1", "POST")}
        rho, rnat = ratio(inj["POST"], inj["P1"]), ratio(nx["POST"], nx["P1"])
        return dict(inj=inj, NX=nx, rho=rho, rho_nat=rnat, d=rho - rnat,
                    d6b=ratio(self.bc.mean(self.dlx("P1", "injPOST")), inj["P1"]))

    def omega(self):
        def W(f):
            return np.array([sum(s["fmt"][f]["W"].values()) for s in self.S], float)
        return ratio(self.ba.mean(W("POST")), self.ba.mean(W("P1")))

    def route(self):
        den = self.bc.mean(self.dlx("POST", "inj", "route"))
        r = {C: one_minus(ratio(self.bc.mean(self.dlx("POST", f"ans_{C}", "route")), den)) for C in ("K", "V", "KV")}
        return dict(denom=den, r=r, VmK=r["V"] - r["K"])

    def gate_R(self):
        if "gate" not in self.S[0]["fmt"]["P1"]:
            return None
        g = [s["fmt"]["P1"]["gate"] for s in self.S]

        def i(name):
            return np.array([0.5 * ((x["S"][name] - x["S"]["none"]) - (x["X"][name] - x["X"]["none"])) for x in g], float)
        return ratio(self.ba.mean(i("H")), self.ba.mean(i("allG")))

    def floor(self):
        a = np.array([s["fmt"][f]["rows"]["none2"] for s in self.S for f in ("P1", "POST")], float)
        b = np.array([s["fmt"][f]["rows"]["none"] for s in self.S for f in ("P1", "POST")], float)
        return float(np.mean(np.abs(a - b)))
