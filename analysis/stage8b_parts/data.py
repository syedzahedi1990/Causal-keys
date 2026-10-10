"""Per-model data of the stage-8 part-B scorer: one Pop per (model, population) from an eval file of
experiments/fresh_factorial.py, with the per-item arrays every statistic is built from.

Scorings sigma: "L" (log p(" w")), "sigma" (the 12 reviewer-named forms), "E" (the emitted forms), and "beta" (generated
answers). For an item in arm f, with rows on the B prompt and d_w(row) = sigma(w | row) - sigma(w | ID@0):
  ID_K = 1/2 [(d_S(K_S@0) - d_S(K_X@0)) + (d_X(K_X@0) - d_X(K_S@0))], ID_V likewise with V_S, V_X, ID_KV with KV_S, KV_X;
  beta_K = 1/2 [(1[a(K_S)=S] - 1[a(K_X)=S]) + (1[a(K_X)=X] - 1[a(K_S)=X])], beta_V likewise (a = the parsed generation).
Mass of a row: the sum over the six candidates of exp sigma(c | row). Coverage of a cell: the minimum over its 16 rows
(13 clamp rows, clean B, S, X) of the mean E mass. Floor of a cell: mean |m^E(ID@0) - m^E(clean B)|, m = E(S) - E(B).
Competent item in an arm: the E-argmax over the six candidates names B in the clean B run and S in the clean S run.
Agreement A of a cell: among the 8 generated rows of every item whose answer names a candidate, the share whose answer is
the E-argmax of that row.
"""
from __future__ import annotations

import functools

import numpy as np

from .stats import boot_of

ROWS13 = ("ID@0", "K_S@0", "V_S@0", "KV_S@0", "K_X@0", "V_X@0", "KV_X@0")
GEN = ("B", "S", "X", "ID@0", "K_S@0", "K_X@0", "V_S@0", "V_X@0")
CLEAN = ("B", "S", "X")


def item_index(key: str) -> int:
    return int("".join(ch for ch in key[-3:] if ch.isdigit()))


def ident(rec, sg, ch):
    """ID_ch of one item under scoring sg in {L, sigma, E}."""
    W = rec["words"]
    iS, iX = W.index(rec["track"]["S"]), W.index(rec["track"]["X"])
    R = rec["rows"]
    d = lambda row, i: R[row][sg][i] - R["ID@0"][sg][i]  # noqa: E731
    return 0.5 * ((d(f"{ch}_S@0", iS) - d(f"{ch}_X@0", iS)) + (d(f"{ch}_X@0", iX) - d(f"{ch}_S@0", iX)))


def beta(rec, ch):
    a, tr = rec["ans"], rec["track"]
    S, X = tr["S"], tr["X"]
    return 0.5 * (((a[f"{ch}_S@0"] == S) - (a[f"{ch}_X@0"] == S)) + ((a[f"{ch}_X@0"] == X) - (a[f"{ch}_S@0"] == X)))


def ff_ident(ff, ch):
    """format_factorial's identity (the stage-1 / 3b scorers' per_core) from a format_factorial item's "m" rows."""
    m = ff["m"]
    d = lambda row, t: m[row]["lp"][t] - m["ID@0"]["lp"][t]  # noqa: E731
    return 0.5 * ((d(f"{ch}_S@0", "S") - d(f"{ch}_X@0", "S")) + (d(f"{ch}_X@0", "X") - d(f"{ch}_S@0", "X")))


class Pop:
    """One model's eval file on one population: the items valid in every arm that was run."""

    def __init__(self, J, key: str):
        self.J, self.key = J, key
        self.P = J["provenance"]
        self.pop = self.P.get("population")
        res = J["results"]
        self.arms = [a for a in self.P.get("arms", []) if a not in (self.P.get("arms_not_run") or [])]
        self.by = {a: {r["key"]: r for r in res if r["arm"] == a} for a in self.arms}
        common = set.intersection(*[set(v) for v in self.by.values()]) if self.by else set()
        self.keys = sorted(common, key=item_index)
        self.missing = sorted(set().union(*[set(v) for v in self.by.values()]) - common, key=item_index) if self.by else []
        rec0 = self.by[self.arms[0]] if self.arms else {}
        self.clusters = [tuple(rec0[k]["cluster"]) for k in self.keys]
        self.idx = np.arange(len(self.keys))
        self.boot = boot_of(self.clusters)
        self.cboot = boot_of(None, len(self.keys))

    def has(self, arm):
        return arm in self.by

    def recs(self, arm):
        return [self.by[arm][k] for k in self.keys]

    def view(self, keep=None):
        return View(self, np.arange(len(self.keys)) if keep is None else np.flatnonzero(keep))

    @functools.lru_cache(maxsize=None)
    def _arr(self, what, arm, *args):
        R = self.recs(arm)
        if what == "id":
            sg, ch = args
            if sg == "beta":
                return np.array([beta(r, ch) for r in R], float)
            return np.array([ident(r, sg, ch) for r in R], float)
        if what == "mass":
            row, sg = args
            src = (lambda r: r["clean"][row]) if row in CLEAN else (lambda r: r["rows"][row])
            return np.array([float(np.exp(src(r)[sg]).sum()) for r in R], float)
        if what == "acc":
            (run,) = args
            return np.array([r["ans"][run] == r["track"][run] for r in R], float)
        if what == "other":
            (run,) = args
            return np.array([r["ans"][run] == "other" for r in R], float)
        if what == "comp":
            return np.array([r["words"][int(np.argmax(r["clean"]["B"]["E"]))] == r["track"]["B"] and
                             r["words"][int(np.argmax(r["clean"]["S"]["E"]))] == r["track"]["S"] for r in R], float)
        if what == "floor":
            def m(v, r):
                W = r["words"]
                return v[W.index(r["track"]["S"])] - v[W.index(r["track"]["B"])]
            return np.array([abs(m(r["rows"]["ID@0"]["E"], r) - m(r["clean"]["B"]["E"], r)) for r in R], float)
        if what == "d":   # format_factorial's d of the m = sigma(S) - sigma(B) readout, row against ID@0
            row, sg = args
            def m(v, r):
                W = r["words"]
                return v[W.index(r["track"]["S"])] - v[W.index(r["track"]["B"])]
            return np.array([m(r["rows"][row][sg], r) - m(r["rows"]["ID@0"][sg], r) for r in R], float)
        if what == "gap":
            (row,) = args
            src = (lambda r: r["clean"][row]) if row in CLEAN else (lambda r: r["rows"][row])
            return np.array([src(r)["gap"] for r in R], float)
        raise KeyError(what)

    def rows_of(self, arm):
        r0 = self.recs(arm)[0] if self.keys else {"rows": {}}
        return list(r0["rows"]) + list(CLEAN)

    def agreement(self, arm, keep=None):
        """(agreements, answers naming a candidate) over the 8 generated rows."""
        num = den = 0
        R = self.recs(arm)
        sel = range(len(R)) if keep is None else keep
        for i in sel:
            r = R[i]
            for g in GEN:
                a = r["ans"][g]
                if a == "other":
                    continue
                v = r["clean"][g]["E"] if g in CLEAN else r["rows"][g]["E"]
                den += 1
                num += r["words"][int(np.argmax(v))] == a
        return num, den


class View:
    """A population's items (all, or a subset such as the competent ones) with the bootstrap of that set."""

    def __init__(self, pop: Pop, idx):
        self.pop, self.idx = pop, np.asarray(idx, int)
        self.n = len(self.idx)
        full = len(self.idx) == len(pop.keys)
        self.boot = pop.boot if full else boot_of([pop.clusters[i] for i in self.idx])
        self.cboot = pop.cboot if full else boot_of(None, self.n)

    def k(self, arm, sg):
        return self.pop._arr("id", arm, sg, "K")[self.idx]

    def v(self, arm, sg):
        return self.pop._arr("id", arm, sg, "V")[self.idx]

    def kv(self, arm, sg):
        return self.pop._arr("id", arm, sg, "KV")[self.idx]

    def mass(self, arm, row, sg):
        return self.pop._arr("mass", arm, row, sg)[self.idx]

    def acc(self, arm, run="B"):
        return self.pop._arr("acc", arm, run)[self.idx]

    def other(self, arm, run="B"):
        return self.pop._arr("other", arm, run)[self.idx]

    def comp(self, arm):
        return self.pop._arr("comp", arm)[self.idx]

    def floor(self, arm):
        return self.pop._arr("floor", arm)[self.idx]

    def d(self, arm, row, sg):
        return self.pop._arr("d", arm, row, sg)[self.idx]

    def gap(self, arm, row):
        return self.pop._arr("gap", arm, row)[self.idx]

    def coverage(self, arm, sg="E"):
        """(minimum over the rows of the mean mass, the row where it is reached)."""
        vals = {row: float(self.mass(arm, row, sg).mean()) for row in self.pop.rows_of(arm)} if self.n else {}
        if not vals:
            return float("nan"), None
        row = min(vals, key=vals.get)
        return vals[row], row
