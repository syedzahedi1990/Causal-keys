"""J-C-READ of P-2026-10-10-J part C on readers/<tag>.json (experiments/stage8_edits.py, stage readers): the same reader
heads for the edits' keys. OPTIONS-AFTER (P1), l = 7, L scores (the six lower-case " w" log-probabilities), every row
differenced against the in-batch self row.

Rows "<donor>|<t>|<spec>": the donor's key at p from block 8 is seen by every head in every row (spec "all", equal to the
K-only clamp row), or by every head except the heads of a set in the six option rows G (spec "H": H*_{>7}; "r0".."r2":
size-matched random sets of blocks > 7), which see B's key; B's value at p is pinned in every row.
  ID_K^D[spec] = 1/2 [(d_S(D_S) - d_S(D_X)) + (d_X(D_X) - d_X(D_S))];  KO_D(spec) = 1 - mean ID_K^D[spec] / mean ID_K^D[all].
Gate (per model): the natural KO_nat(H) >= 0.7 and mean ID_K^nat[all] >= 2 nats. A family D counts when it is effective at
l = 7 (J-C-G5 on the eval file) and mean ID_K^D[all] >= 2 nats (the denominator gate). Criterion per counted family:
KO_D(H) >= 0.7 (effect size) with H0: KO_D(H) <= 0.5 rejected (lower 95 % bound > 0.5), and the mean over the three random
sets KO_D(r) <= 0.25 (effect size).
"""
from __future__ import annotations

import numpy as np

from .stats import Q, boot_of, mean_q, ratio

KO_MIN, KO_LO, KO_RAND, ID_MIN = 0.7, 0.5, 0.25, 2.0


class Readers:
    def __init__(self, J):
        self.J, self.S = J, J["stories"]
        self.n = len(self.S)
        self.boot = boot_of([(s["core"]["base"], s["core"]["source"]) for s in self.S])
        self.donors, self.specs = J["donors"], J["specs"]

    def ID(self, D, spec):
        out = np.full(self.n, np.nan)
        for i, s in enumerate(self.S):
            a, b = s["rows"].get(f"{D}|S|{spec}"), s["rows"].get(f"{D}|X|{spec}")
            if a is None or b is None:
                continue
            iS, iX = s["iS"], s["iX"]
            out[i] = 0.5 * ((a[iS] - b[iS]) + (b[iX] - a[iX]))
        return out

    def m(self, D, spec):
        return Q(*self.boot.mean(self.ID(D, spec)))

    def ko(self, D, spec):
        r = ratio(self.m(D, spec), self.m(D, "all"))
        return Q(1 - r.pt, 1 - r.bs)

    def gate(self):
        k, d = self.ko("nat", "H"), self.m("nat", "all")
        ok = bool(k.pt >= KO_MIN and d.pt >= ID_MIN)
        return ok, f"natural KO(H*) {k.txt()} (>= {KO_MIN}); ID_K^nat {d.txt()} (>= {ID_MIN})"

    def family(self, D):
        d = self.m(D, "all")
        if not d.pt >= ID_MIN:
            return None, f"ID_K^{D} {d.txt()} < {ID_MIN} nats (denominator gate)"
        k = self.ko(D, "H")
        kr = mean_q([self.ko(D, f"r{i}") for i in range(3)])
        ok = bool(k.pt >= KO_MIN and k.lower() > KO_LO and kr.pt <= KO_RAND)
        return ok, f"KO(H*) {k.txt()} (>= {KO_MIN}, H0: <= {KO_LO} rejected); random mean {kr.txt()} (<= {KO_RAND}); ID_K {d.pt:+.2f}"
