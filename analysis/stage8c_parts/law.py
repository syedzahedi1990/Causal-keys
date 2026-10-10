"""The channel-ratio law of P-2026-10-10-J part C on eval/<tag>.json of experiments/stage8_edits.py: per-story identity
contrasts, the cell gates (J-C-G1, J-C-G2, J-C-G4, J-C-G5, J-C-G6), the lambda statistics, the sensitivity gate J-C-G8
and the descriptive kappa / sigma, nu and D.

Rows of a cell (format f, depth l): self, and "<inst>|<C>|<t>" (C in K, V, KV; t in S, X), each stored as its E and L
scores minus those of the in-batch self row, its E mass and its E argmax. For an instance Z (nat, T, R, E1, E2, E3, E4a,
E4b, E5FR, E5DE, E5SYN, E5NL, PAR:Z, PERP:Z, LEX:Z, NONLEX:Z, LEX:nat, NONLEX:nat, SK50, SK80, SV50, SV80) and channel C:
  ID_C^Z = 1/2 [(d_S(C(Z_S)) - d_S(C(Z_X))) + (d_X(C(Z_X)) - d_X(C(Z_S)))]       (d = score minus the self row's)
(letters of S and X under LETTER). SK* rows: K from the interpolated table, V from B (K row) or from t (KV row); their V
row is the natural V row. SV* likewise with K and V exchanged. E4 = the two seeds as a level ([n, 2] arrays).
  psi_C^Z(f, l) = mean ID_C^Z / mean ID_C^nat;  phi = psi_KV;  iota^Z = 1 - (mean ID_K^Z + mean ID_V^Z) / mean ID_KV^Z;
  kappa^Z = mean ID_K^Z / (mean ID_K^Z + mean ID_V^Z) (descriptive), sigma = kappa^nat.
  channel C used in (f, l): mean ID_C^nat >= 2 nats and >= 0.1 mean ID_KV^nat.
  lambda statistics of Z at depth l:  W(f, l) = log psi_K(f, l) - log psi_V(f, l), f in {P1, POST}, both channels used;
  A_L(l) = log psi_K(LETTER, l) - log psi_V(NONE, l); A_P(l) = log psi_K(P1, l) - log psi_V(NONE, l) (K used in the first
  cell, V in NONE); psi floored at 0.01 before the log. A statistic needs its cells evaluable (J-C-G4) and Z effective at
  l (J-C-G5; components: carrying identity); in a cell where Z's iota > 0.5 (the natural cell passing) the statistic is a
  departure ("interaction").
"""
from __future__ import annotations

import numpy as np

from .stats import LOG125, LOG2, LOG05, NAN, Q, boot_of, log_ratio, mean_q, ratio, tost

FORMATS = ("LETTER", "P1", "POST", "NONE")
COMP_FORMATS = ("P1", "NONE")
MIN_ID, MIN_SHARE = 2.0, 0.1            # channel used
G4_ID, G4_IOTA, G4_NEG, G4_COV = 2.0, 0.5, 0.1, 0.8
G5_FLIP, G5_PHI, G5_LO = 0.8, 0.5, 0.3
CARRY_PHI = 0.3                          # E5 efficacy and "carries identity": phi(NONE) >= 0.3 with lower bound > 0
G1_NU, G1_PSI, G1_ID = 0.02, 0.05, 0.25
G2_GAP = 0.5
G6_R = 0.1
MIN_STATS = 3
SEED_PAIRS = {"E4": ("E4a", "E4b")}


def probe(Z):
    """A concrete row prefix of instance Z (the first seed of a seed-level instance)."""
    head, _, base = Z.rpartition(":")
    if base in SEED_PAIRS:
        base = SEED_PAIRS[base][0]
    return f"{head}:{base}" if head else base


class Model:
    """One model's eval file as per-story arrays."""

    def __init__(self, J, test=False):
        self.J, self.test = J, test
        self.P = J["provenance"]
        self.S = J["stories"]
        self.n = len(self.S)
        self.depths = [int(x) for x in J["depths"]]
        self.clusters = [(s["core"]["base"], s["core"]["source"]) for s in self.S]
        self.boot = boot_of(self.clusters)
        self.iS = np.array([s["iS"] for s in self.S])
        self.iX = np.array([s["iX"] for s in self.S])
        self._id = {}

    # ---- raw rows
    def row(self, s, f, l, name):
        return self.S[s]["cells"][f"{f}@{l}"].get(name)

    def has(self, f, l, name):
        return all(self.row(s, f, l, name) is not None for s in range(self.n)) and self.n > 0

    def _rowname(self, Z, C, t):
        if Z.startswith("SK") and C == "V":
            return f"nat|V|{t}"
        if Z.startswith("SV") and C == "K":
            return f"nat|K|{t}"
        return f"{Z}|{C}|{t}"

    def ID(self, f, l, Z, C, field="E"):
        """[n] per-story ID_C^Z (NaN where a row is missing); Z = "E4" gives [n, 2] (the seeds)."""
        if Z in SEED_PAIRS or (":" in Z and Z.split(":")[1] in SEED_PAIRS):
            pre = Z.split(":")[0] + ":" if ":" in Z else ""
            base = Z.split(":")[-1]
            return np.stack([self.ID(f, l, pre + z, C, field) for z in SEED_PAIRS[base]], 1)
        k = (f, l, Z, C, field)
        if k not in self._id:
            off = 0 if field == "E" else 6
            out = np.full(self.n, NAN)
            for s in range(self.n):
                a, b = self.row(s, f, l, self._rowname(Z, C, "S")), self.row(s, f, l, self._rowname(Z, C, "X"))
                if a is None or b is None:
                    continue
                iS, iX = self.iS[s] + off, self.iX[s] + off
                out[s] = 0.5 * ((a[iS] - b[iS]) + (b[iX] - a[iX]))
            self._id[k] = out
        return self._id[k]

    def mean(self, f, l, Z, C, field="E") -> Q:
        return Q(*self.boot.mean(self.ID(f, l, Z, C, field)))

    def psi(self, f, l, Z, C, field="E") -> Q:
        return ratio(self.mean(f, l, Z, C, field), self.mean(f, l, "nat", C, field))

    def flip(self, f, l, Z):
        """Per-story flip rate (mean over t of argmax of the KV row == t); Z = "E4" averages the seeds."""
        if Z in SEED_PAIRS:
            return np.nanmean(np.stack([self.flip(f, l, z) for z in SEED_PAIRS[Z]], 1), 1)
        out = np.full(self.n, NAN)
        for s in range(self.n):
            a, b = self.row(s, f, l, f"{Z}|KV|S"), self.row(s, f, l, f"{Z}|KV|X")
            if a is None or b is None:
                continue
            out[s] = 0.5 * (float(int(a[13]) == int(self.iS[s])) + float(int(b[13]) == int(self.iX[s])))   # not numpy bool + bool (= or)
        return out

    def coverage(self, f, l):
        """Minimum over the natural rows (self, K / V / KV of S and X) of the mean E mass."""
        rows = ["self"] + [f"nat|{C}|{t}" for C in ("K", "V", "KV") for t in ("S", "X")]
        m = []
        for r in rows:
            v = [self.row(s, f, l, r) for s in range(self.n)]
            if any(x is None for x in v):
                return NAN
            m.append(np.mean([x[12] for x in v]))
        return float(min(m))

    def stat(self, l, Z, t, key):
        """Mean over stories of a stored table statistic of instance Z (target t) at depth l (NaN if absent)."""
        v = [self.S[s]["stats"].get(str(l), {}).get(f"{Z}|{t}", {}).get(key) for s in range(self.n)]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else NAN

    # ---- gates
    def cell_ok(self, f, l):
        """J-C-G4 (point estimates): (ok, reason)."""
        if not self.has(f, l, "nat|KV|S"):
            return False, "rows missing"
        K, Vv, KV = (self.mean(f, l, "nat", C).pt for C in ("K", "V", "KV"))
        cov = self.coverage(f, l)
        why = []
        if not KV >= G4_ID:
            why.append(f"ID_KV {KV:.2f} < {G4_ID}")
        iota = 1 - (K + Vv) / KV if KV else NAN
        if not iota <= G4_IOTA:
            why.append(f"iota {iota:.2f} > {G4_IOTA}")
        if not (K >= -G4_NEG * KV and Vv >= -G4_NEG * KV):
            why.append("a channel below -0.1 ID_KV")
        if not cov >= G4_COV:
            why.append(f"coverage {cov:.2f} < {G4_COV}")
        return not why, "; ".join(why)

    def used(self, f, l, C):
        KV = self.mean(f, l, "nat", "KV").pt
        c = self.mean(f, l, "nat", C).pt
        return bool(c >= MIN_ID and c >= MIN_SHARE * KV)

    def iota(self, f, l, Z):
        K, Vv, KV = (self.mean(f, l, Z, C).pt for C in ("K", "V", "KV"))
        return 1 - (K + Vv) / KV if KV else NAN

    def effective(self, l, Z, kind="edit"):
        """J-C-G5 at depth l under NONE: (ok, text). kind "edit": flip rate >= 0.8 x the natural flip rate and phi >= 0.5
        with lower bound > 0.3; kind "carry" (E5, components): phi >= 0.3 with lower bound > 0."""
        ok4, why4 = self.cell_ok("NONE", l)
        if not ok4:
            return None, f"NONE cell not evaluable ({why4})"
        if not self.has("NONE", l, probe(Z) + "|KV|S"):
            return None, "rows missing"
        phi = self.psi("NONE", l, Z, "KV")
        lo = phi.lower()
        if kind == "carry":
            ok = bool(phi.pt >= CARRY_PHI and lo > 0)
            return ok, f"phi(NONE) {phi.txt()} (>= {CARRY_PHI}, lower > 0)"
        fz, fn = np.nanmean(self.flip("NONE", l, Z)), np.nanmean(self.flip("NONE", l, "nat"))
        ok = bool(fz >= G5_FLIP * fn and phi.pt >= G5_PHI and lo > G5_LO)
        return ok, f"flip {fz:.2f} vs natural {fn:.2f}; phi(NONE) {phi.txt()} (>= {G5_PHI}, lower > {G5_LO})"

    def g1(self):
        """J-C-G1, the equivalence control T, over the evaluable cells: nu_T <= 0.02; |psi_C^T - 1| <= 0.05 in each used
        channel; mean |ID_K^T - ID_K^nat| and mean |ID_V^T - ID_V^nat| <= 0.25 nats."""
        bad, n = [], 0
        for l in self.depths:
            nu = max(self.stat(l, "T", t, "nu") for t in ("S", "X"))
            if not nu <= G1_NU:
                bad.append(f"nu_T {nu:.3f} at l={l}")
            for f in FORMATS:
                if not self.cell_ok(f, l)[0]:
                    continue
                n += 1
                for C in ("K", "V"):
                    if self.used(f, l, C):
                        ps = self.psi(f, l, "T", C).pt
                        if not abs(ps - 1) <= G1_PSI:
                            bad.append(f"psi_{C}^T {ps:.3f} in {f}@{l}")
                    d = np.nanmean(np.abs(self.ID(f, l, "T", C) - self.ID(f, l, "nat", C)))
                    if not d <= G1_ID:
                        bad.append(f"mean |ID_{C}^T - ID_{C}^nat| {d:.3f} in {f}@{l}")
        return (not bad and n > 0), (f"{n} evaluable cells; " + ("all within tolerance" if not bad else "; ".join(bad[:6])))

    def sigma(self, f, l) -> Q:
        K, Vv = self.mean(f, l, "nat", "K"), self.mean(f, l, "nat", "V")
        with np.errstate(all="ignore"):
            return Q(K.pt / (K.pt + Vv.pt), K.bs / (K.bs + Vv.bs))

    def kappa(self, f, l, Z) -> Q:
        K, Vv = self.mean(f, l, Z, "K"), self.mean(f, l, Z, "V")
        with np.errstate(all="ignore"):
            return Q(K.pt / (K.pt + Vv.pt), K.bs / (K.bs + Vv.bs))

    def g2(self):
        l = min(self.depths)
        d = self.sigma("LETTER", l)
        n = self.sigma("NONE", l)
        diff = Q(d.pt - n.pt, d.bs - n.bs)
        ok = bool(diff.pt >= G2_GAP and diff.lower() > 0)
        return ok, f"sigma(LETTER, {l}) - sigma(NONE, {l}) {diff.txt()} (>= {G2_GAP}; H0: <= 0 rejected)"

    def g6(self):
        out = {}
        for l in self.depths:
            if not self.has("NONE", l, "R|KV|S"):
                out[l] = (None, "rows missing")
                continue
            r, n = self.mean("NONE", l, "R", "KV").pt, self.mean("NONE", l, "nat", "KV").pt
            out[l] = (bool(abs(r) <= G6_R * n), f"|ID_KV^R| {abs(r):.3f} vs 0.1 ID_KV^nat {G6_R * n:.3f}")
        return out

    # ---- lambda statistics
    def stats_of(self, Z, depths, gate="edit", formats=FORMATS):
        """The lambda statistics of instance Z over ``depths``: list of dicts (code, depth, q, equivalent, interaction),
        and the list of reasons for the depths / statistics left out. gate: "edit" (J-C-G5), "carry", or None."""
        out, skip = [], []
        for l in depths:
            if gate is not None:
                eff, why = self.effective(l, Z, gate)
                if not eff:
                    skip.append(f"l={l}: {'not effective' if eff is False else 'not evaluable'} ({why})")
                    continue
            cand = []
            for f in ("P1", "POST"):
                if f in formats:
                    cand.append((f"W({f},{l})", [(f, "K"), (f, "V")]))
            if "LETTER" in formats:
                cand.append((f"A_L({l})", [("LETTER", "K"), ("NONE", "V")]))
            cand.append((f"A_P({l})", [("P1", "K"), ("NONE", "V")]))
            for code, (ck, cv) in cand:
                cells = {ck[0], cv[0]}
                if not all(self.cell_ok(f, l)[0] for f in cells):
                    skip.append(f"{code}: a cell fails J-C-G4")
                    continue
                if not (self.used(ck[0], l, "K") and self.used(cv[0], l, "V")):
                    skip.append(f"{code}: channel not used")
                    continue
                if not all(self.has(f, l, f"{probe(Z)}|{'V' if Z.startswith('SV') else 'K'}|S") for f in cells):
                    skip.append(f"{code}: rows missing")
                    continue
                q = log_ratio(self.psi(ck[0], l, Z, "K"), self.psi(cv[0], l, Z, "V"))
                inter = [f for f in cells if self.iota(f, l, Z) > G4_IOTA] if not Z.startswith(("SK", "SV")) else []
                out.append({"code": code, "l": l, "q": q, "equivalent": tost(q) and not inter, "interaction": inter})
        return out, skip


def pattern(stats):
    """The pooled lambda (mean over the statistics) and its pattern: equivalent (every statistic), R1 (<= log 0.5),
    R3 (>= log 2) or graded departure."""
    if not stats:
        return None, "no statistic"
    pooled = mean_q([s["q"] for s in stats])
    if all(s["equivalent"] for s in stats):
        return pooled, "equivalent"
    if pooled.pt <= LOG05:
        return pooled, "R1 copy-only"
    if pooled.pt >= LOG2:
        return pooled, "R3 key-flat"
    inside = abs(pooled.pt) < LOG125
    return pooled, "graded departure" + (" (pooled lambda within the margin; some statistic not equivalent)" if inside else "")


def law(m: Model, Z, depths, gate="edit", formats=FORMATS):
    """(met, text, stats, pattern) of the law for one combo; met None when fewer than MIN_STATS statistics."""
    st, skip = m.stats_of(Z, depths, gate, formats)
    pooled, pat = pattern(st)
    if len(st) < MIN_STATS:
        return None, f"{len(st)} statistics (< {MIN_STATS}); left out: {'; '.join(skip[:6]) or 'none'}", st, pat
    met = all(s["equivalent"] for s in st)
    txt = (f"{sum(s['equivalent'] for s in st)}/{len(st)} equivalent; pooled lambda {pooled.txt(0.90)} (90 %) -> {pat}; "
           + ", ".join(f"{s['code']} {s['q'].pt:+.2f} [{s['q'].lower(0.9):+.2f},{s['q'].upper(0.9):+.2f}]"
                       + ("*" if not s["equivalent"] else "") + ("(iota)" if s["interaction"] else "") for s in st))
    return met, txt, st, pat


def old_jc3(m: Model, Z, depths):
    """The design's JC3 rule on kappa (reported for J-C-G8): MAD of |kappa_Z - sigma| over the evaluable cells <= 0.12
    with bootstrap upper bound <= 0.17, and max <= 0.25."""
    gaps = []
    for l in depths:
        for f in FORMATS:
            if m.cell_ok(f, l)[0] and m.has(f, l, f"{Z}|KV|S"):
                k, s = m.kappa(f, l, Z), m.sigma(f, l)
                gaps.append(Q(abs(k.pt - s.pt), np.abs(k.bs - s.bs)))
    if not gaps:
        return None, "no cell"
    mad = mean_q(gaps)
    mx = max(g.pt for g in gaps)
    ok = mad.pt <= 0.12 and mad.upper() <= 0.17 and mx <= 0.25
    return ok, f"MAD {mad.txt()} max {mx:.3f} over {len(gaps)} cells -> old rule {'passes' if ok else 'fails'}"


def g8(m: Model):
    """J-C-G8: T classified equivalent; SK50 a departure with pooled lambda < -log 1.25; SV50 a departure with pooled
    lambda > log 1.25 (no efficacy gate on these rows). Returns (ok, lines)."""
    lines, ok = [], True
    for Z, want in (("T", "equivalent"), ("SK50", "neg"), ("SV50", "pos"), ("SK80", None), ("SV80", None)):
        st, skip = m.stats_of(Z, m.depths, gate=None)
        pooled, pat = pattern(st)
        if not st:
            res = None
        elif want == "equivalent":
            res = all(s["equivalent"] for s in st)
        elif want == "neg":
            res = not all(s["equivalent"] for s in st) and pooled.pt < -LOG125
        elif want == "pos":
            res = not all(s["equivalent"] for s in st) and pooled.pt > LOG125
        else:
            res = None
        o, otxt = old_jc3(m, Z, m.depths)
        lines.append(f"{Z}: {len(st)} statistics, pooled lambda {pooled.txt(0.9) if pooled else 'nan'} -> {pat}"
                     + ("" if want is None else f" (required {'equivalent' if want == 'equivalent' else 'a departure, ' + ('lambda < -log 1.25' if want == 'neg' else 'lambda > log 1.25')}): {'yes' if res else 'no' if res is False else 'not evaluable'}")
                     + f"; {otxt}")
        if want is not None:
            ok = ok and bool(res)
    return ok, lines
