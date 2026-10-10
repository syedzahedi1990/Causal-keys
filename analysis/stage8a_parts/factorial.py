"""Part A of P-2026-10-10-J, the factorial lines: per-model view of factorial/<tag>.json (and explore/<tag>.json), gates
J-A-G1 to J-A-G4, populations C_m(F) and PF_m(F), lines J-A1 to J-A8d, the reported lines and the exploratory report.

Definitions (as in the entry):
  lp_Y(r): the log-probability of Y's decision token after B prompt + w under row r; Delta_Y(r) = lp_Y(r) - lp_Y(ID).
  ID_K = 1/2[(Delta_S(K_S) - Delta_S(K_X)) + (Delta_X(K_X) - Delta_X(K_S))]; ID_V the same with V_S, V_X.
  s_ID = mean ID_K / (mean ID_K + mean ID_V), a ratio of means recomputed in every resample.
  who(r): the entity row r's greedy generation names (B, S, X, Z, D, other; LETA: by letter).
  competent(i, f): who(ID) = B, who(KV_S) = S and who(KV_X) = X in format f. C(F): competent in every f of F.
  prior-free(i, F): the closed-book argmax over the four options is not B (CBOPT; CBLET for LETA) and who(KV_Z) != B in
  every f of F that has the KV_Z row. PF(F) = C(F) and prior-free.
"""
import statistics

import numpy as np

from .common import NAN, est, inside, lower, mean, point, ratio, sid, upper

FORMATS = ("NOM", "OPTA", "OPTB", "MENA", "LETA", "MENB")
FULL = ("NOM", "OPTA", "LETA")
NAMES = {"NOM": "NOM (no mention)", "OPTA": "OPTA (options after)", "OPTB": "OPTB (options before)",
         "MENA": "MENA (sentence after)", "MENB": "MENB (sentence before)", "LETA": "LETA (letters after)"}
G2_MIN, PF_MIN, A1B_MIN, A8_MIN, HYB_MIN = 80, 30, 40, 40, 40
G1_LP, G1_GEN, G3_P, G3_G1, G4_NATS = 0.3, 0.97, 0.5, 0.95, 2.0
KV_ROWS = ("ID", "K_S", "V_S", "K_X", "V_X")


class Res:
    """A line's result in one model: ok True / False / None (not evaluable, with ``why``), the text and components."""

    def __init__(self, ok, txt, comps=(), why=""):
        self.ok, self.txt, self.comps, self.why = ok, txt, list(comps), why


def judge(txt, comps, ev, why):
    if not ev:
        return Res(None, txt, comps, why)
    return Res(all(c.passed for c in comps), txt, comps)


class Model:
    def __init__(self, key, J, X=None, test=False, a_g0=True):
        self.key, self.J, self.X, self.test, self.a_g0 = key, J, X, test, a_g0
        self.P = J.get("provenance", {})
        self.recs = {f: {r["id"]: r for r in F["items"]} for f, F in J.get("formats", {}).items()}
        self.rowset = {f: set(F.get("rows", [])) for f, F in J.get("formats", {}).items()}
        self.cb = J.get("closed_book", {})
        self.art = {i: r["title"] for F in self.recs.values() for i, r in F.items()}
        self.sub = {i: r["sub"] for F in self.recs.values() for i, r in F.items()}
        self.gates = {f: self.gate(f) for f in FORMATS if self.has(f)}

    # ---- populations
    def has(self, f):
        return bool(self.recs.get(f))

    def who(self, f, i, row):
        return self.recs[f][i]["gen"][row]["who"]

    def comp(self, f, i):
        g = self.recs[f][i]["gen"]
        return g["ID"]["who"] == "B" and g["KV_S"]["who"] == "S" and g["KV_X"]["who"] == "X"

    def C(self, *F):
        if not all(self.has(f) for f in F):
            return []
        ids = set.intersection(*[set(self.recs[f]) for f in F])
        return sorted(i for i in ids if all(self.comp(f, i) for f in F))

    def pf(self, i, F):
        for f in F:
            cb = self.cb.get(i, {}).get("CBLET" if f == "LETA" else "CBOPT", {})
            if cb.get("win") in (None, "B"):
                return False
            if f in FULL and "KV_Z" in self.recs[f][i]["gen"] and self.who(f, i, "KV_Z") == "B":
                return False
        return True

    def PF(self, *F):
        return [i for i in self.C(*F) if self.pf(i, F)]

    def arts(self, ids):
        return [self.art[i] for i in ids]

    # ---- measures
    def lp(self, f, ids, row, Y):
        return np.array([self.recs[f][i]["rows"][row]["lp"][Y] for i in ids])

    def idkv(self, f, ids):
        L = lambda r, Y: self.lp(f, ids, r, Y)  # noqa: E731
        k = 0.5 * ((L("K_S", "S") - L("K_X", "S")) + (L("K_X", "X") - L("K_S", "X")))
        v = 0.5 * ((L("V_S", "S") - L("V_X", "S")) + (L("V_X", "X") - L("V_S", "X")))
        return k, v

    def rate(self, f, ids, row, Y):
        return np.array([float(self.who(f, i, row) == Y) for i in ids])

    def g1ok(self, f, ids, row, Y):
        return np.array([float(self.recs[f][i]["gen"][row]["g1"] == self.recs[f][i]["dec"][Y]) for i in ids])

    # ---- gates
    def gate(self, f):
        R = list(self.recs[f].values())
        eS = float(np.mean([max(abs(r["rows"]["KV_S"]["lp"][Y] - r["runs"]["S"][Y]) for Y in r["dec"]) for r in R]))
        eB = float(np.mean([max(abs(r["rows"]["ID"]["lp"][Y] - r["runs"]["B"][Y]) for Y in r["dec"]) for r in R]))
        gS = float(np.mean([r["gen"]["KV_S"]["ids"] == r["gen"]["S_run"]["ids"] for r in R]))
        g1 = eS <= G1_LP and eB <= G1_LP and gS >= G1_GEN
        ids = self.C(f)
        n = len(ids)
        g2 = n >= G2_MIN or (self.test and n > 0)
        if n:
            pB = statistics.median(float(np.exp(self.recs[f][i]["rows"]["ID"]["lp"]["B"])) for i in ids)
            fr = {row: float(self.g1ok(f, ids, row, Y).mean()) for row, Y in (("ID", "B"), ("KV_S", "S"), ("KV_X", "X"))}
        else:
            pB, fr = NAN, {"ID": NAN, "KV_S": NAN, "KV_X": NAN}
        g3 = pB >= G3_P and all(v >= G3_G1 for v in fr.values())
        txt1 = (f"mean max|lp(KV_S) - lp(S run)| {eS:.3f}, mean max|lp(ID) - lp(B run)| {eB:.3f} (<= {G1_LP}); "
                f"g(KV_S) = g(S run) in {gS:.3f} (>= {G1_GEN}) of {len(R)} items")
        txt2 = f"n_comp {n} (>= {G2_MIN}" + (", waived under TEST" if self.test else "") + ")"
        txt3 = (f"median p(dec_B | ID) {pB:.3f} (>= {G3_P}); g1 = dec in ID {fr['ID']:.3f}, KV_S {fr['KV_S']:.3f}, "
                f"KV_X {fr['KV_X']:.3f} (>= {G3_G1})")
        return {"G1": g1, "G2": g2, "G3": g3, "txt": {"G1": txt1, "G2": txt2, "G3": txt3}, "n_comp": n}

    def ev(self, formats, rows=(), gen=()):
        """(evaluable, reason): J-A-G0, and J-A-G1 to G3 in every format; the rows (scored or generated) present."""
        miss = [] if self.a_g0 else ["J-A-G0"]
        for f in formats:
            if not self.has(f):
                red = self.P.get("reduced", {}).get(f)
                miss.append(f"{f} not run" + (f" ({red})" if red else ""))
                continue
            gone = [r for r in rows if r not in self.rowset[f]] + [r for r in gen if r not in next(iter(self.recs[f].values()))["gen"]]
            if gone:
                miss.append(f"{f} rows {gone} not run" + (" (reduced at the deadline)" if self.J["formats"][f].get("reduced") else ""))
            miss += [f"J-A-{g} {f}" for g in ("G1", "G2", "G3") if not self.gates[f][g]]
        return not miss, ", ".join(miss)

    def g4(self, k, v):
        return bool(len(k)) and float(k.mean() + v.mean()) >= G4_NATS

    def floor(self, n, need):
        """A size floor; under TEST it is waived, but an empty population is never evaluable."""
        return n >= need or (self.test and n > 0)


# --------------------------------------------------------------------------- lines
def _sid_line(m, f, ids, lo_thr, hi_floor=None, pt_min=None, pt_max=None, idk_pos=False, idv_pos=False, extra_why=()):
    k, v = m.idkv(f, ids)
    arts = m.arts(ids)
    s, ik, iv = est(arts, sid, k, v), est(arts, mean, k), est(arts, mean, v)
    comps = []
    if pt_min is not None:
        comps += [point(f"s_ID >= {pt_min}", s.pt >= pt_min), lower(s, lo_thr, "s_ID")]
    if pt_max is not None:
        comps += [point(f"s_ID <= {pt_max}", s.pt <= pt_max), upper(s, hi_floor, "s_ID")]
    if idk_pos:
        comps.append(lower(ik, 0, "ID_K"))
    if idv_pos:
        comps.append(lower(iv, 0, "ID_V"))
    g4 = m.g4(k, v)
    txt = (f"n={len(ids)} ({len(set(arts))} articles): s_ID {s}; ID_K {ik}; ID_V {iv}; J-A-G4 mean ID_K + ID_V "
           f"{(k.mean() + v.mean()) if len(k) else NAN:.2f} >= {G4_NATS}: {'yes' if g4 else 'no'}")
    return comps, txt, g4, s


def a1(m):
    ev, why = m.ev(("OPTA",), KV_ROWS)
    ids = m.C("OPTA")
    comps, txt, g4, _ = _sid_line(m, "OPTA", ids, 0.35, pt_min=0.5, idk_pos=True)
    return judge(txt, comps, ev and g4, why or "J-A-G4")


def a1b(m):
    ev, why = m.ev(("OPTA",), KV_ROWS)
    ids = [i for i in m.C("OPTA") if m.recs["OPTA"][i]["nP"] >= 3]
    comps, txt, g4, _ = _sid_line(m, "OPTA", ids, 0.25, pt_min=0.4)
    fl = m.floor(len(ids), A1B_MIN)
    return judge(f"|P| >= 3: {txt}", comps, ev and g4 and fl, why or ("J-A-G4" if not g4 else f"fewer than {A1B_MIN} items with |P| >= 3"))


def a1c(m):
    ev, why = m.ev(("LETA",), KV_ROWS)
    ids = m.C("LETA")
    comps, txt, g4, _ = _sid_line(m, "LETA", ids, 0.35, pt_min=0.5) if ev else ([], "LETA K/V rows not run", False, None)
    return judge(txt, comps, ev and g4, why or "J-A-G4")


def a2(m):
    ev, why = m.ev(("NOM",), KV_ROWS)
    ids = m.C("NOM")
    comps, txt, g4, _ = _sid_line(m, "NOM", ids, None, hi_floor=0.3, pt_max=0.2, idv_pos=True)
    return judge(txt, comps, ev and g4, why or "J-A-G4")


def a3(m):
    ev, why = m.ev(("OPTA", "NOM"), KV_ROWS)
    ids = m.C("OPTA", "NOM")
    kO, vO = m.idkv("OPTA", ids)
    kN, vN = m.idkv("NOM", ids)
    d = est(m.arts(ids), lambda a, b, c, e: sid(a, b) - sid(c, e), kO, vO, kN, vN)
    comps = [point("s_ID(OPTA) - s_ID(NOM) >= 0.4", d.pt >= 0.4), lower(d, 0, "s_ID(OPTA) - s_ID(NOM)")]
    g4 = m.g4(kO, vO) and m.g4(kN, vN)
    return judge(f"n={len(ids)}: s_ID(OPTA) - s_ID(NOM) {d}", comps, ev and g4, why or "J-A-G4")


def a4(m):
    ev, why = m.ev(("OPTA", "OPTB"), KV_ROWS)
    ids = m.C("OPTA", "OPTB")
    kA, _ = m.idkv("OPTA", ids)
    kB, _ = m.idkv("OPTB", ids)
    arts = m.arts(ids)
    r = est(arts, ratio, kB, kA)
    dd = est(arts, mean, kA - kB)
    comps = [point("ID_K(OPTB) / ID_K(OPTA) <= 0.15", r.pt <= 0.15), upper(r, 0.25, "ID_K(OPTB)/ID_K(OPTA)"),
             lower(dd, 0, "ID_K(OPTA) - ID_K(OPTB)")]
    return judge(f"n={len(ids)}: ratio {r}; ID_K(OPTA) - ID_K(OPTB) {dd}; ID_K(OPTB) {est(arts, mean, kB)}", comps, ev, why)


def a5(m):
    ev, why = m.ev(("MENA", "NOM"), KV_ROWS)
    ids = m.C("MENA", "NOM")
    kM, vM = m.idkv("MENA", ids)
    kN, vN = m.idkv("NOM", ids)
    arts = m.arts(ids)
    d = est(arts, lambda a, b, c, e: sid(a, b) - sid(c, e), kM, vM, kN, vN)
    ik = est(arts, mean, kM)
    comps = [point("s_ID(MENA) - s_ID(NOM) >= 0.10", d.pt >= 0.10), lower(d, 0, "s_ID(MENA) - s_ID(NOM)"), lower(ik, 0, "ID_K(MENA)")]
    g4 = m.g4(kM, vM) and m.g4(kN, vN)
    return judge(f"n={len(ids)}: s_ID(MENA) - s_ID(NOM) {d}; ID_K(MENA) {ik}; s_ID(MENA) {est(arts, sid, kM, vM)}",
                 comps, ev and g4, why or "J-A-G4")


def a5b(m):
    ev, why = m.ev(("MENA", "MENB"), KV_ROWS)
    ids = m.C("MENA", "MENB")
    kM, _ = m.idkv("MENA", ids) if ev else (np.zeros(0), None)
    kB, _ = m.idkv("MENB", ids) if ev else (np.zeros(0), None)
    if not ev:
        return Res(None, "MENA / MENB not both run", [], why)
    dd = est(m.arts(ids), mean, kM - kB)
    return judge(f"n={len(ids)}: ID_K(MENA) - ID_K(MENB) {dd}; ID_K(MENB) {est(m.arts(ids), mean, kB)}",
                 [lower(dd, 0, "ID_K(MENA) - ID_K(MENB)")], ev, why)


def cue_rates(m, f, ids):
    key = 0.5 * (m.rate(f, ids, "KS_VX", "S") + m.rate(f, ids, "KX_VS", "X"))
    val = 0.5 * (m.rate(f, ids, "KS_VX", "X") + m.rate(f, ids, "KX_VS", "S"))
    return key, val


def a6_cue(m, f):
    ev, why = m.ev((f,), gen=("KS_VX", "KX_VS"))
    ids = m.C(f)
    if not ev and not ids:
        return Res(None, f"{f}: no competent items", [], why)
    key, val = cue_rates(m, f, ids)
    arts = m.arts(ids)
    K, Vv = est(arts, mean, key), est(arts, mean, val)
    if f == "NOM":
        d = est(arts, mean, val - key)
        comps = [point("value-source rate >= 0.6", Vv.pt >= 0.6), lower(Vv, 0.5, "value-source rate"),
                 point("value - key >= 0.3", d.pt >= 0.3), lower(d, 0, "value - key")]
    else:
        d = est(arts, mean, key - val)
        comps = [point("key-source rate >= 0.6", K.pt >= 0.6), lower(K, 0.5, "key-source rate"),
                 point("key - value >= 0.3", d.pt >= 0.3), lower(d, 0, "key - value")]
    oth = {Y: float(np.mean([m.who(f, i, r) == Y for i in ids for r in ("KS_VX", "KX_VS")])) if ids else NAN for Y in ("B", "Z", "D", "other")}
    return judge(f"{f} n={len(ids)}: key-source {K}; value-source {Vv}; difference {d}; other answers "
                 + ", ".join(f"{Y} {v:.3f}" for Y, v in oth.items()), comps, ev, why)


def a6d(m):
    evO, wO = m.ev(("OPTA",), gen=("KS_VZ",))
    evN, wN = m.ev(("NOM",), gen=("KS_VZ",))
    iO, iN = m.C("OPTA"), m.C("NOM")
    sO = est(m.arts(iO), mean, m.rate("OPTA", iO, "KS_VZ", "S")) if iO else None
    sN = est(m.arts(iN), mean, m.rate("NOM", iN, "KS_VZ", "S")) if iN else None
    if sO is None or sN is None:
        return Res(None, "no competent items", [], ", ".join(x for x in (wO, wN) if x) or "no competent items")
    comps = [point("OPTA S-rate >= 0.6", sO.pt >= 0.6), lower(sO, 0.5, "OPTA S-rate"),
             point("NOM S-rate <= 0.2", sN.pt <= 0.2), upper(sN, 0.3, "NOM S-rate")]
    zO = float(m.rate("OPTA", iO, "KS_VZ", "B").mean())
    return judge(f"OPTA n={len(iO)} S-rate {sO} (B-rate {zO:.3f}); NOM n={len(iN)} S-rate {sN} "
                 f"(Z-rate {float(m.rate('NOM', iN, 'KS_VZ', 'Z').mean()):.3f})", comps, evO and evN, ", ".join(x for x in (wO, wN) if x))


def a6e(m):
    ev, why = m.ev(("OPTA",), gen=("KZ_VS",))
    ids = m.C("OPTA")
    if not ids:
        return Res(None, "no competent items", [], why or "no competent items")
    s = est(m.arts(ids), mean, m.rate("OPTA", ids, "KZ_VS", "S"))
    comps = [point("S-rate >= 0.5", s.pt >= 0.5), lower(s, 0.4, "S-rate")]
    oth = {Y: float(m.rate("OPTA", ids, "KZ_VS", Y).mean()) for Y in ("B", "X", "D", "Z", "other")}
    return judge(f"n={len(ids)}: S-rate {s}; " + ", ".join(f"{Y} {v:.3f}" for Y, v in oth.items()), comps, ev, why)


def quant_acc(m, f, ids, row, Y="S"):
    return np.array([float(m.recs[f][i]["quant"]["rows"][row]["who"] == Y) for i in ids])


def a7_parts(m):
    ev, why = m.ev(("NOM", "OPTA"))
    ids = m.PF("NOM", "OPTA")
    ids = [i for i in ids if "quant" in m.recs["NOM"][i] and "quant" in m.recs["OPTA"][i]]
    fl = m.floor(len(ids), PF_MIN)
    a = {(f, r): quant_acc(m, f, ids, r) for f in ("NOM", "OPTA") for r in ("none", "K2", "V2")}
    return ev and fl, (why or (f"prior-free n {len(ids)} < {PF_MIN}" if not fl else "")), ids, a


def a7(m):
    ev, why, ids, a = a7_parts(m)
    did = (a["NOM", "K2"] - a["NOM", "V2"]) - (a["OPTA", "K2"] - a["OPTA", "V2"])
    e = est(m.arts(ids), mean, did)
    comps = [point("DiD >= 0.10", e.pt >= 0.10), lower(e, 0, "DiD")]
    dr = {(f, r): float((a[f, "none"] - a[f, r]).mean()) if ids else NAN for f in ("NOM", "OPTA") for r in ("K2", "V2")}
    return judge(f"prior-free n={len(ids)}: DiD [drop_NOM(V) - drop_OPTA(V)] - [drop_NOM(K) - drop_OPTA(K)] {e}; drops "
                 + ", ".join(f"{f}({r[0]}) {v:+.3f}" for (f, r), v in dr.items()), comps, ev, why)


def a7b(m):
    ev, why, ids, a = a7_parts(m)
    d = (a["NOM", "none"] - a["NOM", "V2"]) - (a["OPTA", "none"] - a["OPTA", "V2"])
    e = est(m.arts(ids), mean, d)
    return judge(f"prior-free n={len(ids)}: drop_NOM(V) - drop_OPTA(V) {e}", [point(">= 0.10", e.pt >= 0.10), lower(e, 0, "drop_NOM(V) - drop_OPTA(V)")], ev, why)


def a8_pop(m):
    ids = m.C("NOM", "OPTA")
    return [i for i in ids if m.recs["NOM"][i]["sub"] in ("PERSON", "PLACE") and m.recs["NOM"][i].get("stratum", "FT") == "FT"
            and any(m.recs["NOM"][i]["diff_S"])]


def cont_sum(m, f, ids, row):
    out = []
    for i in ids:
        r = m.recs[f][i]
        out.append(sum(x for x, dd in zip(r["rows"][row]["cont"], r["diff_S"]) if dd))
    return np.array(out)


def inter(k, v, kv):
    return (kv - k - v) / kv


def a8_measures(m, ids):
    c = {(f, C): cont_sum(m, f, ids, f"{C}_S") - cont_sum(m, f, ids, "ID") for f in ("NOM", "OPTA") for C in ("K", "V", "KV")}
    dd = {C: m.lp("NOM", ids, f"{C}_S", "S") - m.lp("NOM", ids, "ID", "S") for C in ("K", "V", "KV")}
    return c, dd


def a8(m):
    ev, why = m.ev(("NOM", "OPTA"), KV_ROWS + ("KV_S",))
    ids = a8_pop(m)
    arts = m.arts(ids)
    c, dd = a8_measures(m, ids)
    Ic = est(arts, inter, c["NOM", "K"], c["NOM", "V"], c["NOM", "KV"])
    Id = est(arts, inter, dd["K"], dd["V"], dd["KV"])
    r = est(arts, ratio, c["OPTA", "KV"], c["NOM", "KV"])
    dkv = float(c["NOM", "KV"].mean()) if ids else NAN
    comps = [point("I_cont >= 0.5", Ic.pt >= 0.5), lower(Ic, 0.3, "I_cont(NOM)"),
             point("I_dec <= 0.25", Id.pt <= 0.25), upper(Id, 0.35, "I_dec(NOM)"),
             point("|d_KV^cont(OPTA)/d_KV^cont(NOM)| <= 0.2", abs(r.pt) <= 0.2), inside(r, -0.3, 0.3, "the ratio")]
    fl, base = m.floor(len(ids), A8_MIN), dkv >= 2.0
    txt = (f"n={len(ids)} (leak-free PERSON/PLACE with a continuation token differing from c_B): I_cont(NOM) {Ic}; "
           f"I_dec(NOM) {Id}; d_KV^cont OPTA/NOM {r}; d^cont(NOM) K {c['NOM', 'K'].mean() if ids else NAN:+.2f} "
           f"V {c['NOM', 'V'].mean() if ids else NAN:+.2f} KV {dkv:+.2f} (>= 2 nats: {'yes' if base else 'no'})")
    return judge(txt, comps, ev and fl and base, why or (f"n < {A8_MIN}" if not fl else "d_KV^cont(NOM) < 2 nats"))


def a8d(m):
    ev, why = m.ev(("NOM", "OPTA"), gen=("K_Z",))
    ids = [i for i in m.C("NOM", "OPTA") if m.recs["NOM"][i]["sub"] in ("PERSON", "PLACE")
           and len(m.recs["NOM"][i]["c"]["B"]) - m.recs["NOM"][i]["j"] >= 2]
    arts = m.arts(ids)
    h, ncond = {}, {}
    for f in ("NOM", "OPTA"):
        cond = m.g1ok(f, ids, "K_Z", "B")
        bad = cond * (1 - m.rate(f, ids, "K_Z", "B"))
        h[f] = est(arts, ratio, bad, cond)
        ncond[f] = int(cond.sum())
    comps = [point("h(K_Z, NOM) >= 0.2", h["NOM"].pt >= 0.2), lower(h["NOM"], 0.1, "h(K_Z, NOM)"),
             point("h(K_Z, OPTA) <= 0.05", h["OPTA"].pt <= 0.05), upper(h["OPTA"], 0.1, "h(K_Z, OPTA)")]
    fl = all(m.floor(n, HYB_MIN) for n in ncond.values())
    return judge(f"n={len(ids)}; items with g1(K_Z) = dec_B: NOM {ncond['NOM']}, OPTA {ncond['OPTA']} (>= {HYB_MIN}); "
                 f"h(K_Z, NOM) {h['NOM']}; h(K_Z, OPTA) {h['OPTA']}", comps, ev and fl, why or f"fewer than {HYB_MIN} items with g1 = dec_B")


LINE_FNS = {"J-A1": a1, "J-A1b": a1b, "J-A1c": a1c, "J-A2": a2, "J-A3": a3, "J-A4": a4, "J-A5": a5, "J-A5B": a5b,
            "J-A6a": lambda m: a6_cue(m, "OPTA"), "J-A6b": lambda m: a6_cue(m, "LETA"), "J-A6c": lambda m: a6_cue(m, "NOM"),
            "J-A6d": a6d, "J-A6e": a6e, "J-A7": a7, "J-A7b": a7b, "J-A8": a8, "J-A8d": a8d}


# --------------------------------------------------------------------------- reported
def reported(m, out, s_opta=None):
    """Lines without a verdict: the graded A1 reading, the prior controls, accuracy outcomes on all competent and on
    prior-free items, the decision-token versions of the cue-conflict rates, the old A6 flips and A7 accuracies, s_ID on
    all valid items, the KIVI reconstruction errors and the sign of d_K^cont(NOM)."""
    k = m.key
    out(f"  [{k}] n valid {len(next(iter(m.recs.values()), {}))}; n_comp " + ", ".join(f"{f} {m.gates[f]['n_comp']}" for f in m.gates)
        + f"; frame {m.J.get('frame')!r}")
    ids_all = sorted(m.cb)
    for fmt in ("CBOPT", "CBLET"):
        wins = [m.cb[i].get(fmt, {}).get("win") for i in ids_all]
        v = [w for w in wins if w is not None]
        by = {}
        for sub in ("PERSON", "PLACE", "NUMBER"):
            s = [m.cb[i][fmt]["win"] == "B" for i in ids_all if m.cb[i].get(fmt, {}).get("win") and m.sub.get(i) == sub]
            by[sub] = f"{np.mean(s):.3f} (n {len(s)})" if s else "-"
        out(f"    P_CB(B) {fmt}: {np.mean([w == 'B' for w in v]) if v else NAN:.3f} of {len(v)} (undefined {len(wins) - len(v)}); by type " + ", ".join(f"{a} {b}" for a, b in by.items()))
    if s_opta is not None and not np.isnan(s_opta.pt):
        gr = ("a substantial but not dominant key read on natural text" if 0.35 <= s_opta.pt < 0.5 and s_opta.lo > 0.2 else
              "the template-only reading applies (s_ID lower bound < 0.2)" if s_opta.lo < 0.2 else
              "J-A1's own threshold decides (s_ID >= 0.5)" if s_opta.pt >= 0.5 else "between the graded and the template-only readings")
        out(f"    A-12 graded reading of J-A1: s_ID(OPTA) {s_opta} -> {gr} (with J-A3 met)")
    for f in FULL:
        if not m.has(f):
            continue
        for lab, ids in (("competent", m.C(f)), ("prior-free", m.PF(f))):
            if not ids:
                out(f"    {f} {lab}: n=0")
                continue
            arts = m.arts(ids)
            acc = {r: est(arts, mean, m.rate(f, ids, r, "B")) for r in ("V_Z", "K_Z", "KV_Z") if r in m.recs[f][ids[0]]["gen"]}
            line = f"    {f} {lab} n={len(ids)}: accuracy (B) under " + "; ".join(f"{r} {e}" for r, e in acc.items())
            if "KS_VX" in m.recs[f][ids[0]]["gen"]:
                key, val = cue_rates(m, f, ids)
                line += f"; cue conflict key-source {est(arts, mean, key)} value-source {est(arts, mean, val)}"
            out(line)
        ids = m.C(f)
        if ids and "KS_VX" in m.recs[f][ids[0]]["gen"]:
            arts = m.arts(ids)
            gk = 0.5 * (m.g1ok(f, ids, "KS_VX", "S") + m.g1ok(f, ids, "KX_VS", "X"))
            gv = 0.5 * (m.g1ok(f, ids, "KS_VX", "X") + m.g1ok(f, ids, "KX_VS", "S"))
            out(f"    {f} cue conflict at the decision token (g1): key-source {est(arts, mean, gk)}, value-source {est(arts, mean, gv)}")
        if ids and "K_S" in m.recs[f][ids[0]]["gen"]:
            arts = m.arts(ids)
            out(f"    {f} old A6 flips: phi_dec(K_S) {est(arts, mean, m.g1ok(f, ids, 'K_S', 'S'))}, phi_dec(V_S) "
                f"{est(arts, mean, m.g1ok(f, ids, 'V_S', 'S'))}; full match K_S {est(arts, mean, m.rate(f, ids, 'K_S', 'S'))}, "
                f"V_S {est(arts, mean, m.rate(f, ids, 'V_S', 'S'))}")
    for f in FORMATS:
        if m.has(f) and {"K_S", "K_X", "V_S", "V_X"} <= m.rowset[f]:
            ids = sorted(m.recs[f])
            kk, vv = m.idkv(f, ids)
            mass = float(np.mean([m.recs[f][i]["rows"]["ID"]["mass"] for i in ids]))
            out(f"    {f} all valid items n={len(ids)}: s_ID {est(m.arts(ids), sid, kk, vv)}; option mass (ID) {mass:.3f}")
    for f in FULL:
        ids = [i for i in m.C(f) if "quant" in m.recs[f][i]]
        if ids:
            rk = np.mean([np.mean(m.recs[f][i]["quant"]["relerr"]["K2"]["k"]) for i in ids])
            rv = np.mean([np.mean(m.recs[f][i]["quant"]["relerr"]["V2"]["v"]) for i in ids])
            acc = {r: float(quant_acc(m, f, ids, r).mean()) for r in ("none", "K2", "V2")}
            out(f"    KIVI {f} competent n={len(ids)}: relative error K {rk:.3f}, V {rv:.3f}; faithful (S) accuracy "
                + ", ".join(f"{r} {v:.3f}" for r, v in acc.items()))
    ids = a8_pop(m)
    if ids:
        c, _ = a8_measures(m, ids)
        out(f"    d_K^cont(NOM) on the J-A8 population {est(m.arts(ids), mean, c['NOM', 'K'])} (sign reported; Part D owns the sign account)")


# --------------------------------------------------------------------------- exploratory
def exploratory(m, out):
    k = m.key
    X = (m.X or {}).get("parts", {})
    sk = (m.X or {}).get("provenance", {}).get("skipped_parts", [])
    out(f"  [{k}] explore parts: {sorted(X)}" + (f"; skipped at the deadline: {sk}" if sk else "") + ("" if m.X else " (no explore file)"))
    # E3 moderators
    for f in ("OPTA", "NOM"):
        ids = m.C(f)
        if not ids or not {"K_S", "K_X"} <= m.rowset.get(f, set()):
            continue
        grp = {}
        for i in ids:
            r = m.recs[f][i]
            grp.setdefault(("type", r["sub"]), []).append(i)
            grp.setdefault(("D_in", r["D_in"]), []).append(i)
            grp.setdefault(("|P|", min(r["nP"], 3)), []).append(i)
        line = []
        for g, gi in sorted(grp.items(), key=lambda x: str(x[0])):
            kk, vv = m.idkv(f, gi)
            line.append(f"{g[0]}={g[1]}{'+' if g[0] == '|P|' and g[1] == 3 else ''}: {float(kk.mean() / (kk.mean() + vv.mean())):+.3f} (n {len(gi)})")
        out(f"    E3 {f} s_ID by " + "; ".join(line))
    # E1 / E2
    for f, ch in (("OPTA", "K"), ("NOM", "V")):
        recs = X.get("onset_piece", {}).get(f, [])
        if not recs:
            continue
        mm = lambda r, row: r["rows"][row]["lp"]["S"] - r["rows"][row]["lp"]["B"]  # noqa: E731
        base = np.array([mm(r, f"{ch}_S") - mm(r, "ID") for r in recs])
        parts = [n for n in recs[0]["rows"] if n.startswith(f"{ch}_S@") or n.startswith(f"{ch}_S^")]
        out(f"    E1/E2 {f} ({len(recs)} items, all valid): d_{ch} = m({ch}_S) - m(ID) {base.mean():+.2f} nats; fractions "
            + ", ".join(f"{n} {np.mean([mm(r, n) - mm(r, 'ID') for r in recs]) / base.mean():+.3f}" for n in parts))
    # E4 YEAR
    for f in ("NOM", "OPTA"):
        recs = X.get("year", {}).get(f, [])
        if not recs:
            continue
        vals = {C: [] for C in ("K", "V", "KV")}
        for r in recs:
            cS, cB, j = r["c"]["S"], r["c"]["B"], r["j"]
            t = next((t for t in range(j, len(cS)) if t >= len(cB) or cS[t] != cB[t]), None)
            if t is None:
                continue
            val = (lambda row: r["rows"][row]["lp"]["S"]) if t == j else (lambda row, t=t: r["rows"][row]["cont"][t - j - 1])
            for C in vals:
                vals[C].append(val(f"{C}_S") - val("ID"))
        if vals["KV"]:
            K, Vv, KV = (float(np.mean(vals[C])) for C in ("K", "V", "KV"))
            out(f"    E4 YEAR {f} n={len(vals['KV'])}: at the first token where c_S differs from c_B: d_K {K:+.2f}, d_V {Vv:+.2f}, "
                f"d_KV {KV:+.2f}, I_dec {(KV - K - Vv) / KV:+.3f} (written expectation: about 0, value-only)")
    # LEAK stratum
    for f in ("NOM", "OPTA"):
        recs = X.get("leak", {}).get(f, [])
        comp = [r for r in recs if r.get("gen") and r["gen"]["ID"]["who"] == "B" and r["gen"]["KV_S"]["who"] == "S" and r["gen"]["KV_X"]["who"] == "X"]
        if comp:
            L = lambda r, row, Y: r["rows"][row]["lp"][Y]  # noqa: E731
            kk = np.array([0.5 * ((L(r, "K_S", "S") - L(r, "K_X", "S")) + (L(r, "K_X", "X") - L(r, "K_S", "X"))) for r in comp])
            vv = np.array([0.5 * ((L(r, "V_S", "S") - L(r, "V_X", "S")) + (L(r, "V_X", "X") - L(r, "V_S", "X"))) for r in comp])
            out(f"    LEAK {f} competent n={len(comp)} of {len(recs)}: ID_K {kk.mean():+.2f}, ID_V {vv.mean():+.2f}, s_ID {kk.mean() / (kk.mean() + vv.mean()):+.3f}")
    # E5 quantization extras
    for f, recs in X.get("quant_extra", {}).items():
        if not recs:
            continue
        acc = lambda key, row, Y: np.mean([r[key]["rows"][row]["who"] == Y for r in recs if key in r])  # noqa: E731
        s3 = f"S-base 3-bit accuracy K3 {acc('S3', 'K3', 'S'):.3f}, V3 {acc('S3', 'V3', 'S'):.3f}; " if "S3" in recs[0] else ""
        out(f"    E5 {f} n={len(recs)} (all valid): {s3}unmodified B-base accuracy none {acc('B2', 'none', 'B'):.3f}, "
            f"K2 {acc('B2', 'K2', 'B'):.3f}, V2 {acc('B2', 'V2', 'B'):.3f}")
    # E6 full-string s_ID
    for f in ("NOM", "OPTA"):
        xr = {r["id"]: r for r in X.get("x_target", {}).get(f, [])}
        ids = [i for i in m.C(f) if i in xr]
        if not ids:
            continue
        full = lambda rows, row, Y: rows[row]["lp"][Y] + sum(rows[row]["cont"])  # noqa: E731
        kk, vv = [], []
        for i in ids:
            S_, X_ = m.recs[f][i]["rows"], xr[i]["rows"]
            dS = lambda row: full(S_, row, "S") - full(S_, "ID", "S")  # noqa: E731
            dX = lambda row: full(X_, row, "X") - full(X_, "ID", "X")  # noqa: E731
            kk.append(0.5 * ((dS("K_S") - dS("K_X")) + (dX("K_X") - dX("K_S"))))
            vv.append(0.5 * ((dS("V_S") - dS("V_X")) + (dX("V_X") - dX("V_S"))))
        kk, vv = np.array(kk), np.array(vv)
        out(f"    E6 {f} full string (decision token and later) n={len(ids)}: ID_K {kk.mean():+.2f}, ID_V {vv.mean():+.2f}, s_ID^full {kk.mean() / (kk.mean() + vv.mean()):+.3f}")
