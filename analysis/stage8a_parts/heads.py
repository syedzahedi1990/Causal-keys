"""Part A of P-2026-10-10-J, the head lines on natural text (heads/<tag>.json of experiments/natural_heads.py): gates
J-A-HA-G1 and J-A-HA-G2 (J-A-HA-G0 is the pytest gate), lines J-A-HA1 to J-A-HA3b, the reported and exploratory lines.

  d_full = mean[m(full K_S clamp) - m(clean)] (two single passes, eval's mF and mB), d_G = mean[m(all_G) - m(none)] of
  N*'s sufficiency batch (m = lp(dec_S) - lp(dec_B) at the decision position); for each set (N*, T*, each random set),
  within that set's own sufficiency and knockout batches: R(k) = mean[m(top-k) - m(none)] / mean[m(all_G) - m(none)]
  (for N* the denominator is d_G); KO(k) = 1 - mean[m(all but top-k) - m(none_KO)] / mean[m(all_G, KO) - m(none_KO)]
  (stage 6); random-set values are means over the three draws, each ratio recomputed in every resample. HA3: acc(c) =
  the share of items whose argmax chain over c_S holds under condition c (KV_S clamp, condition's heads mean-ablated at
  Q+); drop(c) = acc(none) - acc(c), paired by item.
"""
import math

import numpy as np

from .common import NAN, est, inside, lower, mean, point, ratio

HEAD_KEYS = ("qwen7", "mistral7")
PF_MIN = 30


def one_minus(a, b):
    return 1 - a / b


class Heads:
    def __init__(self, key, Hj, model=None, test=False, ha_g0=True):
        self.key, self.H, self.m, self.test, self.ha_g0 = key, Hj, model, test, ha_g0
        self.P = Hj.get("provenance", {})
        self.E = Hj.get("eval", [])
        self.sets = Hj.get("sets", {})
        self.KS = self.P.get("KS", [])
        self.kstar = self.P.get("kstar")
        self.ik = self.KS.index(self.kstar) if self.kstar in self.KS else None
        self.arts = [e["title"] for e in self.E]
        self.abl = {f: {r["id"]: r for r in Hj.get("ablate", {}).get(f, [])} for f in ("OPTA", "NOM")}

    def v(self, fn):
        return np.array([fn(e) for e in self.E], float)

    def R(self, s, i):
        return est(self.arts, ratio, self.v(lambda e: e["curves"][s]["suff"][i] - e["curves"][s]["none"]),
                   self.v(lambda e: e["curves"][s]["allG"] - e["curves"][s]["none"]))

    def KO(self, s, i):
        return est(self.arts, one_minus, self.v(lambda e: e["curves"][s]["ko"][i] - e["curves"][s]["ko_none"]),
                   self.v(lambda e: e["curves"][s]["ko_allG"] - e["curves"][s]["ko_none"]))

    def rand_mean(self, i, ko=False):
        xs = []
        for s in ("rand0", "rand1", "rand2"):
            if ko:
                xs += [self.v(lambda e, s=s: e["curves"][s]["ko"][i] - e["curves"][s]["ko_none"]),
                       self.v(lambda e, s=s: e["curves"][s]["ko_allG"] - e["curves"][s]["ko_none"])]
            else:
                xs += [self.v(lambda e, s=s: e["curves"][s]["suff"][i] - e["curves"][s]["none"]),
                       self.v(lambda e, s=s: e["curves"][s]["allG"] - e["curves"][s]["none"])]
        f = (lambda *a: np.mean([1 - a[2 * j] / a[2 * j + 1] for j in range(3)], 0)) if ko else \
            (lambda *a: np.mean([a[2 * j] / a[2 * j + 1] for j in range(3)], 0))
        return est(self.arts, f, *xs)

    def gate1(self):
        d = self.v(lambda e: e["mF"] - e["mB"]).mean() if self.E else NAN
        a = float(np.mean([abs(e["curves"]["N"]["none"] - e["mB"]) for e in self.E])) if self.E else NAN
        b = float(np.mean([abs(e["curves"]["N"]["allT"] - e["mF"]) for e in self.E])) if self.E else NAN
        tol = max(0.5, 0.02 * d) if not math.isnan(d) else NAN
        ok = a <= tol and b <= tol
        return ok, f"mean |m(none) - m(clean)| {a:.3f}, mean |m(all_T) - m(full clamp)| {b:.3f} (<= max(0.5, 0.02 x d_full) = {tol:.3f})"

    def gate2(self):
        if not self.E:
            return False, "no evaluation items"
        dF = est(self.arts, mean, self.v(lambda e: e["mF"] - e["mB"]))
        g = est(self.arts, ratio, self.v(lambda e: e["curves"]["N"]["allG"] - e["curves"]["N"]["none"]), self.v(lambda e: e["mF"] - e["mB"]))
        ok = dF.pt >= 3 and g.pt >= 0.6
        return ok, f"d_full {dF} (>= 3 nats); d_G/d_full {g} (>= 0.6); n={len(self.E)}"

    def ev(self):
        miss = [] if self.ha_g0 else ["J-A-HA-G0"]
        g1, _ = self.gate1()
        g2, _ = self.gate2()
        miss += [] if g1 else ["J-A-HA-G1"]
        miss += [] if g2 else ["J-A-HA-G2"]
        if self.ik is None:
            miss.append("k* not in KS")
        return not miss, ", ".join(miss)


def judge(txt, comps, ev, why):
    from .factorial import Res
    if not ev:
        return Res(None, txt, comps, why)
    return Res(all(c.passed for c in comps), txt, comps)


def ha1(h):
    ev, why = h.ev()
    if h.ik is None or not h.E:
        return judge("no curves", [], False, why or "no curves")
    r, k = h.R("N", h.ik), h.KO("N", h.ik)
    rr, kr = h.rand_mean(h.ik), h.rand_mean(h.ik, ko=True)
    comps = [point("R_N(k*) >= 0.7", r.pt >= 0.7), lower(r, 0.6, "R_N(k*)"), point("KO_N(k*) >= 0.7", k.pt >= 0.7),
             lower(k, 0.6, "KO_N(k*)"), point("R_rand(k*) <= 0.15", rr.pt <= 0.15), point("KO_rand(k*) <= 0.15", kr.pt <= 0.15)]
    return judge(f"k*={h.kstar}, n={len(h.E)}: R_N {r}; KO_N {k}; R_rand {rr}; KO_rand {kr} (means over 3 draws)", comps, ev, why)


def ha2(h):
    ev, why = h.ev()
    if h.ik is None or not h.E:
        return judge("no curves", [], False, why or "no curves")
    r, k = h.R("T", h.ik), h.KO("T", h.ik)
    comps = [point("R_T(k*) >= 0.5", r.pt >= 0.5), lower(r, 0.35, "R_T(k*)"), point("KO_T(k*) >= 0.5", k.pt >= 0.5), lower(k, 0.35, "KO_T(k*)")]
    return judge(f"k*={h.kstar}, n={len(h.E)}: R_T {r}; KO_T {k}; |T* & N*| = {overlap(h)[0]} (hypergeometric P {overlap(h)[1]:.1e})", comps, ev, why)


def overlap(h):
    from scipy.stats import hypergeom
    N, T = {tuple(c) for c in h.sets.get("N", [])}, {tuple(c) for c in h.sets.get("T", [])}
    n_heads = h.P.get("n_heads", 1)
    ov = len(N & T)
    return ov, float(hypergeom.sf(ov - 1, n_heads, len(T), len(N))) if N and T else NAN


def abl_pop(h, f):
    """PF(f) of the same model's factorial, intersected with the ablated items."""
    if h.m is None:
        return []
    return [i for i in h.m.PF(f) if i in h.abl[f]]


def acc(h, f, ids, cond, key="chain"):
    return np.array([float(h.abl[f][i]["conds"][cond][key]) for i in ids])


def ha3a(h):
    ev, why = h.ev()
    if h.m is None:
        return judge("no factorial results for this model (competence and prior-free population)", [], False, "no factorial")
    ev2, why2 = h.m.ev(("OPTA",))
    ids = abl_pop(h, "OPTA")
    fl = len(ids) >= PF_MIN or (h.test and len(ids) > 0)
    arts = [h.m.art[i] for i in ids]
    a0 = acc(h, "OPTA", ids, "none")
    dN = est(arts, mean, a0 - acc(h, "OPTA", ids, "N"))
    dr = {c: float((a0 - acc(h, "OPTA", ids, c)).mean()) if ids else NAN for c in ("rand0", "rand1", "rand2")}
    comps = [point("drop(N*) >= 0.3", dN.pt >= 0.3), lower(dN, 0, "drop(N*)")] + [point(f"drop({c}) <= 0.1", v <= 0.1) for c, v in dr.items()]
    txt = (f"prior-free competent OPTA n={len(ids)}: acc(none) {a0.mean() if ids else NAN:.3f}; drop(N*) {dN}; drops of the random sets "
           + ", ".join(f"{c} {v:+.3f}" for c, v in dr.items()) + f"; drop(T*) {float((a0 - acc(h, 'OPTA', ids, 'T')).mean()) if ids else NAN:+.3f}")
    return judge(txt, comps, ev and ev2 and fl, ", ".join(x for x in (why, why2, "" if fl else f"prior-free n < {PF_MIN}") if x))


def ha3b(h):
    ev, why = h.ev()
    if h.m is None:
        return judge("no factorial results for this model", [], False, "no factorial")
    ev2, why2 = h.m.ev(("NOM",))
    ids = abl_pop(h, "NOM")
    fl = len(ids) >= PF_MIN or (h.test and len(ids) > 0)
    arts = [h.m.art[i] for i in ids]
    dd = est(arts, mean, acc(h, "NOM", ids, "N", "dec") - acc(h, "NOM", ids, "none", "dec"))
    dc = est(arts, mean, acc(h, "NOM", ids, "N", "cont") - acc(h, "NOM", ids, "none", "cont"))
    comps = [point("|delta acc_dec| <= 0.05", abs(dd.pt) <= 0.05), inside(dd, -0.1, 0.1, "delta acc_dec"),
             point("|delta acc_cont| <= 0.05", abs(dc.pt) <= 0.05), inside(dc, -0.1, 0.1, "delta acc_cont")]
    return judge(f"prior-free competent NOM n={len(ids)}: decision accuracy change {dd}; continuation accuracy change {dc}",
                 comps, ev and ev2 and fl, ", ".join(x for x in (why, why2, "" if fl else f"prior-free n < {PF_MIN}") if x))


HEAD_FNS = {"J-A-HA1": ha1, "J-A-HA2": ha2, "J-A-HA3a": ha3a, "J-A-HA3b": ha3b}


def reported(h, out):
    out(f"  [{h.key}] k*={h.kstar}, KS={h.KS}, frame {h.P.get('frame')!r}; ranking items {h.P.get('n_rank')}, evaluation items "
        f"{h.P.get('n_eval')}, ablation items {h.P.get('n_ablate')}")
    for f in ("OPTA", "NOM"):
        ids = sorted(h.abl[f])
        if not ids:
            continue
        line = []
        for c in ("none", "N", "T", "rand0", "rand1", "rand2", "C"):
            a = np.mean([h.abl[f][i]["conds"][c]["chain"] for i in ids])
            b = np.mean([h.abl[f][i]["clean"][c]["chain"] for i in ids])
            mB = np.mean([h.abl[f][i]["clean"][c]["lp"]["B"] - max(h.abl[f][i]["clean"][c]["lp"][Y] for Y in ("S", "X", "D")) for i in ids])
            line.append(f"{c}: faithful(S) {a:.3f}, clean-B chain {b:.3f}, margin m_B {mB:+.2f}")
        out(f"    {f} every ablated item n={len(ids)} (option margin of the unablated prompt under each condition): " + "; ".join(line))


def exploratory(h, out):
    if h.ik is not None and h.E:
        for s in ("N", "T", "rand0"):
            out(f"    curves {s}: R " + " ".join(f"{k}:{h.R(s, i).pt:+.2f}" for i, k in enumerate(h.KS))
                + " | KO " + " ".join(f"{k}:{h.KO(s, i).pt:+.2f}" for i, k in enumerate(h.KS)))
        lay = np.mean([e["layer"]["m"] for e in h.E], 0) - np.mean([e["layer"]["none"] for e in h.E])
        top = np.argsort(-lay)[:5]
        out("    layer profile (m with one layer's heads seeing K_S in G, minus none): top layers " + ", ".join(f"{int(l)}:{lay[l]:+.2f}" for l in top))
        N = [tuple(c) for c in h.sets.get("N", [])]
        out(f"    N* layers: {sorted({c[0] for c in N})}; C* (DLA copy set) overlap with N*: {len(set(N) & {tuple(c) for c in h.sets.get('C', [])})}")
    for f in ("OPTA", "NOM"):
        ids = sorted(h.abl[f])
        if ids:
            a0 = np.mean([h.abl[f][i]["conds"]["none"]["chain"] for i in ids])
            aC = np.mean([h.abl[f][i]["conds"]["C"]["chain"] for i in ids])
            out(f"    C* double dissociation {f} (every ablated item, n={len(ids)}): faithful accuracy none {a0:.3f}, C* {aC:.3f}")
    ex = h.H.get("explore") or []
    if ex:
        def idkv(c):
            L = lambda r, row, Y: r["lp"][f"{row}|{c}"][Y]  # noqa: E731
            k = np.mean([0.5 * ((L(r, "K_S", "S") - L(r, "K_X", "S")) + (L(r, "K_X", "X") - L(r, "K_S", "X"))) for r in ex])
            v = np.mean([0.5 * ((L(r, "V_S", "S") - L(r, "V_X", "S")) + (L(r, "V_X", "X") - L(r, "V_S", "X"))) for r in ex])
            return k, v
        (k0, v0), (k1, v1) = idkv("none"), idkv("N")
        out(f"    ID_K / ID_V with N* ablated at Q+ (OPTA, n={len(ex)}): none {k0:+.2f} / {v0:+.2f}; N* {k1:+.2f} / {v1:+.2f} (rho_K {k1 / k0:+.3f}, ID_V change {v1 - v0:+.2f})")
    elif "explore" in h.P.get("skipped_parts", []):
        out("    ID_K / ID_V under N*: skipped at the deadline")
