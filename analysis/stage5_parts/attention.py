"""Part (a) of P-2026-10-05-G (docs/PREREGISTRATION.md): Gate a and predictions G1-G4 on experiments/remention_attention.py
outputs ({root}/{model}.json + .npz, arms POST = SENTENCE-AFTER and P1 = OPTIONS-AFTER), exactly as drafted. Called by
analysis/stage5_score.py (``score(root)``) or standalone.

Per model and format: odd-indexed cores select heads, even-indexed cores evaluate (fixed parity split, >= 2 per half).
  D_i(l,h) = A^B[r_b -> p] - mean_{j in N_i} A^B[r_j -> p];  H* = top-3 heads by mean D over the selection half
  E = mean over evaluation cores and H* of D;  F = mean gain of A[r_s -> p] under K_S at H* (F_b = the base word's loss)
  G, H2*: the hop-2 analogues (ans -> r_b against ans -> r_j);  Q(m) = G(POST, m) / min over gated-in anchors of G(POST)
  ID_K, d_K, span per casing (lowercase replicates stage 1; capitalised ids beside it); emitted casing = the casing with
  the larger mean candidate mass under clean B.  95 % percentile CIs: 10,000 resamples of cores (even cores for the
  attention statistics, all cores for the key effects); ratios of means recomputed within each resample, cross-model
  ratios resampling the same even-core indices in numerator and denominator; "undefined" when the denominator's point
  estimate is <= 0.
Gate a (per model): lowercase ID_K(P1) > 1.0 with CI excluding 0; anchors ID_K(POST) > 3.0 with CI excluding 0, small
  models CI of ID_K(POST) within [-1, +1]; in cells whose emitted casing is capitalised the capitalised ID_K passes the
  same side (positive cells: > 0 with CI excluding 0; null cells: CI within [-1, +1]); E(P1) >= 0.30 with lower bound
  > 0.15 and F/E(P1) >= 0.5.  A failing model is gated out (statistics exploratory).
G1  anchors, POST: E >= 0.20 (lower > 0.10) and F/E >= 0.5 in 2/2 anchors (an anchor gated out or missing is not
    evaluable, which counts as not met in the k/k line; G3 and G4b take their minimum over the gated-in anchors).
G2  small models, POST: H_diss if E >= 0.20 (lower > 0.10) and F/E >= 0.5 in every gated-in one; H_track if the upper
    bound of E < 0.10 in every one; else mixed. Qualifier "embedding-level heads only" if all H*(POST) lie in layers < 2
    or below the lowest layer of H*(P1); Jaccard of the two H* and d_K reported beside the call.
G3  under H_diss: R_A = E(POST)/E(P1) >= 0.5 and E(m, POST) >= 0.5 x min over gated-in anchors of E(anchor, POST), each
    gated-in small model; under H_track: R_A <= 0.10 each.
G4  (a) anchors, POST: G >= 0.10 with lower > 0.05 in 2/2 (as G1); if instead G's upper bound < 0.10 at both, hop 2 is
    unsuitable and (b) is not evaluable. (b) under H_diss and (a): Q(m) <= 0.5 with upper bound < 1.0 at each gated-in
    small model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_prereg import per_core as per_core_s1  # noqa: E402

SMALL = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"]
ANCHOR = ["Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"]
ARMS = ("POST", "P1")
LOCS = ("box", "basket", "shelf", "drawer", "cabinet", "closet")
SEED, B, K = 20261005, 10000, 3
_idx = {}


def IDX(n):
    if n not in _idx:
        _idx[n] = np.random.default_rng(SEED).integers(0, n, (B, n))
    return _idx[n]


def boot(x):
    x = np.asarray(x, float)
    bs = x[IDX(len(x))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def ratio(a, b):
    """Ratio of means over the same cores; ``b`` may be a list of vectors, in which case the denominator is their minimum."""
    a, bs_ = np.asarray(a, float), [np.asarray(v, float) for v in (b if isinstance(b, list) else [b])]
    den = min(v.mean() for v in bs_)
    if den <= 0:
        return None
    idx = IDX(len(a))
    r = a[idx].mean(1) / np.min([v[idx].mean(1) for v in bs_], 0)
    return a.mean() / den, np.percentile(r, 2.5), np.percentile(r, 97.5)


def fmt(t, d=3):
    return "undefined" if t is None else f"{t[0]:+.{d}f} [{t[1]:+.{d}f},{t[2]:+.{d}f}]"


def verdict(ok):
    return "MET" if ok else "NOT MET"


def top(m, k=K):
    """The k (layer, head) pairs with the largest mean statistic ``m`` [L, H]."""
    return [(int(t // m.shape[1]), int(t % m.shape[1])) for t in np.argsort(m.ravel())[::-1][:k]]


def at(A, heads):
    """``A`` [n, L, H] -> per-core mean over ``heads``."""
    return np.stack([A[:, l, h] for l, h in heads], 1).mean(1)


def key_effects(items, cs):
    """Per core (all cores): ID_K, d_K, span on the ``cs`` ids ("lower" | "cap")."""
    out = {"idK": [], "dK": [], "span": []}
    for it in items:
        b, s, x = it["b"], it["s"], it["x"]
        lp = {c: it["lp"][c][cs] for c in it["lp"]}
        out["idK"].append(0.5 * ((lp["K_S"][s] - lp["K_X"][s]) + (lp["K_X"][x] - lp["K_S"][x])))
        m = {c: lp[c][s] - lp[c][b] for c in lp}
        out["dK"].append(m["K_S"] - m["B"])
        out["span"].append(m["S"] - m["B"])
    return {k: np.array(v) for k, v in out.items()}


def competent(it):
    ids_b = np.array(it["lp"]["B"]["lower"] + it["lp"]["B"]["cap"])
    ids_s = np.array(it["lp"]["S"]["lower"] + it["lp"]["S"]["cap"])
    return int(ids_b.argmax()) % 6 == it["b"] and int(ids_s.argmax()) % 6 == it["s"]


class Cell:
    """One model x format: the arrays [n, L, H] (core order of the JSON), items, split-half heads and statistics."""

    def __init__(self, items, A, n_kv):
        self.items, self.A, self.n = items, A, len(items)
        self.i = np.array([it["i"] for it in items])
        self.sel, self.ev = self.i % 2 == 1, self.i % 2 == 0
        assert self.sel.sum() >= 2 and self.ev.sum() >= 2, (self.sel.sum(), self.ev.sum())
        self.L, self.H = A["D"].shape[1:]
        self.n_kv = n_kv
        D, D2 = A["D"], A["D2"]
        self.mD, self.mD2 = D[self.sel].mean(0), D2[self.sel].mean(0)
        self.Hs, self.H2 = top(self.mD), top(self.mD2)
        self.E_i, self.G_i = at(D[self.ev], self.Hs), at(D2[self.ev], self.H2)
        self.E, self.G = boot(self.E_i), boot(self.G_i)
        self.F_i = at((A["A_KS_rs_p"] - A["A_B_rs_p"])[self.ev], self.Hs)
        self.Fb_i = at((A["A_KS_rb_p"] - A["A_B_rb_p"])[self.ev], self.Hs)
        self.FX_i = at((A["A_KX_rx_p"] - A["A_B_rx_p"])[self.ev], self.Hs)
        self.F, self.Fb, self.FX = boot(self.F_i), boot(self.Fb_i), boot(self.FX_i)
        self.FE = ratio(self.F_i, self.E_i) if self.E[0] > 0 else None
        self.raw = {k: at(A[k][self.ev], self.Hs).mean() for k in ("A_B_rb_p", "A_B_N_p", "A_B_rs_p", "A_KS_rs_p")}
        self.raw2 = {k: at(A[k][self.ev], self.H2).mean() for k in ("A_B_ans_rb", "A_B_ans_N", "A_B_ans_rs")}
        self.hop2_KS = {k: boot(at((A["A_KS_ans_" + k] - A["A_B_ans_" + k])[self.ev], self.H2)) for k in ("rs", "rb")}
        self.key = {cs: key_effects(items, cs) for cs in ("lower", "cap")}
        self.mass = {cs: np.mean([it["mass"]["B"][cs] for it in items]) for cs in ("lower", "cap")}
        self.emitted = "cap" if self.mass["cap"] > self.mass["lower"] else "lower"
        self.competent = np.mean([competent(it) for it in items])
        self.argmax = {c: sorted({it["argmax_tok"][c] for it in items}) for c in ("B", "S", "X")}

    def idk(self, cs):
        return boot(self.key[cs]["idK"])

    def E_at(self, heads, k=None):
        """E over the evaluation half at other heads (cross-format) or the top-k of this cell's own selection."""
        return boot(at(self.A["D"][self.ev], heads if heads else top(self.mD, k)))

    def perm_null(self, reps=200):
        """Max over heads of the selection-half mean excess when a random no-prior candidate plays b."""
        rng, R, out = np.random.default_rng(SEED), self.A["A_B_r_p"][self.sel], []
        Ns = [it["N"] for it, s in zip(self.items, self.sel) if s]
        for _ in range(reps):
            Dn = []
            for A, N in zip(R, Ns):
                j = rng.choice(N)
                Dn.append(A[:, :, j] - A[:, :, [k for k in N if k != j]].mean(-1))
            out.append(np.mean(Dn, 0).max())
        return np.mean(out), np.percentile(out, 95)


def load(root, tag):
    f = Path(root) / f"{tag}.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    z = np.load(Path(root) / f"{tag}.npz")
    prov, cells = d["provenance"], {}
    for arm in ARMS:
        items = [it for it in d["items"] if it["arm"] == arm]
        if not items:
            continue
        A = {k.split("/", 1)[1]: z[k].astype(np.float32) for k in z.files if k.startswith(arm + "/") and k.count("/") == 1}
        cells[arm] = Cell(items, A, prov.get("n_kv"))
    return {"prov": prov, "cells": cells, "z": z}


def jaccard(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def ci_in(t, lo, hi):
    return t[1] >= lo and t[2] <= hi


def gate(m, role, cells):
    """Gate a of the draft: (key) lowercase ID_K criteria, (casing) the emitted-casing consistency, (measure) E(P1)
    and F/E(P1). Returns (gated_in, {name: (ok, text)})."""
    P, Q = cells.get("P1"), cells.get("POST")
    if P is None or Q is None:
        return False, {"missing": (False, "an arm is missing")}
    g = {}
    lo = {a: c.idk("lower") for a, c in (("P1", P), ("POST", Q))}
    g["key P1"] = (lo["P1"][0] > 1.0 and lo["P1"][1] > 0, f"lowercase ID_K(P1) {fmt(lo['P1'], 2)} > 1.0, CI excl. 0")
    if role == "anchor":
        g["key POST"] = (lo["POST"][0] > 3.0 and lo["POST"][1] > 0, f"lowercase ID_K(POST) {fmt(lo['POST'], 2)} > 3.0, CI excl. 0")
    elif role == "small":
        g["key POST"] = (ci_in(lo["POST"], -1.0, 1.0), f"lowercase ID_K(POST) {fmt(lo['POST'], 2)} CI within [-1, +1]")
    else:
        g["key POST"] = (True, f"lowercase ID_K(POST) {fmt(lo['POST'], 2)} (no criterion for an extra model)")
    for a, c in (("P1", P), ("POST", Q)):
        cap = c.idk("cap")
        if c.emitted != "cap":
            g[f"casing {a}"] = (True, f"emitted casing lowercase (capitalised ID_K {fmt(cap, 2)} reported)")
        elif a == "P1" or role == "anchor":
            g[f"casing {a}"] = (cap[0] > 0 and cap[1] > 0, f"emitted casing capitalised: capitalised ID_K {fmt(cap, 2)} > 0, CI excl. 0")
        elif role == "small":
            g[f"casing {a}"] = (ci_in(cap, -1.0, 1.0), f"emitted casing capitalised: capitalised ID_K {fmt(cap, 2)} CI within [-1, +1]")
        else:
            g[f"casing {a}"] = (True, f"emitted casing capitalised: capitalised ID_K {fmt(cap, 2)} (no criterion for an extra model)")
    g["measure"] = (P.E[0] >= 0.30 and P.E[1] > 0.15 and P.FE is not None and P.FE[0] >= 0.5,
                    f"E(P1) {fmt(P.E)} >= 0.30, lower > 0.15; F/E(P1) {fmt(P.FE, 2)} >= 0.5")
    return all(ok for ok, _ in g.values()), g


def report_cell(m, arm, c, out):
    out(f"  {arm:4s} n={c.n} (sel {c.sel.sum()} / eval {c.ev.sum()})  H*={c.Hs}  sel-half D={[round(float(c.mD[l, h]), 3) for l, h in c.Hs]}")
    out(f"       E {fmt(c.E)}  F {fmt(c.F)}  F/E {fmt(c.FE, 2)}  F_b {fmt(c.Fb)}  K_X mirror {fmt(c.FX)}")
    out(f"       raw at H*: A^B[r_b->p] {c.raw['A_B_rb_p']:.3f}  mean_N A^B[r_j->p] {c.raw['A_B_N_p']:.3f}  "
        f"A^B[r_s->p] {c.raw['A_B_rs_p']:.3f}  A^KS[r_s->p] {c.raw['A_KS_rs_p']:.3f}")
    out(f"       hop 2: H2*={c.H2}  G {fmt(c.G)}  raw ans->r_b {c.raw2['A_B_ans_rb']:.3f} ans->N {c.raw2['A_B_ans_N']:.3f}  "
        f"under K_S d(ans->r_s) {fmt(c.hop2_KS['rs'])} d(ans->r_b) {fmt(c.hop2_KS['rb'])}")
    for cs in ("lower", "cap"):
        k = c.key[cs]
        out(f"       {cs:5s} ids: ID_K {fmt(boot(k['idK']), 2)}  d_K {fmt(boot(k['dK']), 2)}  span {k['span'].mean():+.2f}  "
            f"cand-mass {c.mass[cs]:.2f}" + ("  <- emitted" if cs == c.emitted else ""))
    out(f"       competent {c.competent:.2f}  argmax tokens B {c.argmax['B']}")


def score(root, roles=None, out=print, exploratory=True):
    roles = roles or {}
    tags = sorted(f.stem for f in Path(root).glob("*.json"))
    role = {t: roles.get(t, "anchor" if t in ANCHOR else "small" if t in SMALL else "extra") for t in tags}
    order = [t for t in ANCHOR if t in tags] + [t for t in SMALL if t in tags] + [t for t in tags if t not in ANCHOR + SMALL]
    out("== Part (a): re-mention attention to the writing token (POST = SENTENCE-AFTER, P1 = OPTIONS-AFTER)")
    data, gated, gates = {}, {}, {}
    for t in order:
        d = load(root, t)
        data[t] = d
        out(f"\n## {t} ({role[t]}; attn {d['prov'].get('attn_implementation')}, L={d['prov']['n_layers']} H={d['prov']['n_heads']} "
            f"n_kv={d['prov']['n_kv']}, casing_check {d['prov'].get('casing_check')})")
        for arm in ARMS:
            if arm in d["cells"]:
                report_cell(t, arm, d["cells"][arm], out)
            else:
                out(f"  {arm}: MISSING")
        gated[t], gates[t] = gate(t, role[t], d["cells"])
        for name, (ok, txt) in gates[t].items():
            out(f"  gate a [{name:11s}] {txt} -> {'pass' if ok else 'FAIL'}")
        out(f"  Gate a -> {'gated in' if gated[t] else 'GATED OUT (statistics exploratory)'}"
            + ("" if role[t] != "extra" else "; extra model: enters no prediction"))
        if "POST" in d["cells"] and "P1" in d["cells"]:
            P, Q = d["cells"]["P1"], d["cells"]["POST"]
            lowP1 = min(l for l, _ in P.Hs)
            emb = all(l < 2 for l, _ in Q.Hs) or all(l < lowP1 for l, _ in Q.Hs)
            d["jaccard"], d["embedding_only"] = jaccard(Q.Hs, P.Hs), emb
            d["R_A"] = ratio(Q.E_i, P.E_i) if np.array_equal(Q.i, P.i) else None
            out(f"  H*(POST) layers {[l for l, _ in Q.Hs]} vs H*(P1) layers {[l for l, _ in P.Hs]}: Jaccard {d['jaccard']:.2f}, "
                f"embedding-level only: {emb};  R_A = E(POST)/E(P1) {fmt(d['R_A'], 2)}")
    anchors = [t for t in order if role[t] == "anchor" and gated[t]]
    smalls = [t for t in order if role[t] == "small" and gated[t]]
    A_all = [t for t in order if role[t] == "anchor"] + [t for t in ANCHOR if t not in tags and roles.get(t, "anchor") == "anchor"]
    not_in = {t: "missing" if t not in tags else "gated out" for t in A_all if t not in anchors}
    out(f"\nGated-in anchors: {anchors or 'none'};  gated-in small models: {smalls or 'none'}")
    post = lambda t: data[t]["cells"]["POST"]  # noqa: E731

    def E_ok(c):
        return c.E[0] >= 0.20 and c.E[1] > 0.10 and c.FE is not None and c.FE[0] >= 0.5

    def paired(ts):  # the even cores shared by every listed model
        common = set.intersection(*(set(post(t).i[post(t).ev]) for t in ts))
        return {t: np.array([v for i, v in zip(post(t).i[post(t).ev], post(t).E_i) if i in common]) for t in ts}, \
               {t: np.array([v for i, v in zip(post(t).i[post(t).ev], post(t).G_i) if i in common]) for t in ts}

    out("\n== Verdicts (G1-G4; the G2 call is provisional when one small model is evaluable; an anchor gated out or missing counts as not met in G1 and G4a)")
    V = {}  # name -> (True / False / None, text) for analysis/stage5_score.py
    kk = lambda ok: f"{sum(ok.values())}/{len(ok)} anchors" + (f" ({', '.join(f'{t} {w}' for t, w in not_in.items())})" if not_in else "")  # noqa: E731
    # G1
    if anchors:
        ok = {t: t in anchors and E_ok(post(t)) for t in A_all}
        out(f"  G1 anchor (POST E >= 0.20, lower > 0.10, F/E >= 0.5): " + "; ".join(f"{t} E {fmt(post(t).E)} F/E {fmt(post(t).FE, 2)}" for t in anchors)
            + f" -> {kk(ok)} {verdict(all(ok.values()))}")
        g1 = all(ok.values())
        V["G1"] = (g1, kk(ok))
    else:
        out("  G1 anchor: NOT EVALUABLE (no gated-in anchor)")
        g1 = None
        V["G1"] = (None, "no gated-in anchor")
    # G2
    call = None
    if smalls:
        diss = all(E_ok(post(t)) for t in smalls)
        track = all(post(t).E[2] < 0.10 for t in smalls)
        call = "H_diss" if diss else "H_track" if track else "mixed"
        emb = [t for t in smalls if data[t]["embedding_only"]]
        qual = f" (embedding-level heads only at {emb})" if call == "H_diss" and emb else ""
        for t in smalls:
            c = post(t)
            out(f"  G2 {t}: E {fmt(c.E)} F/E {fmt(c.FE, 2)}; H_diss cell {E_ok(c)}, H_track cell {c.E[2] < 0.10}; Jaccard(H*POST, H*P1) "
                f"{data[t]['jaccard']:.2f}, H*(POST) layers {[l for l, _ in c.Hs]}; d_K lower {fmt(boot(c.key['lower']['dK']), 2)} "
                f"cap {fmt(boot(c.key['cap']['dK']), 2)} beside ID_K lower {fmt(c.idk('lower'), 2)} cap {fmt(c.idk('cap'), 2)}")
        out(f"  G2 dissociation: {len(smalls)}/{len(smalls)} gated-in small models -> {'H_diss MET' if diss else 'H_track MET' if track else 'mixed: NOT MET (no account declared)'}{qual}"
            + (" [provisional, one model evaluable]" if len(smalls) == 1 else ""))
        out(f"  CALL: {call}{qual}" + (" (provisional, one model evaluable)" if len(smalls) == 1 else " (full, 2/2)" if len(smalls) == 2 else ""))
        V["G2"] = (call != "mixed", f"{call}{qual}, {len(smalls)}/{len(smalls)} gated-in small models" + (" (provisional)" if len(smalls) == 1 else ""))
    else:
        out("  G2 dissociation: NOT EVALUABLE (no gated-in small model)")
        out("  CALL: no call (0 small models evaluable)")
        V["G2"] = (None, "no gated-in small model")
    # G3
    if call == "H_diss":
        if anchors:
            Es, _ = paired(smalls + anchors)
            lines, oks = [], []
            for t in smalls:
                sc = ratio(Es[t], [Es[a] for a in anchors])
                ra = data[t]["R_A"]
                ok = ra is not None and ra[0] >= 0.5 and sc is not None and sc[0] >= 0.5
                oks.append(ok)
                lines.append(f"{t} R_A {fmt(ra, 2)} E(m)/min anchors {fmt(sc, 2)} -> {verdict(ok)}")
            out(f"  G3 magnitude under H_diss (R_A >= 0.5 and E(m, POST) >= 0.5 x min anchor E(POST)): " + "; ".join(lines) + f" -> {sum(oks)}/{len(oks)} {verdict(all(oks))}")
            V["G3"] = (all(oks), f"under H_diss, {sum(oks)}/{len(oks)}")
        else:
            out("  G3 magnitude under H_diss: NOT EVALUABLE (no gated-in anchor for the scale comparison); R_A: "
                + "; ".join(f"{t} {fmt(data[t]['R_A'], 2)}" for t in smalls))
            V["G3"] = (None, "under H_diss, no gated-in anchor")
    elif call == "H_track":
        oks = {t: data[t]["R_A"] is not None and data[t]["R_A"][0] <= 0.10 for t in smalls}
        out(f"  G3 magnitude under H_track (R_A <= 0.10): " + "; ".join(f"{t} R_A {fmt(data[t]['R_A'], 2)}" for t in smalls)
            + f" -> {sum(oks.values())}/{len(oks)} {verdict(all(oks.values()))}")
        V["G3"] = (all(oks.values()), f"under H_track, {sum(oks.values())}/{len(oks)}")
    else:
        out(f"  G3 magnitude: NOT EVALUABLE (judged only for the account G2 declared; call = {call})")
        V["G3"] = (None, f"call = {call}")
    # G4a
    g4a = None
    if anchors:
        ok = {t: t in anchors and post(t).G[0] >= 0.10 and post(t).G[1] > 0.05 for t in A_all}
        unsuitable = all(post(t).G[2] < 0.10 for t in anchors)
        g4a = all(ok.values())
        out(f"  G4a hop-2 anchor (POST G >= 0.10, lower > 0.05): " + "; ".join(f"{t} G {fmt(post(t).G)}" for t in anchors)
            + f" -> {kk(ok)} {verdict(g4a)}"
            + ("; upper bound < 0.10 at every gated-in anchor: the hop-2 measure is unsuitable, G4b not evaluable" if unsuitable else ""))
        V["G4a"] = (g4a, kk(ok) + ("; measure unsuitable" if unsuitable else ""))
    else:
        out("  G4a hop-2 anchor: NOT EVALUABLE (no gated-in anchor)")
        V["G4a"] = (None, "no gated-in anchor")
    # G4b
    if call == "H_diss" and g4a:
        _, Gs = paired(smalls + anchors)
        lines, oks = [], []
        for t in smalls:
            q = ratio(Gs[t], [Gs[a] for a in anchors])
            ok = q is not None and q[0] <= 0.5 and q[2] < 1.0
            oks.append(ok)
            lines.append(f"{t} Q {fmt(q, 2)} (G {fmt(post(t).G)}) -> {verdict(ok)}")
        out(f"  G4b hop-2 cross-scale (Q(m) <= 0.5, upper < 1.0): " + "; ".join(lines) + f" -> {sum(oks)}/{len(oks)} {verdict(all(oks))}")
        V["G4b"] = (all(oks), f"{sum(oks)}/{len(oks)} gated-in small models")
    else:
        why = "no gated-in small model" if call is None else "G2 did not declare H_diss" if call != "H_diss" else "G4a not met" if g4a is False else "G4a not evaluable"
        out(f"  G4b hop-2 cross-scale: NOT EVALUABLE ({why})")
        V["G4b"] = (None, why)
    if exploratory:
        explore(order, role, data, out)
    return {"gated": gated, "role": role, "call": call, "g1": g1, "g4a": g4a, "verdicts": V}


def ans_row_top5(z, arm, c):
    """Mean attention of the answer row at H2* over the even cores sharing the modal T; columns named by position."""
    rows = {it["i"]: z[f"{arm}/{it['i']}/ans_row"].astype(np.float32) for it in c.items if f"{arm}/{it['i']}/ans_row" in z.files}
    ev = [it for it, e in zip(c.items, c.ev) if e and it["i"] in rows]
    if not ev:
        return "n/a"
    T = max(set(it["T"] for it in ev), key=[it["T"] for it in ev].count)
    ev = [it for it in ev if it["T"] == T]
    m = np.mean([np.mean([rows[it["i"]][l, h] for l, h in c.H2], 0) for it in ev], 0)
    names = {}
    for it in ev[:1]:
        names = {it["p"]: "p", it["q_obj"]: "q_obj", it["q_dist"]: "q_dist", it["ans"]: "ans"} | {r: f"r_{LOCS[j]}" for j, r in enumerate(it["r"])}
    return ", ".join(f"{names.get(int(k), int(k))}={m[k]:.3f}" for k in np.argsort(m)[::-1][:5]) + f" (n={len(ev)})"


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def corr_ci(a, b, f):
    idx = IDX(len(a))
    with np.errstate(invalid="ignore"):
        r = np.array([f(a[i], b[i]) for i in idx[:2000]])
    return f(a, b), np.nanpercentile(r, 2.5), np.nanpercentile(r, 97.5)


def explore(order, role, data, out):
    out("\n== Exploratory (not scored)")
    for t in order:
        d = data[t]
        for arm in ARMS:
            if arm not in d["cells"]:
                continue
            c = d["cells"][arm]
            A, ev = c.A, c.ev
            out(f"  {t} {arm}: layer profile of D (max head per layer, all cores): " + " ".join(f"{v:.2f}" for v in A["D"].mean(0).max(1)))
            nkv = c.n_kv or c.H
            out(f"     H* depth l/L {[round(l / c.L, 2) for l, _ in c.Hs]}, KV groups {[h // (c.H // nkv) for _, h in c.Hs]}; "
                f"H2* depth {[round(l / c.L, 2) for l, _ in c.H2]}; top-1 E {fmt(c.E_at(None, 1))}, top-10 E {fmt(c.E_at(None, 10))}; "
                f"permutation null of the selection max (mean, 95th; 200 permutations) {tuple(round(float(v), 3) for v in c.perm_null())}")
            out(f"     column baselines at H* (even cores): A^B[r_b->p-1] {at(A['A_B_rb_pm1'][ev], c.Hs).mean():.3f}  "
                f"[r_b->p+1] {at(A['A_B_rb_pp1'][ev], c.Hs).mean():.3f}  [r_b->0] {at(A['A_B_rb_0'][ev], c.Hs).mean():.3f};  "
                f"r_b row entropy {at(A['H_B_rb'][ev], c.Hs).mean():.2f} nats")
            dup = at((A["A_B_rinit_qobj"] - A["A_B_N_qobj"])[ev], c.Hs)
            dd = at((A["A_B_rdist_qdist"] - A["A_B_N_qdist"])[ev], c.Hs)
            dd = dd[~np.isnan(dd)]
            out(f"     in-story duplicates at H*: r_init->q_obj excess {fmt(boot(dup))}; r_dist->q_dist excess "
                f"{fmt(boot(dd)) if len(dd) >= 2 else 'n/a'} (n={len(dd)})")
            ansp = A["A_B_ans_p"][ev].mean(0)
            out(f"     ans->p best head {ansp.max():.3f} at {top(ansp, 1)[0]};  hop 2 at H2* under K_S: d(ans->r_s) {fmt(c.hop2_KS['rs'])}")
            out(f"     ans row under clean B, top-5 key columns at H2* (even cores of the modal length): {ans_row_top5(d['z'], arm, c)}")
            out(f"     full-vocabulary argmax under clean B/S/X: {c.argmax['B']} / {c.argmax['S']} / {c.argmax['X']}; "
                f"lowercase mass {c.mass['lower']:.2f}, capitalised {c.mass['cap']:.2f}, competent {c.competent:.2f}")
        if "POST" in d["cells"] and "P1" in d["cells"]:
            P, Q = d["cells"]["P1"], d["cells"]["POST"]
            out(f"  {t} cross-format: P1 heads on POST E_cross {fmt(Q.E_at(P.Hs))}; POST heads on P1 E_cross {fmt(P.E_at(Q.Hs))}; "
                f"paired G(P1) - G(POST) {fmt(boot(P.G_i - Q.G_i)) if np.array_equal(P.i, Q.i) else 'n/a'}")
            if role[t] != "anchor":
                dk = {cs: Q.key[cs]["dK"][Q.ev] for cs in ("lower", "cap")}
                out(f"  {t} d_K vs F_b co-variation (POST, even cores; CIs from 2,000 resamples): lowercase Pearson {fmt(corr_ci(dk['lower'], Q.Fb_i, lambda a, b: np.corrcoef(a, b)[0, 1]), 2)} "
                    f"Spearman {fmt(corr_ci(dk['lower'], Q.Fb_i, spearman), 2)}; emitted ({Q.emitted}) Pearson "
                    f"{fmt(corr_ci(dk[Q.emitted], Q.Fb_i, lambda a, b: np.corrcoef(a, b)[0, 1]), 2)}")
        s1 = Path("results/gpu_stage1") / f"{t}_s0.json"
        if s1.exists():
            res = json.load(open(s1))["results"]
            ref = {a: boot([v["idK"] for v in per_core_s1(res, a).values()]) for a in ARMS}
            out(f"  {t} stage-1 lowercase ID_K (sdpa, batched): POST {fmt(ref['POST'], 2)} P1 {fmt(ref['P1'], 2)} vs in-run "
                f"POST {fmt(d['cells']['POST'].idk('lower'), 2) if 'POST' in d['cells'] else 'n/a'} P1 {fmt(d['cells']['P1'].idk('lower'), 2) if 'P1' in d['cells'] else 'n/a'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/gpu_stage5/attention")
    ap.add_argument("--role", default="", help="override roles, e.g. Qwen2.5-0.5B-Instruct=small (default: 1.5B/3B small, 7B/14B anchor, else extra)")
    ap.add_argument("--no-exploratory", action="store_true")
    a = ap.parse_args(argv)
    roles = dict(x.split("=") for x in filter(None, a.role.split(",")))
    if roles:
        print(f"NOTE: roles overridden {roles}; the preregistered roles are small {SMALL}, anchor {ANCHOR}")
    score(a.root, roles, exploratory=not a.no_exploratory)


if __name__ == "__main__":
    main()
