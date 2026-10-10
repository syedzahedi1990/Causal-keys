"""The part-C scorer of P-2026-10-10-J (analysis/stage8c_score.py, analysis/stage8c_parts) on synthetic results with known
answers: the channel-ratio law (equivalence by TOST; R1 / R3 / graded departure patterns; fewer than 3 statistics ->
NOT EVALUABLE), the cell gate J-C-G4 (coverage), the efficacy gate J-C-G5, the sensitivity gate J-C-G8 (and the old JC3
rule passing the same synthetic rows), J-C1 ... J-C5, J-C-BOUND (MET, NOT MET, NOT EVALUABLE paths), the combination
over combos and models (NOT MET on any evaluable failure), the readers (KO; E3 needs layer 7 passing J-C-G3), J-C6 with
its anchor gate (BIND's usability decides a format's evaluability), the overlap screen with and without a window and
J-C-WIN, the hierarchical bootstrap (determinism, the two stages, the E4 seed level), Holm, J-C-G0 (the files it requires,
the script runs them; without a pass no line is computed and the headline names no outcome), and the whole scorer on a
results directory (sections, exit status). No model is loaded."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import stage8c_score as SC  # noqa: E402
from stage8c_parts import law as LW  # noqa: E402
from stage8c_parts import prakash as PK  # noqa: E402
from stage8c_parts import readers as RD  # noqa: E402
from stage8c_parts import stats as ST  # noqa: E402

from ckeys.edits import POP_SHA256, populations  # noqa: E402
from ckeys.story import LOCATIONS, PAIR_SWAP, pick_x  # noqa: E402

FMT = ("LETTER", "P1", "POST", "NONE")
NAT = {"LETTER": (10.0, 0.2), "P1": (8.0, 4.0), "POST": (3.0, 6.0), "NONE": (0.3, 9.0)}   # (ID_K, ID_V) per format
DEPTHS = (3, 7, 11, 15)


def row(it, ix, a_s, a_x, arg, mass=0.95):
    """A row with d_S = a_s at index it and d_X = a_x at index ix (so a pair of rows has ID = (a_S + a_X) / 2)."""
    d = [0.0] * 6
    d[it] += a_s
    d[ix] += a_x
    return d + d + [mass, arg]


def make_eval(n=64, psi=None, cover=0.95, flip_ok=True, depths=DEPTHS, seed=0, families=("E1", "E2", "E3", "E4a", "E4b"),
              e5=None, comps=None):
    """psi: {inst: (psi_K, psi_V)}; e5: {sub: (psi_K, psi_V)}; comps: {inst: (psi_K, psi_V)} (PERP / NONLEX / PAR / LEX)."""
    rng = np.random.default_rng(seed)
    E = populations()["E"][:n]
    psi = {"T": (1.0, 1.0), "E1": (1.0, 1.0), "E2": (1.0, 1.0), "E3": (1.0, 1.0), "E4a": (1.0, 1.0), "E4b": (1.0, 1.0)} | (psi or {})
    e5 = e5 or {}
    comps = comps or {}
    stories = []
    for si, c in enumerate(E):
        X = pick_x(c)
        iB, iS, iX = LOCATIONS.index(c["base"]), LOCATIONS.index(c["source"]), LOCATIONS.index(X)
        cells, stats = {}, {}
        noise = 1 + 0.05 * rng.standard_normal()
        for l in depths:
            stats[str(l)] = {}
            for f in FMT:
                K, Vv = NAT[f]
                K, Vv = K * noise, Vv * noise
                cell = {"self": [0.0] * 12 + [cover, iB]}

                def put(z, pk, pv, chans=("K", "V", "KV"), arg_ok=True):
                    for C in chans:
                        val = {"K": pk * K, "V": pv * Vv, "KV": pk * K + pv * Vv}[C]
                        for t, it_ in (("S", iS), ("X", iX)):
                            arg = it_ if (arg_ok and C == "KV") else iB
                            cell[f"{z}|{C}|{t}"] = row(iS, iX, val if t == "S" else 0.0, val if t == "X" else 0.0, arg, cover)
                put("nat", 1.0, 1.0)
                put("R", 0.0, 0.0, ("KV",), False)
                for z in ("T",) + tuple(families):
                    put(z, *psi.get(z, (1.0, 1.0)), arg_ok=flip_ok)
                for sub, (pk, pv) in e5.items():
                    put(sub, pk, pv, arg_ok=False)
                for lam, kind in ((0.5, "SK"), (0.8, "SK"), (0.5, "SV"), (0.8, "SV")):
                    z = f"{kind}{int(lam * 100)}"
                    pk, pv = (0.4 if lam == 0.5 else 0.8, 1.0) if kind == "SK" else (1.0, 0.4 if lam == 0.5 else 0.8)
                    put(z, pk, pv, ("K", "KV") if kind == "SK" else ("V", "KV"))
                if f in ("P1", "NONE"):
                    for z, (pk, pv) in comps.items():
                        if z.startswith(("PERP", "NONLEX")):
                            put(z, pk, pv, arg_ok=False)
                if f == "NONE":
                    for z, (pk, pv) in comps.items():
                        if z.startswith(("PAR", "LEX")):
                            put(z, pk, pv, ("KV",), False)
                cells[f"{f}@{l}"] = cell
            for z in ("T",) + tuple(families) + tuple(e5):
                for t in "SX":
                    stats[str(l)][f"{z}|{t}"] = {"nu": 0.0 if z == "T" else 0.5, "cos_k": 0.9, "cos_v": 0.9, "cos_resid": 0.9,
                                                 "proj_k": 1.0, "proj_v": 1.0, "proj_k_readers": 1.0}
        stories.append({"index": si, "core": c, "X": X, "p": 50, "iB": iB, "iS": iS, "iX": iX,
                        "iPiS": LOCATIONS.index(PAIR_SWAP[c["source"]]), "iPiX": LOCATIONS.index(PAIR_SWAP[X]),
                        "cells": cells, "stats": stats})
    prov = {"git_commit": "abc", "dtype": "torch.bfloat16", "attn_implementation": "sdpa", "populations": {"E": POP_SHA256["E"]},
            "calib_sha256": "c" * 64}
    return {"provenance": prov, "depths": list(depths), "formats": list(FMT), "stories": stories}


@pytest.fixture(scope="module")
def good():
    return LW.Model(make_eval(e5={"E5FR": (0.2, 0.6)}, comps={"PERP:E1": (0.1, 0.1), "NONLEX:E1": (0.1, 0.1),
                                                                "PAR:E1": (0.9, 0.9), "LEX:E1": (0.85, 0.85)}))


# --------------------------------------------------------------------------- bootstrap and helpers
def test_bootstrap_two_stage_deterministic_and_seed_level():
    cl = ["a"] * 5 + ["b"] * 3 + ["c"] * 2
    b1, b2 = ST.Boot(cl, nb=2000), ST.Boot(cl, nb=2000)
    assert np.array_equal(b1.W, b2.W) and np.array_equal(b1.SW, b2.SW)
    tot_a = b1.W[:, :5].sum(1)
    assert set(np.unique(tot_a)) <= {0, 5, 10, 15}, "cluster a contributes 5 x (times drawn) stories"
    assert np.allclose(b1.SW.sum(1), 1.0) and set(np.unique(b1.SW)) <= {0.0, 0.5, 1.0}
    x = np.arange(10.0)
    pt, bs = b1.mean(x)
    assert pt == 4.5 and abs(bs.mean() - 4.5) < 0.3
    x2 = np.stack([np.zeros(10), np.ones(10)], 1)          # seed 101 gives 0, seed 102 gives 1
    pt2, bs2 = b1.mean(x2)
    assert pt2 == 0.5 and set(np.round(np.unique(bs2), 6)) <= {0.0, 0.5, 1.0} and 0.3 < bs2.mean() < 0.7


def test_tost_log_ratio_holm_components():
    q = ST.Q(0.0, np.random.default_rng(0).normal(0, 0.05, 10000))
    assert ST.tost(q)
    assert not ST.tost(ST.Q(0.0, np.random.default_rng(0).normal(0, 0.3, 10000)))     # too wide
    assert not ST.tost(ST.Q(0.3, np.random.default_rng(0).normal(0.3, 0.01, 10000)))   # outside
    lr = ST.log_ratio(ST.Q(0.5, np.full(10, 0.5)), ST.Q(1.0, np.full(10, 1.0)))
    assert abs(lr.pt - math.log(0.5)) < 1e-12
    assert abs(ST.log_ratio(ST.Q(-1.0, np.full(3, -1.0)), ST.Q(1.0, np.ones(3))).pt - math.log(0.01)) < 1e-12
    # the Holm components (decision D2): TOST gives H1: lambda > -log 1.25 ('>') and H1: lambda < log 1.25 ('<') with the
    # bootstrap SE; the own decisions are the 90 % interval's
    SC.COMPONENTS.clear()
    SC.comp_tost("J-C3", "x", q)
    (c1, d1, o1), (c2, d2, o2) = SC.COMPONENTS
    assert c1 == c2 == "J-C3" and d1["line"] == "J-C3" and o1 and o2
    assert (d1["direction"], d1["bound"], d2["direction"], d2["bound"]) == (">", -math.log(1.25), "<", math.log(1.25))
    assert abs(d1["se"] - 0.05) < 0.002 and d1["est"] == 0.0
    deg = ST.component("J-C4", "y", ST.Q(1.0, np.ones(50)), 0.5, ">", True)[1]
    assert deg["se"] == 1e-12                                                            # degenerate bootstrap
    SC.COMPONENTS.clear()


# --------------------------------------------------------------------------- the law
def test_law_equivalent_and_departures(good):
    met, txt, st, pat = LW.law(good, "E1", DEPTHS)
    assert met is True and pat == "equivalent" and len(st) == 16, txt       # 4 depths x (W(P1), W(POST), A_L, A_P)
    m = LW.Model(make_eval(psi={"E2": (0.3, 1.0), "E3": (3.0, 1.0), "E4a": (0.7, 1.0), "E4b": (0.7, 1.0)}))
    met2, _, _, pat2 = LW.law(m, "E2", DEPTHS)
    assert met2 is False and pat2.startswith("R1")
    met3, _, _, pat3 = LW.law(m, "E3", DEPTHS)
    assert met3 is False and pat3.startswith("R3")
    met4, _, _, pat4 = LW.law(m, "E4", DEPTHS)
    assert met4 is False and pat4.startswith("graded")
    met5, txt5, _, _ = LW.law(m, "E1", (3,))
    assert met5 is True and "4/4" in txt5
    m6 = LW.Model(make_eval(n=20, depths=(3,)))
    st6, skip6 = m6.stats_of("E1", (3,), formats=("NONE", "P1"))
    assert len(st6) == 2 and LW.law(m6, "E1", (3,), formats=("NONE", "P1"))[0] is None   # W(P1), A_P: < 3 -> NOT EVALUABLE


def test_cell_gates_and_efficacy(good):
    assert good.cell_ok("P1", 3)[0] and good.used("P1", 3, "K") and not good.used("LETTER", 3, "V")
    low = LW.Model(make_eval(n=20, cover=0.5, depths=(3,)))
    ok, why = low.cell_ok("P1", 3)
    assert not ok and "coverage" in why
    assert LW.law(low, "E1", (3,))[0] is None
    eff, txt = good.effective(3, "E1")
    assert eff and "flip 1.00" in txt
    nf = LW.Model(make_eval(n=20, flip_ok=False, depths=(3,)))
    assert nf.effective(3, "E1")[0] is False
    weak = LW.Model(make_eval(n=20, psi={"E1": (0.3, 0.3)}, depths=(3,)))
    assert weak.effective(3, "E1")[0] is False                                            # phi 0.3 < 0.5
    assert good.effective(3, "E4")[0] is True
    g1, t1 = good.g1()
    g2, t2 = good.g2()
    assert g1 and g2, (t1, t2)
    assert all(v[0] for v in good.g6().values())


def test_sensitivity_gate_and_old_rule(good):
    ok, lines = LW.g8(good)
    assert ok, lines
    assert any(x.startswith("SK50") and "old rule passes" in x for x in lines), lines   # the share rule passes the 0.4x rows
    sens = LW.Model(make_eval(n=30, psi={"T": (0.6, 1.0)}, depths=(3,)))
    assert LW.g8(sens)[0] is False                                                        # T not equivalent -> gate fails


# --------------------------------------------------------------------------- the lines through the scorer's combinators
def gates_for(models, g3=None):
    class I:
        test = False
    I.models = models
    G = SC.Gates.__new__(SC.Gates)
    G.I, G.g0 = I, True
    G.model_ok = {k: True for k in models}
    G.g8 = {k: True for k in models}
    G.g3 = g3 if g3 is not None else {3: True, 7: True, 11: True, 15: True}
    G.why = {}
    return G, I


def test_lines_law_met_and_not_met(good):
    G, I = gates_for({"qwen7": good, "mistral7": good, "llama8": good})
    R = SC.lines_law(G, I)
    assert R["J-C1"][0] is True and R["J-C2"][0] is True and R["J-C3"][0] is True and R["J-C4"][0] is True
    assert R["J-C5"][0] is None                                                           # no component carries identity
    bad = LW.Model(make_eval(psi={"E2": (0.3, 1.0)}))
    G2, I2 = gates_for({"qwen7": good, "mistral7": bad, "llama8": good})
    R2 = SC.lines_law(G2, I2)
    assert R2["J-C1"][0] is False and R2["J-C2"][0] is False
    G3, I3 = gates_for({"qwen7": good, "mistral7": good, "llama8": good}, g3={3: False, 7: False, 11: False, 15: False})
    assert SC.lines_law(G3, I3)["J-C3"][0] is None                                       # J-C-G3 fails at every layer
    G4, I4 = gates_for({"qwen7": good})
    assert SC.lines_law(G4, I4)["J-C1"][0] is None                                       # 2 combos < 3


def test_combination_not_met_on_any_evaluable_failure(good):
    """A line over combos or models is NOT MET as soon as an evaluable one fails, whatever the number evaluable; NOT
    EVALUABLE only when none fails and fewer than the minimum are evaluable."""
    assert SC.combine({"a": True, "b": False, "c": None}, 3) is False
    assert SC.combine({"a": True, "b": True, "c": None}, 3) is None
    assert SC.combine({"a": True, "b": True, "c": True}, 3) is True
    assert SC.combine({"a": None}, 1) is None and SC.combine({}, 1) is None
    bad = LW.Model(make_eval(psi={"E2": (0.3, 1.0)}))
    G, I = gates_for({"qwen7": bad})                    # J-C1: 2 evaluable combos (< 3), E2 at l in {3, 7} not met
    v, per, _ = SC.lines_law(G, I)["J-C1"]
    assert sorted(x for x in per.values() if x is not None) == [False, True] and v is False, per
    assert SC.headline({"J-C1": (v, per, [])}).startswith("outcome (e):")


def test_g0_not_passed_computes_no_line_and_no_outcome(good):
    """J-C-G0 failing or not run (g0 False or None): no model is evaluable for any E line, so no pattern or carrier is
    recorded, and the headline names no outcome whatever PATTERNS, CARRIERS and the lexical-code reading hold."""
    carry = LW.Model(make_eval(psi={"E3": (3.0, 1.0)}, comps={"PERP:E1": (0.5, 0.5)}))
    for g0 in (None, False):
        SC.PATTERNS.clear()
        SC.CARRIERS.clear()
        G, I = gates_for({"qwen7": carry, "mistral7": good, "llama8": good})
        G.g0 = g0
        assert not G.ok("qwen7") and not G.ok("qwen7", law=False)
        R = SC.lines_law(G, I)
        assert all(R[c][0] is None for c in R) and len(R) == 6, {c: R[c][0] for c in R}
        assert SC.PATTERNS == {} and SC.CARRIERS == []
        assert SC.reported(G, I, lambda s: None) is None
    SC.PATTERNS["J-C3"] = {"qwen7 E3": "R3 key-flat"}
    SC.CARRIERS.append(("qwen7", "PERP:E1", 3))
    R = {"J-C1": (False, {}, []), "J-C3": (None, {}, [])}
    assert SC.headline(R, True, None) == SC.headline(R, False, False) == "J-C-G0 not passed: no outcome"
    assert SC.headline(R, True, True).startswith("outcome (e):")
    SC.PATTERNS.clear()
    SC.CARRIERS.clear()


def test_j_c5_and_bound():
    carry = LW.Model(make_eval(comps={"PERP:E1": (0.5, 0.5), "NONLEX:E1": (0.4, 0.4)}, e5={"E5FR": (0.2, 0.6)}))
    G, I = gates_for({"qwen7": carry})
    v, _, lines = SC.j_c5(G, I)
    assert v is True, lines
    vb, per, lb = SC.j_bound(G, I)
    assert vb is True and per == {"qwen7 E5FR": True}, lb
    nb = LW.Model(make_eval(e5={"E5FR": (0.9, 0.9)}))                                     # E5 read like the natural edit
    G2, I2 = gates_for({"qwen7": nb})
    assert SC.j_bound(G2, I2)[0] is False
    ng = LW.Model(make_eval(e5={"E5FR": (0.02, 0.05)}))                                   # not gated in at any depth
    G3, I3 = gates_for({"qwen7": ng})
    assert SC.j_bound(G3, I3)[0] is None
    lex = LW.Model(make_eval(comps={"PERP:E1": (0.5, 0.5)}, psi={"E2": (0.3, 1.0)}, e5={"E5FR": (0.2, 0.6)}))
    G4, I4 = gates_for({"qwen7": lex})
    assert SC.j_bound(G4, I4)[0] is False                                                 # E2 below log 0.8 at the same cells


# --------------------------------------------------------------------------- readers
def readers_json(n=60, ko=0.95, ko_r=0.03, idk=8.0):
    E = populations()["E"][:n]
    S = []
    for si, c in enumerate(E):
        X = pick_x(c)
        iB, iS, iX = LOCATIONS.index(c["base"]), LOCATIONS.index(c["source"]), LOCATIONS.index(X)
        rows = {"self": [0.0] * 6}
        for D in ("nat", "E1", "E2", "E3", "E4a"):
            for spec, f in (("all", 1.0), ("H", 1 - ko), ("r0", 1 - ko_r), ("r1", 1 - ko_r), ("r2", 1 - ko_r)):
                for t, it in (("S", iS), ("X", iX)):
                    d = [0.0] * 6
                    d[it] = idk * f
                    rows[f"{D}|{t}|{spec}"] = d
        S.append({"index": si, "core": c, "iB": iB, "iS": iS, "iX": iX, "rows": rows})
    return {"provenance": {"git_commit": "abc"}, "donors": ["nat", "E1", "E2", "E3", "E4a"], "specs": ["all", "H", "r0", "r1", "r2"], "stories": S}


def test_readers():
    R = RD.Readers(readers_json())
    assert R.gate()[0] and abs(R.ko("E1", "H").pt - 0.95) < 1e-9
    assert R.family("E3")[0] is True
    assert RD.Readers(readers_json(ko=0.5)).family("E1")[0] is False
    assert RD.Readers(readers_json(ko_r=0.4)).family("E1")[0] is False
    assert RD.Readers(readers_json(idk=1.0)).family("E1")[0] is None                    # denominator gate


def test_readers_e3_needs_layer_7_passing_g3(good):
    """J-C-READb: E3 at l = 7 is NOT EVALUABLE when layer 7 fails J-C-G3 (or has no J-C-G3 record), and adds no Holm
    component; E4 still counts."""
    RJ = readers_json()
    for g3, want in (({3: True, 7: True, 11: True, 15: True}, True), ({3: True, 7: False, 11: True, 15: True}, None),
                     ({3: True}, None)):
        SC.COMPONENTS.clear()
        G, I = gates_for({"qwen7": good}, g3=g3)
        I.get = lambda sub, k: RJ if sub == "readers" else None
        (va, pa, lines), (vb, pb, _) = SC.j_readers(G, I)
        assert pb["qwen7 E3"] is want and pb["qwen7 E4a"] is True and vb is True, (pb, lines)
        e3 = [d["name"] for c, d, _ in SC.COMPONENTS if "E3" in d["name"]]
        assert (e3 != []) is (want is True), e3
        assert ("qwen7 E3: layer 7 fails J-C-G3: not evaluable" in "\n".join(lines)) is (want is None), lines
    SC.COMPONENTS.clear()


# --------------------------------------------------------------------------- Prakash et al.'s material
def ex_cells(n, arm, fmt, pk, pv, phi=10.0, ok=True, nu=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    for i in range(n):
        e = 1 + 0.02 * rng.standard_normal()
        m = {"r0": 0.0, "r1": phi * e, "r2": pk * phi * e, "r3": pv * phi * e, "r4": phi * e, "r5": 0.0, "r6": 0.0,
             "r2w": 0.0, "r3w": 0.0, "r4w": 0.0}
        out.append({"arm": arm, "format": fmt, "m": m, "m_B": 0.0, "ok_r1": ok, "nu_CAA_ID": nu, "i": i})
    return out


def test_jc6_and_anchor():
    cells = []
    for f, kb in PK.ANCHOR.items():
        cells += ex_cells(40, "BIND", f, kb, 1 - kb) + ex_cells(40, "ID", f, 0.05, 0.9) + ex_cells(40, "CAA", f, 0.05, 0.9)
    J = {"formats": list(PK.ANCHOR), "lstar": 28, "cells": cells}
    v, g7, lines = PK.jc6(J, False)
    assert v is True and g7 is True, lines
    cells2 = [c for c in cells if c["arm"] != "CAA"]
    for f in PK.ANCHOR:
        cells2 += ex_cells(40, "CAA", f, 0.7, 0.3)                                        # key-flat CAA
    assert PK.jc6({"formats": list(PK.ANCHOR), "lstar": 28, "cells": cells2}, False)[0] is False
    cells3 = [c for c in cells if c["arm"] != "BIND"] + [x for f in PK.ANCHOR for x in ex_cells(40, "BIND", f, 0.3, 0.7)]
    v3, g73, _ = PK.jc6({"formats": list(PK.ANCHOR), "lstar": 28, "cells": cells3}, False)
    assert g73 is False and v3 is None                                                     # anchor not reproduced


def test_jc6_bind_usability_decides_evaluability():
    """J-C6: a format is evaluable only when BIND is usable there too (Gates b0 and b2, the kappa rule); BIND failing
    makes that format NOT EVALUABLE (a required format: the line NOT EVALUABLE), never NOT MET."""
    def run(bind_nm):
        cells = list(bind_nm)
        for f, kb in PK.ANCHOR.items():
            cells += ex_cells(40, "ID", f, 0.05, 0.9) + ex_cells(40, "CAA", f, 0.05, 0.9)
            if f != PK.NM:
                cells += ex_cells(40, "BIND", f, kb, 1 - kb)
        return PK.jc6({"formats": list(PK.ANCHOR), "lstar": 28, "cells": cells}, False)
    # BIND under NO-MENTION: kappa 0.617 (the anchor is reproduced) but psi_K + psi_V 0.32 < 0.5 (the kappa rule fails)
    v, g7, lines = run(ex_cells(40, "BIND", PK.NM, 0.2, 0.124))
    assert g7 is True and v is None, lines
    assert any("not evaluable" in x and "BIND kappa rule" in x for x in lines), lines
    v2, g72, lines2 = run(ex_cells(40, "BIND", PK.NM, 0.618, 0.382, phi=2.0))          # BIND Phi 2 < 3 nats (Gate b2)
    assert g72 is True and v2 is None and any("BIND Phi >= 3" in x for x in lines2), lines2
    assert run(ex_cells(40, "BIND", PK.NM, 0.618, 0.382))[0] is True                   # BIND usable: MET as before


def clamp_cells(n, l0, s, fmt=None, scale=10.0):
    """Natural clamp cells with s_ID = s (ID_K = s scale, ID_V = (1 - s) scale)."""
    out = []
    for i in range(n):
        k, v = s * scale, (1 - s) * scale
        lp = {"ID": {"S": 0.0, "X": 0.0}, "K_S": {"S": k, "X": 0.0}, "K_X": {"S": 0.0, "X": k}, "V_S": {"S": v, "X": 0.0},
              "V_X": {"S": 0.0, "X": v}, "KV_S": {"S": k + v, "X": 0.0}, "KV_X": {"S": 0.0, "X": k + v}}
        out.append({"l0": l0, "lp": lp, "i": i} | ({"format": fmt} if fmt else {}))
    return out


def screen_json(n, bind_on, sid):
    rows = {str(l): [{"ok": l >= bind_on, "m_patch": 20.0 if l >= bind_on else 0.0, "m_self": 0.0} for _ in range(n)] for l in range(28)}
    onsets = sorted(sid)
    cells = [c for l0 in onsets for c in clamp_cells(n, l0, sid[l0])]
    WB = [l for l in range(28) if l >= bind_on]
    W = [l for l in WB if (l + 1) in sid and sid[l + 1] >= 0.4]
    return {"population": list(range(n)), "sweep_rows": rows, "onsets": onsets, "clamp_cells": cells, "window": W,
            "l_w": min(W) if W else None}


def test_screen_and_window():
    sid = {l: max(0.0, 0.8 - 0.05 * l) for l in range(28)}
    ev, W, lw, lines, same = PK.screen(screen_json(60, 14, sid), False)
    assert ev and W == [] and lw is None and same                                         # s_ID(15) = 0.05 < 0.4
    ev2, W2, lw2, _, same2 = PK.screen(screen_json(60, 6, sid), False)
    assert ev2 and W2 == [6, 7] and lw2 == 6 and same2                                    # s_ID(7) 0.45, s_ID(8) 0.40
    cells = []
    for f, s in (("NO-MENTION", 0.05), ("QNAMES", 0.5), ("OPTIONS-AFTER", 0.75)):
        cells += ex_cells(60, "BIND", f, s, 1 - s)
    cl = [c for f, s in (("NO-MENTION", 0.05), ("QNAMES", 0.5), ("OPTIONS-AFTER", 0.75)) for c in clamp_cells(60, 7, s, f)]
    v, wl = PK.window({"l_w": 6, "cells": cells, "clamp_cells": cl}, False)
    assert v is True, wl
    cells_b = []
    for f in ("NO-MENTION", "QNAMES", "OPTIONS-AFTER"):
        cells_b += ex_cells(60, "BIND", f, 0.8, 0.2)                                       # flat-high: the law fails
    assert PK.window({"l_w": 6, "cells": cells_b, "clamp_cells": cl}, False)[0] is False


def test_lexical_code_counts_only_evaluable_models():
    """Outcome (b) reads the lexical-code reading of the effective families of the evaluable models only (J-C-G1, J-C-G2,
    J-C-G8 passed; E3 at the layers passing J-C-G3): a model removed from the law lines cannot decide it."""
    fams = ("E1", "E2", "E3", "E4a", "E4b")
    lexy = LW.Model(make_eval(n=24, depths=(3,), comps={f"{c}:{z}": (0.9, 0.9) for c in ("PAR", "LEX") for z in fams}))
    flat = LW.Model(make_eval(n=24, depths=(3,), comps={f"{c}:{z}": (0.1, 0.1) for c in ("PAR", "LEX") for z in fams}))
    G, I = gates_for({"qwen7": lexy, "mistral7": flat})
    assert SC.reported(G, I, lambda s: None) is False                                    # both evaluable: mistral7 decides
    G.model_ok["mistral7"] = False
    assert SC.reported(G, I, lambda s: None) is True                                     # mistral7 not evaluable: left out
    G2, I2 = gates_for({"qwen7": flat}, g3={3: True})
    G2.model_ok["qwen7"] = False
    assert SC.reported(G2, I2, lambda s: None) is None                                   # nothing counted


def test_holm_sensitivity_family_and_verdicts():
    """D2: the family is the interval components of the R-class lines with a verdict (J-C1, class M, is left out); a MET
    line whose component Holm no longer rejects becomes NOT MET under Holm; the helper is analysis/stage8_holm.py."""
    SC.COMPONENTS.clear()
    rng = np.random.default_rng(0)
    q_far = ST.Q(0.0, rng.normal(0.0, 0.02, 10000))          # far inside +-log 1.25: both components rejected by any rule
    q_edge = ST.Q(0.13, rng.normal(0.13, 0.05, 10000))       # 90 % interval inside; the '<' component's normal p ~ 0.03
    SC.comp_tost("J-C3", "far", q_far)
    SC.comp_tost("J-C4", "edge", q_edge)
    SC.comp_tost("J-C1", "m-class", q_edge)                  # class M: not in the family
    R = {"J-C1": (True, {}, []), "J-C3": (True, {}, []), "J-C4": (True, {}, [])}
    lines = []
    SC.holm_sensitivity(R, lines.append)
    text = "\n".join(lines)
    assert "NOT COMPUTED" not in text, text
    assert "over the 4 interval components" in text, text
    j3 = next(x for x in lines if x.strip().startswith("J-C3:"))
    j4 = next(x for x in lines if x.strip().startswith("J-C4:"))
    assert "2 components" in j3 and "no decision changes" in j3 and "verdict unchanged (MET)" in j3, j3
    assert "2 components" in j4 and "MET -> NOT MET under Holm" in j4, j4
    assert not any(x.strip().startswith("J-C1:") for x in lines)
    SC.COMPONENTS.clear()


def test_headline_outcomes():
    R = lambda **v: {k.replace("_", "-"): (x, {}, []) for k, x in v.items()}  # noqa: E731
    SC.PATTERNS.clear()
    assert SC.headline(R(J_C3=True, J_C4=True, J_C_BOUND=True)).startswith("outcome (a):")
    assert SC.headline(R(J_C5=True, J_C_BOUND=False)).startswith("outcome (a'):")
    assert SC.headline(R(J_C3=False, J_C_BOUND=True)).startswith("outcome (c):")
    SC.CARRIERS.clear()
    assert SC.headline(R(J_C1=True, J_C3=None, J_C5=None), lexical_all=True).startswith("outcome (b):")
    SC.CARRIERS.append(("qwen7", "PERP:E1", 7))   # a carrying component without a statistic excludes (b)
    assert SC.headline(R(J_C1=True, J_C3=None, J_C5=None), lexical_all=True).startswith("outcome (f):")
    SC.CARRIERS.clear()
    assert SC.headline(R(J_C1=False)).startswith("outcome (e):")
    SC.PATTERNS["J-C3"] = {"qwen7 E3": "graded departure"}
    assert SC.headline(R(J_C1=True, J_C3=False)).startswith("outcome (d):")
    SC.PATTERNS.clear()
    assert SC.headline(R(J_C1=True)).startswith("outcome (f): CONSISTENT")


# --------------------------------------------------------------------------- the whole scorer on a results directory
def test_g0_requires_the_shared_files_the_script_runs(tmp_path):
    """J-C-G0 requires every test of tests/test_generate.py (frame discovery), tests/test_stage8_populations.py (G6
    across the parts) and tests/test_stage8_holm.py (D2) besides the part's files, and the GPU script's pytest step runs
    every file the gate requires."""
    shared = {"tests/test_generate.py": 5, "tests/test_stage8_populations.py": 5, "tests/test_stage8_holm.py": 7}
    assert all(SC.G0_FILES.get(f, 0) >= n for f, n in shared.items()), SC.G0_FILES
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "gpu_stage8c.sh").read_text()
    call = re.search(r"^s8_pytest ((?:.*\\\n)*.*)$", sh, re.M).group(1)
    assert set(SC.G0_FILES) <= set(call.replace("\\\n", " ").split()), call
    (tmp_path / "logs").mkdir()
    hd = "==== 2026-10-10T00:00:00Z python -m pytest tests/...\n"
    full = hd + "".join(f"{fn}::test_{i} PASSED\n" for fn, n in SC.G0_FILES.items() for i in range(n))
    old = hd + "".join(f"{fn}::test_{i} PASSED\n" for fn, n in SC.G0_FILES.items() if fn not in shared for i in range(n))
    skip = full.replace("tests/test_generate.py::test_4 PASSED", "tests/test_generate.py::test_4 SKIPPED")
    for log, want in ((full, True), (old, False), (skip, False)):
        (tmp_path / "logs" / "pytest.log").write_text(log)
        assert SC.gate_g0(tmp_path, lambda s: None) is want


def test_main_on_results_dir(tmp_path, good):
    root = tmp_path / "res"
    J = make_eval(e5={"E5FR": (0.2, 0.6)}, comps={"PERP:E1": (0.1, 0.1), "NONLEX:E1": (0.1, 0.1)})
    for k in ("qwen7", "mistral7", "llama8"):
        (root / "eval").mkdir(parents=True, exist_ok=True)
        json.dump(J, open(root / "eval" / f"{k}.json", "w"))
    (root / "readers").mkdir()
    for k in ("qwen7", "mistral7"):
        json.dump(readers_json(), open(root / "readers" / f"{k}.json", "w"))
    (root / "calib").mkdir()
    sae_rec = {str(l): {"fve": {"-1": 0.8, "0": 0.95, "1": 0.85}, "published_fve": 0.93, "fve_at_p": 0.5, "kF": 8, "beta": 1.0}
               for l in DEPTHS}
    json.dump({"provenance": {"git_commit": "abc"}, "calib_pt_sha256": "c" * 64, "sae": sae_rec, "frames": []},
              open(root / "calib" / "qwen7.json", "w"))
    (root / "logs").mkdir()
    log = "==== 2026-10-10T00:00:00Z python -m pytest tests/...\n" + "".join(
        f"{fn}::test_{i} PASSED\n" for fn, n in SC.G0_FILES.items() for i in range(n))
    (root / "logs" / "pytest.log").write_text(log)
    rc = SC.main(["--results", str(root)])
    text = (root / "STAGE8C_SCORE.txt").read_text()
    for sec in ("PROVENANCE", "POPULATION", "GATES", "PREDICTIONS", "HEADLINE", "REPORTED", "SUMMARY", "EXPLORATORY"):
        assert sec in text, sec
    assert "J-C-G0  FP32 unit tests" in text and "-> MET" in text
    assert re.search(r"J-C1\s+M A prior 0.80\s+MET", text) and re.search(r"J-C-BOUND\s+R A prior 0.50\s+MET", text), text[-3000:]
    assert re.search(r"J-C6\s+M A prior 0.85\s+NOT EVALUABLE", text)
    assert "account lines, class R:" in text and rc == 0, rc
    (root / "logs" / "pytest.log").write_text(log.replace("PASSED", "FAILED", 1))
    SC.main(["--results", str(root)])
    assert re.search(r"J-C1\s+M A prior 0.80\s+NOT EVALUABLE", (root / "STAGE8C_SCORE.txt").read_text())
    (root / "logs" / "pytest.log").unlink()                     # J-C-G0 not run: no line computed, no outcome, exit 2
    rc = SC.main(["--results", str(root)])
    text = (root / "STAGE8C_SCORE.txt").read_text()
    assert "HEADLINE (C-6, pre-committed; the Section-5 outcome of the entry): J-C-G0 not passed: no outcome\n" in text
    assert re.search(r"J-C1\s+M A prior 0.80\s+NOT EVALUABLE", text) and SC.PATTERNS == {} and rc == 2, rc


import re  # noqa: E402
