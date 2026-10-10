"""The part-D scorer of P-2026-10-10-J (analysis/stage8d_score.py, analysis/stage8d_parts) on synthetic inputs with known
answers: the seeded story bootstrap and ratio recomputation, the one-sided tests and Holm, the 2/2 combination, each
line's statistic from rows built to give a chosen value, the gates and NOT EVALUABLE paths, the D6 decision table, and an
end-to-end scoring of a synthetic results directory."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import stage8d_score as SC  # noqa: E402
from stage8d_parts import lines as LN  # noqa: E402
from stage8d_parts.stats import Boot, Q, boot, combine, ratio  # noqa: E402
from stage8d_parts.stats import Tests as OneSided  # noqa: E402

LOC = ("box", "basket", "shelf", "drawer", "cabinet", "closet")


def core(i):
    """A synthetic core: B, S, X, I, D at fixed indices (B=0, S=1, X=2, I=3, D=4)."""
    return dict(agent="Alice", other="Bob", object="toy" if i % 2 else "cup", distractor="pen", initial=LOC[3],
                distractor_location=LOC[4], base=LOC[0], source=LOC[1])


IX = {"B": 0, "S": 1, "X": 2, "I": 3, "D": 4}
I4 = {"B": 0, "S": 1, "X": 2, "Subj": 3}


def base_lp(k=6, top="B", lift=5.0):
    v = np.zeros(k)
    v[(I4 if k == 4 else IX)[top]] = lift
    return v - np.log(np.exp(v).sum())


def row(base=None, k=6, **d):
    """A normalised log-prob vector equal to ``base`` except that the named words' log-probs change by exactly d; the
    other words absorb the mass proportionally (so their log-probs shift by one common constant)."""
    base = base_lp(k) if base is None else np.asarray(base, float)
    ix = I4 if len(base) == 4 else IX
    new = base.copy()
    idx = {ix[w]: dv for w, dv in d.items()}
    for j, dv in idx.items():
        new[j] = base[j] + dv
    rest = [j for j in range(len(base)) if j not in idx]
    fixed = float(np.exp([new[j] for j in idx]).sum()) if idx else 0.0
    assert fixed < 1, d
    new[rest] = base[rest] + np.log((1 - fixed) / np.exp(base[rest]).sum())
    return new.tolist()


# --------------------------------------------------------------------------- statistics
def test_bootstrap_seeded_and_ratio_recomputed():
    b1, b2 = Boot(30), Boot(30)
    assert np.array_equal(b1.W, b2.W) and b1.W.sum(1).min() == 30
    rng = np.random.default_rng(0)
    a, c = rng.normal(2, 1, 30), rng.normal(4, 1, 30)
    r = ratio(b1.mean(a), b1.mean(c))
    assert r.pt == pytest.approx(a.mean() / c.mean())
    want = (b1.W @ a) / (b1.W @ c)
    assert np.allclose(r.bs, want)
    assert boot(30) is boot(30)


def test_one_sided_tests_and_holm_components():
    q = Q(1.0, np.linspace(0.5, 1.5, 10001))
    sink = []
    t = OneSided("X", sink)
    assert t.lower_gt(q, 0.4, "a") and not t.lower_gt(q, 0.6, "b")
    assert t.upper_lt(q, 1.6, "c") and not t.upper_lt(q, 1.4, "d")
    assert t.inside(q, 0.4, 1.6, "e") and not t.inside(q, 0.6, 1.6, "f")
    assert len(sink) == len(t.items) == 8
    # each test is one Holm component of analysis/stage8_holm.py: {line, name, est, se, bound, direction}
    code, d, own = sink[0]
    assert code == "X" and own is True and set(d) == {"line", "name", "est", "se", "bound", "direction"}
    assert d["line"] == "X" and d["est"] == 1.0 and d["bound"] == 0.4 and d["direction"] == ">"
    assert d["se"] == pytest.approx(float(np.std(q.bs, ddof=1)))
    assert sink[2][1]["direction"] == "<" and sink[2][1]["bound"] == 1.6 and sink[1][2] is False
    assert [x[1]["direction"] for x in sink[4:6]] == [">", "<"]          # equivalence = two one-sided components
    assert len({x[1]["name"] for x in sink}) == 8


def test_risk_class_follows_the_prior():
    """Decision D1: L = prior >= 0.9, M = 0.8 <= prior < 0.9, R = prior < 0.8 (the class is relabelled, the prior kept)."""
    for code, (cls, kind, prior, *_rest) in SC.LINES.items():
        want = "L" if prior >= 0.9 else "M" if prior >= 0.8 else "R"
        assert cls == want, (code, cls, prior)
        assert kind == "A"
    assert SC.LINES["J-D4"][0] == "M" and SC.LINES["J-D8-Q"][0] == "M"


def test_holm_family_contract(tmp_path, monkeypatch):
    """The family handed to analysis/stage8_holm.py: the interval components of the R-class lines with a verdict, from
    the models where the line is evaluable, every component of such a line whatever its point conditions, names unique;
    a MET line with a component Holm no longer rejects is reported as NOT MET under Holm."""
    import types
    calls = []

    def fake(components):
        calls.append([dict(c) for c in components])
        return [dict(c, p=0.5, threshold=0.025, reject=False) for c in reversed(components)]   # any order
    monkeypatch.setitem(sys.modules, "stage8_holm", types.SimpleNamespace(holm=fake))
    build(tmp_path, mistral_bind=False, qwen_ioi_v=-1.0)
    rc = SC.main(["--results", str(tmp_path), "--test"])
    text = (tmp_path / "STAGE8D_SCORE.txt").read_text()
    assert rc == 0 and len(calls) == 1, text
    fam = calls[0]
    lines = {c["line"] for c in fam}
    rlines = {c for c, x in SC.LINES.items() if x[0] == "R"}
    assert lines <= rlines and "J-D1" in lines and "J-D-HOP2" in lines
    assert "J-D5" not in lines                     # NOT EVALUABLE (one evaluable model): no component
    assert not lines & {"J-D4", "J-D8-Q", "J-D7", "J-D6a"}     # M and L lines are not in the family
    names = [(c["line"], c["name"]) for c in fam]
    assert len(set(names)) == len(names)
    assert all(set(c) == {"line", "name", "est", "se", "bound", "direction"} for c in fam)
    # J-D-HOP2 is NOT MET on a point condition (carry 0.1 in INLINE) in both models; its tests are still in the family
    hop = [c["name"] for c in fam if c["line"] == "J-D-HOP2"]
    assert sorted(hop) == sorted(f"{k} {a} carry(top10) H0: <= 0.2" for k in ("qwen7", "mistral7") for a in ("Q_OUT", "INLINE"))
    assert "J-D1: 2 components; 2 decision(s) change under Holm" in text
    assert "verdict MET -> NOT MET under Holm" in text and "J-D-ROUTE-IOI: " in text


def test_holm_with_the_shared_helper(tmp_path):
    """analysis/stage8_holm.py (common part) on this part's family: normal p from the bootstrap SE, step-down at 0.025."""
    import math
    import stage8_holm
    q = Q(0.30, np.random.default_rng(0).normal(0.30, 0.1, 10000))
    sink = []
    OneSided("J-D1", sink).lower_gt(q, 0.0, "qwen7 x")
    got = stage8_holm.holm([d for _, d, _ in sink])
    z = (0.30 - 0.0) / float(np.std(q.bs, ddof=1))
    assert len(got) == 1 and got[0]["line"] == "J-D1" and abs(got[0]["p"] - 0.5 * math.erfc(z / math.sqrt(2))) < 1e-9
    build(tmp_path, mistral_bind=False, qwen_ioi_v=-1.0)
    SC.main(["--results", str(tmp_path), "--test"])
    text = (tmp_path / "STAGE8D_SCORE.txt").read_text()
    assert "Holm sensitivity (analysis/stage8_holm.py" in text and "NOT COMPUTED" not in text
    assert any(l.strip().startswith("J-D1: 2 components") for l in text.splitlines())


def test_combine_rule():
    assert combine({"a": True, "b": True}, 2) is True
    assert combine({"a": True, "b": None}, 2) is None
    assert combine({"a": False, "b": None}, 2) is False
    assert combine({"a": None, "b": None}, 2) is None
    assert combine({"x": True, "y": True, "z": None}, 2) is True
    assert combine({"x": True, "y": None, "z": None}, 2) is None
    assert combine({"x": True}, 1) is True


# --------------------------------------------------------------------------- synthetic files
def inject_json(n=40, loo=5.0, pi_ok=True, kn_like_subB=True, route=(0.9, 0.1, 0.95), kv_scale=1.0, controls=0.1):
    """Rows with exact six-way log-prob changes: ID_K = 4, ID_inj^LOO = loo (looX lifts X by loo, or by 1 when not
    pi_ok), N_X = 4, iota(move) = 4.5 / 4, ID_inj(KV) / ID_inj(K) = kv_scale, r_ans(C) = route."""
    S = []
    for i in range(n):
        rows = {"none": row(), "none2": row(), "K_S": row(S=4, X=0), "K_X": row(X=4, S=0), "subB": row(B=-3)}
        rows["K_N"] = list(rows["subB"]) if kn_like_subB else row(S=3, X=0)
        rows["looS"], rows["looX"] = row(S=loo, X=0), row(X=loo if pi_ok else 1.0, S=0)
        rows["moveS"], rows["move"] = row(S=4.5, X=0), row(X=4.5, S=0)
        rows["kvS"], rows["kvX"] = row(S=4.5 * kv_scale, X=0), row(X=4.5 * kv_scale, S=0)
        for nm, al in (("add0.5", 1), ("add1", 2), ("add2", 4)):
            rows[nm] = row(X=al)
        for nm in ("iso0", "iso1", "iso2", "hspan0", "hspan1", "hspan2", "perm", "choices", "question", "postflag"):
            rows[nm] = row(X=0.05)
        for nm in ("orth", "meanH", "active"):
            rows[nm] = row(X=4.5 * controls)
        rows["own"] = row(X=4.6)
        rk, rv, rkv = route
        rt = {"none": row(), "move": row(X=4.5), "ans_K": row(X=4.5 * (1 - rk)), "ans_V": row(X=4.5 * (1 - rv)),
              "ans_KV": row(X=4.5 * (1 - rkv)), "other_KV": row(X=4.4)}
        S.append(dict(index=i, core=core(i), ix=dict(IX), rows=rows, route=rt))
    return {"provenance": {}, "stories": S}


def test_inject_lines_known_values():
    x = LN.Inject(inject_json())
    d = x.d1()
    assert d["ratio"].pt == pytest.approx(5 / 4) and d["pi_x"] == 1.0 and x.floor() == 0.0
    assert x.NX().pt == pytest.approx(4.0)
    assert d["iota"]["move"].pt == pytest.approx(4.5 / 4)
    r = x.d4()
    assert r["r"]["K"].pt == pytest.approx(0.9) and r["r"]["V"].pt == pytest.approx(0.1) and r["KmV"].pt == pytest.approx(0.8)
    k = x.kn()
    assert k["r"].pt == pytest.approx(1.0) and k["beta"].pt == pytest.approx(1.0) and abs(k["spec"].pt) < 1e-9
    assert x.addr().pt == pytest.approx(1.0)
    assert LN.Inject(inject_json(kv_scale=0.5)).addr().pt == pytest.approx(0.5)
    c = x.d2()["controls"]["orth"]
    assert c["frac"].pt == pytest.approx(0.1) and c["diff"].pt == pytest.approx(4.5 * 0.9 / 4)
    y = LN.Inject(inject_json(pi_ok=False))
    assert y.d1()["pi_x"] == 0.0
    z = LN.Inject(inject_json(kn_like_subB=False)).kn()
    assert z["spec"].pt == pytest.approx(3.0)


def ablate_json(n=30, flag=0.2, pc=0.9):
    rng = np.random.default_rng(1)
    S = []
    for i in range(n):
        base = 4 + rng.normal(0, 0.3)
        res = {}
        for c, f in (("none", 1), ("flag", flag), ("pc1", pc), ("meanH", pc), ("iso0", 1), ("iso1", 1), ("iso2", 1),
                     ("flag@init_dloc", 1), ("flag@B", 1)):
            res[c] = dict(idK=base * f, idV=3.0, dK=0, dV=0, mass=0.9, argmax=LOC[0], base_ok=True, init_ans=False)
        S.append(dict(index=i, core=core(i), res=res))
    return {"provenance": {}, "stories": S}


def test_ablate_rho():
    a = LN.Ablate(ablate_json())
    assert a.rho("flag").pt == pytest.approx(0.2) and a.rho("pc1").pt == pytest.approx(0.9)
    assert a.rates("none") == (1.0, 0.0)


def bind_json(n=40, psi_obj=1.0, psi_dis=1.0):
    S = []
    for i in range(n):
        strat = "object_first" if i % 2 == 0 else "distractor_first"
        psi = psi_obj if strat == "object_first" else psi_dis
        views = {}
        for v, tgt in (("direct", "B"), ("other_agent", "I"), ("irrelevant_object", "D")):
            b = base_lp(6, tgt)
            views[v] = {"none": b.tolist(), "init": row(b, X=psi if v == "other_agent" else 0.0),
                        "dloc": row(b, X=0.0), "ev": row(b, X=1.0 if v == "direct" else 0.0), "iso": b.tolist(), "K_X": row(b, X=4.0)}
        S.append(dict(index=i, core=core(i), stratum=strat, ix=dict(IX), views=views))
    return {"provenance": {}, "stories": S}


def test_bind_psi_and_strata():
    b = LN.Bind(bind_json())
    assert b.acc("other_agent") == 1.0 and b.acc("irrelevant_object") == 1.0 and b.acc("direct") == 1.0
    p = b.psi()
    assert p["pooled"].pt == pytest.approx(1.0) and abs(p["diff"].pt) < 1e-9 and p["n_obj"] == 20
    q = LN.Bind(bind_json(psi_obj=2.0, psi_dis=-2.0)).psi()
    assert q["diff"].pt == pytest.approx(4.0) and q["diff"].lo() > 0 and q["pooled"].pt == pytest.approx(0.0)
    assert b.psi_ev().pt == pytest.approx(2.0)          # [1 - 0] - [0 - 1]


def q_recs(n=30, arm="Q_OUT", idk=-3.0, idv=-5.0, R=0.9, KO=0.95, rnd=0.0, comp=True, ioi=False, carry=0.7, route=(0.9, 0.2, 0.95)):
    rng = np.random.default_rng(2)
    k = 4 if ioi else 6
    recs = []
    for i in range(n):
        if ioi:
            item = dict(idK=idk + rng.normal(0, 0.1), idV=idv + rng.normal(0, 0.1), idKV=0, id4K=0, id4V=0,
                        two_B=1.0 if comp else 0.0, four_B=1.0, LD_B=3.0 + rng.normal(0, 0.1), mass_B=0.6)
        else:
            word = (LOC[2] if arm == "Q_OUT" else LOC[0]) if comp else (LOC[0] if arm == "Q_OUT" else LOC[2])
            item = dict(idK=idk + rng.normal(0, 0.1), idV=idv + rng.normal(0, 0.1), mass=0.9, argmax=word)
        dG = -4.0 if idk < 0 else 4.0
        trS = {"none": 0.0, "H": dG * R, "rand0": dG * rnd, "rand1": dG * rnd, "rand2": dG * rnd, "allG": dG,
               "koH": dG * (1 - KO), "korand0": dG, "korand1": dG, "korand2": dG, "allGc": dG * 0.6, "allT": dG * 1.1}
        trX = {kk: -v for kk, v in trS.items()}
        b = base_lp(k)
        hop2 = {"none": b.tolist(), "inj": row(b, X=-2.0), "top10": row(b, X=-2.0 * (1 - carry)), "rand10": row(b, X=-2.0),
                "all": row(b, X=-0.2)}
        rows = {"none": b.tolist(), "none2": b.tolist(), "K_S": row(b, S=-3), "K_X": row(b, X=-3), "V_S": row(b, S=-3),
                "V_X": row(b, X=-3), "inj": row(b, X=-1.0), "iso0": b.tolist(), "iso1": b.tolist(), "iso2": b.tolist(),
                "injIOI": row(b, X=-1), "moveIOI": row(b, X=-1), "injP1": row(b, X=-0.3), "iso": b.tolist()}
        rec = dict(index=i, core=core(i), ix=dict(IX), item=item, transfer={"S": trS, "X": trX}, hop2=hop2, rows=rows)
        if ioi:
            rk, rv, rkv = route

            def mrow(dm):
                return row(b, S=dm, B=0.0)        # m = l_S - l_B changes by exactly dm
            rec["route"] = {"none": b.tolist(), "K_S": mrow(-3), "K_S+ans_K": mrow(-3 * (1 - rk)), "K_S+ans_V": mrow(-3 * (1 - rv)),
                            "K_S+ans_KV": mrow(-3 * (1 - rkv)), "K_S+other_KV": mrow(-3), "inj": mrow(-2),
                            "inj+ans_K": mrow(-0.2), "inj+ans_V": mrow(-2), "inj+ans_KV": mrow(0), "inj+other_KV": mrow(-2)}
        recs.append(rec)
    return recs


def test_cells_transfer_competence_route_hop2():
    c = LN.Cell(q_recs(), "Q_OUT")
    ok, _ = c.competence()
    assert ok
    t = c.transfer()
    assert t["R"].pt == pytest.approx(0.9) and t["KO"].pt == pytest.approx(0.95) and t["Rrand"].pt == pytest.approx(0.0)
    assert t["dGc"].pt == pytest.approx(0.6 / 1.1) and t["dG"].pt == pytest.approx(-4.0)
    assert c.hop2()["carry"]["top10"].pt == pytest.approx(0.7) and c.hop2()["carry"]["all"].pt == pytest.approx(0.9)
    assert not LN.Cell(q_recs(comp=False), "Q_OUT").competence()[0]
    assert c.behaviour()["none"] == 1.0
    io = LN.Cell(q_recs(arm="INLINE", ioi=True, idv=2.0), "INLINE")
    assert io.competence()[0]
    r = io.route()["K_S"]
    assert r["r"]["K"].pt == pytest.approx(0.9) and r["r"]["V"].pt == pytest.approx(0.2) and r["r"]["KV"].pt == pytest.approx(0.95)
    assert io.route()["inj"]["r"]["KV"].pt == pytest.approx(1.0)
    assert not LN.Cell(q_recs(arm="INLINE", ioi=True, comp=False), "INLINE").competence()[0]


def diss_json(n=60, rho=-0.2, rho_nat=-0.2, comp_frac=1.0, with_gate=True, route=(0.1, 0.7)):
    S = []
    for i in range(n):
        comp = i < int(comp_frac * n)
        fmt = {}
        for f in ("P1", "POST"):
            inj = 2.0 if f == "P1" else 2.0 * rho
            nx = 4.0 if f == "P1" else 4.0 * rho_nat
            b = base_lp(6)
            r = dict(rows={"none": b.tolist(), "none2": b.tolist(), "injP1": row(b, X=inj), "injPOST": row(b, X=inj * 0.9),
                           "iso": b.tolist(), "K_X": row(b, X=nx), "K_S": b.tolist()},
                     cleanB=b.tolist() if comp else base_lp(6, "S").tolist(), cleanS=base_lp(6, "S").tolist(),
                     W={"3": 1.0, "7": 2.0} if f == "P1" else {"3": 0.9, "7": 1.8})
            if f == "POST":
                rk, rv = route
                r["route"] = {"none": b.tolist(), "inj": row(b, X=inj), "ans_K": row(b, X=inj * (1 - rk)),
                              "ans_V": row(b, X=inj * (1 - rv)), "ans_KV": b.tolist()}
            if f == "P1" and with_gate:
                r["gate"] = {"S": {"none": 0.0, "H": 3.6, "allG": 4.0}, "X": {"none": 0.0, "H": -3.6, "allG": -4.0}}
            fmt[f] = r
        S.append(dict(index=i, core=core(i), ix=dict(IX), fmt=fmt))
    return {"provenance": {}, "stories": S, "kappa_POST_P1": 0.95}


def test_diss_statistics():
    d = LN.Diss(diss_json(comp_frac=0.5))
    assert len(d.ci) == 30
    x = d.rho()
    assert x["rho"].pt == pytest.approx(-0.2) and x["rho_nat"].pt == pytest.approx(-0.2)
    assert abs(x["d"].pt) < 1e-9 and x["d6b"].pt == pytest.approx(0.9)
    assert d.omega().pt == pytest.approx(0.9)
    assert d.gate_R().pt == pytest.approx(0.9)
    r = d.route()
    assert r["VmK"].pt == pytest.approx(0.6) and r["r"]["KV"].pt == pytest.approx(1.0)
    assert LN.Diss(diss_json(with_gate=False)).gate_R() is None


# --------------------------------------------------------------------------- end to end
def write(root, st, key, obj):
    f = root / st / f"TEST_{key}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(obj))


def prov(extra=None):
    return {"git_commit": "abc", "populations": dict(SC.POP_SHA), "sets_sha256": "s", "flags_sha256": "f"} | (extra or {})


def build(root, mistral_bind=True, qwen_ioi_v=2.0, pytest_ok=True):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    lines = ["==== 2026-10-10T00:00:00Z python -m pytest tests/test_flag.py -v -rA"]
    for fn, n in SC.G0_FILES.items():
        for i in range(n):
            lines.append(f"{fn}::t{i} " + ("PASSED" if pytest_ok or i else "SKIPPED"))
    (root / "logs" / "pytest.log").write_text("\n".join(lines) + "\n")
    for k in ("qwen7", "mistral7", "qwen1.5", "qwen3b"):
        write(root, "preflight", k, {"provenance": prov(), "sha": dict(SC.POP_SHA), "overlap": {}, "ioi": {"E_idx": [], "F_idx": []}})
        write(root, "sets", k, {"provenance": prov(), "sets": {"kstar": 5, "hop_top": [], "H": []}, "sets_sha256": "s", "layers": [3],
                                "a3_inrun_overlap_H": 5})
        write(root, "fit", k, {"provenance": prov(), "pt_sha256": "f", "summary": {
            "wcos_K_KV": 0.97, "kappa_POST_P1": 0.95, "identity_share": {"share": {"3": 0.4}, "null": 0.1},
            "logit_lens": {"3": [0.05, 0.02]}, "geometry": {"c": {"3": 0.9}, "cross_cos": [[1.0]]}, "cos": {}}})
    for k in ("qwen7", "mistral7"):
        write(root, "inject", k, {"provenance": prov(), **inject_json()})
        write(root, "ablate", k, {"provenance": prov(), **ablate_json()})
        if k == "qwen7" or mistral_bind:
            write(root, "bind", k, {"provenance": prov(), **bind_json()})
        q7 = k == "qwen7"
        res = {"Q_IN": q_recs(arm="Q_IN", idk=3.0, idv=5.0), "Q_OUT": q_recs(),
               "P1": [{kk: v for kk, v in r.items() if kk != "rows"} for r in q_recs(arm="Q_IN", idk=3.0)],
               "INLINE": q_recs(arm="INLINE", ioi=True, idv=qwen_ioi_v if q7 else 2.0, carry=0.1,
                                route=(0.2, 0.5, 0.3) if q7 else (0.9, 0.2, 0.95)),
               "INLINE_CHAT": q_recs(arm="INLINE", ioi=True, idv=2.0), "AFTER": []}
        write(root, "sign", k, {"provenance": prov(), "results": res})
    write(root, "diss", "qwen1.5", {"provenance": prov(), **diss_json()})
    write(root, "diss", "qwen3b", {"provenance": prov(), **diss_json(rho=0.05, rho_nat=0.05)})
    write(root, "diss", "qwen7", {"provenance": prov(), **diss_json(rho=0.25, rho_nat=0.25, with_gate=False)})


def verdicts(text):
    out = {}
    for code in SC.LINES:
        m = [l for l in text.splitlines() if l.startswith(f"  {code} ")]
        assert m, code
        for v in ("NOT EVALUABLE", "NOT MET", "MET"):
            if f" {v} " in m[0]:
                out[code] = v
                break
    return out


def test_end_to_end(tmp_path):
    build(tmp_path, mistral_bind=False, qwen_ioi_v=-1.0)
    rc = SC.main(["--results", str(tmp_path), "--test"])
    text = (tmp_path / "STAGE8D_SCORE.txt").read_text()
    assert rc == 0, text
    v = verdicts(text)
    for code in ("J-D1", "J-D2", "J-D3", "J-D4", "J-D-ADDR", "J-D-KN", "J-D7", "J-D-SIGN-Q", "J-D8-Q", "J-D8-IOI", "J-D6a", "J-D6", "J-D6-ROUTE"):
        assert v[code] == "MET", (code, text)
    assert v["J-D5"] == "NOT EVALUABLE"            # mistral7 has no binding file: one evaluable model of two
    assert v["J-D-SIGN-IOI"] == "NOT MET"           # qwen7's IOI value read is negative
    assert v["J-D-ROUTE-IOI"] == "NOT MET"          # qwen7: r_ans(KV) 0.3 < 0.6
    assert v["J-D-HOP2"] == "NOT MET" and "[intermediate]" in text      # carry 0.7 in Q_OUT, 0.1 in INLINE
    assert "(a) MEDIATED, READ BY VALUE AT 1.5B" in text
    for sec in ("GATES", "PREDICTIONS", "REPORTED", "SUMMARY", "EXPLORATORY"):
        assert sec in text
    assert "account lines, class R:" in text and "Holm" in text


def test_gate_failures_make_lines_not_evaluable(tmp_path):
    build(tmp_path, pytest_ok=False)
    SC.main(["--results", str(tmp_path), "--test"])
    v = verdicts((tmp_path / "STAGE8D_SCORE.txt").read_text())
    assert set(v.values()) == {"NOT EVALUABLE"}


def test_d6_table_branches():
    def q(pt, lo, hi):
        return Q(pt, np.linspace(lo, hi, 101))
    R = {"J-D6": (True, {}, []), "J-D6-ROUTE": (True, {}, [])}
    assert SC.d6_table(R, {}).startswith("(a)")
    R["J-D6-ROUTE"] = (False, {}, [])
    assert SC.d6_table(R, {}).startswith("(b)")
    R["J-D6"] = (False, {}, [])
    rho = {"qwen1.5": {"rho_nat": q(-0.2, -0.3, -0.1), "rho": q(0.0, -0.1, 0.1)}}
    assert SC.d6_table(R, rho).startswith("(c)")
    rho = {"qwen1.5": {"rho_nat": q(-0.2, -0.3, -0.1), "rho": q(-0.3, -0.4, -0.2)}}
    assert SC.d6_table(R, rho).startswith("(d)")


def test_pins_equal_the_experiment():
    from experiments import stage8_flag as s8
    assert SC.POP_SHA == s8.POP_SHA and SC.SETS_SHA == s8.SETS_SHA
