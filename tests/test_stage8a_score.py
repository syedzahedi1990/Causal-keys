"""The stage-8 part-A scorer (analysis/stage8a_score.py, analysis/stage8a_parts) on synthetic results with known answers:
the two-stage cluster bootstrap (reproducible, articles then items), the interval criteria and their p-values, the
cross-model combination (>= 3 evaluable with a fresh family; IUT), every J-A line MET on data built to meet it, the NOT MET
and NOT EVALUABLE paths (a failing model, too few models, no fresh family, a failed gate, too few competent or
prior-free items, a format skipped at the deadline), Holm, the pytest gate parser and the full report; the heads
population check, the pytest step of scripts/gpu_stage8a.sh, and the script's and the runbook's statements that the code
decides (compute, network hosts, download tries, the GPU check)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import stage8a_score as sc  # noqa: E402
from stage8a_parts import common as cm  # noqa: E402
from stage8a_parts import factorial as fa  # noqa: E402
from stage8a_parts import heads as hd  # noqa: E402

FMTS = ("NOM", "OPTA", "OPTB", "MENA", "LETA", "MENB")
DEC = {"B": 11, "S": 12, "X": 13, "Z": 14, "D": 15}
# per format: (mean ID_K, mean ID_V) of the synthetic rows
KV = {"NOM": (0.2, 8.0), "OPTA": (6.0, 2.0), "OPTB": (0.1, 2.5), "MENA": (2.0, 6.0), "LETA": (8.0, 0.5), "MENB": (0.0, 7.0)}
CUE = {"NOM": {"KS_VX": "X", "KX_VS": "S", "KS_VZ": "Z", "KZ_VS": "S"},
       "OPTA": {"KS_VX": "S", "KX_VS": "X", "KS_VZ": "S", "KZ_VS": "S"},
       "LETA": {"KS_VX": "S", "KX_VS": "X", "KS_VZ": "S", "KZ_VS": "S"}}


def rec(i, f, rng, kv=None, comp=True, nP=3, sub="PERSON", hybrid=False):
    k, v = kv or KV[f]
    k, v = k + rng.normal(0, 0.5), v + rng.normal(0, 0.5)
    lp = {"ID": dict.fromkeys(DEC, -1.0) | {"B": -0.05}}   # p(dec_B | ID) 0.95 (Gate J-A-G3)
    lp["K_S"] = {"S": -1 + k, "X": -1 - k / 2} | {Y: -1.0 for Y in "BZD"}
    lp["K_X"] = {"S": -1.0, "X": -1 + k / 2} | {Y: -1.0 for Y in "BZD"}
    lp["V_S"] = {"S": -1 + v, "X": -1 - v / 2} | {Y: -1.0 for Y in "BZD"}
    lp["V_X"] = {"S": -1.0, "X": -1 + v / 2} | {Y: -1.0 for Y in "BZD"}
    lp["KV_S"] = {"S": -1 + 11 + rng.normal(0, 0.3)} | {Y: -1.0 for Y in "BXZD"}
    lp["KV_X"] = {"X": -1 + 10.0} | {Y: -1.0 for Y in "BSZD"}
    for r in ("K_Z", "V_Z", "KV_Z", "KS_VX", "KX_VS", "KS_VZ", "KZ_VS"):
        lp[r] = dict.fromkeys(DEC, -1.0)
    if f == "NOM":   # A8: continuation conjunctive, decision additive
        lp["K_S"]["S"], lp["V_S"]["S"], lp["KV_S"]["S"] = -1 + 1.0, -1 + 10.0, -1 + 11.0
    cont = {"ID": [0.0, 0.0], "K_S": [-1.5, -1.5], "V_S": [0.5, 0.5], "KV_S": [4.0, 4.0]} if f == "NOM" else \
        {"ID": [0.0, 0.0], "K_S": [0.0, 0.0], "V_S": [0.0, 0.0], "KV_S": [0.05, 0.05]}
    rows = {n: {"lp": lp[n], "mass": 0.99, "argmax": DEC["B"], "cont": cont.get(n, [0.0, 0.0])} for n in lp}
    gen = {n: {"ids": [DEC["B"], 7], "text": "", "who": "B", "g1": DEC["B"]} for n in lp}
    gen["S_run"] = {"ids": [DEC["S"], 7], "text": "", "who": "S", "g1": DEC["S"]}
    gen["KV_S"] = {"ids": [DEC["S"], 7], "text": "", "who": "S" if comp else "B", "g1": DEC["S"]}
    gen["KV_X"] = {"ids": [DEC["X"], 7], "text": "", "who": "X", "g1": DEC["X"]}
    gen["KV_Z"]["who"] = "Z"
    for r, Y in CUE.get(f, {}).items():
        gen[r]["who"] = Y
    if f == "NOM" and hybrid:
        gen["K_Z"]["who"] = "other"
    q = {"rows": {"none": {"who": "S"}, "K2": {"who": "S"}, "V2": {"who": "other" if f == "NOM" else "S"}},
         "relerr": {"K2": {"k": [0.2, 0.2]}, "V2": {"v": [0.4, 0.4]}}}
    r = dict(id=f"q{i:03d}", art=i % 20, title=f"art{i % 20:02d}", sub=sub, D_in=False, stratum="FT", P=list(range(nP)), nP=nP,
             T=100, j=0, w=[], dec=dict(DEC) if f != "LETA" else {Y: DEC[Y] for Y in "BSXD"}, c={"S": [12, 21, 22], "B": [11, 31, 32]},
             diff_S=[True, True], rows=rows, runs={"B": lp["ID"], "S": lp["KV_S"], "X": lp["KV_X"], "Z": lp["KV_Z"]}, gen=gen)
    if f in ("NOM", "OPTA", "LETA"):
        r["quant"] = q
    return r


def model_json(key, n=100, seed=0, kv=None, drop=(), skip_menb=False, n_comp=None, broken_g1=False, cb_win="S"):
    rng = np.random.default_rng(seed)
    out = {"provenance": {"model_key": key, "reduced": {}, "skipped_items": []}, "frame": " ", "formats": {}, "closed_book": {}}
    hyb = set(np.random.default_rng(seed + 100).choice(range(1, n, 2), int(0.3 * (n // 2)), replace=False).tolist())   # 30 % of PERSON
    for f in FMTS:
        if f in drop or (f == "MENB" and skip_menb):
            if f == "MENB" and skip_menb:
                out["provenance"]["reduced"]["MENB"] = "skipped: projected past the deadline"
            continue
        items = [rec(i, f, rng, (kv or {}).get(f), comp=(n_comp is None or i < n_comp), sub="PERSON" if i % 2 else "NUMBER",
                     hybrid=i in hyb) for i in range(n)]
        if broken_g1:
            for r in items:
                r["runs"]["S"] = {Y: x + 1.0 for Y, x in r["runs"]["S"].items()}
        rows = list(items[0]["rows"]) if f in ("NOM", "OPTA", "LETA") else ["ID", "K_S", "V_S", "KV_S", "K_X", "V_X", "KV_X"]
        out["formats"][f] = {"items": items, "skipped": [], "rows": rows, "reduced": False}
    for i in range(n):
        out["closed_book"][f"q{i:03d}"] = {"CBOPT": {"win": cb_win}, "CBLET": {"win": cb_win}}
    out["provenance"]["population_sha256"] = None
    return out


def M(key, **kw):
    return fa.Model(key, model_json(key, **kw), None, test=False, a_g0=True)


def verdicts(models):
    res = {}
    for code, fn in fa.LINE_FNS.items():
        per = {k: fn(m) for k, m in models.items()}
        res[code] = (cm.comb_models({k: r.ok for k, r in per.items()}), per)
    return res


@pytest.fixture(scope="module")
def four():
    return {k: M(k, seed=s) for s, k in enumerate(("llama8", "gemma9", "qwen7", "mistral7"))}


def test_bootstrap_is_two_stage_and_reproducible():
    arts = [f"a{i % 7}" for i in range(40)] + ["a0"] * 5
    b1, b2 = cm.Boot(arts, nb=2000), cm.Boot(arts, nb=2000)
    assert np.array_equal(b1.W, b2.W)
    sizes = {a: arts.count(a) for a in set(arts)}
    for w in b1.W[:200]:   # each resample = a multiset of whole articles' sizes; items only within their articles
        per_art = {a: sum(w[i] for i, x in enumerate(arts) if x == a) for a in sizes}
        assert all(per_art[a] % sizes[a] == 0 for a in sizes)
        assert sum(per_art[a] // sizes[a] for a in sizes) == len(sizes)
    x = np.arange(45, dtype=float)
    e = cm.est(arts, cm.mean, x)
    assert abs(e.pt - x.mean()) < 1e-12 and e.lo < e.pt < e.hi and e.drop == 0
    e2 = cm.est(arts, cm.mean, x)
    assert (e.lo, e.hi) == (e2.lo, e2.hi)
    c = cm.lower(e, e.pt + 100, "x")
    assert not c.passed and c.tests[0][1] is False and c.tests[0][2] == {"est": e.pt, "se": e.se, "bound": e.pt + 100, "direction": ">"}
    c = cm.lower(e, -100, "x")
    assert c.passed and c.tests[0][1] is True
    assert abs(e.se - float(np.std(e.bs, ddof=1))) < 1e-12 and e.se > 0
    c = cm.upper(e, 100, "x")
    assert c.passed and c.tests[0][2]["direction"] == "<"
    c = cm.inside(cm.est(arts, cm.mean, np.zeros(45)), -0.1, 0.1, "z")
    assert c.passed and [t[2]["direction"] for t in c.tests] == [">", "<"] and [t[2]["bound"] for t in c.tests] == [-0.1, 0.1]
    assert cm.point("x", True).tests == []


def test_combination_rules_and_holm():
    assert cm.comb_models({"llama8": True, "qwen7": True, "mistral7": True}) is True
    assert cm.comb_models({"qwen7": True, "mistral7": True, "x": True}) is None          # no fresh family
    assert cm.comb_models({"llama8": True, "qwen7": True, "mistral7": None}) is None      # two evaluable
    assert cm.comb_models({"llama8": True, "qwen7": False, "mistral7": None}) is False
    assert cm.comb_both({"qwen7": True, "mistral7": True}) is True and cm.comb_both({"qwen7": True}) is None
    assert cm.comb_both({"qwen7": None, "mistral7": False}) is False


def test_classes_follow_the_recorded_priors():
    """Entry J, G4 (decision D1): L needs a prior >= 0.9, M >= 0.8, R is every line below 0.8."""
    for code, (cls, prior, _) in cm.LINES.items():
        assert cls in ("L", "M", "R", "D"), code
        assert (cls == "R") == (prior < 0.8) or cls == "D", (code, cls, prior)
        assert cls != "L" or prior >= 0.9, (code, cls, prior)
        assert cls != "M" or prior >= 0.8, (code, cls, prior)
    assert set(fa.LINE_FNS) | set(hd.HEAD_FNS) == set(cm.LINES)


def test_every_line_met_on_data_built_to_meet_it(four):
    res = verdicts(four)
    for code, (v, per) in res.items():
        assert v is True, (code, {k: (r.ok, r.why, r.txt) for k, r in per.items()})
    m = four["llama8"]
    k, v = m.idkv("OPTA", m.C("OPTA"))
    assert abs(cm.sid(k.mean(), v.mean()) - 0.75) < 0.05
    assert m.gates["NOM"]["G1"] and m.gates["NOM"]["n_comp"] == 100


def test_not_met_and_not_evaluable_paths(four):
    bad = dict(four, gemma9=M("gemma9", seed=9, kv={"OPTA": (1.0, 6.0)}))
    r = fa.a1(bad["gemma9"])
    assert r.ok is False and cm.comb_models({k: fa.a1(m).ok for k, m in bad.items()}) is False
    two = {k: four[k] for k in ("llama8", "qwen7")}
    assert cm.comb_models({k: fa.a1(m).ok for k, m in two.items()}) is None
    nofresh = {k: four[k] for k in ("qwen7", "mistral7")} | {"olmo7": M("olmo7", seed=7)}
    assert cm.comb_models({k: fa.a1(m).ok for k, m in nofresh.items()}) is None
    g1 = M("llama8", seed=3, broken_g1=True)
    assert not g1.gates["OPTA"]["G1"] and fa.a1(g1).ok is None and "J-A-G1 OPTA" in fa.a1(g1).why
    few = M("llama8", seed=4, n_comp=60)
    assert few.gates["OPTA"]["n_comp"] == 60 and fa.a1(few).ok is None and "J-A-G2" in fa.a1(few).why
    menb = M("llama8", seed=5, skip_menb=True)
    r = fa.a5b(menb)
    assert r.ok is None and "MENB not run" in r.why and fa.a5(menb).ok is True
    prior = M("llama8", seed=6, cb_win="B")          # every item has the closed-book prior on B: no prior-free items
    assert fa.a7(prior).ok is None and "prior-free n 0" in fa.a7(prior).why
    assert fa.a7(four["llama8"]).ok is True
    tinyg4 = M("llama8", seed=8, kv={"OPTA": (0.6, 0.6)})   # s_ID's own denominator collapses in OPTA: NOT MET
    assert fa.a1(tinyg4).ok is False and "J-A-G4" in fa.a1(tinyg4).txt and fa.a1(tinyg4).undef
    assert fa.a2(tinyg4).ok is True and fa.a3(tinyg4).ok is False and fa.a4(tinyg4).ok is not None
    test_m = fa.Model("llama8", model_json("llama8", n=5, seed=1), None, test=True)
    assert test_m.gates["OPTA"]["G2"] and fa.a8d(test_m).ok is not None


def test_a8_and_hybrid_values(four):
    m = four["qwen7"]
    ids = fa.a8_pop(m)
    c, dd = fa.a8_measures(m, ids)
    Ic = fa.inter(c["NOM", "K"].mean(), c["NOM", "V"].mean(), c["NOM", "KV"].mean())
    assert abs(Ic - (8 - (-3) - 1) / 8) < 1e-9          # the synthetic continuation: K -3, V +1, KV +8 nats
    r = fa.a8d(m)
    assert r.ok is True and "h(K_Z, NOM) +0.300" in r.txt


def heads_json(drop_n=0.5, rand_drop=0.0, n=80, nab=100):
    rng = np.random.default_rng(1)
    KS = [1, 2, 5, 10, 20, 40, 80]
    ev = []
    for i in range(n):
        base, full = 0.0, 10 + rng.normal(0, 1)
        allG = 0.9 * full
        cv = {}
        for s, (fr, fk) in {"N": (0.9, 0.95), "T": (0.6, 0.6), "rand0": (0.0, 0.0), "rand1": (0.02, 0.01), "rand2": (0.01, 0.02)}.items():
            suff = [allG * fr * (j + 1) / len(KS) if KS[j] < 40 else allG * fr for j in range(len(KS))]
            ko = [allG * (1 - fk) if KS[j] >= 40 else allG for j in range(len(KS))]
            cv[s] = {"suff": suff, "none": base, "allG": allG, "allT": full, "ko": ko, "ko_none": base, "ko_allG": allG}
        ev.append({"id": f"q{i:03d}", "title": f"art{i % 20:02d}", "mB": base, "mF": full, "curves": cv,
                   "layer": {"m": [0.0] * 4, "none": 0.0, "allG": allG}})
    abl = {}
    for f in ("OPTA", "NOM"):
        recs = []
        for i in range(nab):
            conds = {c: {"chain": True, "dec": True, "cont": True} for c in ("none", "N", "T", "rand0", "rand1", "rand2", "C")}
            if f == "OPTA" and i < drop_n * nab:
                conds["N"] = {"chain": False, "dec": False, "cont": False}
            if f == "OPTA" and i < rand_drop * nab:
                conds["rand1"] = {"chain": False, "dec": False, "cont": False}
            recs.append({"id": f"q{i:03d}", "title": f"art{i % 20:02d}", "conds": conds,
                         "clean": {c: {"chain": True, "lp": {"B": -0.1, "S": -5.0, "X": -5.0, "D": -5.0}} for c in conds}})
        abl[f] = recs
    return {"provenance": {"KS": KS, "kstar": 40, "n_heads": 784}, "eval": ev, "ablate": abl,
            "sets": {"N": [[l, h] for l in range(20) for h in (0, 1)], "T": [[l, h] for l in range(20) for h in (1, 2)], "C": []}}


def test_head_lines(four):
    h = hd.Heads("qwen7", heads_json(), four["qwen7"])
    assert h.gate1()[0] and h.gate2()[0]
    for code, fn in hd.HEAD_FNS.items():
        r = fn(h)
        assert r.ok is True, (code, r.why, r.txt)
    assert abs(h.R("N", h.ik).pt - 0.9) < 1e-6 and abs(h.KO("T", h.ik).pt - 0.6) < 1e-6
    assert hd.ha3a(hd.Heads("qwen7", heads_json(drop_n=0.1), four["qwen7"])).ok is False
    assert hd.ha3a(hd.Heads("qwen7", heads_json(rand_drop=0.3), four["qwen7"])).ok is False
    assert hd.ha3a(hd.Heads("qwen7", heads_json(), None)).ok is None
    assert hd.overlap(h)[0] == 20


def test_heads_ratios_use_each_sets_own_batch():
    """R(k) and KO(k) of a set divide by that set's own all_G - none (knockout: all_G,KO - none_KO), not by N*'s d_G;
    d_full is m(full K_S clamp) - m(clean), eval's mF - mB. The docstrings of the scorer part and of the experiment say so."""
    J = heads_json()
    for e in J["eval"]:
        c = e["curves"]["T"]
        c.update(none=1.0, allG=c["allG"] / 2, ko_none=2.0, ko_allG=c["ko_allG"] / 2)
    h = hd.Heads("qwen7", J)
    i, T = h.ik, [e["curves"]["T"] for e in J["eval"]]
    r = np.mean([c["suff"][i] - c["none"] for c in T]) / np.mean([c["allG"] - c["none"] for c in T])
    ko = 1 - np.mean([c["ko"][i] - c["ko_none"] for c in T]) / np.mean([c["ko_allG"] - c["ko_none"] for c in T])
    assert abs(h.R("T", i).pt - r) < 1e-9 and abs(h.KO("T", i).pt - ko) < 1e-9
    dG = np.mean([e["curves"]["N"]["allG"] - e["curves"]["N"]["none"] for e in J["eval"]])
    assert abs(r - np.mean([c["suff"][i] - c["none"] for c in T]) / dG) > 0.1
    assert f"d_full {cm.est(h.arts, cm.mean, h.v(lambda e: e['mF'] - e['mB']))} " in h.gate2()[1]
    doc = " ".join(hd.__doc__.split())
    assert "d_full = mean[m(full K_S clamp) - m(clean)]" in doc and "m(ID)" not in doc
    assert "within that set's own sufficiency and knockout batches" in doc
    ex = " ".join((ROOT / "experiments" / "natural_heads.py").read_text().split('"""')[1].split())
    assert "d_full = mean[m(full K_S clamp) - m(clean)]" in ex and "the denominators of that set's R(k) and KO(k)" in ex


def test_heads_population_takes_the_first_valid_items():
    """The heads population check: the ranking items are the first min(60, n) of the n R items valid in NOM and OPTA
    under the model's frame, the evaluation items the first min(80, n) of the n valid E items (n from the heads file's
    provenance n_valid; exactly 60 and 80 without it). Fewer valid items under a frame other than ' ' is no MISMATCH."""
    items = json.load(open(sc.ITEMS))
    Rr = [i["id"] for i in sorted(items, key=lambda i: i["rank"]) if i["split"] == "R"]
    Ee = [i["id"] for i in items if i["split"] == "E"]

    def check(n_rank, n_eval, n_valid=None):
        J = heads_json(n=n_eval)
        for e, i in zip(J["eval"], Ee):
            e["id"] = i
        J["rank"] = {"items": [{"id": i} for i in Rr[:n_rank]]}
        if n_valid:
            J["provenance"]["n_valid"] = dict(zip("RE", n_valid))
        lines = []
        ok = sc.population({}, {}, {"qwen7": hd.Heads("qwen7", J)}, lines.append, test=False)
        assert ("MISMATCH" in lines[-1]) is not ok
        return ok

    assert check(60, 80, (71, 185)) and check(60, 80)              # the usual case, with and without the record
    assert check(55, 80, (55, 185)) and check(60, 70, (66, 70))    # fewer than 60 valid R items or 80 valid E items
    assert not check(55, 80) and not check(50, 80, (55, 185)) and not check(60, 70, (66, 80))


def test_pytest_gate_parser(tmp_path):
    (tmp_path / "logs").mkdir()
    good = "".join(f"tests/test_natural_clamp.py::t{i} PASSED\n" for i in range(14)) + "".join(f"tests/test_kvquant.py::k{i} PASSED\n" for i in range(4))
    (tmp_path / "logs" / "pytest.log").write_text("==== test session starts ====\n" + good.replace("t0 PASSED", "t0 FAILED")
                                                  + "==== test session starts ====\n" + good)
    lines = []
    assert sc.gate_tests(tmp_path, sc.A_G0, "J-A-G0", lines.append) is True
    (tmp_path / "logs" / "pytest.log").write_text("==== test session starts ====\n" + good.replace("t3 PASSED", "t3 SKIPPED"))
    assert sc.gate_tests(tmp_path, sc.A_G0, "J-A-G0", lines.append) is False
    assert sc.gate_tests(tmp_path, sc.HA_G0, "J-A-HA-G0", lines.append) is None


def test_pytest_step_runs_the_gates_and_the_shared_tests(tmp_path):
    """scripts/gpu_stage8a.sh: its s8_pytest call runs J-A-G0's and J-A-HA-G0's files, the scorer's tests and the tests
    every part runs before any model (tests/test_stage8_populations.py, rule G6; tests/test_stage8_holm.py, the shared
    Holm helper); the gate parser reads J-A-G0 and J-A-HA-G0 from a log of that whole session."""
    sh = (ROOT / "scripts" / "gpu_stage8a.sh").read_text()
    call = re.search(r"^s8_pytest ((?:.*\\\n)*.*)$", sh, re.M).group(1).replace("\\\n", " ").split()
    shared = {"tests/test_stage8_populations.py", "tests/test_stage8_holm.py"}
    assert set(sc.A_G0) | set(sc.HA_G0) | {"tests/test_stage8a_score.py"} | shared <= set(call), call
    assert len(call) == len(set(call)) and all((ROOT / f).is_file() for f in call)
    need = sc.A_G0 | sc.HA_G0
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "pytest.log").write_text("==== test session starts ====\n" + "".join(
        f"{f}::test_{i} PASSED\n" for f in call for i in range(need.get(f, 3))))
    lines = []
    assert sc.gate_tests(tmp_path, sc.A_G0, "J-A-G0", lines.append) is True
    assert sc.gate_tests(tmp_path, sc.HA_G0, "J-A-HA-G0", lines.append) is True


def test_full_report(tmp_path, four):
    for k in ("llama8", "gemma9", "qwen7", "mistral7"):
        J = model_json(k, seed={"llama8": 11, "gemma9": 12, "qwen7": 13, "mistral7": 14}[k])
        ids = sorted(r["id"] for r in J["formats"]["NOM"]["items"])
        J["provenance"]["population_sha256"] = __import__("hashlib").sha256("\n".join(ids).encode()).hexdigest()
        (tmp_path / "factorial").mkdir(exist_ok=True)
        (tmp_path / "factorial" / f"TEST_{k}.json").write_text(json.dumps(J))
    (tmp_path / "heads").mkdir()
    (tmp_path / "heads" / "TEST_qwen7.json").write_text(json.dumps(heads_json()))
    rc = sc.main(["--results", str(tmp_path), "--test"])
    text = (tmp_path / "STAGE8A_SCORE.txt").read_text()
    assert rc == 0, text[-3000:]
    for sec in ("PROVENANCE", "POPULATION", "GATES", "PREDICTIONS", "REPORTED", "SUMMARY", "EXPLORATORY"):
        assert sec in text
    assert "J-A1      [M, prior 0.80]" in text and "-> MET" in text
    assert "J-A-HA1   [R, prior 0.75] sparse natural readers (N*) -> NOT EVALUABLE" in text   # one head model only
    assert "account lines, class R:" in text and "Holm sensitivity" in text and "J-A6a: no component decision changes" in text


def test_holm_family_is_the_r_lines_interval_components(four):
    """D2: the family holds the interval components of the R-class lines only (each model with a verdict; an 'inside'
    criterion gives two), each with its estimate, bootstrap SE, bound and direction; a component the interval rule
    passes on a wide interval but Holm does not reject is reported, and the verdict change is flagged."""
    res = verdicts(four)
    comps, where = sc.holm_family(res)
    assert comps and {c["line"] for c in comps} == {c for c in res if cm.LINES[c][0] == "R"}
    assert all(set(c) >= {"line", "name", "est", "se", "bound", "direction"} and c["direction"] in "<>" for c in comps)
    n_a6a = sum(c["line"] == "J-A6a" for c in comps)
    assert n_a6a == 4 * 2        # two lower-bound tests in each of four models
    lines = []
    sc.summary(res, lines.append)
    text = "\n".join(lines)
    assert "account lines, class L: 1 lines" in text and "account lines, class M: 5 lines" in text
    assert "account lines, class R: 10 lines: 10 MET" in text    # the head lines are not in this synthetic result
    assert "observed 10 against expected 4.65" in text and "Brier" in text
    assert "J-A6a: no component decision changes under Holm; verdict unchanged (MET)" in text
    # a model whose J-A6a key-source lower bound sits just above 0.5: passed by the interval, not by a strict Holm
    r = res["J-A6a"][1]["llama8"]
    t = r.comps[1].tests[0]
    r.comps[1].tests[0] = (t[0], True, dict(t[2], est=0.52, se=0.0105))
    lines = []
    sc.summary(res, lines.append)
    text = "\n".join(lines)
    assert "J-A6a: 1 component decision(s) change under Holm: llama8: key-source rate > 0.5" in text
    assert "verdict MET -> NOT MET under Holm" in text


def test_compute_estimate_agrees_with_the_runbook():
    """The script header's compute estimate is the runbook's Part A row (the entry's Compute): about 3.3 GPU-h for the
    core, about 4.1 h with every exploratory pass."""
    sh = (ROOT / "scripts" / "gpu_stage8a.sh").read_text()
    core, total = re.search(r"Compute estimate \(entry, Compute\): about ([\d.]+) GPU-h for the core, about ([\d.]+) h", sh).groups()
    rb = (ROOT / "docs" / "GPU_RUNBOOK.md").read_text().splitlines()
    row = next(x for x in rb if x.startswith("| A | `scripts/gpu_stage8a.sh`"))
    assert f"about {core} h core, {total} h in all" in row and (core, total) == ("3.3", "4.1")


def test_runbook_stage8_names_what_the_code_does():
    """docs/GPU_RUNBOOK.md, Stage 8: every host the stage-8 code downloads from (the GPU scripts' curl lines, the
    fetcher's Hub URL, Part C's release) is named under What to rent; a download is tried up to three times per source
    (the fetcher's default tries); the GPU check reads the first device against 75 GiB (MINGIB)."""
    rb = (ROOT / "docs" / "GPU_RUNBOOK.md").read_text()
    s8 = rb[rb.index("## Stage 8"):]
    rent = next(x for x in s8.split("\n\n") if x.startswith("**What to rent.**"))
    fv = (ROOT / "scripts" / "fetch_verified.py").read_text()
    common = (ROOT / "scripts" / "stage8_common.sh").read_text()
    code = "".join((ROOT / "scripts" / f"gpu_stage8{x}.sh").read_text() for x in "abcd") + fv
    code += re.search(r'^RELEASE_URL = "(.*)"$', (ROOT / "ckeys" / "causaltom.py").read_text(), re.M).group(1) + "/"
    hosts = set(re.findall(r"https://([a-z0-9.-]+)/", code))
    assert {"huggingface.co", "rajpurkar.github.io", "github.com", "anonymous.4open.science"} <= hosts
    assert all(h in rent for h in hosts), sorted(h for h in hosts if h not in rent)
    assert re.search(r"def fetch\(.*?tries=3\b", fv, re.S) and "tried up to three times per source" in s8
    assert "get_device_properties(0)" in common and "MINGIB=${MINGIB:-75}" in common
    assert "the first GPU has less than 75 GiB" in s8 and "no GPU with ≥ 75 GiB is visible" not in s8
