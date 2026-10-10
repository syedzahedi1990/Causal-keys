"""analysis/stage8b_score.py (preregistration J, part B; gate JB-G0 items 8 and 9) on synthetic results with known
answers: every line MET; NOT MET paths (per-family sentences); NOT EVALUABLE paths (competence, anchor, too few
evaluable models, a model-level gate, the deadline); the coverage switch to the behavioural counterpart and the floor;
own-arm undefinedness (NOT MET) against anchor undefinedness (NOT EVALUABLE); the N4 fallback slot; the pytest gate;
the bootstrap's determinism and cluster structure; the combination rules and Holm.

The generator writes, per item, six-word vectors whose identity effects are known: rows K_S / K_X move the S / X word by
the item's key effect K, rows V_S / V_X by its value effect V, the KV rows by K + V; every row and the clean B run give
the base word a bonus of BONUS nats, so the E-argmax (and the generated answer, which the generator takes from it) names
S or X in a key row exactly when K > BONUS; base levels set the minimum row mass of a cell to its target coverage.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import stage8b_score as S  # noqa: E402
from stage8b_parts import lines as ln  # noqa: E402
from stage8b_parts import stats as st  # noqa: E402

from ckeys import fresh  # noqa: E402
from experiments.format_factorial import LABEL, row_specs  # noqa: E402

BONUS, NL = 3.0, 24
ROWS = [f"{LABEL[(k, v)]}@{l0}" for k, v, l0 in row_specs(NL)]
GEN = ("B", "S", "X", "ID@0", "K_S@0", "K_X@0", "V_S@0", "V_X@0")
# per arm: (mean ID_K, mean ID_V) in nats under E (L and sigma equal unless overridden)
F_ARMS = {"AFTER": (12.0, 2.0), "BEFORE": (-0.5, 10.0), "NONE": (0.4, 10.0), "POST": (4.0, 6.0), "PRE": (0.0, 10.0),
          "POST-NULL": (0.3, 10.0)}
S0_ARMS = {"P1": (12.0, 2.0), "AFTER": (12.0, 2.0), "BEFORE": (-0.5, 10.0), "POST": (4.0, 6.0), "PRE": (0.0, 10.0),
           "NONE": (0.4, 10.0)}


def vec(base_mass, bonus_word, words):
    b = np.log(base_mass / (5 + np.exp(BONUS)))
    v = np.full(6, b)
    v[words.index(bonus_word)] += BONUS
    return v


def make_items(pop, arms, rng, cov=None, Lmass=None, acc=None, other=None, eff=None, idnoise=0.0, n=None):
    """Synthetic eval records. cov/Lmass/acc/other: {arm: value}; eff: {arm: (K, V)} overriding the defaults."""
    out = []
    items = fresh.population(pop)[:n] if n else fresh.population(pop)
    for arm, (K0, V0) in arms.items():
        K0, V0 = (eff or {}).get(arm, (K0, V0))
        for it in items:
            W, tr = list(it.words), it.track
            k, v = K0 + rng.normal(0, 1.0), V0 + rng.normal(0, 1.0)
            rows, clean = {}, {}
            masses = {"E": (cov or {}).get(arm, 0.95), "sigma": 0.9, "L": (Lmass or {}).get(arm, 0.95)}
            for sg, ms in masses.items():
                for r in ROWS:
                    x = vec(ms, tr["B"], W)
                    if r == "K_S@0":
                        x[W.index(tr["S"])] += k
                    elif r == "K_X@0":
                        x[W.index(tr["X"])] += k
                    elif r == "V_S@0":
                        x[W.index(tr["S"])] += v
                    elif r == "V_X@0":
                        x[W.index(tr["X"])] += v
                    elif r == "KV_S@0":
                        x[W.index(tr["S"])] += k + v
                    elif r == "KV_X@0":
                        x[W.index(tr["X"])] += k + v
                    elif r == "ID@0" and idnoise:
                        x[W.index(tr["S"])] += idnoise
                    rows.setdefault(r, {"gap": 0.01})[sg] = x.tolist()
                for run in "BSX":
                    clean.setdefault(run, {"gap": 0.01, "top": []})[sg] = vec(ms, tr[run], W).tolist()
            ans = {}
            for g in GEN:
                e = clean[g]["E"] if g in "BSX" else rows[g]["E"]
                ans[g] = W[int(np.argmax(e))]
            if rng.random() > (acc or {}).get(arm, 1.0):
                ans["B"] = tr["S"]
            if rng.random() < (other or {}).get(arm, 0.0):
                ans["B"] = "other"
            rec = {**it.meta(), "arm": arm, "pos": 40, "len": 120, "n_layers": NL, "scorer": "trie", "nodes": 37,
                   "rows": rows, "clean": clean, "ans": ans, "gen": {g: " " + a for g, a in ans.items()}}
            if pop == "S0":
                tid = {t: W.index(w) for t, w in tr.items()}
                rec["plain"] = {"m": {r: {"m": rows[r]["L"][tid["S"]] - rows[r]["L"][tid["B"]],
                                          "lp": {t: rows[r]["L"][j] for t, j in tid.items()}} for r in ROWS},
                                "clean": {}}
            out.append(rec)
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_model(root, key, rng, test=False, pops=("F",), scorer="trie", g3_scorer="trie", tok_ok=True, **kw):
    tag = ("TEST_" if test else "") + key
    for d in ("tokcheck", "frames", "g3", "eval", "verified"):
        (root / d).mkdir(parents=True, exist_ok=True)
    prov0 = {"git_commit": "abc", "dtype": "torch.bfloat16", "attn_implementation": "eager" if key.startswith("gemma") else "sdpa",
             "model": key, "model_key": key, "test_mode": test}
    (root / "tokcheck" / f"{tag}.json").write_text(json.dumps({"pass": tok_ok, "fails": [] if tok_ok else ["x"], "provenance": prov0}))
    (root / "frames" / f"{tag}.json").write_text(json.dumps({"frames": [], "counts": {}, "provenance": prov0}))
    (root / "g3" / f"{tag}.json").write_text(json.dumps({"pass": g3_scorer == "trie", "scorer": g3_scorer, "provenance": prov0,
                                                        "arms": {"NONE": {"stat": {"n": 30, "abs_ds": 0.001, "mean_abs_dL": 0.01, "tol_dL": 0.5}}}}))
    if not test:
        (root / "verified" / f"{key}.json").write_text(json.dumps({"repo": key, "revision": "r", "files": {}}))
    for pop in pops:
        arms = S0_ARMS if pop == "S0" else F_ARMS
        res = make_items(pop, arms, rng, **kw.get(pop, {}))
        P = prov0 | {"population": pop, "population_sha256": fresh.POP_SHA256[pop], "arms": list(arms), "arms_not_run": [],
                     "skipped_items": [], "frames_sha256": sha(root / "frames" / f"{tag}.json"),
                     "g3_sha256": sha(root / "g3" / f"{tag}.json"), "scorer": scorer}
        if pop == "F" and kw.get("not_run"):
            P["arms_not_run"] = kw["not_run"]
            P["arms"] = list(arms)
            res = [r for r in res if r["arm"] not in kw["not_run"]]
        (root / "eval" / f"{tag}_{pop}.json").write_text(json.dumps({"provenance": P, "results": res}))


def write_jb8(root, mass=0.9, shift=0.0):
    (root / "jb8").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    LOCS = list(fresh.LEX1)
    res = []
    for i in range(60):
        b, s = rng.choice(6, 2, replace=False)
        t = [j for j in range(6) if j not in (b, s)][0]
        core = {"id": f"c{i}", "base": LOCS[b], "source": LOCS[s], "target": LOCS[t]}
        for arm in ("P1", "NONE", "BEFORE", "POST", "LETTER"):
            runs = {}

            def put(name, m):
                cand = np.full(6, -5.0)
                cand[t] += m / 2
                cand[s] -= m / 2
                e = cand.copy()
                e[t] += shift * m / 2
                runs[name] = {"cand": cand.tolist(), "argmax": 0, "E": e.tolist(), "sig": cand.tolist(),
                              "mass": {"L": 0.5, "sig": 0.8, "E": mass}}
            put("S", -6 + rng.normal()), put("T", 6 + rng.normal()), put("B", 0.0)
            for sd in (101, 102, 103):
                put(f"m3_{sd}", 4 + rng.normal()), put(f"pca_{sd}", -4 + rng.normal())
                put(f"addition_{sd}", 0 + rng.normal()), put(f"addition_v_{sd}", 1 + rng.normal())
            res.append({"core": core, "arm": arm, "runs": runs})
    (root / "jb8" / "mistral.json").write_text(json.dumps({"provenance": {"score": "E", "frames_sha256": None}, "results": res}))


def pytest_log(root, fail=None, skip=None):
    (root / "logs").mkdir(parents=True, exist_ok=True)
    lines = ["============================= test session starts =============================="]
    for f, n in S.G0_FILES.items():
        for i in range(n):
            tid = f"{f}::test_{i}"
            w = "FAILED" if fail == tid else "SKIPPED" if skip == tid else "PASSED"
            lines.append(f"{tid} {w}")
    (root / "logs" / "pytest.log").write_text("\n".join(lines) + "\n")


def full(root, seed=0, jb8=True, per=None, skip_models=(), **common):
    rng = np.random.default_rng(seed)
    pytest_log(root)
    for k in ln.P4:
        if k not in skip_models:
            write_model(root, k, rng, pops=("F", "S0"), **{"S0": {"Lmass": {"POST": 0.2, "PRE": 0.2, "NONE": 0.2}}},
                        **common, **(per or {}).get(k, {}))
    for k in ln.N4 + (ln.SMALL,):
        if k not in skip_models:
            kw = dict(common, **(per or {}).get(k, {}))
            if k == ln.SMALL and "F" not in kw:
                kw["F"] = {"eff": {"POST": (0.2, 6.0)}}
            write_model(root, k, rng, **kw)
    if jb8:
        write_jb8(root)


def score(root, test=False):
    rc = S.main(["--results", str(root)] + (["--test"] if test else []))
    txt = (root / "STAGE8B_SCORE.txt").read_text()
    return rc, txt


def verdict(txt, code):
    for l in txt.splitlines():
        if l.startswith(f"  {code} ") or l.startswith(f"  {code:13s} "):
            return l.split("-> ")[-1].split(" (")[0]
    raise KeyError(code)


def per_model(txt, code, key):
    on = False
    for l in txt.splitlines():
        if l.startswith("  J-B"):
            on = l.startswith(f"  {code} ")
        elif on and l.strip().startswith(f"{key}:"):
            return l
    raise KeyError((code, key))


@pytest.fixture(scope="module")
def allmet(tmp_path_factory):
    root = tmp_path_factory.mktemp("allmet")
    full(root)
    return root, score(root)


def test_every_line_met(allmet):
    root, (rc, txt) = allmet
    assert rc in (0, 2), txt[-3000:]
    for code in ln.ORDER:
        assert verdict(txt, code) == "MET", (code, "\n".join(l for l in txt.splitlines() if code in l)[:3000])
    assert "account lines, class L: 3 lines: 3 MET" in txt and "account lines, class M: 7 lines: 7 MET" in txt
    assert "account lines, class R: 6 lines: 6 MET" in txt and "measurement-validity lines, class R: 5 lines: 5 MET" in txt
    for f in ("tab_fresh.tex", "tab_mass.tex", "tab_invariance.tex"):
        assert (root / f).read_text().startswith(r"\begin{tabular}")
    assert "Holm sensitivity" in txt and "J-B-G0  FP32 exactness" in txt and "-> MET" in txt.split("J-B-G0  FP32")[1][:2000]


def test_not_met_paths_and_family_sentences(tmp_path):
    bad = {"F": {"eff": {"POST": (0.3, 6.0)}}}
    full(tmp_path, per={"llama8": bad, "gemma9": bad, "qwen14": {"F": {"eff": {"BEFORE": (3.0, 10.0)}}}})
    rc, txt = score(tmp_path)
    assert verdict(txt, "J-B3-N4") == "NOT MET" and verdict(txt, "J-B3-P4f") == "MET"
    assert verdict(txt, "J-B2-P4f") == "NOT MET" and "NOT MET" in per_model(txt, "J-B2-P4f", "qwen14")
    assert verdict(txt, "J-B-NULL-N4") == "NOT MET"           # delta(POST - POST-NULL) has no positive read in two families
    fam = [l for l in txt.splitlines() if l.strip().startswith("J-B3-N4:")][0]
    assert "exactly two (llama8, gemma9)" in fam
    assert verdict(txt, "J-B-LB-N4") == "MET"


def test_not_evaluable_paths(tmp_path):
    inc = {"F": {"acc": {"NONE": 0.5}}}
    full(tmp_path, jb8=False, per={"phi4": inc, "falcon7": inc, "olmo7": {"F": {"eff": {"AFTER": (0.5, 12.0)}}},
                                   "mistral7": {"g3_scorer": "cached"}}, skip_models=("gemma9",))
    rc, txt = score(tmp_path)
    for code in ("J-B1-N4", "J-B4-N4", "J-B5-N4"):
        assert verdict(txt, code) == "NOT EVALUABLE"            # llama8 only: competence fails in two, gemma9 absent
    assert "competence" in per_model(txt, "J-B1-N4", "phi4")
    assert "J-B-G3" in per_model(txt, "J-B1-P4f", "mistral7")  # evaluated with the trie although G3 chose the fallback
    assert "anchor" in per_model(txt, "J-B3-P4f", "olmo7") and "NOT EVALUABLE" in per_model(txt, "J-B3-P4f", "olmo7")
    assert verdict(txt, "J-B3-P4f") == "NOT EVALUABLE"         # qwen7, qwen14 only
    assert verdict(txt, "J-B8") == "NOT EVALUABLE" and "not run" in per_model(txt, "J-B8", "mistral24")
    assert verdict(txt, "J-B6c") == "NOT EVALUABLE"


def test_coverage_beta_counterpart_and_floor(tmp_path):
    full(tmp_path, per={"llama8": {"F": {"cov": {"POST": 0.6}}}, "gemma9": {"F": {"cov": {"BEFORE": 0.5}}},
                        "falcon7": {"F": {"idnoise": 2.0}}})
    rc, txt = score(tmp_path)
    l3 = "\n".join(txt.splitlines())
    assert "behavioural counterpart: coverage below 0.8 (POST 0.60)" in l3
    assert "r^beta(POST)" in l3
    assert "no behavioural counterpart" in per_model(txt, "J-B-LB-N4", "gemma9") or "NOT EVALUABLE" in per_model(txt, "J-B-LB-N4", "gemma9")
    assert "BF16 batch floor" in per_model(txt, "J-B1-N4", "falcon7")
    assert verdict(txt, "J-B7") == "NOT MET"                    # llama8 and gemma9 cells below 0.8 coverage -> 2 of 4


def test_own_undefined_is_not_met_and_anchor_is_not_evaluable():
    class Fake:
        pass
    b = st.boot_of(None, 50)
    rng = np.random.default_rng(1)
    k, v = rng.normal(0.1, 0.01, 50), rng.normal(0.1, 0.01, 50)
    ka, va = rng.normal(10, 1, 50), rng.normal(2, 1, 50)
    s = st.est(b, ln._s, k, v, ka, va)                        # D = 0.2 < 0.2 x 12: undefined by the arm itself
    c = st.lower(s, 0.1, 0.95, "s")
    assert not c.passed and c.ne is None and "not met" in c.label
    r = st.est(b, lambda x, a, y: (x / a, True, ln._anchor(a, y)), k, rng.normal(0.5, 0.1, 50), rng.normal(10, 1, 50))
    c = st.upper(r, 0.1, 0.95, "r")
    assert c.ne and "anchor" in c.ne
    assert st.decide([c, st.Comp("x", True)]).ok is None and st.decide([st.Comp("x", False), st.Comp("y", True)]).ok is False


def test_fallback_slot_and_test_mode(tmp_path):
    full(tmp_path, jb8=False, skip_models=("llama8",))
    write_model(tmp_path, "yi9", np.random.default_rng(9))
    (tmp_path / "FETCH_FAILED.txt").write_text("2026-10-10T00:00:00Z FETCH REFUSED llama8 (exit 1)\n")
    rc, txt = score(tmp_path)
    assert "'llama8': 'yi9'" in txt and verdict(txt, "J-B1-N4") == "MET"
    assert "llama8->yi9: MET" in txt


def test_fallback_slot_after_a_failed_tokenizer_check(tmp_path):
    """The pipeline moves a failed tokcheck file aside (scripts/stage8_common.sh, S8_OUTPUTS), so the replaced model
    leaves no file; COMMIT.txt's record names it. Another N4 model without F results (a deadline) keeps its own slot."""
    full(tmp_path, jb8=False, skip_models=("phi4",))
    write_model(tmp_path, "yi9", np.random.default_rng(9))
    J = json.loads((tmp_path / "eval" / "falcon7_F.json").read_text())
    J["provenance"]["arms_not_run"], J["results"] = list(F_ARMS), []          # falcon7: every arm cut by the deadline
    (tmp_path / "eval" / "falcon7_F.json").write_text(json.dumps(J))
    (tmp_path / "COMMIT.txt").write_text("2026-10-10T01:00:00Z fallback: yi9 replaces phi4 (its tokenizer check (J-B-G0b) "
                                         "failed; no output of phi4 exists)\n")
    rc, txt = score(tmp_path)
    assert "'phi4': 'yi9'" in txt and "'falcon7': 'falcon7'" in txt
    assert "phi4->yi9: MET" in per_model(txt, "J-B1-N4", "phi4->yi9")
    assert "NOT EVALUABLE" in per_model(txt, "J-B1-N4", "falcon7") and verdict(txt, "J-B1-N4") == "MET"


def test_pytest_gate(tmp_path):
    pytest_log(tmp_path, skip="tests/test_fresh.py::test_1")
    out = []
    assert S.gate_g0(tmp_path, out.append) is False
    p = tmp_path / "logs" / "pytest.log"
    p.write_text(p.read_text().replace("tests/test_fresh.py::test_1 SKIPPED", "tests/test_fresh.py::test_1_cached_tokenizer SKIPPED"))
    assert S.gate_g0(tmp_path, out.append) is True
    pytest_log(tmp_path, fail="tests/test_surface.py::test_0")
    assert S.gate_g0(tmp_path, out.append) is False
    assert S.gate_g0(tmp_path / "nothing", out.append) is None


def test_model_gate_failures_outside_test(tmp_path):
    full(tmp_path, jb8=False, per={"qwen7": {"tok_ok": False}})
    (tmp_path / "verified" / "olmo7.json").unlink()
    rc, txt = score(tmp_path)
    assert "J-B-G0b failed" in per_model(txt, "J-B1-P4f", "qwen7") and "J-B-G1" in per_model(txt, "J-B1-P4f", "olmo7")
    assert verdict(txt, "J-B1-P4f") == "NOT EVALUABLE"         # two of four evaluable


def test_bootstrap_deterministic_and_two_stage():
    labs = [(1, "a", "b")] * 3 + [(2, "a", "b")] * 2 + [(1, "c", "d")] * 4 + [(2, "x", "y")]
    b1, b2 = st.Boot(labs), st.Boot(labs)
    assert np.array_equal(b1.W, b2.W) and b1.W.shape == (st.NB, 10)
    assert np.allclose(b1.W.sum(1), b1.tot)
    for w in b1.W[:200]:   # within a cluster the weights sum to (times drawn) x (cluster size)
        for c, size in (((1, "a", "b"), 3), ((2, "a", "b"), 2), ((1, "c", "d"), 4), ((2, "x", "y"), 1)):
            idx = [i for i, l in enumerate(labs) if l == c]
            assert w[idx].sum() % size == 0
    assert st.boot_of(tuple(labs)) is st.boot_of(tuple(labs))
    core = st.Boot(None, 10)
    assert not np.array_equal(core.W, b1.W) and np.array_equal(core.W, st.Boot(None, 10).W)
    e = st.est(b1, st.mean, np.arange(10.0))
    assert e.pt == 4.5 and e.bound(0.95, "lo") < 4.5 < e.bound(0.95, "hi")
    assert e.bound(0.9875, "lo") <= e.bound(0.95, "lo")


def test_combination_rules():
    assert st.comb_every({"a": True, "b": True, "c": True, "d": None}) is True
    assert st.comb_every({"a": True, "b": False, "c": True, "d": True}) is False
    assert st.comb_every({"a": True, "b": True, "c": None, "d": None}) is None
    assert st.comb_k_of({"a": True, "b": True, "c": True, "d": False}) is True
    assert st.comb_k_of({"a": True, "b": True, "c": False, "d": None}) is False
    assert st.comb_k_of({"a": True, "b": True, "c": None, "d": None}) is None
    e = st.Est(1.0, np.r_[np.full(97, 1.0), np.full(3, -1.0)], 100)
    assert abs(st.pval(e, lambda b: b <= 0) - 4 / 101) < 1e-12


def test_risk_class_follows_the_recorded_prior():
    """Common part, G4: L = implied by data in hand, prior >= 0.9; M = prior >= 0.8; R = prior < 0.8."""
    for code, (cls, kind, prior, *_rest) in ln.LINES.items():
        assert kind in ("A", "V") and 0 < prior < 1, code
        assert (cls == "L" and prior >= 0.9) or (cls == "M" and prior >= 0.8) or (cls == "R" and prior < 0.8), (code, cls, prior)


def _holm_res():
    """Three lines in one model: an R account line (J-B3-N4: one interval component met only at 95 %, one point), an M
    account line (J-B1-N4) and an R validity line (J-B6b); only J-B3-N4's interval component is in the Holm family."""
    b = st.boot_of(None, 60)
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 60)
    x = x - x.mean() + 0.30                                             # mean 0.30, bootstrap SE ~ 0.13
    e = st.est(b, st.mean, x)
    lo = st.lower(e, 0.0, 0.95, "r(POST)")                              # 95 %: rejected (lower bound > 0)
    res = {"J-B3-N4": (True, {"llama8": st.Res(True, "", [lo, st.point("r(POST) >= 0.15", True)])}),
           "J-B1-N4": (True, {"llama8": st.Res(True, "", [st.lower(e, 0.0, 0.95, "s_ID(AFTER)")])}),
           "J-B6b": (True, {"qwen7": st.Res(True, "", [st.lower(e, 0.0, 0.95, "x")])})}
    return res, lo, e


def test_holm_family_and_the_shared_helper_contract(monkeypatch):
    import types
    res, lo, e = _holm_res()
    calls = []

    def fake(components):
        calls.append([dict(c) for c in components])
        return [dict(c, p=0.5, threshold=0.025, reject=False) for c in components]
    monkeypatch.setitem(sys.modules, "stage8_holm", types.SimpleNamespace(holm=fake))
    out = []
    ch = S.holm_sensitivity(res, out.append)
    assert len(calls) == 1 and [c["line"] for c in calls[0]] == ["J-B3-N4"]          # R-class account lines only
    c = calls[0][0]
    assert set(c) == {"line", "name", "est", "se", "bound", "direction"} and c["direction"] == ">" and c["bound"] == 0.0
    assert abs(c["est"] - 0.30) < 1e-9 and abs(c["se"] - float(np.std(e.bs, ddof=1))) < 1e-12 and 0.05 < c["se"] < 0.3
    assert lo.passed and ch == {"J-B3-N4": (True, True)}                    # the decision and the verdict change
    assert any("decisions changed by Holm: llama8: H0 r(POST) <= 0 rejected -> not rejected" in l for l in out)
    upp = st.upper(e, 1.0, 0.9875, "r(POST)")
    assert upp.holm["direction"] == "<" and upp.holm["bound"] == 1.0


def test_holm_with_the_shared_helper():
    """analysis/stage8_holm.py (common part) on the family: p from the bootstrap SE, step-down at 0.025."""
    import stage8_holm
    res, lo, e = _holm_res()
    got = stage8_holm.holm([d for *_, d in S.holm_family(res)])
    assert len(got) == 1 and got[0]["line"] == "J-B3-N4"
    z = (e.pt - 0.0) / float(np.std(e.bs, ddof=1))
    assert abs(got[0]["p"] - 0.5 * math.erfc(z / math.sqrt(2))) < 1e-9
    out = []
    S.holm_sensitivity(res, out.append)
    assert any(l.strip().startswith("J-B3-N4: 1 components") for l in out)
