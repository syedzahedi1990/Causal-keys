"""Part (b) of P-2026-10-05-H: the port of Prakash et al.'s generator (pool hash, release hashes, the n = 320 pitfall),
positions and lengths of all 320 pairs in every format (Qwen2.5; Llama-3 BOS-shifted when its tokenizer is cached), the
exchange's exactness at Qwen2.5-0.5B FP32 (B+KV_M == M, the self row == B, the batched M row == the unbatched M, the clamp
self row == B, at 1e-3 in the log-probs, the entry's FP32 Gate b0 bound), the resid-patch row placement, and the scorer's rules on synthetic fixtures
(kappa evaluability on point estimates and per resample, the > 5 % dropped-fraction rule, interaction-carried, the l* tie
rule, l* re-derived and asserted against lstar.json, every gate and verdict line). Needs the release checkout
($PRAKASH_REPO, default /home/user/nix07/mind)."""
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.stage6_parts import prakash as sc
from ckeys import causaltom as ct
from ckeys.interventions import blocks
from experiments import prakash_swap as ps

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA_TOK = "unsloth/Meta-Llama-3.1-8B-Instruct"   # the Llama-3 tokenizer (the 70B's is re-checked in the preflight)
torch.set_grad_enabled(False)
NM, QN, OA, LA = ct.FORMATS


@pytest.fixture(scope="module")
def rel():
    try:
        return ct.load()
    except AssertionError as ex:
        pytest.fail(f"release checkout unusable ({ex}); set PRAKASH_REPO to a checkout of {ct.RELEASE_URL} at {ct.RELEASE_SHA}")


@pytest.fixture(scope="module")
def pairs(rel):
    return ct.pool(rel)


@pytest.fixture(scope="module")
def qwen():
    return AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32, attn_implementation="sdpa").eval(), AutoTokenizer.from_pretrained(QWEN)


def test_pool_hash_and_release(rel, pairs, tmp_path):
    assert ct.pool_hash(pairs) == ct.POOL_SHA256 and len(pairs) == 320
    p = pairs[0]
    assert p["clean_prompt"].startswith("Instruction: 1. Track the belief of each character") and p["clean_prompt"].endswith("contains?\nAnswer:")
    assert p["target"] == " " + [s for s in p["clean_states"] if s != p["clean_ans"]][0]
    other = ct.reversed_sentence_counterfacts(rel, 600, random.Random(10))   # the q draws follow every entity draw
    assert all(a["clean_story"] == b["clean_story"] for a, b in zip(pairs, other))
    assert sum(a["clean_prompt"] == b["clean_prompt"] for a, b in zip(pairs, other)) == 174
    for f in ct.FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(rel.root) / f, tmp_path / f)
    assert ct.load(str(tmp_path)).hashes == rel.hashes
    d = tmp_path / "data/synthetic_entities/drinks.json"
    d.write_text(d.read_text().replace("beer", "ale "))
    with pytest.raises(AssertionError, match="drinks.json"):
        ct.load(str(tmp_path))


def _all_positions(tok, rel, pairs, nb):
    for pr in pairs:
        m = ct.meta(rel, pr)
        assert m["S"] not in m["states"] and m["X"] not in m["states"] and sorted(m["cand"]) == sorted(m["states"] + [m["S"], m["X"]])
        for f in ct.FORMATS + ct.EXTRA_FORMATS:
            L = ct.locate(tok, rel, pr, f, m)   # asserts lengths, positions, punctuation ids, the cf spans, B/S/X at p
            assert L["T"] == ct.LENGTHS[nb][f] and L["P"] == list(ct.POS[nb]) and L["n_bos"] == nb
            assert L["bind_dst"] == [L["P"][2], L["P"][3], L["P"][0], L["P"][1]] and L["id_span"] == [L["p"], L["p"] + 1]
            w = ct.readout_words(f, m)
            assert all(len(tok.encode(" " + v, add_special_tokens=False)) == 1 for v in w.values())


def test_positions_lengths_qwen(rel, pairs):
    _all_positions(AutoTokenizer.from_pretrained(QWEN), rel, pairs, 0)


def test_positions_lengths_llama3(rel, pairs):
    try:
        tok = AutoTokenizer.from_pretrained(LLAMA_TOK, local_files_only=True)
    except Exception:
        pytest.skip(f"{LLAMA_TOK} tokenizer not cached")
    _all_positions(tok, rel, pairs, 1)


def test_preflight(rel, pairs):
    tok = AutoTokenizer.from_pretrained(QWEN)
    pf = ps.preflight(tok, rel, pairs[:40], list(ct.FORMATS))
    assert pf["pool_sha256"] != ct.POOL_SHA256   # 40 pairs only; the full pool is hashed in test_pool_hash_and_release
    assert pf["formats"][OA] == {"length": 196, "positions": [154, 155, 166, 167]} and pf["punct_ids"] == [13, 624]


@pytest.mark.parametrize("fmt", [NM, OA])
def test_exchange_exact(qwen, rel, pairs, fmt):
    model, tok = qwen
    L = ps.item(tok, rel, pairs[0], fmt, 0)
    nL = len(blocks(model))
    ex, cl = ps.run_pair(model, tok, L, [("BIND", 8), ("ID", 3)], [9], True, True, nL, True, lambda s: None)
    assert len(ex) == 2 and len(cl) == 1
    tol = 1e-3   # Gate b0's FP32 bound: a batched row vs an unbatched run, measured 2e-6 (1 thread) to 5e-5 (2-4 threads) here
    for e in ex:
        assert all(abs(e["lp"]["r4"][k] - e["lp"]["r1"][k]) <= tol for k in ps.ROLES), e["b0"]   # B+KV_M == M
        assert all(abs(e["lp"]["r0"][k] - e["lp_B"][k]) <= tol for k in ps.ROLES), e["b0"]       # self row == unbatched B
        assert abs(e["m"]["r1"] - e["m_M"]) <= tol, e["b0"]                                        # batched M row == unbatched M
        assert abs(e["m"]["r1"] - e["m"]["r0"]) > 1e-2             # the patch does something
        for r in ("r5", "r6", "r2w", "r3w", "r4w"):
            assert np.isfinite(e["m"][r])
    for c in cl:
        assert all(abs(c["lp"]["ID"][k] - c["lp_B"][k]) <= tol for k in ps.ROLES)   # the clamp self row == B


def test_resid_patch_rows(qwen, rel, pairs):
    """The patch lands only in the active rows and only at the destination positions; the sweep's in-batch self rows
    are the clean run and its patched rows the single patched run."""
    model, tok = qwen
    items = [ps.item(tok, rel, pairs[i], NM, i) for i in (0, 1)]
    l = 6
    L = items[0]
    a = L["arm"]["BIND"]
    with ps.capture_resid(model, [l], [a["src"]]) as R:
        ps.logprobs(model, L["ids"]["C"])
    with ps.resid_patch(model, l, [None, a["dst"], None], [None, R[l][0], None]), ps.capture_resid(model, [l], [list(range(len(L["ids"]["B"])))] * 3) as H:
        lp = ps.logprobs(model, [L["ids"]["B"]] * 3)
    h = H[l]   # the block-l output as block l+1 reads it (a later hook sees the patched output)
    tol = 1e-3   # batch rows vs unbatched runs (host-dependent rounding); the patch moves the residual by O(1)
    assert torch.equal(h[1, a["dst"]], R[l][0]) and (h[1, a["dst"]] - h[0, a["dst"]]).abs().max() > 0.1
    assert torch.allclose(h[0], h[2], atol=tol, rtol=1e-4)
    other = [t for t in range(h.shape[1]) if t not in a["dst"]]
    with ps.capture_resid(model, [l], [list(range(h.shape[1]))]) as RB:
        lpB = ps.logprobs(model, L["ids"]["B"])
    assert torch.allclose(h[0], RB[l][0], atol=tol, rtol=1e-4) and torch.allclose(h[1, other], RB[l][0][other], atol=tol, rtol=1e-4)
    with ps.resid_patch(model, l, [a["dst"]], [R[l][0]]):
        lpM = ps.logprobs(model, L["ids"]["B"])
    assert (lp[0] - lpB[0]).abs().max() <= tol and (lp[2] - lpB[0]).abs().max() <= tol and (lp[1] - lpM[0]).abs().max() <= tol
    assert (lpM[0] - lpB[0]).abs().max() > 0.01
    rows, summ = ps.sweep(model, tok, items, "ID", [l], bs=2)
    for r, it in zip(rows[l], items):
        b = it["arm"]["ID"]
        with ps.capture_resid(model, [l], [b["src"]]) as RD:
            ps.logprobs(model, it["ids"]["S"])
        with ps.resid_patch(model, l, [b["dst"]], [RD[l][0]]):
            mp = ps.mval(ps.logprobs(model, it["ids"]["B"])[0], it, "ID")
        assert abs(r["m_patch"] - mp) <= tol and abs(r["m_self"] - ps.mval(ps.logprobs(model, it["ids"]["B"])[0], it, "ID")) <= tol


# --------------------------------------------------------------------------- scorer fixtures
def _ex_cells(pk, pv, n=40, phi=6.0, noise=0.2, seed=0, b0=0.01):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = phi + rng.normal(0, 1)
        e = lambda: rng.normal(0, noise)  # noqa: E731
        m = {"r0": 0.0, "r1": f, "r2": pk * f + e(), "r3": pv * f + e(), "r4": f + b0, "r5": (1 - pk) * f + e(), "r6": (1 - pv) * f + e(),
             "r2w": pk * f + e(), "r3w": pv * f + e(), "r4w": 0.9 * f}
        out.append({"i": i, "q": i % 2, "m": m, "m_B": b0, "m_M": f, "ok_r1": True, "ok_B": True, "ok_C": True})
    return out


def test_evaluability_rule():
    ok = sc.Ex(_ex_cells(0.3, 0.6), False)
    assert ok.rule and ok.drop <= 0.05 and abs(ok.kappa - 1 / 3) < 0.05 and "kappa" in ok.state() and "NOT EVALUABLE" not in ok.state()
    proto = sc.Ex(_ex_cells(1.54, -0.61), False)   # the 0.5B prototype's NO-MENTION cell
    assert not proto.rule and "NOT EVALUABLE" in proto.state() and "interaction-carried" not in proto.state()
    inter = sc.Ex(_ex_cells(0.1, 0.2), False)
    assert not inter.rule and abs(inter.inter - 0.7) < 0.05 and "interaction-carried" in inter.state()
    edge = sc.Ex(_ex_cells(0.27, 0.27, n=12, noise=1.5), False)   # point estimate passes, many resamples do not
    assert edge.rule and edge.drop > 0.05 and "CI FAILED" in edge.state()
    neg = sc.Ex(_ex_cells(-0.12, 0.9), False)                      # psi_K below the -0.1 floor
    assert not neg.rule
    assert sc.Ex(_ex_cells(-0.08, 0.9), False).rule                 # the NO-MENTION two-lookback signature stays evaluable
    d = sc.diff_ci(sc.Ex(_ex_cells(0.8, 0.15, seed=1), False), sc.Ex(_ex_cells(0.05, 0.9, seed=2), False))
    assert d[0] > 0.6 and d[1] > 0 and d[3] <= 0.05
    assert not sc.Ex(_ex_cells(0.3, 0.6, b0=0.5), False).b0 and sc.Ex(_ex_cells(0.3, 0.6, b0=0.5e-3), True).b0


def test_tie_rule():
    assert sc.tie_rule({0: 0.5, 1: 0.695, 2: 0.70, 3: 0.70}) == 1 and ps.tie_rule({0: 0.5, 1: 0.695, 2: 0.70, 3: 0.70}) == 1
    assert sc.tie_rule({3: 0.0, 5: 0.0}) == 3 and sc.tie_rule({10: 0.9, 4: 0.8, 6: 0.9}) == 6
    iia = {l: k / 150 for l, k in enumerate([0, 10, 104, 105, 106, 105])}
    assert sc.tie_rule(iia) == 3 and ps.tie_rule(iia) == 3


def write_synth(root, model="Qwen2.5-14B-Instruct", n=40, nL=48, ls=28, li=4, iia=0.95,
                kap=None, sid=None, kap_id=None, id_at_ls=True):
    """A synthetic part-(b) directory: H_read pattern by default (kappa follows s_ID; H7, H8, H9, H11 met, H10 not);
    id_at_ls: the ID edit also reproduces at l* (H11 Part 2 and Gate b3(b) evaluable)."""
    kap = kap or {NM: (0.03, 0.92), QN: (0.4, 0.55), OA: (0.85, 0.12), LA: (0.9, 0.08), "QNAMES2": (0.2, 0.7)}
    kap_id = kap_id or {NM: (0.08, 0.9), QN: (0.45, 0.5), OA: (0.88, 0.1), LA: (0.9, 0.1)}
    sid = sid or {NM: 0.06, QN: 0.45, OA: 0.9, LA: 0.95}
    d = Path(root) / model
    d.mkdir(parents=True, exist_ok=True)
    prov = {"args": {"model": model}, "git_commit": "synthetic"}
    W = lambda f, body: json.dump({"provenance": prov} | body, open(d / f, "w"))  # noqa: E731
    W("preflight.json", {"pool_sha256": ct.POOL_SHA256, "n_pool": 320, "n_bos": 0, "release": {"sha": ct.RELEASE_SHA, "files": dict(ct.FILES)},
                         "formats": {f: {"length": ct.LENGTHS[0][f], "positions": list(ct.POS[0])} for f in ct.FORMATS}})
    W("filter.json", {"pairs": [{"i": i, "ok": i % 7 != 3} for i in range(320)], "n_pass": 274, "accuracy": 274 / 320})
    pop = [i for i in range(320) if i % 7 != 3][:n]
    for arm, l0 in (("BIND", ls), ("ID", li)):
        hit = lambda l: l0 <= l <= l0 + 4 or (arm == "ID" and id_at_ls and l == ls)  # noqa: E731
        rows = {str(l): [{"i": i, "m_patch": 5.0 if hit(l) else 0.1, "m_self": 0.0,
                          "ok": hit(l) and k < iia * n} for k, i in enumerate(pop)] for l in range(nL)}
        W(f"sweep_{arm}_{NM}.json", {"arm": arm, "format": NM, "layers": list(range(nL)), "population": pop, "rows": rows})
    W("lstar.json", {"lstar": ls, "lstar_ID": li, "population": pop})
    cells = []
    for f, (pk, pv) in kap.items():
        cells += [c | {"arm": "BIND", "depth": ls, "format": f, "i": i} for c, i in zip(_ex_cells(pk, pv, n), pop)]
    for f, (pk, pv) in kap_id.items():
        for dep in (li, ls):
            cells += [c | {"arm": "ID", "depth": dep, "format": f, "i": i} for c, i in zip(_ex_cells(pk, pv, n, seed=dep), pop)]
    W("exchange.json", {"lstar": ls, "lstar_ID": li, "population": pop, "cells": cells})
    rng = np.random.default_rng(5)
    cl = []
    for f, s in sid.items():
        for l0 in sorted({0, 3, 14, ls + 1, li + 1}):
            for i in pop:
                k, v = 10 * s + rng.normal(0, 0.5), 10 * (1 - s) + rng.normal(0, 0.5)
                lp = {"ID": {"S": -5.0, "X": -5.0, "s_q": -0.1, "other": -6.0}}
                for ch, x in (("K", k), ("V", v), ("KV", 1.1 * (k + v))):
                    lp[f"{ch}_S"] = {"S": -5.0 + x, "X": -5.0, "s_q": -1.0, "other": -6.0}
                    lp[f"{ch}_X"] = {"S": -5.0, "X": -5.0 + x, "s_q": -1.0, "other": -6.0}
                cl.append({"format": f, "l0": l0, "i": i, "q": i % 2, "lp": lp})
    W("clamp.json", {"lstar": ls, "lstar_ID": li, "population": pop, "cells": cl})
    return d


def _score(root, test=False):
    lines = []
    r = sc.score(root, out=lines.append, test=test)
    text = "\n".join(lines)
    for g in ("Gate b0", "Gate b1", "Gate b2", "Gate b3"):
        assert g in text, g
    for h in ("H7", "H8", "H9", "H10", "H11", "H12"):
        assert f"   {h} " in text and h in r["verdicts"], h
    return r, text


def test_scorer_h_read(tmp_path):
    write_synth(tmp_path)
    r, text = _score(tmp_path)
    v = {h: ok for h, (ok, _) in r["verdicts"].items()}
    assert v == {"H7": True, "H8": True, "H9": True, "H10": False, "H11": True, "H12": None}, text
    assert r["gates"]["Qwen2.5-14B-Instruct"]["b3"] is True and "NOT RUN" in text
    assert "interaction-carried" not in text and "overlap with the 81st-160th" in text and "flat-high" not in text
    write_synth(tmp_path / "x", id_at_ls=False)   # the ID edit does not reproduce at l*: Gate b3(b) fails, H11 Part 2 not evaluable
    r, text = _score(tmp_path / "x")
    assert r["gates"]["Qwen2.5-14B-Instruct"]["b3"] is False and r["verdicts"]["H11"][0] is True and "Part 2 NOT EVALUABLE" in text


def test_scorer_h_binding_and_rules(tmp_path):
    flat = {NM: (0.03, 0.92), QN: (0.05, 0.9), OA: (0.1, 0.85)}
    write_synth(tmp_path, kap=flat, kap_id={NM: (0.08, 0.9), QN: (0.45, 0.5), OA: (1.54, -0.61)})
    r, text = _score(tmp_path)
    v = {h: ok for h, (ok, _) in r["verdicts"].items()}
    assert v["H10"] is True and v["H7"] is False and v["H9"] is False and v["H11"] is None, text   # kappa_ID(OA) not evaluable
    assert r["gates"]["Qwen2.5-14B-Instruct"]["b3"] is False and "not evaluable as a dissociation at this depth" in text
    oa_out = {NM: (0.03, 0.92), QN: (0.05, 0.9), OA: (0.1, 0.2)}   # OPTIONS-AFTER fails the rule: H7, H9 and H10 all not evaluable
    write_synth(tmp_path / "h", kap=oa_out)
    r, text = _score(tmp_path / "h")
    v = {h: ok for h, (ok, _) in r["verdicts"].items()}
    assert v["H10"] is None and v["H7"] is None and v["H9"] is None and "must be evaluable" in r["verdicts"]["H10"][1], text
    write_synth(tmp_path / "q", kap={NM: (0.03, 0.92), QN: (0.1, 0.2), OA: (0.1, 0.85)})   # QNAMES alone fails: H10 on NM and OA
    r = _score(tmp_path / "q")[0]
    assert r["verdicts"]["H10"][0] is True and "QNAMES not evaluable" in r["verdicts"]["H10"][1]
    write_synth(tmp_path / "fo", kap={NM: (0.1, 0.2), QN: (0.85, 0.1), OA: (0.9, 0.08)})   # flat-high needs H10 evaluable
    r = _score(tmp_path / "fo")[0]
    assert r["verdicts"]["H10"][0] is None and "flat-high" not in r["verdicts"]["H10"][1]
    write_synth(tmp_path / "fh", kap={NM: (0.9, 0.05), QN: (0.85, 0.1), OA: (0.9, 0.08)})
    assert "flat-high" in _score(tmp_path / "fh")[0]["verdicts"]["H10"][1]
    write_synth(tmp_path / "b", iia=0.5)        # Gate b1 fails: the arm's predictions are not evaluable
    r, text = _score(tmp_path / "b")
    assert r["verdicts"]["H7"][0] is None and r["verdicts"]["H11"][0] is None and "Gate b1" in text
    d = write_synth(tmp_path / "c")
    j = json.load(open(d / "lstar.json"))
    json.dump(j | {"lstar": 27}, open(d / "lstar.json", "w"))
    with pytest.raises(AssertionError, match="re-derived"):
        sc.score(tmp_path / "c", out=lambda s: None)


def test_scorer_llama_and_missing(tmp_path):
    write_synth(tmp_path)
    write_synth(tmp_path, model="Meta-Llama-3-70B-Instruct", nL=80, ls=34, li=6)
    r, _ = _score(tmp_path)
    assert r["verdicts"]["H12"][0] is True
    nm_fail = {NM: (0.1, 0.2), QN: (0.4, 0.55), OA: (0.85, 0.12)}   # H8 not met and H7/H9/H10 not evaluable at both sizes
    write_synth(tmp_path / "n", kap=nm_fail, kap_id={NM: (0.1, 0.2), QN: (0.1, 0.2), OA: (0.1, 0.2)})
    write_synth(tmp_path / "n", model="Meta-Llama-3-70B-Instruct", nL=80, ls=34, li=6, kap=nm_fail, kap_id={NM: (0.1, 0.2), QN: (0.1, 0.2), OA: (0.1, 0.2)})
    r, text = _score(tmp_path / "n")
    v = {h: ok for h, (ok, _) in r["verdicts"].items()}
    assert v["H8"] is False and v["H11"] is None and v["H12"] is True and "H11 Part 1 NOT EVALUABLE at 14B (left out)" in text, text
    write_synth(tmp_path / "m", kap_id={NM: (0.1, 0.2), QN: (0.1, 0.2), OA: (0.1, 0.2)}, kap={NM: (0.1, 0.2), QN: (0.1, 0.2), OA: (0.1, 0.2)})
    write_synth(tmp_path / "m", model="Meta-Llama-3-70B-Instruct", nL=80, ls=34, li=6)
    r, text = _score(tmp_path / "m")   # H8 NOT MET at 14B (psi_V < 0.5) but MET at 70B: not reproduced
    assert r["verdicts"]["H12"][0] is False and "H8 NOT MET, MET: no" in text, text
    write_synth(tmp_path / "o", model="Meta-Llama-3-70B-Instruct", nL=80, ls=34, li=6)   # no 14B results
    r, text = _score(tmp_path / "o")
    assert r["verdicts"]["H12"][0] is None and "no Qwen2.5-14B results" in text
    (tmp_path / "e").mkdir()
    r, text = _score(tmp_path / "e")
    assert all(ok is None for ok, _ in r["verdicts"].values()) and "MISSING" in text
