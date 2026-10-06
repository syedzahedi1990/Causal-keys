"""experiments/remention_attention.py and analysis/stage5_parts/attention.py (prereg G, part a) at Qwen2.5-0.5B, FP32, CPU:
positions and candidate ids, P1/AFTER identity through the last option word, clamp exactness of the eager instrument,
an end-to-end run with the scorer, and the scorer's rules on synthetic data."""
import io
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.stage5_parts import attention as sc
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import encode, raw_prompt
from ckeys.interventions import blocks, hooks
from ckeys.story import LOCATIONS, make_cores, record
from experiments.remention_attention import (ARRAYS, cased_ids, encode_item, item_2a, positions, probe, run, write)

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def mt():
    return (AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32, attn_implementation="eager").eval(),
            AutoTokenizer.from_pretrained(QWEN))


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(QWEN)


def test_positions_core0(tok):
    core = make_cores(1, random.Random(0))[0]
    cid = cased_ids(tok)
    assert cid["cap_single"] and cid["lower"] == [3745, 14024, 27645, 26482, 21921, 31944]
    assert cid["cap"] == [8261, 33117, 70346, 48260, 32946, 99140] and len(set(cid["lower"] + cid["cap"])) == 12
    for arm, r, ans in (("POST", [68, 71, 74, 77, 80, 83], 108), ("P1", [76, 78, 80, 82, 84, 86], 102)):
        enc = encode_item(tok, core, arm)
        pos = positions(tok, core, enc, arm, cid["lower"])
        assert pos["p"] == 55 and pos["r"] == r and pos["ans"] == ans == pos["T"] - 1
        assert pos["q_obj"] == 44 and pos["q_dist"] == 33 and pos["s"] in pos["N"] and pos["x"] in pos["N"] and 3 <= len(pos["N"]) <= 4
        assert len(set(enc["ids"]["B"][0, r].tolist())) == 6


def test_all_cores_valid(tok):
    """The entry's per-story asserts hold on all 150 seed-0 cores under both arms (tokenizer only)."""
    cid = cased_ids(tok)["lower"]
    same = 0
    for core in make_cores(150, random.Random(0)):
        for arm in ("POST", "P1"):
            enc = encode_item(tok, core, arm)
            assert enc is not None
            pos = positions(tok, core, enc, arm, cid)
            assert len(pos["N"]) == (4 if core["initial"] == core["distractor_location"] else 3)
        same += core["initial"] == core["distractor_location"]
    assert same == 40


def test_p1_after_identical(tok):
    for core in make_cores(150, random.Random(0)):
        r = record(core, "direct", core["base"])
        a, b = (encode(tok, raw_prompt(arm, r["story"], r["query"]))[0] for arm in ("P1", "AFTER"))
        last = max(positions(tok, core, encode_item(tok, core, "P1"), "P1", cased_ids(tok)["lower"])["r"])
        assert torch.equal(a[: last + 1], b[: last + 1])


def test_clamp_exactness(mt):
    """Self-clamp of the base key leaves attentions and logits unchanged; a K+V clamp from the source run reproduces the
    source run's attention in every row after p and its logits, rows before p untouched; the K_S attention of ``probe``
    equals the committed single-position k_proj hook."""
    model, tok = mt
    core = make_cores(1, random.Random(0))[0]
    enc = encode_item(tok, core, "POST")
    p, nL = enc["p"], len(blocks(model))
    ib, is_ = enc["ids"]["B"], enc["ids"]["S"]

    def fwd(ids):
        o = model(ids, use_cache=False, output_attentions=True)
        return torch.stack([a[0] for a in o.attentions]), torch.log_softmax(o.logits[0, -1].float(), -1)

    with capture_kv(model, [p], range(nL)) as kb:
        A_b, lp_b = fwd(ib)
    with capture_kv(model, [p], range(nL)) as ks:
        A_s, lp_s = fwd(is_)
    with clamp_kv(model, [p], {k: v[0] for k, v in kb.items()}, range(nL)):
        A_self, lp_self = fwd(ib)
    assert torch.allclose(A_self, A_b, atol=1e-6) and torch.allclose(lp_self, lp_b, atol=1e-5)
    with clamp_kv(model, [p], {k: v[0] for k, v in ks.items()}, range(nL)):
        A_kv, lp_kv = fwd(ib)
    assert torch.allclose(A_kv[:, :, p + 1:], A_s[:, :, p + 1:], atol=1e-5) and torch.allclose(lp_kv, lp_s, atol=1e-4)
    assert torch.allclose(A_kv[:, :, :p], A_b[:, :, :p], atol=1e-6)
    # probe's K_S == the factorial's K_S@0 row (K from S, V held at B's); a key-only hook (p's own values drift) differs
    out = probe(model, enc["ids"], p, [60, enc["T"] - 1], [p, 0], conds=("B", "S", "K_S"))
    with clamp_kv(model, [p], {(l, "k"): ks[(l, "k")][0] for l in range(nL)} | {(l, "v"): kb[(l, "v")][0] for l in range(nL)}, range(nL)):
        A_h, lp_h = fwd(ib)
    assert torch.allclose(out["att"]["K_S"], A_h[:, :, [60, enc["T"] - 1]][:, :, :, [p, 0]], atol=1e-6)
    assert torch.allclose(out["lp"]["K_S"], lp_h, atol=1e-5) and torch.allclose(out["att"]["B"], A_b[:, :, [60, enc["T"] - 1]][:, :, :, [p, 0]], atol=1e-6)
    assert not torch.allclose(out["att"]["K_S"], out["att"]["B"])
    Ks = {l: ks[(l, "k")][0, 0] for l in range(nL)}
    hs = []
    for l in range(nL):
        def hk(_m, _i, o, l=l):
            o = o.clone(); o[:, p] = Ks[l]; return o
        hs.append(blocks(model)[l].self_attn.k_proj.register_forward_hook(hk))
    with hooks(hs):
        _, lp_k = fwd(ib)
    assert (lp_k[cased_ids(tok)["lower"]] - lp_h[cased_ids(tok)["lower"]]).abs().max() > 1e-3


def test_item_arrays(mt):
    model, tok = mt
    core = make_cores(1, random.Random(0))[0]
    it, ar, row = item_2a(model, tok, core, "P1", cased_ids(tok))
    L, H = len(blocks(model)), model.config.num_attention_heads
    assert set(ar) == set(ARRAYS) | {"D", "D2", "H_B_rb", "A_B_r_p", "A_B_ans_r"} and ar["D"].shape == (L, H) and row.shape == (L, H, it["T"])
    assert ar["A_B_r_p"].shape == (L, H, 6) and np.allclose(ar["A_B_r_p"][:, :, it["b"]], ar["A_B_rb_p"], atol=1e-3)
    assert np.allclose(ar["D"], ar["A_B_rb_p"] - ar["A_B_N_p"], atol=2e-3) and np.isnan(ar["A_B_rdist_qdist"]).all() == it["same_init_dist"]
    assert all(len(it["lp"][c]["lower"]) == 6 and len(it["lp"][c]["cap"]) == 6 for c in it["lp"]) and 0 < it["mass"]["B"]["lower"] <= 1
    # the clean-B ans row sums to 1 and its p column matches the saved array
    assert np.allclose(row.astype(np.float32).sum(-1), 1, atol=2e-3) and np.allclose(row[:, :, it["p"]], ar["A_B_ans_p"], atol=1e-3)


def test_end_to_end_scorer(mt):
    """n = 4 (a 2/2 parity split), both arms, written like the experiment and scored: every verdict line is printed,
    with NOT EVALUABLE for the cells 0.5B cannot fill, and the G2 path runs when 0.5B stands in as a small model."""
    model, tok = mt
    cores = make_cores(4, random.Random(0))
    items, arrays, cid = run(model, tok, cores, ["POST", "P1"], log=lambda s: None)
    assert len(items) == 8 and arrays["POST/D"].shape[0] == 4
    out = Path(tempfile.mkdtemp())

    class A:  # provenance() reads vars(a)
        model, n, arms, dtype = QWEN, 4, "POST,P1", "float32"
    write(out, QWEN, A(), items, arrays, cid, model)
    assert (out / "Qwen2.5-0.5B-Instruct.json").exists() and (out / "Qwen2.5-0.5B-Instruct.npz").exists()
    buf = io.StringIO()
    res = sc.score(out, out=lambda s: buf.write(s + "\n"))
    txt = buf.getvalue()
    for line in ("  G1 anchor", "  G2 dissociation", "  CALL:", "  G3 magnitude", "  G4a hop-2", "  G4b hop-2", "Gate a ->", "== Exploratory"):
        assert line in txt, line
    assert "G1 anchor: NOT EVALUABLE" in txt and "G2 dissociation: NOT EVALUABLE" in txt and res["call"] is None
    buf = io.StringIO()
    res = sc.score(out, {"Qwen2.5-0.5B-Instruct": "small"}, out=lambda s: buf.write(s + "\n"))
    txt = buf.getvalue()
    assert res["call"] in ("H_diss", "H_track", "mixed", None) and ("CALL: " in txt)
    assert "G3 magnitude" in txt and "G4b hop-2 cross-scale: NOT EVALUABLE" in txt


# ----------------------------------------------------------------------------- scorer rules on synthetic data
def synth(out, tag, n, E, F, G, idk_post, idk_p1, cap=False, L=4, H=4, seed=0):
    """Fake run: D peaks at heads (2, 1..3) with mean E (POST) / 0.6 (P1), F gain (0.8 under P1) at those heads, D2 peaks at (3, 0..2) with mean G;
    in-run ID_K (lowercase) = idk_post / idk_p1 with cap ids = 0.8 x that; capitalised mass dominates if ``cap``."""
    rng = np.random.default_rng(seed)
    items, arrays = [], {}
    for arm, e, f, g, idk in (("POST", E, F, G, idk_post), ("P1", 0.6, 0.8, 0.4, idk_p1)):
        acc = {}
        for i in range(n):
            core = {"base": "cabinet", "source": "drawer", "initial": "basket", "distractor_location": "basket", "object": "lamp",
                    "distractor": "coin", "agent": "Alice", "other": "Bob"}
            b, s, x = 4, 3, 0
            ar = {k: rng.normal(0, 0.01, (L, H)).astype(np.float32) for k in ARRAYS}
            ar["A_B_rb_p"][2, 1:] += e + rng.normal(0, 0.02)       # three signal heads (E is the mean over the top-3)
            ar["A_KS_rs_p"][2, 1:] += f * e + rng.normal(0, 0.02)
            ar["A_B_ans_rb"][3, :3] += g + rng.normal(0, 0.02)
            ar["D"] = ar["A_B_rb_p"] - ar["A_B_N_p"]
            ar["D2"] = ar["A_B_ans_rb"] - ar["A_B_ans_N"]
            ar["H_B_rb"] = np.ones((L, H), np.float32)
            ar["A_B_r_p"] = rng.normal(0, 0.01, (L, H, 6)).astype(np.float32)
            ar["A_B_ans_r"] = rng.normal(0, 0.01, (L, H, 6)).astype(np.float32)
            for k, v in ar.items():
                acc.setdefault(k, []).append(v)
            lp = {c: {"lower": [-3.0] * 6, "cap": [-3.0] * 6} for c in ("B", "S", "X", "K_S", "K_X")}
            v = idk + rng.normal(0, 0.1)
            lp["K_S"]["lower"][s] += v; lp["K_X"]["lower"][x] += v
            lp["K_S"]["cap"][s] += 0.8 * v; lp["K_X"]["cap"][x] += 0.8 * v
            lp["B"]["lower"][b] = lp["B"]["cap"][b] = -0.5
            lp["S"]["lower"][s] = lp["S"]["cap"][s] = -0.5
            mass = {c: {"lower": 0.2 if cap else 0.9, "cap": 0.7 if cap else 0.05} for c in lp}
            items.append({"i": i, "arm": arm, "core": core, "X": "box", "b": b, "s": s, "x": x, "init": 1, "dist": 1, "N": [0, 3, 5],
                          "p": 55, "T": 109, "ans": 108, "q_obj": 44, "q_dist": 33, "r": [68, 71, 74, 77, 80, 83], "same_init_dist": True,
                          "lp": lp, "argmax": {c: 0 for c in lp}, "argmax_tok": {c: " cabinet" for c in lp}, "mass": mass})
            arrays[f"{arm}/{i}/ans_row"] = np.zeros((L, H, 109), np.float16)
        arrays |= {f"{arm}/{k}": np.stack(v) for k, v in acc.items()}
    prov = {"attn_implementation": "eager", "n_layers": L, "n_heads": H, "n_kv": 2, "casing_check": "ok"}
    json.dump({"provenance": prov, "items": items}, open(out / f"{tag}.json", "w"))
    np.savez_compressed(out / f"{tag}.npz", **arrays)


def run_score(out, roles=None):
    buf = io.StringIO()
    res = sc.score(out, roles, out=lambda s: buf.write(s + "\n"), exploratory=False)
    return res, buf.getvalue()


def test_rules_h_diss():
    out = Path(tempfile.mkdtemp())
    synth(out, "Qwen2.5-7B-Instruct", 40, E=0.5, F=0.8, G=0.3, idk_post=5.5, idk_p1=20.0, cap=True)
    synth(out, "Qwen2.5-14B-Instruct", 40, E=0.6, F=0.8, G=0.4, idk_post=12.0, idk_p1=35.0, cap=True)
    synth(out, "Qwen2.5-1.5B-Instruct", 40, E=0.5, F=0.8, G=0.05, idk_post=-0.4, idk_p1=2.8)
    synth(out, "Qwen2.5-3B-Instruct", 40, E=0.45, F=0.7, G=0.08, idk_post=0.0, idk_p1=10.0, cap=True)
    res, txt = run_score(out)
    assert all(res["gated"].values()) and res["call"] == "H_diss" and res["g1"] and res["g4a"]
    assert "G1 anchor" in txt and "2/2 anchors MET" in txt.split("G1 anchor")[1].split("\n")[0]
    assert "H_diss MET" in txt and "CALL: H_diss (full, 2/2)" in txt
    assert "G3 magnitude under H_diss" in txt and "2/2 MET" in txt.split("G3 magnitude")[1].split("\n")[0]
    assert "G4a hop-2 anchor" in txt and "2/2 anchors MET" in txt.split("G4a hop-2")[1].split("\n")[0]
    assert "G4b hop-2 cross-scale" in txt and "2/2 MET" in txt.split("G4b hop-2")[1].split("\n")[0]


def test_rules_h_track_and_gates():
    out = Path(tempfile.mkdtemp())
    synth(out, "Qwen2.5-7B-Instruct", 40, E=0.5, F=0.8, G=0.02, idk_post=5.5, idk_p1=20.0, cap=True)
    synth(out, "Qwen2.5-14B-Instruct", 40, E=0.6, F=0.8, G=0.03, idk_post=2.0, idk_p1=35.0, cap=True)   # fails key POST gate
    synth(out, "Qwen2.5-1.5B-Instruct", 40, E=0.0, F=0.0, G=0.0, idk_post=-0.4, idk_p1=2.8)
    synth(out, "Qwen2.5-3B-Instruct", 40, E=0.02, F=0.0, G=0.0, idk_post=0.0, idk_p1=10.0, cap=True)
    res, txt = run_score(out)
    assert res["gated"]["Qwen2.5-7B-Instruct"] and not res["gated"]["Qwen2.5-14B-Instruct"]
    assert "Gated-in anchors: ['Qwen2.5-7B-Instruct']" in txt and res["call"] == "H_track"
    assert res["verdicts"]["G2"][0] is False and "NOT MET (alternative H_track declared)" in txt   # G2 is met only under H_diss
    assert "1/2 anchors (Qwen2.5-14B-Instruct gated out) NOT MET" in txt.split("G1 anchor")[1].split("\n")[0]   # a gated-out anchor counts as not met
    assert res["verdicts"]["G1"] == (False, "1/2 anchors (Qwen2.5-14B-Instruct gated out)") and res["g1"] is False
    assert "G3 magnitude under H_track" in txt and "2/2 MET" in txt.split("G3 magnitude")[1].split("\n")[0]
    assert "G4a hop-2 anchor" in txt and "0/2 anchors (Qwen2.5-14B-Instruct gated out) NOT MET" in txt.split("G4a hop-2")[1].split("\n")[0] and "unsuitable" in txt
    assert "G4b hop-2 cross-scale: NOT EVALUABLE (G2 did not declare H_diss)" in txt


def test_rules_mixed_and_casing():
    out = Path(tempfile.mkdtemp())
    synth(out, "Qwen2.5-7B-Instruct", 40, E=0.5, F=0.8, G=0.3, idk_post=5.5, idk_p1=20.0, cap=True)
    synth(out, "Qwen2.5-14B-Instruct", 40, E=0.6, F=0.8, G=0.4, idk_post=12.0, idk_p1=35.0, cap=True)
    synth(out, "Qwen2.5-1.5B-Instruct", 40, E=0.5, F=0.8, G=0.05, idk_post=-0.4, idk_p1=2.8)
    synth(out, "Qwen2.5-3B-Instruct", 40, E=0.0, F=0.0, G=0.08, idk_post=0.0, idk_p1=10.0, cap=True)
    res, txt = run_score(out)
    assert res["call"] == "mixed" and "no account declared" in txt and "G3 magnitude: NOT EVALUABLE" in txt
    assert "G4b hop-2 cross-scale: NOT EVALUABLE" in txt
    # a small model whose capitalised ID_K (emitted) sits outside [-1, 1] is gated out although the lowercase passes
    out2 = Path(tempfile.mkdtemp())
    synth(out2, "Qwen2.5-3B-Instruct", 40, E=0.5, F=0.8, G=0.08, idk_post=1.2, idk_p1=10.0, cap=True)
    items = json.load(open(out2 / "Qwen2.5-3B-Instruct.json"))
    for it in items["items"]:
        if it["arm"] == "POST":
            it["lp"]["K_S"]["lower"][it["s"]] = -3.0 + 0.5; it["lp"]["K_X"]["lower"][it["x"]] = -3.0 + 0.5   # lowercase ID_K 0.5
            it["lp"]["K_S"]["cap"][it["s"]] = -3.0 + 1.6; it["lp"]["K_X"]["cap"][it["x"]] = -3.0 + 1.6         # capitalised 1.6
    json.dump(items, open(out2 / "Qwen2.5-3B-Instruct.json", "w"))
    res, txt = run_score(out2)
    assert not res["gated"]["Qwen2.5-3B-Instruct"] and "casing POST" in txt and "FAIL" in txt.split("casing POST")[1].split("\n")[0]
    assert "CALL: no call" in txt
