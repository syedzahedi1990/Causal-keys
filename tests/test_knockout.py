"""ckeys.knockout at Qwen2.5-0.5B, FP32, CPU: 4D-mask exactness under sdpa and eager, the row/column group finder on
seed-0 core 0 (Qwen2.5 and Mistral tokenizers), target ids; experiments/attention_knockout.run_item (M8 makes ID_K and ID_V
vanish, M0 rows equal the clean runs, chunked == single batch, the emission pass) and the part (b) scorer."""
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv, stack_rows
from ckeys.encoding import chat_text, raw_prompt
from ckeys.interventions import blocks
from ckeys.knockout import groups, knockout_mask, knockout_masks, mask_for_model, masks_for_model, target_ids
from ckeys.story import make_cores, pick_x, record
from experiments.attention_knockout import masks_of

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def mt():
    return AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval(), AutoTokenizer.from_pretrained(QWEN)


def item(tok, arm, core=None):
    core = core or make_cores(1, random.Random(0))[0]
    rb, rs = record(core, "direct", core["base"]), record(core, "direct", core["source"])
    tb = chat_text(tok, raw_prompt(arm, rb["story"], rb["query"]))
    enc = tok(tb, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
    ib, off = enc.input_ids, enc.offset_mapping[0].tolist()
    is_ = tok(chat_text(tok, raw_prompt(arm, rs["story"], rs["query"])), add_special_tokens=False, return_tensors="pt").input_ids
    p = (ib[0] != is_[0]).nonzero().item()
    return core, tb, ib, is_, off, p, groups(tok, tb, ib, off, p, core, arm, pick_x(core), rb["query"])


def test_groups_core0_qwen(mt):
    tok = mt[1]
    for arm, a in (("AFTER", 101), ("P1", 102), ("POST", 108), ("NONE", 87)):
        g = item(tok, arm)[-1]
        assert g["p"] == 55 and g["a"] == a and g["c_prev"] == 54 and g["C_init"] == [33, 44], (arm, g)
        want = {"AFTER": [76, 78, 80, 82, 84, 86], "P1": [76, 78, 80, 82, 84, 86], "POST": [68, 71, 74, 77, 80, 83], "NONE": []}[arm]
        assert g["R_cand"] == want and len(g["R_trk"]) == (3 if want else 0) and set(g["R_trk"]) | set(g["R_oth"]) == set(want)
        assert min(g["R_q"]) > 55 and g["R_all"] == list(range(56, a + 1)) and a not in g["R_post"] and a in g["R_tail"]
        assert set(g["R_tail"]) == set(g["R_all"]) - set(want) and not set(g["R_post"]) & set(want)
        assert all(r > max(g["R_q"]) for r in g["R_post"]) and (not want or all(r > max(want) for r in g["R_post"]) or arm == "POST")


def test_groups_core0_mistral():
    tok = AutoTokenizer.from_pretrained(MISTRAL)
    for arm, a, want in (("AFTER", 104, [79, 81, 83, 85, 87, 89]), ("P1", 105, [79, 81, 83, 85, 87, 89]),
                         ("POST", 109, [67, 70, 73, 76, 79, 82]), ("NONE", 88, [])):
        g = item(tok, arm)[-1]
        assert g["p"] == 54 and g["a"] == a and g["C_init"] == [32, 43] and g["R_cand"] == want, (arm, g)


def test_target_ids():
    q = target_ids(AutoTokenizer.from_pretrained(QWEN))
    assert len(q["loc_mass"]) == 12 and all(q[w]["cap_single"] for w in q if w != "loc_mass")
    m = target_ids(AutoTokenizer.from_pretrained(MISTRAL))
    assert len(m["loc_mass"]) == 8 and [w for w in m if w != "loc_mass" and m[w]["cap_single"]] == ["box", "basket"]


def test_mask_shapes():
    m = knockout_mask(5, [(3, 1)], B=2)
    assert m.dtype == torch.bool and m.shape == (2, 1, 5, 5) and not m[1, 0, 3, 1] and m[1, 0, 3, 0] and not m[0, 0, 1, 2]
    e = knockout_mask(5, [(3, 1)], dtype=torch.bfloat16, impl="eager")
    assert e.dtype == torch.bfloat16 and e[0, 0, 3, 1] == torch.finfo(torch.bfloat16).min and e[0, 0, 3, 0] == 0
    assert knockout_masks(5, [[], [(3, 1)]]).shape == (2, 1, 5, 5)
    with pytest.raises(AssertionError):
        knockout_mask(5, [(3, 0)])


def test_knockout_exact(mt):
    model, tok = mt
    core, tb, ib, is_, off, p, g = item(tok, "AFTER")
    T, nL = ib.shape[1], len(blocks(model))
    cand = g["R_cand"]
    pairs = [(r, p) for r in cand]
    ans = [(T - 1, p)]
    lg = {}
    for impl in ("sdpa", "eager"):
        model.set_attn_implementation(impl)
        lg0 = model(ib, use_cache=False).logits[0, -1]
        lg1 = model(ib, attention_mask=mask_for_model(model, T, []), use_cache=False).logits[0, -1]
        assert torch.allclose(lg0, lg1, atol=1e-4), impl                       # (i) causal 4D mask is a no-op
        lgK = model(ib, attention_mask=mask_for_model(model, T, pairs), use_cache=False, logits_to_keep=1).logits[0, -1]
        assert (lgK - lg0).abs().max() > 1e-2
        lgA = model(ib, attention_mask=mask_for_model(model, T, ans), use_cache=False, logits_to_keep=1).logits[0, -1]
        lgb = model(ib.expand(3, -1), attention_mask=masks_for_model(model, T, [[], pairs, ans]), use_cache=False, logits_to_keep=1).logits[:, -1]
        for row, ref in zip(lgb, (lg0, lgK, lgA)):                                  # (iv) per-row masks == separate runs
            assert torch.allclose(row, ref, atol=1e-4), impl
        lg[impl] = lgK
        if impl == "eager":                                                          # (ii) weights exactly 0, rows renormalised
            a0 = model(ib, use_cache=False, output_attentions=True).attentions
            aK = model(ib, attention_mask=mask_for_model(model, T, pairs), use_cache=False, output_attentions=True).attentions
            assert all(aK[l][0, :, r, p].abs().max().item() == 0.0 for l in range(nL) for r in cand)
            keep = [j for j in range(T) if j != p]
            for r in cand:
                ren = a0[0][0, :, r][:, keep] / (1 - a0[0][0, :, r, p:p + 1])
                assert torch.allclose(aK[0][0, :, r][:, keep], ren, atol=1e-5)
            other = [i for i in range(T) if i not in cand]
            assert torch.allclose(aK[0][0, :, other], a0[0][0, :, other], atol=1e-6)
    assert torch.allclose(lg["sdpa"], lg["eager"], atol=1e-4)                       # (iii) sdpa == eager under the mask
    # (v) with K/V clamps from S at p: KV_S without knockout == source run; under M8 every clamp row == self-clamp row
    model.set_attn_implementation("sdpa")
    L = range(nL)
    with capture_kv(model, [p], L) as tB:
        model(ib, use_cache=False, logits_to_keep=1)
    with capture_kv(model, [p], L) as tS:
        lgS = model(is_, use_cache=False, logits_to_keep=1).logits[0, -1]
    rows = [lambda l, ch: "B", lambda l, ch: "S" if ch == "k" else "B", lambda l, ch: "S" if ch == "v" else "B", lambda l, ch: "S"]
    tabs = stack_rows({"B": {k: v[0] for k, v in tB.items()}, "S": {k: v[0] for k, v in tS.items()}}, rows, L)
    with clamp_kv(model, [p], tabs, L):
        full = model(ib.expand(4, -1), attention_mask=mask_for_model(model, T, [], B=4), use_cache=False, logits_to_keep=1).logits[:, -1]
        cut = model(ib.expand(4, -1), attention_mask=mask_for_model(model, T, [(r, p) for r in g["R_all"]], B=4), use_cache=False, logits_to_keep=1).logits[:, -1]
    assert torch.allclose(full[3], lgS, atol=1e-4) and (full[1] - full[0]).abs().max() > 1e-2
    assert torch.allclose(cut[1:], cut[0].expand(3, -1), atol=1e-5)


# ---- experiments/attention_knockout.py on the tiny model (part (b) of P-2026-10-05-G)
def test_run_item_exactness(mt, tmp_path):
    from experiments import attention_knockout as ak
    model, tok = mt
    model.set_attn_implementation("sdpa")
    core = make_cores(1, random.Random(0))[0]
    tidx = target_ids(tok)
    items = {arm: ak.run_item(model, tok, core, arm, "cpu", tidx, emit_steps=12, chunk=5) for arm in ("AFTER", "NONE")}
    it = items["AFTER"]
    assert set(it["m"]) == set(ak.MASKS) and set(items["NONE"]["m"]) == set(ak.NONE_MASKS) and it["pos"] == 55 and it["len"] == 102
    for arm, r in items.items():
        for M in r["m"]:
            rows = r["m"][M]
            assert abs(rows["ID"]["m"] - rows["ID2"]["m"]) < 1e-5, (arm, M)              # duplicate self-clamp row: noise floor
            if M == "M8":                                                                 # nothing after p sees p: every clamp row == self-clamp
                assert all(abs(rows[k]["m"] - rows["ID"]["m"]) < 1e-4 and abs(rows[k]["lp"]["S"] - rows["ID"]["lp"]["S"]) < 1e-4 for k in rows), (arm, M)
        P = {M: ak.per_core([r], arm, M) for M in r["m"]}
        v = {M: next(iter(P[M].values())) for M in P}
        assert abs(v["M8"]["idK"]) < 1e-4 and abs(v["M8"]["idV"]) < 1e-4 and abs(v["M8"]["dKV"]) < 1e-4, (arm, v["M8"])
        assert v["M0"]["floor_clean"] < 1e-4 and abs(r["m"]["M0"]["KV_S"]["m"] - r["clean"]["S"]["m"]) < 1e-4, arm   # M0 rows == clean runs
        assert abs(r["m"]["M0"]["KV_X"]["m"] - r["clean"]["X"]["m"]) < 1e-4 and v["M0"]["n_init"] == 2
        assert v["M0"]["dKV"] != 0 and abs(v["M0"]["dKV"]) > 1e-2
    g = it["groups"]
    pairs = ak.mask_pairs(g)
    assert pairs["M1"] == [(r, 55) for r in g["R_cand"]] and pairs["M3"] == [(101, 55)] and len(pairs["M2"]) == 12
    assert set(pairs["M4"]) == set(pairs["M1"]) | set(pairs["M3"]) and set(pairs["M5"]) | set(pairs["M6"]) == set(pairs["M1"])
    assert set(pairs["M7"]) | set(pairs["M1"]) == set(pairs["M8"]) and all(c == 54 for _, c in pairs["M2b"])
    assert abs(it["m"]["M1"]["ID"]["m"] - it["m"]["M0"]["ID"]["m"]) > 1e-3                 # M1 is not a silent no-op
    chunked = ak.run_item(model, tok, core, "AFTER", "cpu", tidx, emit_steps=0, chunk=0)   # one 96-row batch == chunks of 5 masks
    assert all(abs(chunked["m"][M][k]["m"] - it["m"][M][k]["m"]) < 1e-4 for M in it["m"] for k in it["m"][M])
    em = items["NONE"]["emit"]
    assert em["a_emit"] is None or (em["a_emit"] >= items["NONE"]["len"] - 1 and set(em["m"]) == {"M0", "M3e"})
    if em["a_emit"] == items["NONE"]["len"] - 1:                                             # a_emit == a: M3' is M3
        assert all(abs(em["m"]["M3e"][k]["m"] - items["NONE"]["m"]["M3"][k]["m"]) < 1e-4 for k in em["m"]["M3e"])
    # the part scorer prints every gate and prediction verdict on a one-model, one-core output
    import io
    from analysis.stage5_parts import knockout as sc
    json.dump({"provenance": {"skipped_items": 0, "vocab": {}, "attn_implementation": "sdpa"}, "results": list(items.values())}, open(tmp_path / "Tiny_s0.json", "w"))
    buf = io.StringIO()
    sc.score(str(tmp_path), ["Tiny", "Qwen2.5-14B-Instruct"], stage3b=str(tmp_path / "none"), out=lambda s: buf.write(s + "\n"))
    txt = buf.getvalue()
    for line in ("Gate b ->", "G5 ->", "G6 ->", "G7 ->", "G8 ->", "Qwen2.5-14B-Instruct: MISSING", "Gate b not measured in ['Qwen2.5-14B-Instruct']"):
        assert line in txt, line   # a missing file is MISSING (Gate b None), reported apart from an M8 failure of a measured model


# ---- the part (b) scorer's rules on synthetic knockout files
def synth_knockout(root, model, n=150, seed=0, gate_ok=True, takeover=True, replaced=False):
    """Items in attention_knockout's JSON shape: per arm x mask the 8 clamp rows with chosen ID_K / ID_V / span, the
    readouts (restricted argmax, on-target flags, location mass). ``takeover``: M1 keeps the answer and raises ID_V
    (H_redundant); ``replaced``: M1 drops ID_V and the answer (H_replaced); ``gate_ok`` False leaves ID_K under M8."""
    rng, cores = np.random.default_rng(seed), make_cores(n, random.Random(0))
    V = {("AFTER", "M0"): (20, 6, 26, 0.9), ("AFTER", "M2"): (20, 6, 26, 0.9), ("AFTER", "M3"): (18, 6, 24, 0.9), ("AFTER", "M8"): (0, 0, 0, 0.1),
         ("P1", "M0"): (20, 6, 26, 0.9), ("P1", "M3"): (18, 6, 24, 0.9), ("P1", "M8"): (0, 0, 0, 0.1),
         ("POST", "M0"): (6, 10, 16, 0.9), ("POST", "M1"): (1.5, 12, 14, 0.9), ("POST", "M8"): (0, 0, 0, 0.1),
         ("NONE", "M0"): (1, 16, 20, 0.9), ("NONE", "M3"): (1, 6, 10, 0.9), ("NONE", "M8"): (0, 0, 0, 0.1)}
    for f in ("AFTER", "P1"):
        V[f, "M1"] = (2, 2, 6, 0.3) if replaced else (2, 12, 14, 0.9) if takeover else (2, 6, 14, 0.9)
        V[f, "M4"] = (2, 1, 6, 0.3) if replaced else (2, 5, 8, 0.9)
    for f in ("AFTER", "P1", "POST"):
        for M in ("M2", "M2b", "M3", "M5", "M6", "M7", "Mq", "Mpost"):
            V.setdefault((f, M), V[f, "M0"])
        V.setdefault((f, "M4"), V[f, "M1"])
    for M in ("Mq", "Mpost"):
        V[("NONE", M)] = V["NONE", "M0"]
    res = []
    for arm in ("AFTER", "P1", "POST", "NONE"):
        for core in cores:
            m = {}
            for M in masks_of(arm):
                idK, idV, span, loc = V[arm, M]
                if M == "M8" and not gate_ok:
                    idK, loc = 3.0, 0.9
                e = rng.normal(0, 0.3)
                rows = {}
                for lab in ("ID", "ID2", "K_S", "V_S", "KV_S", "K_X", "V_X", "KV_X"):
                    lp = {"S": -3.0, "B": -0.5, "X": -3.0, "init": -4.0}
                    lp["S"] += (idK + e) * (lab == "K_S") + (idV + e) * (lab == "V_S") + (span + e) * (lab == "KV_S")
                    lp["X"] += (idK + e) * (lab == "K_X") + (idV + e) * (lab == "V_X") + (span + e) * (lab == "KV_X")
                    answer = M != "M8" and loc > 0.5
                    rows[lab] = {"lp": lp, "m": lp["S"] - lp["B"], "mass": 0.9, "loc_mass": loc, "top5": [[1, 2, 3, 4, 5], [-0.1] * 5],
                                 "argmax_cand": (core["source"] if lab in ("KV_S", "V_S") else core["base"]) if answer else "shelf",
                                 "on": {"B": answer and lab not in ("KV_S", "V_S"), "S": answer and lab in ("KV_S", "V_S")}}
                m[M] = rows
            clean = {k: {"lp": dict(m["M0"]["ID"]["lp"]), "m": m["M0"]["ID"]["m"], "argmax_cand": core["source" if k == "S" else "base"]} for k in ("B", "S", "X")}
            res.append({"core": core, "X": pick_x(core), "arm": arm, "view": "direct", "pos": 55, "len": 102, "n_layers": 24,
                        "groups": {"C_init": [33, 44]}, "clean": clean, "m": m})
    Path(root).mkdir(parents=True, exist_ok=True)
    prov = {"skipped_items": 0, "attn_implementation": "sdpa", "dtype": "torch.bfloat16", "vocab": {}, "c_init_counts": {"2": n}, "len_core0": {}}
    json.dump({"provenance": prov, "results": res}, open(Path(root) / f"{model}_s0.json", "w"))


def test_scorer_synthetic_rules(tmp_path):
    import io
    from analysis.stage5_parts import knockout as sc

    def run(root):
        buf = io.StringIO()
        r = sc.score(str(root), stage3b=str(root / "none"), out=lambda s: buf.write(s + "\n"))
        return r, buf.getvalue()

    for m in sc.MODELS:
        synth_knockout(tmp_path, m)
    r, txt = run(tmp_path)
    assert all(r["gate"].values()) and r["G5"] and r["G6"] and r["G7"] and not r["H_replaced"] and r["G8"] == (True, True, True)
    assert "G7 -> H_redundant MET; H_replaced NOT MET" in txt and "ALL GATES MET" in txt
    # H_replaced in every model: G7 not met, the alternative met; partial takeover when neither
    for m in sc.MODELS:
        synth_knockout(tmp_path, m, replaced=True)
    r, txt = run(tmp_path)
    assert not r["G7"] and r["H_replaced"] and "H_replaced MET" in txt and r["G8"][2] is False   # G8c needs G7a
    for m in sc.MODELS:
        synth_knockout(tmp_path, m, takeover=False)
    r, txt = run(tmp_path)
    assert not r["G7"] and not r["H_replaced"] and "partial takeover" in txt
    # Gate b fails in one model (ID_K survives M8): the part is flagged
    synth_knockout(tmp_path, sc.MODELS[1], gate_ok=False)
    r, txt = run(tmp_path)
    assert r["gate"][sc.MODELS[1]] is False and "GATE b FAILED in ['Qwen2.5-14B-Instruct']" in txt
