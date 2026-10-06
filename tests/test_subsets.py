"""ckeys.subsets (prereg G, part c): the arm rules and preflight, tokenisation and span checks on 300 cores (seeds 0 and
1) x the Qwen2.5 / Mistral-7B / OLMo-2 tokenizers, the registered arms through format_factorial.run_item and the
row-restricted groups at Qwen2.5-0.5B FP32 (S6 exact to POST), and analysis/stage5_parts/subsets.py on synthetic data."""
import io
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ckeys.subsets as sub
import experiments.row_restricted_keys as rr
from analysis.stage5_parts import subsets as sc
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.encoding import ALL_ARMS, ARM_BUILDERS, arm_span, build_prompt, candidate_ids, chat_text, raw_prompt
from ckeys.interventions import blocks
from ckeys.story import LOCATIONS, make_cores, pick_x, record
from experiments.format_factorial import run_item

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
TOKS = {"qwen": (QWEN, 55), "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", 54), "olmo": ("allenai/OLMo-2-1124-7B-Instruct", 58)}
torch.set_grad_enabled(False)


def cores300():
    return make_cores(150, random.Random(0)) + make_cores(150, random.Random(1))


def test_rules_and_registry():
    assert sub.check_rules(2000).startswith("subset rules ok")
    assert sub.check_rules(5, AutoTokenizer.from_pretrained(QWEN)).endswith("(token level, p checked)")
    assert set(sub.SUBSET_ARMS) <= set(ARM_BUILDERS) and all(ARM_BUILDERS[a].needs_core for a in sub.SUBSET_ARMS)
    assert [a for a in ALL_ARMS if a in sub.SUBSET_ARMS] == list(sub.SUBSET_ARMS)
    core = make_cores(1, random.Random(0))[0]
    X = pick_x(core)
    assert (core["base"], core["source"], X) == ("cabinet", "drawer", "box") and sub.others(core, X) == ["basket", "closet", "shelf"]
    r = record(core, "direct", core["base"])
    assert sub.named_set("S2", core, X) == ("box", "drawer") and sub.named_set("S3out", core, X) == ("basket", "cabinet", "closet")
    assert sub.sentence(("box", "drawer")) == "The room has a box and a drawer." and sub.sentence(("box",)) == "The room has a box."
    assert build_prompt("S3", r["story"], r["query"], core, X).endswith(
        "Bob does not see this happen. The room has a box, a drawer and a cabinet.\nQuestion: Where does Alice believe the lamp is?\nAnswer with one word.\nAnswer:")
    assert build_prompt("L3out", r["story"], r["query"], core, X).endswith("\nChoices: basket, cabinet, closet\nAnswer with one word.\nAnswer:")
    assert build_prompt("S6", r["story"], r["query"], core, X) == raw_prompt("POST", r["story"], r["query"])
    assert build_prompt("L6", r["story"], r["query"], core, X) == raw_prompt("AFTER", r["story"], r["query"])
    assert arm_span("L2", r["story"], r["query"], core, X) == "Choices: box, drawer" and ARM_BUILDERS["S4"].meta(core, X) == {"named": ["box", "basket", "drawer", "cabinet"], "k": 4}
    with pytest.raises(ValueError):
        raw_prompt("S2", r["story"], r["query"])  # core-dependent


@pytest.mark.parametrize("fam", list(TOKS))
def test_tokenisation_and_spans(fam):
    name, p_ref = TOKS[fam]
    tok = AutoTokenizer.from_pretrained(name)
    cid = candidate_ids(tok, "NONE")
    task = rr.get_task("belief", view="direct")
    bad, lens = [], {}
    for ci, core in enumerate(cores300()):
        X = pick_x(core)
        for arm in sub.SUBSET_ARMS:
            named = sub.named_set(arm, core, X)
            enc, text = {}, {}
            for k, loc in (("B", core["base"]), ("S", core["source"]), ("X", X)):
                r = record(core, "direct", loc)
                text[k] = chat_text(tok, build_prompt(arm, r["story"], r["query"], core, X))
                enc[k] = tok(text[k], add_special_tokens=False, return_offsets_mapping=True)
            ids = {k: v.input_ids for k, v in enc.items()}
            if len({len(v) for v in ids.values()}) != 1:
                bad.append((ci, arm, "len")); continue
            d = [i for i in range(len(ids["B"])) if ids["B"][i] != ids["S"][i]]
            dx = [i for i in range(len(ids["B"])) if ids["B"][i] != ids["X"][i]]
            if d != [p_ref] or dx != [p_ref]:
                bad.append((ci, arm, "p", d, dx)); continue
            after = [LOCATIONS[cid.index(t)] for t in ids["B"][p_ref + 1:] if t in cid]
            if after != list(named):
                bad.append((ci, arm, "named", after)); continue
            g = task.groups(tok, text["B"], torch.tensor([ids["B"]]), enc["B"].offset_mapping, p_ref, core, arm)
            if len(g["mention_words"]) != len(named) or [LOCATIONS[cid.index(ids["B"][i])] for i in g["mention_words"]] != list(named) or min(g["mention"]) <= p_ref:
                bad.append((ci, arm, "groups", g["mention_words"]))
            if ci == 0:
                lens[arm] = len(ids["B"])
    assert not bad, bad[:5]
    if fam == "qwen":  # the spec's core-0 lengths
        assert lens == {"S2": 97, "S3": 100, "S3out": 100, "S3half": 100, "S4": 103, "S4out": 103, "S6": 109, "L2": 94, "L3": 96,
                        "L3out": 96, "L4": 98, "L4out": 98, "L6": 102}


@pytest.fixture(scope="module")
def mt():
    return AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval(), AutoTokenizer.from_pretrained(QWEN)


def test_run_item_and_splice_exact(mt):
    model, tok = mt
    core = make_cores(1, random.Random(0))[0]
    s6, post, s2 = (run_item(model, tok, core, a, "direct", "cpu") for a in ("S6", "POST", "S2"))
    assert s6["pos"] == post["pos"] == s2["pos"] == 55 and s6["len"] == post["len"] == 109 and s2["len"] == 97
    assert s6["arm_meta"] == {"named": list(LOCATIONS), "k": 6} and s2["arm_meta"]["named"] == ["box", "drawer"] and "arm_meta" not in post
    for k in s6["m"]:  # identical prompts, identical FP32 numbers
        assert abs(s6["m"][k]["m"] - post["m"][k]["m"]) < 1e-5 and all(abs(s6["m"][k]["lp"][t] - post["m"][k]["lp"][t]) < 1e-5 for t in ("S", "B", "X"))
    assert abs(s2["m"]["ID@0"]["m"] - s2["clean"]["B"]["m"]) < 1e-4  # self-clamp == clean
    res = rr.run(model, tok, rr.get_task("belief", view="direct"), ["S2", "L3out"], 1, log=lambda s: None)
    for r, k in zip(res, (2, 3)):
        assert r["p"] == 55 and r["sizes"]["mention_words"] == k and abs(r["m"]["none"] - r["m_B"]) < 1e-4 and abs(r["m"]["all"] - r["m_S"]) > 1e-3
    raw_b, raw_s = rr.get_task("belief", view="direct").prompts(core, "S2")  # all rows == the single-sequence K_S clamp (exact)
    _, ib, _ = rr.encode_with_offsets(tok, raw_b)
    _, is_, _ = rr.encode_with_offsets(tok, raw_s)
    L = range(len(blocks(model)))
    with capture_kv(model, [55], L, "k") as K:
        model(is_, use_cache=False)
    with clamp_kv(model, [55], K, L, "k"):
        lp = torch.log_softmax(model(ib, use_cache=False).logits[0, -1].float(), -1)
    iS, iB = rr.get_task("belief", view="direct").targets(tok, core, "S2")
    assert abs(res[0]["m"]["all"] - (lp[iS] - lp[iB]).item()) < 1e-4
    assert "rows=mention_words" in rr.summarize(res, ["S2"])


# ---- scorer on synthetic factorial / splice files
def synth_item(core, arm, idK, idV, rng):
    X, rows = pick_x(core), {}
    for k in ("ID@0", "K_S@0", "V_S@0", "KV_S@0", "K_X@0", "V_X@0", "KV_X@0"):
        lp, e = {t: -3.0 for t in ("S", "B", "X", "init")}, rng.normal(0, 0.3)
        lp["S"] += (idK + e) * (k == "K_S@0") + (idV + e) * (k == "V_S@0")
        lp["X"] += (idK + e) * (k == "K_X@0") + (idV + e) * (k == "V_X@0")
        rows[k] = {"m": lp["S"] - lp["B"], "lp": lp}
    clean = {r: {"lp": {}, "m": 0.0, "mass": 0.5, "argmax_cand": core["source"] if r == "S" else core["base"]} for r in ("B", "S", "X")}
    return {"core": core, "X": X, "arm": arm, "view": "direct", "pos": 55, "len": 100, "n_layers": 24, "clean": clean, "m": rows}


def write_synth(root, model, spec, seed=0, n=150, skipped=0, splice=True):
    rng, cores = np.random.default_rng(seed + 1), make_cores(n, random.Random(seed))
    Path(root, "factorial").mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": {"skipped_items": skipped}, "results": [synth_item(c, a, k, v, rng) for a, (k, v) in spec.items() for c in cores]},
              open(Path(root, "factorial", f"{model}_s{seed}.json"), "w"))
    if splice:
        Path(root, "row_restricted").mkdir(parents=True, exist_ok=True)
        res = []
        for arm in ("S2", "S3", "L2", "L3", "S3out", "L3out", "S6", "L6"):
            for c in cores[:60]:
                e, fw = rng.normal(0, 0.2), 0.7 if arm != "L2" else 0.2
                m = {"self": 0.1, "story_tail": 0.0, "question": 0.1, "mention": 5 * fw + 0.3 + e, "mention_words": 5 * fw + e, "rest_after_p": 1.0, "all": 5 + e, "none": 0.0}
                res.append({"core": c, "arm": arm, "p": 55, "T": 100, "sizes": {g: 3 for g in m if g != "none"}, "m_B": -1.0, "m_S": 4.0, "m": {g: v - 1.0 for g, v in m.items()}})
        json.dump({"provenance": {"skipped_items": 0}, "results": res}, open(Path(root, "row_restricted", f"{model}_direct.json"), "w"))


def good_spec(post, none, after):
    return {"NONE": (none, 16.0), "POST": (post, 12.0), "AFTER": (after, 6.0), "S2": (0.8 * post, 12.0), "S3": (0.9 * post, 12.0),
            "S3out": (none + 0.2, 13.0), "S3half": (0.5 * post, 12.0), "S4": (0.95 * post, 12.0), "S4out": (0.9 * post, 12.0),
            "L2": (0.75 * after, 6.0), "L3": (0.85 * after, 6.0), "L3out": (none + 0.3, 15.0), "L4": (0.9 * after, 6.0), "L4out": (0.85 * after, 6.0)}


def test_scorer_synthetic(tmp_path):
    refs = {m: sc.REF3B[m] for m in sc.PRIMARY}
    for m, r in refs.items():
        write_synth(tmp_path, m, good_spec(r["POST"], r["NONE"], r["AFTER"]))
        write_synth(tmp_path, m, good_spec(r["POST"], r["NONE"], r["AFTER"]), seed=1, splice=False)
    buf = io.StringIO()
    res = sc.score(tmp_path, out=lambda s: buf.write(s + "\n"))
    txt = buf.getvalue()
    assert res["final"] == {"G9 S": True, "G9 L": True, "G10 S": True, "G10 L": True, "G11": True, "G12": False}  # L2 f_words 0.2
    assert "Gate 0 passed in 3/3" in txt and txt.count("Gate 0 -> passed") == 3 and "replicated in 3/3" in txt
    for line in ("G9 S ", "G9 L ", "G10 S", "G10 L", "G11 ", "G12 "):
        assert any(l.strip().startswith(line.strip()) and ("MET" in l) for l in txt.splitlines()), line
    assert "L2    n=60" in txt and "9/12 cells -> NOT MET" in txt
    # one model fails Gate 0 (POST off by 2 nats, skipped item): its predictions are not evaluable, every 3/3 line fails
    m = sc.PRIMARY[0]
    write_synth(tmp_path, m, good_spec(refs[m]["POST"] + 2.0, refs[m]["NONE"], refs[m]["AFTER"]), skipped=1)
    buf = io.StringIO()
    res = sc.score(tmp_path, out=lambda s: buf.write(s + "\n"))
    assert not any(res["final"].values()) and "Gate 0 passed in 2/3" in buf.getvalue() and "NOT EVALUABLE for this model (Gate 0 not passed)" in buf.getvalue()
    # --test mode: the 0.5B-like model has no reference and a null key effect -> no reference notice, lower bounds fail
    write_synth(tmp_path, "Qwen2.5-0.5B-Instruct", good_spec(-0.5, 0.1, 0.3), n=3)
    buf = io.StringIO()
    sc.score(tmp_path, test=True, out=lambda s: buf.write(s + "\n"))
    t = buf.getvalue()
    assert "TEST MODE" in t and "no stage-3b reference for Qwen2.5-0.5B-Instruct (TEST MODE" in t and "lower CI of ID_K(POST) > 0  ] " in t
    assert "G11" in t and "G12" in t and "G10 S" in t and "replicated in" in t
