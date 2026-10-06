"""Part (e) of preregistration G: ckeys.ioi (cores, arms, tokenisation on 200 cores x 3 tokenizers), the GPT-2 small FP32
exactness of experiments/ioi_factorial.py through the fused c_attn and Qwen2.5-0.5B through k_proj/v_proj, the IOI
RowTask of row_restricted_keys, the attention probe, and analysis/stage5_parts/ioi.py on synthetic fixtures (sign
handling with a negative ID_K) and on the TEST_MODE output (every gate and verdict line printed)."""
import io
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import experiments.ioi_attention as ia
import experiments.ioi_factorial as ff
import experiments.row_restricted_keys as rr
from analysis.stage5_parts import ioi as sc
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.ioi import (ARMS, LIST_ARMS, NAMES, SEED, OBJECTS, PLACES, TEMPLATES, RowTask, check_occurrences, encode_runs, identity_measures,
                       inline, listing, make_cores, name_ids, raw_prompt, sentence)
from ckeys.interventions import blocks

torch.set_grad_enabled(False)
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
# spec ranges (p, length) per tokenizer and arm, 200 seed-0 cores
RANGES = {"gpt2": ("gpt2", False, {"PLAIN": (2, 7, 15, 21), "AFTER": (2, 7, 29, 35)}),
          "qwen": ("Qwen/Qwen2.5-7B-Instruct", True, {"AFTER": (25, 30, 63, 69), "QUESTION": (25, 30, 61, 67), "PLAIN": (1, 6, 14, 20)}),
          "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", True, {"AFTER": (24, 29, 64, 70), "QUESTION": (24, 29, 62, 68)})}


def test_cores_and_arms():
    assert (len(TEMPLATES), len(NAMES), len(PLACES), len(OBJECTS)) == (15, 197, 19, 18) and not {"cafe", "necklace", "snack"} & (set(PLACES) | set(OBJECTS))
    cores = make_cores(200, random.Random(0))
    c0 = cores[0]
    assert c0 == dict(template=6, pattern="ABBA", place="beach", object="ticket", io_b="Ruth", io_s="Erik", io_x="Joan", subj="Charles", order=[0, 1, 3, 2])
    assert sentence(c0, "Ruth") == "When Ruth and Charles got a ticket at the beach, Charles decided to give it to"
    assert sentence(dict(c0, template=0, pattern="BABA"), "Ruth") == "Then, Charles and Ruth went to the beach. Charles gave a ticket to"
    assert listing(c0) == "Choices: Ruth, Erik, Charles, Joan"
    assert inline(c0, "Ruth", False) == "When Ruth and Charles got a ticket at the beach ( Ruth, Erik, Charles, Joan), Charles decided to give it to"
    assert inline(c0, "Ruth", True) == "( Ruth, Erik, Charles, Joan) When Ruth and Charles got a ticket at the beach, Charles decided to give it to"
    assert [sum(c["order"].index(0) == s for c in cores) for s in range(4)] == [40, 55, 58, 47]
    assert sum(c["pattern"] == "ABBA" for c in cores) == 100 and all(c["pattern"] == ("ABBA" if i % 2 == 0 else "BABA") for i, c in enumerate(cores))
    hist = [sum(c["template"] == t for c in cores) for t in range(15)]
    assert min(hist) == 9 and max(hist) == 22 and len({len({c[k] for k in ("io_b", "io_s", "io_x", "subj")}) for c in cores}) == 1
    sam = dict(c0, io_b="Sam", subj="Samuel", template=0, pattern="BABA")  # whole-word insertion, not str.index
    assert inline(sam, "Sam", False) == "Then, Samuel and Sam went to the beach ( Sam, Erik, Samuel, Joan). Samuel gave a ticket to"
    assert raw_prompt("AFTER", c0, "Ruth", True) == ("Complete the sentence with the right name.\n\nSentence: " + sentence(c0, "Ruth") +
                                                     "\nChoices: Ruth, Erik, Charles, Joan\nAnswer with one name.\nAnswer:")
    assert raw_prompt("BEFORE", c0, "Ruth", True) == ("Complete the sentence with the right name.\nChoices: Ruth, Erik, Charles, Joan\n\nSentence: " +
                                                      sentence(c0, "Ruth") + "\nAnswer with one name.\nAnswer:")
    assert raw_prompt("QUESTION", c0, "Ruth", True) == ("Complete the sentence with the right name.\n\nSentence: " + sentence(c0, "Ruth") +
                                                        "\nQuestion: Which name completes the sentence?\nAnswer with one name.\nAnswer:")
    assert raw_prompt("AFTER", c0, "Ruth", False) == sentence(c0, "Ruth") + "\nChoices: Ruth, Erik, Charles, Joan\nAnswer:"
    assert raw_prompt("BEFORE", c0, "Ruth", False) == "Choices: Ruth, Erik, Charles, Joan\n" + sentence(c0, "Ruth") + "\nAnswer:"
    assert raw_prompt("QUESTION", c0, "Ruth", False) == sentence(c0, "Ruth") + "\nQuestion: Which name completes the sentence?\nAnswer:"
    assert raw_prompt("PLAIN", c0, "Ruth", True) == sentence(c0, "Ruth") and raw_prompt("INLINE", c0, "Ruth", True) == inline(c0, "Ruth", False)


@pytest.mark.parametrize("fam", list(RANGES))
def test_tokenisation(fam):
    name, chat, ranges = RANGES[fam]
    tok = AutoTokenizer.from_pretrained(name)
    cores = make_cores(200, random.Random(0))
    for w in NAMES + PLACES + OBJECTS:
        assert len(tok.encode(" " + w, add_special_tokens=False)) == 1, w
    for arm in ARMS:
        ps, lens = [], []
        for c in cores:
            ids = encode_runs(tok, c, arm, chat)
            assert ids is not None, (fam, arm, c)
            check_occurrences(tok, c, arm, ids["B"])
            if fam != "qwen":
                assert ids["B"][0, 0].item() == tok.bos_token_id
            ps.append((ids["B"][0] != ids["S"][0]).nonzero().item())
            lens.append(ids["B"].shape[1])
        if arm in ranges:
            assert (min(ps), max(ps), min(lens), max(lens)) == ranges[arm], (fam, arm, min(ps), max(ps), min(lens), max(lens))
    with pytest.raises(AssertionError):  # one IO occurrence too many fails the occurrence check
        ids = encode_runs(tok, cores[0], "AFTER", chat)["B"]
        check_occurrences(tok, cores[0], "AFTER", torch.cat([ids, torch.tensor([[name_ids(tok, cores[0])["B"]]])], 1))


def _clean_identity(r):
    """ID_KV computed from the clean runs (the exact clamp makes KV_S / KV_X rows equal to the clean S / X runs)."""
    b, s, x = (r["clean"][k]["lp"] for k in ("B", "S", "X"))
    return 0.5 * ((s["S"] - b["S"] - (x["S"] - b["S"])) + (x["X"] - b["X"] - (s["X"] - b["X"])))


def test_gpt2_fp32_exact():
    model, tok = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval(), AutoTokenizer.from_pretrained("gpt2")
    cores = make_cores(4, random.Random(0))
    for arm in ("PLAIN", "INLINE", "INLINE_BEFORE", "AFTER"):
        for c in cores[:2]:
            r = ff.run_item(model, tok, c, arm, chat=False, bos=None, device="cpu")
            assert r["floor_B"] <= 1e-3 and r["floor_S"] <= 1e-3 and r["n_layers"] == 12 and not r["chat"]
            m = identity_measures(r)
            assert abs(m["idKV"] - _clean_identity(r)) < 2e-3  # KV rows reproduce the clean runs
            assert set(r["m"]) == {"ID@0", "K_S@0", "V_S@0", "KV_S@0", "K_X@0", "V_X@0", "KV_X@0", "K_S@4", "V_S@4", "KV_S@4"}
            assert abs(m["d"]["KV_S@0"] - (r["clean"]["S"]["m"] - r["clean"]["B"]["m"])) < 2e-3
            assert abs(m["d"]["K_S@0"]) + abs(m["d"]["V_S@0"]) > 0.05  # the single channels do something
    # the batched K_S row equals a single-row clamp of (K from S, V from B) through the same c_attn slices; a key-only
    # hook differs (p attends to itself, so its own later-layer values drift), which is why V is held at B's
    c, arm = cores[0], "AFTER"
    ids = encode_runs(tok, c, arm, False)
    p, L = (ids["B"][0] != ids["S"][0]).nonzero().item(), range(12)
    kv = {}
    for name in ("B", "S"):
        with capture_kv(model, [p], L) as K:
            model(ids[name], use_cache=False)
        kv[name] = {k: v[0] for k, v in K.items()}
    with clamp_kv(model, [p], {(l, "k"): kv["S"][(l, "k")] for l in L} | {(l, "v"): kv["B"][(l, "v")] for l in L}, L):
        lp = torch.log_softmax(model(ids["B"], use_cache=False).logits[0, -1].float(), -1)
    with clamp_kv(model, [p], {(l, "k"): kv["S"][(l, "k")] for l in L}, L, "k"):
        lpk = torch.log_softmax(model(ids["B"], use_cache=False).logits[0, -1].float(), -1)
    r = ff.run_item(model, tok, c, arm, False, None, "cpu")
    tid = name_ids(tok, c)
    assert abs((lp[tid["S"]] - lp[tid["B"]]).item() - r["m"]["K_S@0"]["m"]) < 1e-4
    assert abs((lpk[tid["S"]] - lpk[tid["B"]]).item() - r["m"]["K_S@0"]["m"]) > 1e-2


def test_qwen_chat_exact():
    model, tok = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval(), AutoTokenizer.from_pretrained(QWEN)
    c = make_cores(1, random.Random(0))[0]
    for arm in ("AFTER", "PLAIN"):
        r = ff.run_item(model, tok, c, arm, chat=True, bos=None, device="cpu")
        assert r["chat"] == (arm == "AFTER") and r["floor_B"] <= 1e-3 and r["floor_S"] <= 1e-3
        assert abs(identity_measures(r)["idKV"] - _clean_identity(r)) < 2e-3
    assert r["pos"] in range(1, 7) and ff.auto_chat(QWEN) and not ff.auto_chat("gpt2") and not ff.auto_chat("Qwen/Qwen2.5-7B")


def test_rowtask_gpt2():
    model, tok = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval(), AutoTokenizer.from_pretrained("gpt2")
    task = RowTask(chat=False)
    res = rr.run(model, tok, task, ["AFTER", "INLINE", "QUESTION"], 2, task.seed, log=lambda s: None)
    assert len(res) == 6
    for r in res:
        assert abs(r["m"]["none"] - r["m_B"]) < 1e-4 and abs(r["m"]["all"] - r["m_B"]) > 1e-2
        if r["arm"] in ("AFTER", "INLINE"):
            assert r["sizes"]["options"] == 4 and r["sizes"]["options_sx"] == 2 and r["sizes"]["choices"] >= 8
            assert set(r["m"]) == {"self", "choices", "options", "options_sx", "sentence_tail", "tail", "rest_after_p", "all", "none"}
        else:
            assert "options" not in r["sizes"] and r["sizes"]["tail"] >= 8
        assert r["sizes"]["sentence_tail"] >= 5 and r["sizes"]["rest_after_p"] == 0
        if r["arm"] == "AFTER":
            assert r["sizes"]["tail"] == 3  # '\n', 'Answer', ':'
    c, L = task.cores(2)[0], range(12)
    raw_b, raw_s = task.prompts(c, "AFTER")
    assert not task.chat and task.prompts(c, "AFTER") == (raw_prompt("AFTER", c, c["io_b"], False), raw_prompt("AFTER", c, c["io_s"], False))
    _, ib, _ = rr.encode_with_offsets(tok, raw_b, chat=False)
    _, is_, _ = rr.encode_with_offsets(tok, raw_s, chat=False)
    p = (ib[0] != is_[0]).nonzero().item()
    with capture_kv(model, [p], L, "k") as K:
        model(is_, use_cache=False)
    iS, iB = task.targets(tok, c, "AFTER")
    with clamp_kv(model, [p], K, L, "k"):
        lp = torch.log_softmax(model(ib, use_cache=False).logits[0, -1].float(), -1)
    r = next(x for x in res if x["arm"] == "AFTER")
    assert abs(r["m"]["all"] - (lp[iS] - lp[iB]).item()) < 1e-4
    chat_task = RowTask()
    chat_task.prompts(c, "AFTER")
    assert chat_task.chat
    chat_task.prompts(c, "PLAIN")
    assert not chat_task.chat
    assert "rows=options_sx" in rr.summarize(res, ["AFTER"]) and "rows=tail" in rr.summarize(res, ["QUESTION"])


def test_attention_probe():
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32, attn_implementation="eager").eval()
    mats, n, skipped = ia.run(model, tok, make_cores(2, random.Random(0)), log=lambda s: None)
    assert n == 2 and skipped == 0 and set(mats) == set(ia.QUANT) and np.array(mats["PLAIN END->p"]).shape == (12, 12)
    M = {q: np.array(v) for q, v in mats.items()}
    assert M["PLAIN S2->S1"][3, 0] > 0.5 and M["INLINE IO_B(list)->p"][3, 0] > 0.5   # duplicate-token head 3.0 fires on the repeat
    assert M["INLINE IO_S(list)->p"][3, 0] < 0.1 < M["INLINE K_S: IO_S(list)->p"][3, 0]  # and follows the clamped key
    assert M["PLAIN END->p"][9, 9] > 0.5                                             # name mover 9.9 reads the IO
    assert "top-5" in ia.summarize(mats, n)


# ---- scorer fixtures: items in the ioi_factorial JSON shape with chosen ID_K / ID_V / ID_KV per arm
def _item(core, arm, idK, idV, idKV, rng, L=12):
    base = {"S": -6.0, "B": -0.5, "X": -6.5, "Subj": -2.0}
    noise = lambda: rng.normal(0, 0.15)  # noqa: E731

    def row(ch, which, val):
        lp = dict(base)
        lp[which] += val + noise()
        return {"m": lp["S"] - lp["B"], "lp": lp}

    m = {"ID@0": {"m": base["S"] - base["B"], "lp": dict(base)}}
    for ch, val in (("K", idK), ("V", idV), ("KV", idKV)):
        m[f"{ch}_S@0"], m[f"{ch}_X@0"] = row(ch, "S", val), row(ch, "X", val)
    for ch in ("K", "V", "KV"):
        m[f"{ch}_S@{round(0.3 * L)}"] = m[f"{ch}_S@0"]
    clean = {"B": {"lp": dict(base), "m": base["S"] - base["B"]}, "S": {"lp": m["KV_S@0"]["lp"], "m": m["KV_S@0"]["m"]},
             "X": {"lp": m["KV_X@0"]["lp"], "m": m["KV_X@0"]["m"]}}
    return {"core": core, "arm": arm, "chat": arm in ("AFTER", "BEFORE", "QUESTION"), "pos": 3, "len": 20, "n_layers": L,
            "pattern": core["pattern"], "template": core["template"], "clean": clean, "m": m, "floor_B": 1e-5, "floor_S": 1e-5}


def _fixture(root, model, spec, n=40, seed=1):
    rng, cores = np.random.default_rng(seed), make_cores(n, random.Random(SEED))
    res = [_item(c, arm, *spec[arm], rng) for arm in spec for c in cores]
    (root / "ioi").mkdir(parents=True, exist_ok=True)
    json.dump({"provenance": {"skipped_items": 0, "chat": True, "bos": None, "args": {"dtype": "float32"}, "assert_exact": True, "label": "synthetic"},
               "results": res}, open(root / "ioi" / f"{model}_s{SEED}.json", "w"))


def _score(root, **kw):
    buf = io.StringIO()
    out = sc.score(root, out=lambda s: buf.write(s + "\n"), **kw)
    return out, buf.getvalue()


def test_scorer_sign_handling(tmp_path):
    base = {"PLAIN": (0.05, 8.0, 8.0), "QUESTION": (0.1, 9.0, 10.0), "BEFORE": (0.1, 2.0, 2.0)}
    _fixture(tmp_path, "Qwen2.5-7B-Instruct", base | {"AFTER": (4.0, 1.0, 5.0)})      # selection read
    _fixture(tmp_path, "Mistral-7B-Instruct-v0.3", base | {"AFTER": (-4.0, 1.0, 5.0)})  # inhibitory read
    _fixture(tmp_path, "gpt2", {"PLAIN": (0.05, 8.0, 8.0), "INLINE": (-1.5, 6.0, 5.0), "INLINE_BEFORE": (0.1, 6.0, 6.0)})
    r, text = _score(tmp_path)
    q, mi, g = r["pred"]["Qwen2.5-7B-Instruct"], r["pred"]["Mistral-7B-Instruct-v0.3"], r["pred"]["gpt2"]
    assert all(r["gates"][m][a] for m in r["gates"] for a in r["gates"][m])
    assert q["G18"] and mi["G18"] and g["G18"] and r["final"]["G18"]
    assert q["G19a"] and q["G19b"] and q["G20"] and q["G21a"] and q["G21b"]
    assert mi["G19a"] and mi["G19b"] is False and mi["G20"] is None and mi["G21a"] and mi["G21b"]  # sgn = -1 fixes the contrasts
    assert "ALTERNATIVE: inhibitory key read" in text and "inhibitory key read at ['Mistral-7B-Instruct-v0.3']" in text
    assert r["final"]["G19a"] and not r["final"]["G19b"] and not r["final"]["G20"] and r["final"]["G21a"] and r["final"]["G21b"]
    assert g["G21c"] and g["G22a"] and g["G22b"] and r["final"]["G22a"] and r["final"]["G22b"] and r["final"]["G21c"]
    assert re.search(r"Mistral.*\n(.*\n)*?.*AFTER ID_K -4\.\d\d \[.*\] sign -", text)
    assert "signed contrast sgn x [ID_K(AFTER) - ID_K(QUESTION)] +4." in text and "dropped cores 0" in text
    # an additive mixture and a positive INLINE read are reported as such
    _fixture(tmp_path, "Mistral-7B-Instruct-v0.3", base | {"AFTER": (4.0, 5.0, 8.0)})
    _fixture(tmp_path, "gpt2", {"PLAIN": (0.05, 8.0, 8.0), "INLINE": (1.5, 6.0, 5.0), "INLINE_BEFORE": (0.1, 6.0, 6.0)})
    r, text = _score(tmp_path)
    assert r["pred"]["Mistral-7B-Instruct-v0.3"]["G20"] is False and "ALTERNATIVE: additive mixture" in text
    assert r["pred"]["gpt2"]["G22a"] and r["pred"]["gpt2"]["G22b"] is False
    # G21a's contrast is evaluated where G19a holds (a null AFTER gives a noise sign): not evaluable here; G22b stands on its own
    _fixture(tmp_path, "Qwen2.5-7B-Instruct", base | {"AFTER": (1.0, 1.0, 5.0)})       # |mean| < 2.0: G19a not met
    _fixture(tmp_path, "gpt2", {"PLAIN": (0.05, 8.0, 8.0), "INLINE": (-1.5, 6.0, 5.0), "INLINE_BEFORE": (-2.0, 6.0, 6.0)})  # vs INLINE_BEFORE fails (a)
    r, text = _score(tmp_path)
    assert r["pred"]["Qwen2.5-7B-Instruct"]["G19a"] is False and r["pred"]["Qwen2.5-7B-Instruct"]["G21a"] is None and r["pred"]["Qwen2.5-7B-Instruct"]["G20"] is None
    assert "G19a not met: the contrast is not evaluable -> NOT EVALUABLE" in text.split("## Qwen2.5-7B-Instruct")[1].split("G21c")[0]
    assert r["pred"]["gpt2"]["G22a"] is False and r["pred"]["gpt2"]["G22b"] is True and r["pred"]["gpt2"]["G21c"] is False
    assert r["models"]["G18"] == ["gpt2", "Qwen2.5-7B-Instruct", "Mistral-7B-Instruct-v0.3"] and r["models"]["G22b"] == ["gpt2"] and r["models"]["G20"] == sc.PAIR
    _fixture(tmp_path, "Qwen2.5-7B-Instruct", base | {"AFTER": (4.0, 1.0, 5.0)})
    # a failing gate (four-way) makes the cell not evaluable
    f = tmp_path / "ioi" / f"gpt2_s{SEED}.json"
    d = json.load(open(f))
    for it in d["results"]:
        if it["arm"] == "INLINE":
            it["clean"]["B"]["lp"]["Subj"] = 0.0
    json.dump(d, open(f, "w"))
    r, text = _score(tmp_path)
    assert r["gates"]["gpt2"]["INLINE"] is False and r["pred"]["gpt2"]["G22a"] is None and "G22a  GPT-2 small in-sentence re-mention opens a key read -> NOT EVALUABLE" in text
    # missing precondition (ID_KV < 3) -> G18 not evaluable
    _fixture(tmp_path, "gpt2", {"PLAIN": (0.05, 1.0, 1.0)})
    r, text = _score(tmp_path)
    assert r["pred"]["gpt2"]["G18"] is None and re.search(r"precondition ID_KV \+[01]\.\d\d \[.*\] >= 3 -> FAILS;.*-> NOT EVALUABLE", text)


def test_scorer_on_test_mode_output(tmp_path):
    ff.main(["--test", "--out", str(tmp_path / "ioi")])                                       # Qwen2.5-0.5B, n = 3, all arms
    ff.main(["--model", "gpt2", "--n", "2", "--test", "--out", str(tmp_path / "ioi")])         # GPT-2 small (--test keeps a given --model, n = 3)
    files = sorted(f.name for f in (tmp_path / "ioi").glob("*.json"))
    assert files == ["Qwen2.5-0.5B-Instruct_s1.json", "gpt2_s1.json"]
    d = json.load(open(tmp_path / "ioi" / "Qwen2.5-0.5B-Instruct_s1.json"))
    assert d["provenance"]["test_mode"] and d["provenance"]["assert_exact"] and d["provenance"]["skipped_items"] == 0 and len(d["results"]) == 18
    assert {r["arm"] for r in d["results"]} == set(ARMS) and all(r["chat"] == (r["arm"] in ("AFTER", "BEFORE", "QUESTION")) for r in d["results"])
    assert max(max(r["floor_B"], r["floor_S"]) for r in d["results"]) <= 1e-3
    r, text = _score(tmp_path, test=True)
    assert "TEST MODE" in text and r["pred"].keys() == {"gpt2", "Qwen2.5-0.5B-Instruct"}
    for m in ("gpt2", "Qwen2.5-0.5B-Instruct"):
        for arm in ARMS:
            assert re.search(rf"gate e {arm}\s+n=\s*\d+ two-way .* -> (passed|FAILED)", text), (m, arm)
    for pred in ("G18 ", "G19a", "G19b", "G20 ", "G21a", "G21b", "G21c", "G22a", "G22b"):
        assert re.search(rf"^  {pred}.*-> (MET|NOT MET|NOT EVALUABLE)", text, re.M), pred
    assert "7B pair = ['Qwen2.5-0.5B-Instruct']" in text  # which tiny-model cells are evaluable depends on the cores
    assert "ioi factorial MISSING" not in text
    (tmp_path / "ioi" / "gpt2_s1.json").unlink()
    r, text = _score(tmp_path, test=True)
    assert "ioi factorial MISSING" in text and r["final"]["G22a"] is False


def test_exact_violations_recorded_without_assert(tmp_path, monkeypatch):
    """--assert-exact no (IOI_EXACT=no) keeps the file and still counts the FP32 items above the 1e-3 floor; auto raises."""
    run = ff.run_item

    def noisy(*a, **k):
        r = run(*a, **k)
        return r if r is None else r | {"floor_B": 0.02}
    monkeypatch.setattr(ff, "run_item", noisy)
    monkeypatch.setenv("TEST_MODE", "1")   # FP32 on the CPU, keeps n = 1
    for flag, raises in (("no", False), ("auto", True)):
        argv = ["--model", "gpt2", "--n", "1", "--arms", "PLAIN", "--assert-exact", flag, "--out", str(tmp_path / flag)]
        if raises:
            with pytest.raises(SystemExit, match="clamp not exact in 1 items"):
                ff.main(argv)
        else:
            ff.main(argv)
        p = json.load(open(tmp_path / flag / "gpt2_s1.json"))["provenance"]
        assert p["assert_exact"] is raises and p["exact_violations"] == 1 and p["max_floor"] == 0.02 and p["device"] == "cpu"
