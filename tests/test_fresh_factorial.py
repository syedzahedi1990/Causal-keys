"""Gate JB-G0 (preregistration J, part B; FP32, CPU, 1e-4 nats): the hooks and scorers that experiments/fresh_factorial.py
runs on the GPU, each against an independently computed reference.
1. The one-pass trie score (and score_cached, the JB-G3 fallback) equals separate plain forward passes for every form
   (ckeys.surface.score_reference), at Qwen2.5-0.5B under per-row K/V clamps, on real F prompts of BOTH lexicons.
2. run_item's clamp batch: a row's E, Sigma and L equal score_reference under a separately constructed single-row clamp.
3. run_item's generation batch (clean rows unclamped, five rows with per-row tables) equals cache-free stepwise argmax
   decoding (ckeys.generate.greedy_reference) of each row alone under its own separately constructed clamp.
4. The lower-case fields of run_item equal format_factorial.run_item on 2 S0 items; plain_item equals it too, and the
   JB-G3 statistics pass in FP32 and fail on a perturbed record.
5. experiments/paper1_frames.py --score E: L equals the old lp_rows (every run of run_core, locations and letters); the
   default path writes byte-identical output to the file before the flag was added.
6. score_cached equals score_reference on tiny random Gemma-2 (eager, sliding layers) and Phi-3 (fused qkv) models.
7. Frame admission rejects a frame that breaks the disjointness rule (in calibration, and per core under --score E).
8. The tokenizer check (J-B-G0b) fails when the null sentence would not have the after-sentence's length.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma2Config, Phi3Config

from ckeys import fresh
from ckeys.clamp import capture_kv, clamp_kv
from ckeys.generate import greedy_reference
from ckeys.interventions import blocks
from ckeys.story import LOCATIONS, PAIR_SWAP
from ckeys.surface import FRAMES_SIGMA, FormSet, score, score_reference
from experiments import fresh_factorial as ff
from experiments.format_factorial import run_item as ff_run_item

NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOL = 1e-4
ROOT = Path(__file__).resolve().parents[1]
SMALL = {"sigma": (" ", "The "), "E": (" ", "The ", " In the ", " {a} put it in the ")}
MIN = {"sigma": (" ",), "E": (" ", " {a} put it in the ")}
torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def qwen():
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_implementation="sdpa").eval()
    return model, tok


def close(a, b):
    return torch.allclose(torch.as_tensor(a), torch.as_tensor(b), atol=TOL, rtol=0)


def same(r1, r2, fs, i1=slice(None), i2=slice(None)):
    for name in ("L", *fs.sets):
        for w in fs.words:
            assert close(r1[name][w][i1], r2[name][w][i2]), (name, w, (r1[name][w][i1] - r2[name][w][i2]).abs().max())
    assert close(r1["first"][i1], r2["first"][i2]) and close(r1["gap"][i1], r2["gap"][i2])


def item_of(lex, k=0):
    return [it for it in fresh.population("F") if it.lex == lex][k]


def kv_of(model, ids, pos, nL):
    with capture_kv(model, [pos], range(nL)) as t:
        model(ids, use_cache=False)
    return {k: v[0] for k, v in t.items()}


@pytest.mark.parametrize("lex", [1, 2])
def test_trie_and_cached_equal_plain_passes_under_clamps(qwen, lex):
    model, tok = qwen
    nL = len(blocks(model))
    it = item_of(lex)
    ids, pos = ff.encode_item(tok, it, "POST")
    fs = FormSet(tok, [it.track[t] for t in ("B", "S", "X")], SMALL, names=it.names)   # the words the identity uses
    assert len(fs) > 6 and all(w in fresh.LEXICONS[lex] for w in fs.words)
    kv = {n: kv_of(model, ids[n], pos, nL) for n in "BSX"}
    # per-row tables: row 0 the B run's own K/V (identity), row 1 K from S, row 2 V from X, all layers
    tabs = {(l, ch): torch.stack([kv["B"][(l, ch)], kv["S" if ch == "k" else "B"][(l, ch)], kv["B" if ch == "k" else "X"][(l, ch)]])
            for l in range(nL) for ch in "kv"}
    x = ids["B"].expand(3, -1)
    with clamp_kv(model, [pos], tabs, range(nL)):
        rt, rc, rr = score(model, x, fs), ff.score_cached(model, x, fs, chunk=7), score_reference(model, x, fs)
    same(rt, rr, fs)
    same(rc, rr, fs)
    plain = torch.log_softmax(model(ids["B"], use_cache=False).logits[:, -1].float(), -1)
    assert close(rt["first"][0:1], plain)                       # the identity row's answer position = a plain pass
    assert not close(rt["first"][1], rt["first"][0])             # the clamps act


def test_run_item_rows_equal_separate_clamps(qwen):
    model, tok = qwen
    nL = len(blocks(model))
    it = item_of(2, 1)
    fs = FormSet(tok, it.words, MIN, names=it.names)
    rec = ff.run_item(model, tok, it, "AFTER", fs, nL, "cpu", gen=False)
    ids, pos = ff.encode_item(tok, it, "AFTER")
    kv = {n: kv_of(model, ids[n], pos, nL) for n in "BSX"}
    l0 = round(0.3 * nL)
    for row, donor_k, donor_v, start in (("K_S@0", "S", "B", 0), (f"KV_S@{l0}", "S", "S", l0)):
        tab = {(l, ch): kv[donor_k if ch == "k" else donor_v][(l, ch)] for l in range(start, nL) for ch in "kv"}
        with clamp_kv(model, [pos], tab, range(start, nL)):
            ref = score_reference(model, ids["B"], fs)
        for name in ("L", "sigma", "E"):
            assert close(rec["rows"][row][name], [float(ref[name][w][0]) for w in fs.words]), (row, name)
    ref = score_reference(model, ids["S"], fs)
    assert close(rec["clean"]["S"]["E"], [float(ref["E"][w][0]) for w in fs.words])


def test_generation_batch_equals_stepwise_reference(qwen):
    model, tok = qwen
    nL = len(blocks(model))
    it = item_of(2, 2)
    fs, _ = fresh.form_set(tok, it)
    rec = ff.run_item(model, tok, it, "NONE", fs, nL, "cpu", gen=True, max_new=6)
    ids, pos = ff.encode_item(tok, it, "NONE")
    kv = {n: kv_of(model, ids[n], pos, nL) for n in "BSX"}
    stop = fresh.stop_fn(tok, it.words)
    for kd, vd, l0, name in ff.GEN_ROWS:
        if l0 is None:
            g = greedy_reference(model, tok, ids[kd], max_new=6, stop=stop)[0]
            x = ids[kd]
        else:
            tab = {(l, ch): kv[kd if ch == "k" else vd][(l, ch)] for l in range(nL) for ch in "kv"}
            with clamp_kv(model, [pos], tab, range(nL)):
                g = greedy_reference(model, tok, ids["B"], max_new=6, stop=stop)[0]
            x = ids["B"]
        assert rec["gen"][name] == fresh.gen_text(tok, x[0].tolist(), g), (name, rec["gen"][name], tok.decode(g))
        assert rec["ans"][name] == fresh.parse(rec["gen"][name], it.words)
    assert rec["ans"]["K_S@0"] in set(it.words) | {"other"}


def test_lowercase_fields_equal_format_factorial_run_item(qwen):
    model, tok = qwen
    nL = len(blocks(model))
    S0 = fresh.population("S0")
    for it, arm in ((S0[0], "P1"), (S0[1], "POST")):
        fs, _ = fresh.form_set(tok, it)
        rec = ff.run_item(model, tok, it, arm, fs, nL, "cpu", gen=False)
        ref = ff_run_item(model, tok, it.core, arm, "direct", "cpu")
        assert rec["pos"] == ref["pos"] and rec["len"] == ref["len"] and rec["track"]["X"] == ref["X"]
        for n in "BSX":
            a, b = rec["ff"]["clean"][n], ref["clean"][n]
            assert a["argmax_cand"] == b["argmax_cand"] and abs(a["mass"] - b["mass"]) < TOL and abs(a["m"] - b["m"]) < TOL
            assert all(abs(a["lp"][t] - b["lp"][t]) < TOL for t in b["lp"])
        assert set(rec["ff"]["m"]) == set(ref["m"])
        for row in ref["m"]:
            assert abs(rec["ff"]["m"][row]["m"] - ref["m"][row]["m"]) < TOL
            assert all(abs(rec["ff"]["m"][row]["lp"][t] - ref["m"][row]["lp"][t]) < TOL for t in ref["m"][row]["lp"])
        pl = ff.plain_item(model, tok, it, arm, nL, "cpu")
        W = list(it.words)
        for row in ref["m"]:   # the plain path is format_factorial's own passes: equal up to float summation order
            assert all(abs(pl["rows"][row]["L"][W.index(it.track[t])] - ref["m"][row]["lp"][t]) < 1e-5 for t in ("S", "B", "X"))
        st = ff.g3_compare([rec], [pl])
        assert st["pass"] and st["abs_ds"] < 1e-4 and st["mean_abs_dL"] < 1e-4
        bad = json.loads(json.dumps(pl))
        for row in bad["rows"]:
            bad["rows"][row]["L"] = [x + (1.0 if j % 2 else -1.0) * st["ID_KV_plain"] for j, x in enumerate(bad["rows"][row]["L"])]
        assert not ff.g3_compare([rec], [bad])["pass"]


def native_core(i=0):
    import random
    rng = random.Random(100 + i)
    from ckeys.story import make_cores
    c = make_cores(1, rng)[0]
    return dict(c, id=f"t{i}", target=PAIR_SWAP[c["source"]])


def test_paper1_frames_score_E_keeps_L(qwen):
    import experiments.paper1_frames as pf
    model, tok = qwen
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    bases = {(o, s): torch.linalg.qr(torch.randn(model.config.hidden_size, 16, generator=g))[0].T.contiguous()
             for o in pf.OBJECTIVES for s in pf.SEEDS}
    core = native_core()
    for arm in ("P1", "LETTER"):
        pf.SCORE.update(mode="L", frames=(), fs=None)
        a = pf.run_core(model, tok, core, arm, bases, "cpu")
        pf.SCORE.update(mode="E", frames=(" {a} put it in the ",), fs=None)
        try:
            b = pf.run_core(model, tok, core, arm, bases, "cpu")
        finally:
            pf.SCORE.update(mode="L", frames=(), fs=None)
        assert set(a["runs"]) == set(b["runs"]) and len(a["runs"]) > 20
        for k in a["runs"]:
            assert set(a["runs"][k]) == {"cand", "argmax"}               # the default path stores what it always stored
            assert close(a["runs"][k]["cand"], b["runs"][k]["cand"]), (arm, k)
            assert a["runs"][k]["argmax"] == b["runs"][k]["argmax"]
            assert set(b["runs"][k]) == {"cand", "argmax", "E", "sig", "mass"}
            assert all(e >= c - TOL for e, c in zip(b["runs"][k]["E"], b["runs"][k]["cand"]))   # E sums over more forms
            assert b["runs"][k]["mass"]["E"] >= b["runs"][k]["mass"]["L"] - TOL


def _old_paper1_frames():
    """experiments/paper1_frames.py as it was before --score was added (the parent of the commit that added it, or HEAD
    while the change is uncommitted)."""
    run = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True, check=True).stdout  # noqa: E731
    added = run("log", "--format=%H", "-S", '"--score"', "--", "experiments/paper1_frames.py").split()
    rev = (added[-1] + "^") if added else "HEAD"
    src = run("show", f"{rev}:experiments/paper1_frames.py")
    assert '"--score"' not in src
    return src


def test_paper1_frames_default_path_byte_identical(tmp_path):
    src = _old_paper1_frames()
    (tmp_path / "old").mkdir()
    old = tmp_path / "old" / "paper1_frames_old.py"
    old.write_text(src)
    p1 = tmp_path / "p1" / "gpu" / "component_data"
    p1.mkdir(parents=True)
    (p1 / "native_story_120.json").write_text(json.dumps({"stories": [native_core(0), native_core(1)]}))
    outs = []
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"), CUDA_VISIBLE_DEVICES="")
    for script in (str(old), str(ROOT / "experiments" / "paper1_frames.py")):
        out = tmp_path / "out"
        cmd = [sys.executable, script, "--model", "qwen", "--p1-root", str(tmp_path / "p1"), "--arms", "NONE",
               "--model-override", NAME, "--bases-override", "896", "--n", "1", "--out", str(out)]
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True)
        f = out / "TEST_Qwen2.5-0.5B-Instruct.json"
        outs.append(f.read_bytes())
        f.unlink()
    assert outs[0] == outs[1]


@pytest.mark.parametrize("cfg", [
    lambda: Gemma2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, max_position_embeddings=256, sliding_window=16,
                         query_pre_attn_scalar=16, attn_logit_softcapping=50.0, final_logit_softcapping=30.0),
    lambda: Phi3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, max_position_embeddings=256, pad_token_id=0)])
def test_score_cached_tiny_families(cfg):
    torch.manual_seed(0)
    c = cfg()
    c._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(c).float().eval()

    class T:
        def __call__(self, s, add_special_tokens=False):
            return type("E", (), {"input_ids": [3 + (ord(ch) % 120) for ch in s]})()

        def decode(self, ids):
            return "".join(chr(i - 3) for i in ids)
    fs = FormSet(T(), ["box", "shelf", "map"], {"sigma": (" ", " the "), "E": (" ", " the ", " On the ")})
    ids = torch.randint(3, 120, (2, 24), generator=torch.Generator().manual_seed(1))
    P = [9]
    other = ids.clone()
    other[:, P] = 7
    nL = len(blocks(model))
    with capture_kv(model, P, range(nL)) as kv:
        model(other[:1], use_cache=False)
    with clamp_kv(model, P, {k: v[0] for k, v in kv.items()}, range(nL), which="v"):
        same(ff.score_cached(model, ids, fs, chunk=5), score_reference(model, ids, fs), fs)


def test_frame_admission_rejects_prefix_conflicts():
    class T:
        def __call__(self, s, add_special_tokens=False):
            return type("E", (), {"input_ids": [3 + (ord(ch) % 120) for ch in s]})()

        def decode(self, ids):
            return "".join(chr(i - 3) for i in ids)
    C = fresh.population("C")
    it1, it2 = next(i for i in C if i.lex == 1), next(i for i in C if i.lex == 2)
    adm, rej = ff.admit_frames(T(), [it1, it2], [" {a} put it in the ", "box", "bin", " **"])
    assert adm == [" {a} put it in the ", " **"] and set(rej) == {"box", "bin"}
    assert ff.admit_frames(T(), [it2], ["box"]) == (["box"], {})      # "box" + a lexicon-2 word is no proper prefix
    fs, dropped = fresh.form_set(T(), it1, [" {a} put it in the ", "box"])
    assert dropped == ["box"] and f" {it1.names['a']} put it in the box" in fs.sets["E"]["box"].values()
    assert FRAMES_SIGMA[0] + "box" in fs.sets["sigma"]["box"].values()


def _char_tok():
    class T:
        def __call__(self, s, add_special_tokens=False):
            return type("E", (), {"input_ids": [3 + (ord(ch) % 120) for ch in s]})()

        def decode(self, ids):
            return "".join(chr(i - 3) for i in ids)
    return T()


def test_paper1_frames_score_formset_drops_a_conflicting_frame():
    import experiments.paper1_frames as pf
    core = native_core()
    pf.SCORE.update(mode="E", frames=(" {a} put it in the ", "box"), fs=None)
    try:
        fs, dropped = pf.score_formset(_char_tok(), "P1", core)       # "box" + "box" extends the form "" + "box"
    finally:
        pf.SCORE.update(mode="L", frames=(), fs=None)
    assert dropped == ["box"] and f" {core['agent']} put it in the shelf" in fs.sets["E"]["shelf"].values()
    pf.SCORE.update(mode="E", frames=(" {a} put it in the ",), fs=None)
    try:
        assert pf.score_formset(_char_tok(), "P1", core)[1] == []
    finally:
        pf.SCORE.update(mode="L", frames=(), fs=None)


def test_tokcheck_requires_one_sentence_length(monkeypatch, tmp_path):
    import argparse
    P = {n: fresh.population(n)[:2] for n in ("F", "C", "S0")}
    monkeypatch.setattr(fresh, "population", lambda name: P[name])
    a = argparse.Namespace(stage="tokcheck", model=NAME, revision=None, key="qwen7", test=True, out=str(tmp_path),
                           tag="TEST_qwen7", attn="auto", dtype="float32", max_new=16)
    assert ff.stage_tokcheck(a) == 0
    monkeypatch.setattr(fresh, "NULL_NOUNS", fresh.NULL_NOUNS[:5] + ("hippopotamus",))   # several tokens
    assert ff.stage_tokcheck(a) == 1
    rep = json.loads((tmp_path / "tokcheck" / "TEST_qwen7.json").read_text())
    assert any("differ between the lexicons, the null nouns or the orders" in f for f in rep["fails"]), rep["fails"]
