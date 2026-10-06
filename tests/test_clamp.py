"""ckeys.clamp: K/V sites, position-set clamps (Qwen2.5-0.5B and GPT-2 small, FP32, CPU) and the regression of the
refactored experiments/format_factorial.py and experiments/row_restricted_keys.py against their committed versions."""
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.clamp import capture_kv, clamp_kv, kv_sites, stack_rows
from ckeys.encoding import encode_raw
from ckeys.interventions import blocks

ROOT = Path(__file__).resolve().parents[1]
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
BEFORE_REFACTOR = "6e253bb"  # last commit with the single-position hooks inside the experiment scripts
PY = sys.executable
torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def qwen():
    return AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float32).eval(), AutoTokenizer.from_pretrained(QWEN)


@pytest.fixture(scope="module")
def gpt2():
    return AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval(), AutoTokenizer.from_pretrained("gpt2")


def committed(path, name):
    try:
        src = subprocess.run(["git", "show", f"{BEFORE_REFACTOR}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as ex:   # a failure, not a skip: the regression must run on the GPU box's preflight
        pytest.fail(f"git show {BEFORE_REFACTOR}:{path} failed ({ex}): the regression needs the pinned commit (shallow or rewritten clone?)")
    f = Path(tempfile.mkdtemp()) / f"{name}.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, f


def logp(model, ids):
    return torch.log_softmax(model(ids, use_cache=False).logits[:, -1].float(), -1)


def test_kv_sites(qwen, gpt2):
    (km, ks), (vm, vs) = kv_sites(qwen[0], 3)
    at = blocks(qwen[0])[3].self_attn
    assert km is at.k_proj and vm is at.v_proj and ks == slice(None) and vs == slice(None)
    (km, ks), (vm, vs) = kv_sites(gpt2[0], 3)
    D = gpt2[0].config.hidden_size
    assert km is vm is blocks(gpt2[0])[3].attn.c_attn and ks == slice(D, 2 * D) and vs == slice(2 * D, 3 * D)


def ioi_pair(tok):
    b = encode_raw(tok, "When Ruth and Charles got a ticket at the beach, Charles decided to give it to", bos=True)
    s = encode_raw(tok, "When Erik and Charles got a ticket at the beach, Charles decided to give it to", bos=True)
    assert b.shape == s.shape and b[0, 0] == tok.bos_token_id
    d = (b[0] != s[0]).nonzero().flatten().tolist()
    assert len(d) == 1
    return b, s, d[0]


def test_gpt2_clamp_exact(gpt2):
    """GPT-2 (fused c_attn, absolute positions): self-clamp == clean and the KV clamp from the source run == source."""
    model, tok = gpt2
    L = range(len(blocks(model)))
    ib, is_, p = ioi_pair(tok)
    with capture_kv(model, [p], L) as tb:
        lb = logp(model, ib)
    with capture_kv(model, [p], L) as ts:
        ls = logp(model, is_)
    with clamp_kv(model, [p], tb, L):
        assert torch.allclose(logp(model, ib), lb, atol=1e-4)
    with clamp_kv(model, [p], ts, L):
        assert torch.allclose(logp(model, ib), ls, atol=1e-4)
    with clamp_kv(model, [p], ts, L, which="k"):  # the key alone does not reproduce the source run
        assert not torch.allclose(logp(model, ib), ls, atol=1e-2)
    # several positions, per-row tables, per_row selection: row 0 self, row 1 source K/V at all three positions
    P = [p, p + 1, p + 3]
    with capture_kv(model, P, L) as tb3:
        logp(model, ib)
    with capture_kv(model, P, L) as ts3:
        logp(model, is_)
    tabs = {k: torch.stack([tb3[k][0], ts3[k][0]]) for k in tb3}
    with clamp_kv(model, P, tabs, L):
        out = logp(model, ib.expand(2, -1))
    assert torch.allclose(out[0], lb[0], atol=1e-4) and torch.allclose(out[1], ls[0], atol=1e-4)
    with clamp_kv(model, P, tabs, L, per_row=torch.tensor([False, False])):
        out = logp(model, ib.expand(2, -1))
    assert torch.allclose(out, lb.expand(2, -1), atol=1e-4)


def test_qwen_clamp_matches_single_position_hooks(qwen):
    model, tok = qwen
    L = range(len(blocks(model)))
    ids = tok.apply_chat_template([{"role": "user", "content": "The apple is in the box. Ben moves the apple to the shelf. Where is it?"}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = ids if torch.is_tensor(ids) else ids["input_ids"]
    p = (ids[0] == tok.encode(" shelf", add_special_tokens=False)[0]).nonzero()[0].item()
    with capture_kv(model, [p], L) as t:
        lb = logp(model, ids)
    g = torch.Generator().manual_seed(0)
    rnd = {k: 3 * torch.randn_like(v[0]) for k, v in t.items()}
    hs = []  # the committed hook form: overwrite out[:, p]
    for l in L:
        at = blocks(model)[l].self_attn
        for mod, ch in ((at.k_proj, "k"), (at.v_proj, "v")):
            hs.append(mod.register_forward_hook(lambda _m, _i, out, r=rnd[(l, ch)]: out.clone().index_copy_(1, torch.tensor([p]), r[None].expand(out.shape[0], -1, -1).clone())))
    ref = logp(model, ids)
    for h in hs:
        h.remove()
    with clamp_kv(model, [p], rnd, L):
        assert torch.allclose(logp(model, ids), ref, atol=1e-4)
    with clamp_kv(model, [p], {k: v[0] for k, v in t.items()}, L):
        assert torch.allclose(logp(model, ids), lb, atol=1e-4)
    rows = [lambda l, ch: "B", lambda l, ch: "R" if ch == "k" else "B"]
    tabs = stack_rows({"B": {k: v[0] for k, v in t.items()}, "R": rnd}, rows, L)
    assert tabs[(0, "k")].shape == (2, 1, t[(0, "k")].shape[-1])
    with clamp_kv(model, [p], tabs, L, per_row=torch.tensor([True, False])):
        out = logp(model, ids.expand(2, -1))
    assert torch.allclose(out, lb.expand(2, -1), atol=1e-4)


def test_format_factorial_regression(qwen):
    """Refactored run_item == the committed single-position version: the item dict byte for byte (JSON), and S/B/X
    log-probs of every row, idK, idV within 1e-4 as the explicit statement of the same."""
    import experiments.format_factorial as new
    old, _ = committed("experiments/format_factorial.py", "ff_committed")
    model, tok = qwen
    cores = new.make_cores(4, random.Random(0))
    for arm in ("P1", "NONE", "POST", "LETTER", "BEFORE"):
        for core in cores:
            a, b = old.run_item(model, tok, core, arm, "direct", "cpu"), new.run_item(model, tok, core, arm, "direct", "cpu")
            assert (a is None) == (b is None)
            if a is None:
                continue
            assert a["pos"] == b["pos"] and a["len"] == b["len"] and set(a["m"]) == set(b["m"]) and "arm_meta" not in b
            assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (arm, "the standard arms must stay byte-identical")
            for k in a["m"]:
                for t in ("S", "B", "X", "init"):
                    assert abs(a["m"][k]["lp"][t] - b["m"][k]["lp"][t]) < 1e-4, (arm, k, t)
            for run in ("B", "S", "X"):
                assert abs(a["clean"][run]["m"] - b["clean"][run]["m"]) < 1e-4
                assert a["clean"][run]["argmax_cand"] == b["clean"][run]["argmax_cand"]
    idk = lambda r: 0.5 * ((r["m"]["K_S@0"]["lp"]["S"] - r["m"]["K_X@0"]["lp"]["S"]) + (r["m"]["K_X@0"]["lp"]["X"] - r["m"]["K_S@0"]["lp"]["X"]))  # noqa: E731
    idv = lambda r: 0.5 * ((r["m"]["V_S@0"]["lp"]["S"] - r["m"]["V_X@0"]["lp"]["S"]) + (r["m"]["V_X@0"]["lp"]["X"] - r["m"]["V_S@0"]["lp"]["X"]))  # noqa: E731
    a, b = old.run_item(model, tok, cores[0], "P1", "direct", "cpu"), new.run_item(model, tok, cores[0], "P1", "direct", "cpu")
    assert abs(idk(a) - idk(b)) < 1e-4 and abs(idv(a) - idv(b)) < 1e-4
    assert new.summarize([b] * 5).splitlines()[0].startswith("P1      direct      [all] n=  5")


def test_row_restricted_regression():
    """Refactored row_restricted_keys == the committed version on 2 cores (P1, POST): every group's m to 1e-4."""
    _, f = committed("experiments/row_restricted_keys.py", "rr_committed")
    outs = []
    for script in (f, ROOT / "experiments/row_restricted_keys.py"):
        out = Path(tempfile.mkdtemp())
        subprocess.run([PY, str(script), "--model", QWEN, "--n", "2", "--arms", "P1,POST", "--out", str(out)],
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}, check=True,
                       capture_output=True, text=True)
        d = json.load(open(out / "Qwen2.5-0.5B-Instruct_direct.json"))
        outs.append(d["results"] if isinstance(d, dict) else d)   # the refactored script writes {"provenance", "results"}
    a, b = outs
    assert len(a) == len(b) == 4
    for ra, rb in zip(a, b):
        assert ra["arm"] == rb["arm"] and ra["p"] == rb["p"] and ra["sizes"] == rb["sizes"] and set(ra["m"]) == set(rb["m"])
        assert abs(ra["m_B"] - rb["m_B"]) < 1e-4 and abs(ra["m_S"] - rb["m_S"]) < 1e-4
        for g in ra["m"]:
            assert abs(ra["m"][g] - rb["m"][g]) < 1e-4, (ra["arm"], g)
