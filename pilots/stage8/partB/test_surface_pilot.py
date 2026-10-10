"""Exactness pilot for the trie scorer (FP32, CPU):
 (1) Qwen2.5-0.5B-Instruct (trained weights), sdpa and eager, clean and under a key clamp at p: trie == separate passes;
 (2) tiny random-weight models of every Part-B architecture (Qwen2, Mistral, Llama [Llama-3.1, Falcon3], OLMo2, Gemma2
     eager, Phi3): trie == separate passes, and the answer-position log-probs equal a plain forward."""
import random, sys, time
sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ckeys.encoding import encode, build_prompt
from ckeys.story import LOCATIONS, make_cores, pick_x, record
from ckeys.clamp import capture_kv, clamp_kv, stack_rows
from surface import FormSet, score, score_reference

torch.manual_seed(0)
TOL = 1e-4


def check(model, tok, ids, fs, label, clamp=None):
    if clamp is None:
        a, b = score(model, ids, fs), score_reference(model, ids, fs)
    else:
        with clamp():
            a = score(model, ids, fs)
        with clamp():
            b = score_reference(model, ids, fs)
    err = max((a["forms"][w] - b["forms"][w]).abs().max().item() for w in fs.words)
    plain = torch.log_softmax(model(ids, use_cache=False).logits[:, -1].float(), -1) if clamp is None else None
    e2 = (a["first"] - plain).abs().max().item() if plain is not None else float("nan")
    print(f"  {label:40s} nodes={len(fs):3d} max|trie-ref| over forms {err:.2e}  max|first-plain| {e2:.2e}  {'OK' if err < TOL and (e2 != e2 or e2 < TOL) else 'FAIL'}")
    return err


def trained():
    m = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(m)
    core = make_cores(1, random.Random(8))[0]
    X = pick_x(core)
    for impl in ("sdpa", "eager"):
        model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32, attn_implementation=impl).eval()
        for arm in (("POST",) if impl == "sdpa" else ("NONE",)):
            for plus in ((True,) if impl == "sdpa" else (False,)):
                fs = FormSet(tok, [core["base"], core["source"], X], plus=plus)
                ids = {}
                for nm, loc in (("B", core["base"]), ("S", core["source"])):
                    r = record(core, "direct", loc)
                    ids[nm] = encode(tok, build_prompt(arm, r["story"], r["query"], core, X))
                p = (ids["B"][0] != ids["S"][0]).nonzero().item()
                nL = model.config.num_hidden_layers
                check(model, tok, ids["B"], fs, f"Qwen0.5B {impl} {arm} plus={plus} clean")
                kv = {}
                for nm in ("B", "S"):
                    with capture_kv(model, [p], range(nL)) as t:
                        model(ids[nm], use_cache=False)
                    kv[nm] = {k: v[0] for k, v in t.items()}
                tabs = stack_rows(kv, [lambda l, ch: "S" if ch == "k" else "B", lambda l, ch: "B" if ch == "k" else "S"], range(nL))
                two = ids["B"].expand(2, -1)
                check(model, tok, two, fs, f"Qwen0.5B {impl} {arm} plus={plus} K_S/V_S", clamp=lambda: clamp_kv(model, [p], tabs, range(nL)))


def tiny():
    from transformers import (Qwen2Config, MistralConfig, LlamaConfig, Olmo2Config, Gemma2Config, Phi3Config)
    common = dict(vocab_size=152000, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    fs = FormSet(tok, ["shelf", "box", "closet"], plus=True)
    ids = torch.randint(0, 150000, (2, 40))
    cfgs = {"Qwen2": (Qwen2Config, {}), "Mistral": (MistralConfig, {}), "Llama": (LlamaConfig, {}), "Olmo2": (Olmo2Config, {}),
            "Gemma2": (Gemma2Config, dict(head_dim=16, attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
                                          query_pre_attn_scalar=16, sliding_window=4096)),
            "Phi3": (Phi3Config, dict(pad_token_id=0))}
    for name, (C, extra) in cfgs.items():
        cfg = C(**common, **extra)
        for impl in (("eager",) if name == "Gemma2" else ("sdpa", "eager")):
            try:
                model = AutoModelForCausalLM.from_config(cfg, attn_implementation=impl).float().eval()
                check(model, tok, ids, fs, f"tiny {name} {impl}")
            except Exception as e:
                print(f"  tiny {name} {impl}: ERROR {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    print("transformers", transformers.__version__, "torch", torch.__version__)
    t0 = time.time()
    tiny()
    sys.stdout.flush()
    trained()
    print(f"done in {time.time() - t0:.0f}s")
