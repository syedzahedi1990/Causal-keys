"""Do later choice tokens read the critical token by identity matching on its KEY?

For base runs, source runs and base runs with the source key swapped in at the critical token
(all layers), measure attention from each choice-list token to the critical position.
"""
import argparse, random, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ckeys.interventions import blocks, capture, hooks
from ckeys.story import LOCATIONS, make_cores, prompt, record

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--view", default="world")
a = ap.parse_args()
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()
cand = [tok.encode(" " + l, add_special_tokens=False)[0] for l in LOCATIONS]
nL = len(blocks(model))

def enc(text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt")
    return ids if torch.is_tensor(ids) else ids["input_ids"]

agg = {k: torch.zeros(nL) for k in ["base:B", "base:S", "base:other", "src:B", "src:S", "src:other", "Kswap:B", "Kswap:S", "Kswap:other"]}
cnt = 0
for core in make_cores(a.n, random.Random(1)):
    ib = enc(prompt(record(core, a.view, core["base"]), True)); is_ = enc(prompt(record(core, a.view, core["source"]), True))
    if ib.shape != is_.shape: continue
    pos = (ib[0] != is_[0]).nonzero().item()
    # choice-list positions: last occurrence of each location token
    cpos = {l: (ib[0] == cand[i]).nonzero().flatten()[-1].item() for i, l in enumerate(LOCATIONS)}
    assert all(p > pos for p in cpos.values())
    with capture(model, range(nL), "k") as K:
        out_s = model(is_, output_attentions=True)
    Ks = {l: K[l][0, pos].clone() for l in range(nL)}
    out_b = model(ib, output_attentions=True)
    hs = []
    for l in range(nL):
        def hk(_m, _i, out, l=l):
            out = out.clone(); out[:, pos] = Ks[l]; return out
        hs.append(blocks(model)[l].self_attn.k_proj.register_forward_hook(hk))
    with hooks(hs):
        out_k = model(ib, output_attentions=True)
    for name, out in (("base", out_b), ("src", out_s), ("Kswap", out_k)):
        for l in range(nL):
            att = out.attentions[l][0].mean(0)  # [q, k] averaged over heads
            for role, locs in (("B", [core["base"]]), ("S", [core["source"]]),
                               ("other", [x for x in LOCATIONS if x not in (core["base"], core["source"])])):
                agg[f"{name}:{role}"][l] += sum(att[cpos[x], pos].item() for x in locs) / len(locs)
    cnt += 1
print(f"n={cnt}; mean attention (avg over heads) from choice-list token to critical token, by layer band")
bands = [(0, 7), (7, 14), (14, 21), (21, nL)]
print("cond:choice   " + "  ".join(f"L{a}-{b-1}" for a, b in bands) + "   max-layer")
for k, v in agg.items():
    v = v / cnt
    print(f"{k:13s} " + "  ".join(f"{v[a:b].mean():.4f}" for a, b in bands) + f"   L{int(v.argmax())}={v.max():.3f}")
