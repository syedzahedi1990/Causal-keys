"""Part B pilot 5 (CPU, FP32, clean runs only, no clamps): candidate mass under lowercase / SIGMA / SIGMA+ scoring and
greedy generations, per arm, on calibration-type cores (seed 20261011, never used for evaluation). Previews the forms a
model family emits after the "Answer:" prefill. Disclosed as a pilot; enters no verdict."""
import random, re, sys, time
sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ckeys.encoding import encode, build_prompt, WRAPPER_USED
from ckeys.story import LOCATIONS, make_cores, pick_x, record
from surface import FormSet, score

torch.set_grad_enabled(False)
PAT = re.compile(r"(?i)\b(box|basket|shelf|drawer|cabinet|closet)")


def main(name, n, arms):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    fsS, fsP = FormSet(tok, LOCATIONS), FormSet(tok, LOCATIONS, plus=True)
    cores = make_cores(n, random.Random(20261011))
    print(f"{name}: SIGMA nodes {len(fsS)}, SIGMA+ nodes {len(fsP)}", flush=True)
    for arm in arms:
        t0 = time.time()
        ml, ms, mp, acc, gens, top = [], [], [], [], [], {}
        for core in cores:
            r = record(core, "direct", core["base"])
            ids = encode(tok, build_prompt(arm, r["story"], r["query"], core, pick_x(core)))
            sc = score(model, ids, fsP)
            ml.append(sum(sc["lower"][w][0].exp().item() for w in LOCATIONS))
            mp.append(sum(sc["sigma"][w][0].exp().item() for w in LOCATIONS))
            ms.append(sum(torch.logsumexp(torch.stack([sc["forms"][w][0, fsP.seqs[w].index(s)] for s in fsS.seqs[w]]), 0).exp().item() for w in LOCATIONS))
            cur = ids.clone()
            for _ in range(6):
                nxt = model(cur, use_cache=False, logits_to_keep=1).logits[:, -1].argmax(-1)
                cur = torch.cat([cur, nxt[:, None]], 1)
                txt = tok.decode(cur[0, ids.shape[1]:])
                if PAT.search(txt) or "\n" in txt or nxt.item() == tok.eos_token_id:
                    break
            gens.append(txt)
            m = PAT.search(txt)
            acc.append(bool(m) and m.group(1).lower() == core["base"])
            p = sc["first"][0].exp()
            for t in p.topk(4).indices.tolist():
                top[tok.decode([t])] = top.get(tok.decode([t]), 0) + p[t].item() / n
        print(f"== {arm:6s} ({time.time() - t0:.0f}s) mass low {np.mean(ml):.3f} SIGMA {np.mean(ms):.3f} SIGMA+ {np.mean(mp):.3f} "
              f"| gen-acc(B) {np.mean(acc):.2f} | top first: " + ", ".join(f"{k!r}:{v:.2f}" for k, v in sorted(top.items(), key=lambda x: -x[1])[:5]), flush=True)
        print(f"   generations: {gens[:6]}", flush=True)
    print("system_merged", WRAPPER_USED["system_merged"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","))
