"""Part B pilot 4 (CPU, FP32): the stage-8 Part-B measurement on a small model and fresh cores (seed 20261011).
Rows at l0 = 0: ID (self-clamp), K_S, K_X, V_S, V_X, and the clean B/S/X runs. Per row: lowercase, SIGMA and SIGMA+
candidate log-probs (trie scorer), candidate masses, and greedy generation (<= 8 tokens, use_cache=False, early stop)
parsed to a candidate. Prints ID_K / ID_V / s_ID under each scoring, masses, behavioural beta_K / beta_V, the agreement of
the SIGMA argmax with the generated answer, top first tokens, and timing. Disclosed as a pilot; enters no verdict."""
import random, re, sys, time, json
sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ckeys.encoding import encode, build_prompt
from ckeys.story import LOCATIONS, make_cores, pick_x, record
from ckeys.clamp import capture_kv, clamp_kv, stack_rows
from surface import FormSet, score

torch.set_grad_enabled(False)
PAT = re.compile(r"(?i)\b(box|basket|shelf|drawer|cabinet|closet)")
ROWS = [("B", "B"), ("S", "B"), ("X", "B"), ("B", "S"), ("B", "X")]     # (key donor, value donor)
RN = ["ID", "K_S", "K_X", "V_S", "V_X"]


def generate(model, tok, ids, clamp, max_new=6):
    R = ids.shape[0]
    cur = ids.clone()
    done = [False] * R
    texts = [""] * R
    for _ in range(max_new):
        with clamp():
            lg = model(cur, use_cache=False, logits_to_keep=1).logits[:, -1]
        nxt = lg.argmax(-1)
        cur = torch.cat([cur, nxt[:, None]], 1)
        for r in range(R):
            if not done[r]:
                texts[r] = tok.decode(cur[r, ids.shape[1]:])
                if PAT.search(texts[r]) or nxt[r].item() == tok.eos_token_id or "\n" in texts[r]:
                    done[r] = True
        if all(done):
            break
    return texts


def parse(t):
    m = PAT.search(t)
    return m.group(1).lower() if m else "other"


def main(model_name, n, arms, seed=20261011):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
    nL = model.config.num_hidden_layers
    cores = make_cores(n, random.Random(seed))
    fsS = FormSet(tok, LOCATIONS, plus=False)
    fsP = FormSet(tok, LOCATIONS, plus=True)
    print(f"{model_name}: SIGMA trie nodes {len(fsS)}, SIGMA+ nodes {len(fsP)}")
    summary = {}
    for arm in arms:
        t0 = time.time()
        recs = []
        firsttok = {}
        for core in cores:
            X = pick_x(core)
            ids, kv = {}, {}
            for nm, loc in (("B", core["base"]), ("S", core["source"]), ("X", X)):
                r = record(core, "direct", loc)
                ids[nm] = encode(tok, build_prompt(arm, r["story"], r["query"], core, X))
            p = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()
            assert len(p) == 1
            p = p[0]
            for nm in ("B", "S", "X"):
                with capture_kv(model, [p], range(nL)) as t:
                    model(ids[nm], use_cache=False, logits_to_keep=1)
                kv[nm] = {k: v[0] for k, v in t.items()}
            tabs = stack_rows(kv, [lambda l, ch, r=r: r[0] if ch == "k" else r[1] for r in ROWS], range(nL))
            batch = ids["B"].expand(len(ROWS), -1)
            cl = lambda: clamp_kv(model, [p], tabs, range(nL))
            with cl():
                sc = score(model, batch, fsP)
            # SIGMA (12 forms) from the SIGMA+ pass: restrict to the SIGMA sequences
            sig = {w: torch.logsumexp(torch.stack([sc["forms"][w][:, fsP.seqs[w].index(s)] for s in fsS.seqs[w]], 1), 1) for w in LOCATIONS}
            gens = generate(model, tok, batch, cl)
            clean = torch.cat([ids["B"], ids["S"], ids["X"]])
            scc = score(model, clean, fsP)
            sigc = {w: torch.logsumexp(torch.stack([scc["forms"][w][:, fsP.seqs[w].index(s)] for s in fsS.seqs[w]], 1), 1) for w in LOCATIONS}
            gc = generate(model, tok, clean, lambda: torch.no_grad())
            top = scc["first"][0].exp()
            for tid in top.topk(5).indices.tolist():
                firsttok[tok.decode([tid])] = firsttok.get(tok.decode([tid]), 0) + top[tid].item() / n
            recs.append(dict(core=core, X=X,
                             low={w: sc["lower"][w].tolist() for w in LOCATIONS}, sig={w: sig[w].tolist() for w in LOCATIONS},
                             plus={w: sc["sigma"][w].tolist() for w in LOCATIONS}, gen=gens,
                             mass_low=sum(scc["lower"][w][0].exp().item() for w in LOCATIONS),
                             mass_sig=sum(sigc[w][0].exp().item() for w in LOCATIONS),
                             mass_plus=sum(scc["sigma"][w][0].exp().item() for w in LOCATIONS),
                             gen_clean=gc, argmax_clean=[max(LOCATIONS, key=lambda w: sigc[w][j].item()) for j in range(3)]))
        dt = time.time() - t0
        out = {}
        for sc_name in ("low", "sig", "plus"):
            def ID(ch):
                v = []
                for r in recs:
                    S, X = r["core"]["source"], r["X"]
                    L = r[sc_name]
                    d = lambda w, row: L[w][RN.index(row)] - L[w][0]
                    v.append(0.5 * ((d(S, f"{ch}_S") - d(S, f"{ch}_X")) + (d(X, f"{ch}_X") - d(X, f"{ch}_S"))))
                return np.mean(v)
            k, v = ID("K"), ID("V")
            out[sc_name] = (k, v, k / (k + v) if k + v > 0.5 else float("nan"))
        ans = [[parse(g) for g in r["gen"]] for r in recs]
        P = lambda row, who: np.mean([a[RN.index(row)] == (r["core"]["source"] if who == "S" else r["X"]) for a, r in zip(ans, recs)])
        bK = 0.5 * ((P("K_S", "S") - P("K_X", "S")) + (P("K_X", "X") - P("K_S", "X")))
        bV = 0.5 * ((P("V_S", "S") - P("V_X", "S")) + (P("V_X", "X") - P("V_S", "X")))
        agree = [max(LOCATIONS, key=lambda w: r["sig"][w][j]) == a[j] for r, a in zip(recs, ans) for j in range(5) if a[j] != "other"]
        other = np.mean([a == "other" for r, A in zip(recs, ans) for a in A])
        acc = np.mean([parse(r["gen_clean"][0]) == r["core"]["base"] for r in recs])
        print(f"\n== {arm} (n={n}, {dt:.0f}s)  mass low {np.mean([r['mass_low'] for r in recs]):.3f}  SIGMA {np.mean([r['mass_sig'] for r in recs]):.3f}  SIGMA+ {np.mean([r['mass_plus'] for r in recs]):.3f}  gen-acc(B) {acc:.2f}  gen-other {other:.2f}")
        for k in ("low", "sig", "plus"):
            print(f"   {k:4s}: ID_K {out[k][0]:+7.3f}  ID_V {out[k][1]:+7.3f}  s_ID {out[k][2]:+.3f}")
        print(f"   behaviour: beta_K {bK:+.2f}  beta_V {bV:+.2f};  SIGMA-argmax agrees with generation {np.mean(agree) if agree else float('nan'):.2f} (n={len(agree)})")
        print(f"   top first tokens (clean B): " + ", ".join(f"{k!r}:{v:.2f}" for k, v in sorted(firsttok.items(), key=lambda x: -x[1])[:6]))
        print(f"   example generations clean B: {[r['gen_clean'][0] for r in recs[:4]]}")
        summary[arm] = dict(out=out, bK=bK, bV=bV, sec=dt)
    json.dump(summary, open(f"/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partB/mini_{model_name.split('/')[-1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3].split(","))
