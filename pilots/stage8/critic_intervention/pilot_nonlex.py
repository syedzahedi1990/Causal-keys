"""Critic pilot for Part C (CPU, FP32, Qwen2.5-1.5B-Instruct): do NON-LEXICAL identity edits obey the matched-depth law?

Edits at the output of block l at the writing token p (all additive, alpha = 1):
  T     natural residual h_t (tautological control)
  CO    lexical mean difference from neutral sentences with the English word (Part C's E2)
  SYN   same, with the synonym (crate, hamper, ledge, compartment, cupboard, wardrobe) -- no English token form of t
  FR    same, with the French translation (boite, panier, ...)
  CI    in-task mean difference (Part C's E1)
  CIperp CI with its component in the 5-dim span of the English lexical differences removed
Reports per format (NONE, P1, LETTER) and depth: phi_E = ID_KV^E / ID_KV^nat (full-run rows), flip rate under NONE,
transfer ratio phi(LETTER)/phi(NONE) and phi(P1)/phi(NONE), and residual cosine with the natural displacement.
"""
import json
import random
import sys
import time

import torch

sys.path.insert(0, "/home/user/Causal-keys")
sys.path.insert(0, "/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partC")
from pilot_c import NEUTRAL, cand_tables, ids_for, resid_at, run_rows, score  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from ckeys.encoding import encode  # noqa: E402
from ckeys.story import LOCATIONS, make_cores, pick_x  # noqa: E402
from ckeys.variants import _F  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEPTHS = [9, 13, 17]
ARMS = ["NONE", "P1", "LETTER"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = sys.argv[2] if len(sys.argv) > 2 else "pilot_nonlex.json"
FORMS = {"CO": dict(zip(LOCATIONS, LOCATIONS)), "SYN": dict(zip(LOCATIONS, _F["SYN"])), "FR": dict(zip(LOCATIONS, _F["FR"]))}
FR_SENT = [  # French neutral frames for French words (a word in an English frame is also tried: FRen)
    "Le vieux {x} du couloir a été repeint au printemps.",
    "Elle a acheté un nouveau {x} au marché samedi.",
    "Chaque {x} du magasin avait une petite étiquette.",
    "Un {x} poussiéreux se trouvait dans le coin du grenier.",
    "Le musée expose un {x} sculpté du dix-huitième siècle.",
    "Notre voisin vend un {x} solide à un prix correct.",
]


def last_pos(tok, ids, word):
    w = tok.encode(" " + word, add_special_tokens=False)
    seq = ids[0].tolist()
    hits = [i + len(w) - 1 for i in range(len(seq) - len(w) + 1) if seq[i:i + len(w)] == w]
    assert len(hits) == 1, (word, hits)
    return hits[0]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    t0 = time.time()
    D = model.config.hidden_size
    # lexical means per family
    mu = {}
    for fam, fm in FORMS.items():
        sents = FR_SENT if fam == "FR" else NEUTRAL[:8]
        mu[fam] = {x: {l: torch.zeros(D) for l in DEPTHS} for x in LOCATIONS}
        for s in sents:
            for x in LOCATIONS:
                ids = encode(tok, s.format(x=fm[x]), prefill="")
                st = resid_at(model, ids, last_pos(tok, ids, fm[x]), DEPTHS)
                for l in DEPTHS:
                    mu[fam][x][l] += st[l] / len(sents)
    # in-task means (held-out cores)
    held = make_cores(12, random.Random(202))
    mu["CI"] = {x: {l: torch.zeros(D) for l in DEPTHS} for x in LOCATIONS}
    for core in held:
        for x in LOCATIONS:
            ids = ids_for(tok, core, "NONE", x)
            ref = ids_for(tok, core, "NONE", LOCATIONS[0] if x != LOCATIONS[0] else LOCATIONS[1])
            p = (ids[0] != ref[0]).nonzero().flatten().tolist()
            st = resid_at(model, ids, p[0], DEPTHS)
            for l in DEPTHS:
                mu["CI"][x][l] += st[l] / len(held)
    # lexical subspace (English CO differences), orthonormal basis per depth
    Q = {}
    for l in DEPTHS:
        M = torch.stack([mu["CO"][x][l] - mu["CO"][LOCATIONS[0]][l] for x in LOCATIONS[1:]], 1)
        Q[l], _ = torch.linalg.qr(M)
    print(f"means {time.time() - t0:.0f}s", flush=True)
    fams = ["T", "CO", "SYN", "FR", "CI", "CIperp"]
    cores = make_cores(4 * N, random.Random(101))[:N]
    out = []
    for k, core in enumerate(cores):
        X = pick_x(core)
        B, S = core["base"], core["source"]
        item = {"core": core, "X": X, "arms": {}, "cos": {}}
        for arm in ARMS:
            ids = {n: ids_for(tok, core, arm, loc) for n, loc in (("B", B), ("S", S), ("X", X))}
            p = (ids["B"][0] != ids["S"][0]).nonzero().flatten().tolist()[0]
            h = {n: resid_at(model, ids[n], p, DEPTHS) for n in ("B", "S", "X")}
            names, edits = ["self"], [(None, None)]
            for l in DEPTHS:
                for t, tl in (("S", S), ("X", X)):
                    dn = h[t][l] - h["B"][l]
                    vec = {"T": dn}
                    for fam in ("CO", "SYN", "FR", "CI"):
                        vec[fam] = mu[fam][tl][l] - mu[fam][B][l]
                    c = vec["CI"]
                    vec["CIperp"] = c - Q[l] @ (Q[l].T @ c)
                    for fam in fams:
                        names.append(f"{fam}_{t}@{l}")
                        edits.append((l, h["B"][l] + vec[fam]))
                        if arm == "NONE":
                            item["cos"][f"{fam}_{t}@{l}"] = [
                                torch.nn.functional.cosine_similarity(vec[fam], dn, 0).item(),
                                (vec[fam].norm() / dn.norm()).item()]
            lp, _ = run_rows(model, ids["B"], p, edits)
            tabs = cand_tables(tok, arm)
            c = score(lp, tabs, "marg")
            item["arms"][arm] = {"names": names, "scores": c.tolist(), "iS": LOCATIONS.index(S), "iX": LOCATIONS.index(X),
                                 "iB": LOCATIONS.index(B)}
            print(f"core {k} {arm} {time.time() - t0:.0f}s", flush=True)
        out.append(item)
        json.dump(out, open(OUT, "w"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
