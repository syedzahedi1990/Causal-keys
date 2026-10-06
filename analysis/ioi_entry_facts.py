"""The facts quoted by entry G, part (e), regenerated from ckeys/ioi.py: core 0 (the three runs, the listing and the
INLINE forms), the IO_B slot counts, the template histogram, and per tokenizer x arm the p and length ranges and the
IO / S occurrence counts over the 200 seed-0 cores (GPT-2 raw with BOS; Qwen2.5 chat for AFTER/BEFORE/QUESTION, no
BOS; Mistral-v0.3 chat with BOS; Qwen2.5 raw = the base model). Writes --out (default
results/gpu_stage5/ioi/ENTRY_FACTS.txt); the pipeline re-runs it and refuses to start on a diff, so the entry and
the generator cannot disagree. Tokenizers only, no weights.
"""
import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from ckeys.ioi import ARMS, LIST_ARMS, SEED, TEMPLATES, arm_chat, encode_runs, inline, listing, make_cores, name_ids, raw_prompt, sentence  # noqa: E402

TOKENIZERS = (("gpt2", "gpt2", False), ("Qwen2.5", "Qwen/Qwen2.5-7B-Instruct", True), ("Qwen2.5 raw", "Qwen/Qwen2.5-7B-Instruct", False),
              ("Mistral-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", True))


def facts(n=200, seed=SEED, tokenizers=TOKENIZERS):
    cores = make_cores(n, random.Random(seed))
    c0 = cores[0]
    lines = [f"ckeys/ioi.py make_cores({n}, random.Random({seed})): {len(TEMPLATES)} templates, reduced pools",
             f"core 0: {c0}",
             f"  B: {sentence(c0, c0['io_b'])}", f"  S: {sentence(c0, c0['io_s'])}", f"  X: {sentence(c0, c0['io_x'])}",
             f"  listing: {listing(c0)}", f"  INLINE: {inline(c0, c0['io_b'], False)}", f"  INLINE_BEFORE: {inline(c0, c0['io_b'], True)}",
             f"  AFTER (chat user turn): {raw_prompt('AFTER', c0, c0['io_b'], True)!r}",
             f"  AFTER (raw): {raw_prompt('AFTER', c0, c0['io_b'], False)!r}",
             "IO_B list slot counts (slot 0..3): " + " ".join(f"{s}:{k}" for s, k in sorted(collections.Counter(c["order"].index(0) for c in cores).items())),
             "pattern counts: " + str(dict(collections.Counter(c["pattern"] for c in cores))),
             "template histogram (index: count): " + " ".join(f"{t}:{k}" for t, k in sorted(collections.Counter(c["template"] for c in cores).items()))]
    for label, name, chat in tokenizers:
        tok = AutoTokenizer.from_pretrained(name)
        lines.append(f"tokenizer {label} ({name}; chat {chat}; BOS {tok.bos_token!r})")
        for arm in ARMS:
            fails, ps, lens, nio, ns = 0, [], [], collections.Counter(), collections.Counter()
            for c in cores:
                ids = encode_runs(tok, c, arm, chat)
                if ids is None:
                    fails += 1
                    continue
                tid = name_ids(tok, c)
                ps.append((ids["B"][0] != ids["S"][0]).nonzero().item())
                lens.append(ids["B"].shape[1])
                nio[(ids["B"][0] == tid["B"]).sum().item()] += 1
                ns[(ids["B"][0] == tid["Subj"]).sum().item()] += 1
            ok = fails == 0 and set(nio) == {2 if arm in LIST_ARMS else 1} and set(ns) <= ({2, 3} if arm in LIST_ARMS else {2})
            lines.append(f"  {arm:14s} chat={arm_chat(arm, chat)!s:5s} skipped={fails} p {min(ps)}..{max(ps)} len {min(lens)}-{max(lens)} "
                         f"IO-count {dict(sorted(nio.items()))} S-count {dict(sorted(ns.items()))} -> {'ok' if ok else 'CHECK FAILED'}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default="results/gpu_stage5/ioi/ENTRY_FACTS.txt")
    a = ap.parse_args(argv)
    s = facts(a.n, a.seed)
    print(s)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        open(a.out, "w").write(s + "\n")


if __name__ == "__main__":
    main()
