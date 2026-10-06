"""Gate d2 of preregistration G, part (d): does the model map each surface form back to its English word?

Six-way mapping prompts (same chat wrapper and "Answer:" prefill as the factorial; argmax over the six English
candidate tokens at the answer position), one per form family: TITLE / UPPER / PLURAL ("the same word as"), SYN
("means almost the same as"), FR / DE ("Translate the French/German word ... into English"). A family is competent
when >= 5/6 words map correctly; the per-word result gates the primary population of the SYN/FRMIX/DEMIX/FR/DE cells.
Output: <out>/<model>.json = {"provenance": ..., "families": {family: {word: {"form", "pick", "correct", "top1"}}}}.
TEST_MODE (--test or TEST_MODE=1): Qwen2.5-0.5B-Instruct, FP32, CPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ckeys.encoding import candidate_ids, encode
from ckeys.story import LOCATIONS
from ckeys.variants import _F as FORMS
from experiments.format_factorial import provenance

FAMILIES = ("TITLE", "UPPER", "PLURAL", "SYN", "FR", "DE")
SIX = ", ".join(LOCATIONS)
SIX_OR = ", ".join(LOCATIONS[:-1]) + " or " + LOCATIONS[-1]
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def prompt(family: str, form: str) -> str:
    if family in ("FR", "DE"):
        return (f"Translate the {'French' if family == 'FR' else 'German'} word '{form}' into English. Choose from: {SIX}."
                "\nAnswer with one word.\nAnswer:")
    rel = "means almost the same as" if family == "SYN" else "is the same word as"
    return f"Which of these six words {rel} '{form}': {SIX_OR}?\nAnswer with one word.\nAnswer:"


@torch.no_grad()
def run(model, tok, families=FAMILIES):
    dev, cid = next(model.parameters()).device, candidate_ids(tok, "POST")
    out = {}
    for fam in families:
        out[fam] = {}
        for word, form in zip(LOCATIONS, FORMS[fam]):
            lp = torch.log_softmax(model(encode(tok, prompt(fam, form)).to(dev), use_cache=False, logits_to_keep=1).logits[0, -1].float(), -1)
            pick = LOCATIONS[int(lp[cid].argmax())]
            out[fam][word] = {"form": form, "pick": pick, "correct": pick == word, "top1": tok.decode([int(lp.argmax())]),
                              "lp": {w: lp[i].item() for w, i in zip(LOCATIONS, cid)}}
    return out


def competent(fam: dict) -> bool:
    return sum(w["correct"] for w in fam.values()) >= 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", default="results/gpu_stage5/competence")
    ap.add_argument("--test", action="store_true", help=f"TEST_MODE: {TEST_MODEL}, FP32, CPU")
    a = ap.parse_args()
    test = bool(a.test or os.environ.get("TEST_MODE"))
    if test:
        a.model, a.dtype, a.device_map = TEST_MODEL, "float32", None
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    kw = {"dtype": getattr(torch, a.dtype), "revision": a.revision} | ({"device_map": a.device_map} if a.device_map else {})
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    if not a.device_map and torch.cuda.is_available() and not test:
        model = model.to("cuda")
    fams = run(model, tok)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = f"{a.out}/{a.model.split('/')[-1]}.json"
    json.dump({"provenance": provenance(a) | {"attn_implementation": model.config._attn_implementation}, "families": fams}, open(f, "w"))
    for fam, words in fams.items():
        print(f"{fam:7s} {sum(w['correct'] for w in words.values())}/6 {'competent' if competent(words) else 'NOT competent'}  "
              + "  ".join(f"{w['form']}->{w['pick']}{'' if w['correct'] else '!'}" for w in words.values()))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
