# Stage 8 design pilots (seen before finalisation of preregistration P-2026-10-10-J)

CPU pilots run while stage 8 was designed and critiqued, before the entry was finalised. Every number here that bears on
a line of the entry is listed in the entry's "Seen before finalisation" sections. No study model at 7B or above was run;
the pilots used Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B and tokenizers only.

- `partA/`: natural-text items (earlier builder versions; the committed builder is `ckeys/squad_items.py`), tokenizer
  validity, the 0.5B effect and continuation pilots, power. The SQuAD file and the intermediate item files are not kept
  (they are rebuilt by `scripts/build_stage8a_items.py`).
- `partB/`: emitted-form scorer prototype, Hub hash capture, coverage and exactness pilots, power simulations.
- `partC/`: identity-edit pilots at 0.5B and 1.5B, power.
- `partD/`: flag, ablation, sign, dissociation and IOI pilots at 0.5B and 1.5B.
- `critic_*/`: the critics' own checks (closed-book prior and cue conflict; JB9 chance and lexicon checks; share
  compression and non-lexical steering; the leave-one-word-out flag, K_N and the value-channel sign).

Paths inside these scripts point at the scratch directory where they ran; they are kept as run, not as maintained code.
