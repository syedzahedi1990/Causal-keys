# Paper 2 proposal: *Looked Up, Not Copied*

**Working title:** Looked Up, Not Copied: The Readout Decides Whether In-Context State Travels Through Attention Keys or Values, and What Interventions Change

**Status (2026-10-03, after code review):**
- **Gate G1 was not met as originally written.** No-option formats still show 1–2 nats of key effect.
- **The option-listing version of C1 is exploratory.** It was adopted *after* seeing the data. A fresh-seed CPU test and an out-of-sample GPU test are preregistered in `docs/PREREGISTRATION.md`.
- **Measurement code was audited** by an independent review workflow, and the verified issues are fixed.
- **The reader is localised** at 1.5B (Section 3b″).
- **GPU stages have not started** ($0 spent). The Colab notebook for stage 1 is ready.

---

## 1. One-paragraph pitch

A prompt writes some state into context, such as "Alice watches as the lamp is moved to the cabinet". Later tokens can read that state token in two ways: by matching its attention key or by copying its value.

We show that the readout decides which:
- **Answer options listed after the state token** (a multiple-choice line, as in Paper 1): the listed option words look the state up by key identity.
  - With Paper 1's encoder, keys carry a key share of 0.37 [0.34, 0.41] of the answer log-odds at Qwen2.5-1.5B, and about 0.35 at 0.5B (pilot, n=6).
  - Lettered options give 0.71.
  - Paper 1's 24B/72B fixed-value data give 0.73 / 0.745. That is a related but different estimand (blocks 6+, values fixed at the target run, a patched recipient), so it is **not** on the same scale curve. Matched-estimand scale points are a GPU-stage question.
- **No listed options after the state token** (free answer, options before the story, or a neutral re-mention sentence): keys carry about 0.02–0.08 of the log-odds and almost no identity. The value is copied.

This split has consequences for interventions. Paper 1's learned remap was fit under a multiple-choice readout, and it put more of its edit into the channel that readout weights: it is 96% complete in keys but only 73% in values (Qwen2.5-72B). That explains Paper 1's open "why keys?" result. It also predicts how such interventions will (fail to) transfer to free-form answers. And it implies that attribution methods which freeze attention patterns (the QK side) miss most of the state's effect in multiple-choice-style prompts.

We ship **kvaudit**, a one-call key/value/format audit for any intervention or interpretability claim.

## 2. How this direction was chosen

1. **Broad search (39-agent workflow).** Seven literature scouts covered 262 papers, mostly 2025–2026. Six lens-diverse ideation agents produced 18 ideas, merged into 8 candidates. Each candidate got two adversarial novelty checks and a review panel, then a judge ranked them.
   - The winner was "key/value channel accounting", the most thesis-coherent and cheapest option.
   - The runner-up, "eval-awareness steering validation", had the best hiring pull, but its central claim had already been made by four 2026 groups, and it cost $400–900.
2. **Falsification on Paper 1's own released data, at zero cost.** The judge's central prediction was that keys act as mere pointers, so the remap should need the target mentioned in context. It is false: the remap writes the target equally often whether or not the target is mentioned in the story body. Meanwhile, Paper 1's fixed-value study shows that the critical-token keys, not the values, decide the answer.
3. **New CPU experiments** isolated the variable that matters: whether the candidates are re-mentioned after the state token (Section 3).
4. **Second workflow (8 agents).** Two novelty sweeps, two adversarial critics and three independent designs, plus a synthesis, gave the plan below. I verified its key numbers independently (`analysis/paper1_fixed_value_logodds.py`).

## 3. Evidence so far

### 3a. Paper 1 fixed-value data (Qwen2.5-72B, Mistral-24B; critical-token values clamped to the target run's V_T in blocks 6+)

The table gives the mean log p(S) − log p(T) on direct questions (48 distinct stories × 3 fits). "Endpoint" rows are runs with no swap; the other rows keep V_T fixed and vary the key source.

| Cell | Qwen2.5-72B | Mistral-24B |
|---|---|---|
| endpoint S / endpoint T | +25.9 / −26.1 | +9.7 / −9.7 |
| natural source keys, values fixed at V_T | +12.7 | +4.3 |
| natural target keys, values fixed at V_T | −26.1 | −9.7 |
| **key share s_K** | **0.745** | **0.730** |
| learned remap M: completeness in keys κ_K | 0.96 [0.93, 0.98] | 0.74–0.76 [0.70, 0.80] |
| learned remap M: completeness in values κ_V | 0.73 [0.64, 0.80] | 0.50–0.56 [0.38, 0.67] |

CIs are 95% story-cluster bootstraps (48 stories; fits kept together). The two values per Mistral cell are the P and M frames. κ_V assumes the key and value effects add.

Argmax rates (≈100% key-determined) overstate this; log-odds are the right scale.

### 3b. New natural key/value factorial (unpatched stories; clamp the critical token's key and/or value to the source run's)

Qwen2.5-1.5B, argmax rate of answering S among items answered correctly in both clean runs:

| Format | keys only → S | values only → S | both → S |
|---|---|---|---|
| choices listed after story | 32–50% | 28–43% | 100% |
| no choices (free answer) | **0%** | 81–98% | 100% |
| choices listed before story | **0%** | 100% | 100% |

- This early script used no system prompt or prefill and did not save the clean-run log-probs. It is **superseded** by 3b′, and its argmax rates condition on competence (the n for choices-before was only 9–11 of 50).
- SmolLM2-1.7B shows the same pattern without choices (keys 0%, values 86%).

### 3b′. Gate G1 with Paper 1's exact encoder (Qwen2.5-1.5B, n=40, log-odds, 95% CIs)

Results for direct questions. Identity is measured as a double difference against a third location X that is absent from the story body, which cancels the non-specific "weaken the base" effect.

| Format | Key effect | Value effect | Key share | Identity carried by keys |
|---|---|---|---|---|
| Paper 1 (choices after) | +6.0 [5.3, 6.9] | +10.1 | 0.37 | **+2.6** |
| Lettered options after | +9.3 | +3.7 | **0.71** | **+3.2** |
| Choices before the story | +1.2 | +13.1 | 0.08 | −0.2 |
| No choices | +1.9 | +21.1 | 0.08 | +0.3 |
| Neutral sentence re-mentioning all six locations after the story | +0.4 [−0.1, 0.8] | +14.3 | 0.02 | −0.6 |

- **Additivity.** Keys and values add approximately in P1, BEFORE, NONE and POST, where all interaction CIs include 0. LETTER has a positive interaction of +2.0 [0.3, 3.7] nats, about 13% of the joint effect. (Recovery = 1.000 is an identity, not evidence of additivity.)
- **Population.** These numbers pool all 40 items. Restricted to competent items, POST's key effect is +0.95 and NONE's is +2.6.
- **Identity CIs.** The identity column was computed by hand. It is now implemented with bootstrap CIs in `experiments/format_factorial.py`.

**Refinement (post hoc, exploratory until confirmed):** a plain re-mention does not open the key channel. **Listed answer options** after the state token do, and more option-like formats (letters) give a larger key share. Without options, keys only slightly weaken the base answer.

### 3b″. Who reads the key? Exact row-restricted key swap (attention computed twice per layer and spliced by row)

| Rows allowed to see the source key | Paper 1 format | Lettered options |
|---|---|---|
| The six location words in the options line | **87%** of the full key effect | **98%** |
| Question tokens | 0% | 1% |
| Story after the state token | −1% | 0% |
| Instruction, chat tokens, prefill, answer position | 0% | 0% |

### 3c. Mechanism probe

This probe used an earlier setup: world view, no system prompt or prefill, n=25, no CIs.

With the base story's critical key replaced by the source run's key, the **source word in the choices line** attends to the critical token as it does in the source run. Its peak attention at layer 16 is 0.35 vs 0.15 for non-matching words. The base word's early-layer attention falls to the non-matching level, though at layer 16 it stays above it. So later candidate mentions look the state token up by key identity (`results/choice_attention_qwen1.5b_world.txt`).

## 4. Claims (to be preregistered after gate G2)

- **C1, option-listing gate** (post hoc; see `docs/PREREGISTRATION.md`).
  - Without listed answer options after the state token, identity(K) is inside an equivalence margin of ±1 nat.
  - With listed options after it, identity(K) > 1 nat.
- **C2, additivity.** Key and value effects add in log-odds (interaction ≤ 15% of the joint effect).
- **C3, reader.** The key effect is read by the re-mentioned candidate tokens. It is localised to identified heads and layers, and knocking out candidate→state attention removes it.
- **C4, scale (exploratory, no direction preregistered).** How the multiple-choice key share varies across Qwen2.5 sizes, measured with one matched estimand (l0 = 0 and l0 = 0.0625·L).
- **C5, channel completeness.** Interventions fit under a readout put more of their edit into that readout's dominant channel.
  - DAS fit under multiple-choice: key-biased.
  - DAS fit free-form: value-complete.
  - Unfitted PCA: complete in both.
- **C6, transfer law (headline consequence).** An intervention's free-form effect ≈ its value completeness κ_V. Out-of-sample prediction for Paper 1's released remap bases: **κ_V = 0.73 [0.64, 0.80] at 72B and 0.50–0.56 [0.38, 0.67] at 24B**. The 24B interval is wide, so that test is weak.
- **C7, attribution blind spot.** Freezing attention patterns keeps only ≈ (1 − s_K) of the state's effect under re-mention readouts.
- **C8, generality.** The same sign pattern holds on CausalToM (Prakash et al.) and MIB MCQA.

## 5. Positioning: what is known vs new

**Known (cited up front, not claimed):**
- Multiple-choice answers are selected via QK at option tokens: Lieberum et al. 2023; Tulchinskii et al. 2024; Anthropic's QK attributions (Kamath et al. 2025); Sun et al. 2026; Wong et al. 2026.
- Options must attend back to the context: Ok & Lee 2026.
- For free-form answers, binding is in keys and payload in values: Prakash et al. lookbacks; Oh & Demberg 2026; Wu & Shomali 2026.
- Steering and function vectors do not transfer across answer formats: Opiełka et al. 2026; Gao et al. 2026.

**New (two independent sweeps, arXiv/OpenReview/web through Sept 2026, found none of these):**
- A causal key/value decomposition at an in-context state token showing that a post-state re-mention opens a key channel that carries most of the answer at scale.
- The dissociation of re-mention from answer format.
- Localisation of the reader to re-mention tokens.
- The finding that fitting under a readout biases an intervention's channel, with a quantitative transfer law.
- A quantified blind spot for attention-frozen attribution.

Novelty is estimated at 6.5/10 for the bare mechanism and 7.5/10 for the full package. Wong et al. 2026 explicitly list format-dependence as untested.

**Thesis fit:**
- **Paper 1:** what an intervention changes, beyond its fitted answer; keys carry the learned-vs-PCA difference.
- **Paper 2:** *why* keys. The readout determines the channel, and the channel determines what an intervention changes and whether it transfers.

## 6. Plan and gates

| Stage | What | Where | Cost | Gate / decision |
|---|---|---|---|---|
| X1 | Format factorial (7 arms) with Paper 1's exact encoder, 0.5B/1.5B | CPU (running) | $0 | G1: key effect ≥ 3 nats in Paper 1's format and ≈ 0 with no re-mention |
| X2 | Reader localisation (row-restricted key swaps, head/group windows, knockouts); identity/role/omit/shuffle controls | CPU | $0 | C3/C4 thresholds |
| X3 | DAS (intended + remap) fit under multiple-choice vs free-form, PCA, mean-difference steering; channel completeness; cross-format transfer; attention-freeze, at 1.5B | Colab (L4 ~10 h) | ~$10 | G2: channel bias and transfer law at 1.5B, then freeze the preregistration |
| X4 | Scale and families: Qwen2.5-3B/7B/14B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B; X3 core at 7B | Colab (A100 ~25 h) | ~$30 | C4, C6 pooled |
| X5 | CausalToM and MIB MCQA format-transfer cards | Colab (A100 ~6 h) | ~$7 | C8 |
| X6 | **Paper 1 bases at 24B, 32B and 72B:** remap and PCA evaluated free-form (out-of-sample transfer-law test), full key/value factorial, attention-freeze | Vast 1×H100 ~12 h, then 2×H100 ~10 h | ~$95 | 72B only if the 24B result is clean |
| X7 | kvaudit package (pre-RoPE GQA-aware clamps, exact row splice, format generators, cards) | CPU, rolling | $0 | ships regardless |

- **Expected total:** about $150–200 (cap $300).
- **Timeline:** CPU gates by about Oct 12; Colab Oct 12–Nov 14; Vast Nov 15–22; writing until Dec 14; arXiv about Dec 15; ICML 2027 in late January.
- **Fallback if C1 fails:** keys carry the state for some other reason, such as a head-pattern code. The paper then becomes a key/value accounting and transfer-law paper, still anchored on the Paper 1 numbers.

## 7. Decisions needed from the author

1. **Compute route.** Either:
   - a Colab Pro notebook that writes results to Google Drive (I can read them), or
   - Vast instances I drive myself, which needs `VAST_API_KEY` set as an environment variable in the cloud-environment settings (a new session picks it up).
2. **`HF_TOKEN` (read-only) as an environment variable,** plus licence acceptance for the Llama-3.x, Gemma-2 and Mistral models.
3. **Paper 1 policy.**
   - Paper 2 reports two refinements of Paper 1's readings, using Paper 1's own data: the key share is ≈0.73 rather than ≈100%, and the "availability" result concerns story-body mentions. Is that OK?
   - Should the arXiv post wait for the ICLR anonymity period?
4. **Budget sign-off:** about $200 expected, $300 cap.
