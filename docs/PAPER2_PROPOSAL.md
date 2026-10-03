# Paper 2 proposal: *Looked Up, Not Copied*

**Working title:** Looked Up, Not Copied: The Readout Decides Whether In-Context State Travels Through Attention Keys or Values, and What Interventions Change

**Status (2026-10-03):** direction chosen and pilot evidence in hand. CPU gate experiments are running. GPU stages are not started yet ($0 spent).

---

## 1. One-paragraph pitch

A prompt writes some state into context, such as "Alice watches as the lamp is moved to the cabinet". Later tokens can read that state token in two ways: by matching its attention key or by copying its value.

We show that the readout decides which:
- **Candidates listed after the state token** (a multiple-choice line, as in Paper 1): each listed candidate looks the state up by key identity. Keys then carry a share of the answer log-odds that grows with scale: about 0.31 at 0.5B, about 0.5 at 1.5B, 0.73 at 24B and 0.745 at 72B.
- **No later re-mention** (free-form answer): the key share is exactly 0 and the value is copied.

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
| learned remap M: completeness in keys / in values | 0.96 / 0.73 | 0.76 / ≈0.53 |

Argmax rates (≈100% key-determined) overstate this; log-odds are the right scale.

### 3b. New natural key/value factorial (unpatched stories; clamp the critical token's key and/or value to the source run's)

Qwen2.5-1.5B, argmax rate of answering S among items answered correctly in both clean runs:

| Format | keys only → S | values only → S | both → S |
|---|---|---|---|
| choices listed after story | 32–50% | 28–43% | 100% |
| no choices (free answer) | **0%** | 81–98% | 100% |
| choices listed before story | **0%** | 100% | 100% |

- In log-odds, keys and values **add** (interaction ≈ 0) when choices come after the story, with key share 0.44–0.52.
- SmolLM2-1.7B shows the same pattern without choices (keys 0%, values 86%).

### 3c. Mechanism probe

With the base story's critical key replaced by the source run's key, the **source word in the choices line** attends to the critical token exactly as it does in the source run. Its peak attention at layer 16 is 0.35 vs 0.15 for non-matching words. The base word's attention drops to baseline. So later candidate mentions look the state token up by key identity (`results/choice_attention_qwen1.5b_world.txt`).

## 4. Claims (to be preregistered after gate G2)

- **C1, re-mention gate.**
  - Key effect ≈ 0 with no post-state re-mention: choices before the story, no choices, a neutral listing sentence before the story, cloze.
  - Key effect > 0 with a post-state re-mention: choices after, lettered options, or a neutral sentence after the story with a free-form answer.
- **C2, additivity.** Key and value effects add in log-odds (interaction ≤ 15% of the joint effect).
- **C3, reader.** The key effect is read by the re-mentioned candidate tokens. It is localised to identified heads and layers, and knocking out candidate→state attention removes it.
- **C4, scale.** In Qwen2.5 from 0.5B to 72B (plus Llama, Gemma and Mistral), the key share of the multiple-choice answer rises with scale.
- **C5, channel completeness.** Interventions fit under a readout put more of their edit into that readout's dominant channel.
  - DAS fit under multiple-choice: key-biased.
  - DAS fit free-form: value-complete.
  - Unfitted PCA: complete in both.
- **C6, transfer law (headline consequence).** An intervention's free-form effect ≈ its value completeness κ_V. Out-of-sample prediction for Paper 1's released remap bases: **0.73 ± 0.15 at 72B and ≈0.5 ± 0.15 at 24B**.
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
