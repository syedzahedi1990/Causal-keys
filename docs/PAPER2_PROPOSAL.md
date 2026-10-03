# Paper 2 proposal: *Looked Up, Not Copied*

**Working title:** *The Readout Decides: Whether In-Context State Is Looked Up Through Attention Keys or Copied From Values, and Why Component-Level Explanations of Interventions Do Not Transfer Across Answer Formats*

**Status (2026-10-03, after GPU stages 1, 2, 3 and 3b; paper draft in `paper/`, main text 8 pages):**
- Every planned GPU experiment is complete. Predictions and their outcomes, met and not met, are in `docs/PREREGISTRATION.md`. Compute spent: two short Vast runs.
- **Gate G1:** not met as originally written. The narrowed "options only" claim, adopted post hoc at 1.5B, passed a fresh-seed preregistered test, then failed to generalise at ≥ 7B, where plain re-mentions also open the key channel.
- **Stage 1** (P-2026-10-03-B): predictions 1 and 2 met in 5/5 open models.
- **Stage 2** (P-2026-10-03-C):
  - Paper 1 reproduced item by item (99.4–100%).
  - C3 met in both models.
  - The transfer-law predictions C1/C2 were **not** supported (met only marginally at 72B).
  - The C4 key-share threshold was met only at 32B; its BEFORE control was met in all three models.
- **Stage 3** (P-2026-10-03-D): all predictions met.
  - Paint and schedule tasks: OPT − NONE > 0 in 5/5 models per task; BEFORE ≤ 0.5 nats in 10/10 cells.
  - Localisation at 7–14B: option words recover 0.92–0.98 of the key effect; the neutral sentence's candidate words recover 0.67–0.73 (0.19–0.22 recovered by no single group).
- **Stage 3b** (P-2026-10-04-E): all predictions met.
  - Instruction-matched 2×2: position decides; LIST-AFTER 8.7–35.8 nats, LIST-BEFORE and SENTENCE-BEFORE ≤ +0.01.
  - Role control: the writing token's key or value carries ≤ 0.02 of the belief-role effect, including at 72B.
  - Environment re-runs reproduce the 14B and 32B key shares (0.91, 0.86); the fall beyond 14B is real.
  - Value-only exchange: ψ_V(NONE) = 0.80 (Mistral-24B) and 0.90 (Qwen-72B), above ψ_K.
- **Exploratory cross-check (paper Fig. 2c):** the remap's key share ψ_K/(ψ_K+ψ_V) tracks the unpatched model's identity key share ID_K/(ID_K+ID_V) across 5 formats × 2 models (Pearson r = 0.98; largest gap under SENTENCE-AFTER).
- Measurement code was audited by an independent review workflow, and the verified issues are fixed.

---

## 1. Findings (what the paper can now claim)

**F1. Mechanism: later re-mentions read in-context state through the state token's key.**
- **Setting:** a story writes a location at a critical token, and the critical token's key and value are clamped to those of another run.
- **Result:** the location's identity travels through the key whenever the candidate answers are re-mentioned after the state token, and the re-mentioned candidate words are what read it.
  - Exact row-restricted swaps at 1.5B: the option words recover 87–98% of the key effect, the question and answer positions about 0%.
  - The read happens in early-to-mid layers and is concentrated in one KV group.
- **Structural control:** when the options come before the story, keys carry no identity, in all 10 models (≤ 0.01 nats).
- **Without any re-mention:** keys carry at most about 1 nat, under 5% of the option-listing effect.
- **Scale and families:** the effect holds from 0.5B to 72B in four families (Qwen2.5, Qwen3, Mistral, OLMo-2).
  - Key share of the multiple-choice answer, Qwen2.5: 0.37 (1.5B), 0.69, 0.82, 0.91 (14B), 0.86 (32B), 0.76 (72B).
  - Other models: Mistral-7B 0.92, Mistral-24B 0.78, OLMo-2-7B 0.73, Qwen3-8B 0.84.
  - From 7B up, a neutral sentence re-mentioning the candidates opens the key channel at roughly ⅓–½ the strength of an options list.

**F2. Consequence: component-level explanations of an intervention are readout-relative.**
- **Behaviour is readout-invariant.** Paper 1's learned remap moves the answer by the same fraction of the natural S→T shift in every format: Mistral-24B φ 0.71–0.75, Qwen-72B φ 0.86–0.91. PCA source transfer is 100% in every format.
- **The key-only exchange is not.** Swapping critical-token keys between the learned and PCA runs (Paper 1's central causal result):
  - transfers the remap strongly with options after the story (key addition ψ 0.53–0.89, removal ρ 0.76–0.99);
  - transfers it weakly with a neutral re-mention (ψ about 0.2);
  - transfers essentially nothing for free-form or options-before readouts (ψ ≤ 0.07).
- **Interpretation:** the remap is written redundantly into the key and value channels. Paper 1's "keys carry the learned–PCA difference" is true of its multiple-choice readout and false of a free-form one.
- **Lesson:** where an intervention's effect "lives" depends on how the model is read out. Mechanistic attributions measured under multiple-choice formats, which are common in interpretability benchmarks and evals, need not transfer to free-form behaviour.

**F3. Negative result.** Channel completeness estimated under clamping (κ_V from Paper 1's fixed-value design) does **not** predict free-form transfer. It underestimates how fully the edit is carried by values (Mistral-24B: predicted 0.53, observed 0.71).

**Tool.** kvaudit (planned): exact key/value clamps, row-restricted splices, format arms and transfer cards, as one call.

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

**Depth and KV group (only the option-word rows see the swapped key):**

| Rows/layers seeing the swapped key | Paper 1 format | Lettered options |
|---|---|---|
| Layers 0–3 | −1% | −1% |
| **Layers 4–7** | **19%** | **35%** |
| Layers 8–11 | 12% | 5% |
| Layers 12–15 | 13% | 3% |
| Layers 16–27 | ≤ 2% | ≤ 1% |
| KV group 0 / KV group 1 | 19% / **46%** | 6% / **65%** |

- The read happens in early-to-mid layers (4–15 of 28) and is concentrated in one KV group.
- Single 4-layer windows sum to only about 45% of the all-layer effect, so the read compounds across layers. This matches Paper 1's finding that narrow block bands recover only part of the key effect.
- Fractions are of the all-layer, choice-words-only effect at n=40 (`results/row_restricted_windows/`).

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
