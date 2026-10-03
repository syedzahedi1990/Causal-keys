# Writing brief for Paper 2 (ICML 2027 submission draft)

## Hard rules for every writer
1. **Numbers.**
   - Every number reported from our experiments must be written as a macro from `paper/numbers.tex`, e.g. `\shareQwenFourteen`, `\psiOptQwen`.
   - Or it must come from an auto-generated table (`paper/tables/*.tex`), or appear verbatim in `docs/PREREGISTRATION.md`.
   - Never type a result number that is not in one of those sources.
   - Rounded prose such as "about three quarters" must match the macro value.
2. **Citations.**
   - Cite only keys in `paper/references.bib`.
   - Describe a cited work only in ways its abstract supports. When unsure, describe it more weakly.
   - Paper 1 is `anonymous2026fitted`. Refer to it in the third person ("Anonymous (2026) showed..."), because the paper is double-blind.
3. **Honesty.**
   - Report what failed. The preregistered transfer law (C1/C2) failed for Mistral-24B and was marginal for Qwen-72B.
   - The C4 threshold was met only at Qwen-32B.
   - The 1.5B "options-only" refinement was post hoc. It passed a same-model fresh-seed test but did not generalise to ≥ 7B, where neutral re-mentions also open the key channel.
   - Never claim more than the data show.
4. **Style.**
   - Use LaTeX with the ICML 2026 style. No `\section*` for main sections.
   - Prose should be concise and concrete: no hype, no "groundbreaking", no rhetorical questions.
   - Use the present tense for findings.
5. **Placeholders.** Stage-3 results (new tasks; localisation at 7–14B) are pending. Mark where they go with `\todo{...}` and do not invent them.

## Working title
**Looked Up or Copied? The Readout Decides Which Attention Channel Carries In-Context State and Intervention Effects**

## One-paragraph story
Interpretability claims about where a model stores and reads in-context state, and about which component carries an intervention's effect, are usually measured under one readout format. Multiple-choice prompts are common, both in benchmarks and in Paper 1. Later tokens can see a context token only through its attention key (which decides whether they attend to it) and its value (what they copy). We clamp these two channels separately at the single token where a story writes a state. We show that the readout format decides which channel carries the state:
- When candidate answers are re-mentioned after the state token, the re-mentioned words look the state up through the state token's **key**.
- When nothing re-mentions them, the state is **copied** from the value.

Consequence: the same learned intervention from Paper 1, which changes behaviour equally under every format, appears to be "carried by keys" only under multiple-choice readouts. So component-level explanations of interventions are **readout-relative**.

## Setup facts (for Methods)
**Task.** The synthetic belief/world-state stories are ported exactly from Paper 1:
- "Everyone initially sees that the {o} is in the {init}. Everyone initially sees that the {d} is in the {dloc}. {A} watches as the {o} is moved to the {LOC}. {B} does not see this happen."
- Question: "Where does {A} believe the {o} is?" (direct view).
- Six single-token locations: box, basket, shelf, drawer, cabinet, closet.
- Base (B) and source (S) stories differ only at the critical token {LOC}.

**Encoder.** Exactly Paper 1's: the system turn "You are a helpful assistant.", the user turn holding the raw prompt, the generation prompt with thinking disabled, and an "Answer:" prefill. Scoring uses full-vocabulary log-probabilities at the answer position.

**Format arms** (identical story; only the candidate listing or instructions change):
- **P1** (Paper 1): "Choices: box, basket, …" after the question, "Answer with exactly one choice."
- **LETTER:** "A) box, B) basket, …" after the question; the answer is a letter.
- **POST:** the neutral sentence "The room has a box, a basket, a shelf, a drawer, a cabinet and a closet." appended after the story; free-form answer.
- **NONE:** no candidates; "Answer with one word."
- **BEFORE:** the choices are listed *before* the story; free-form answer. Under the causal mask, listed tokens cannot attend to the state token.

**Clamp.** At the critical token, the cached key (pre-RoPE `k_proj` output) and/or value (`v_proj` output) read by later tokens is set to the source run's, for all layers ≥ l0 (l0 = 0 primary; depth-matched l0 = 0.0625·L gives identical results). Because later tokens see a context token only through its key and value, clamping both reproduces the source run exactly; this is verified to numerical precision.

**Metrics.**
- **Log-odds:** m = log p(S) − log p(B). d_C = m(C) − m(identity row in the same batch).
- **Key share:** dK / (dK + dV).
- **Interaction:** dKV − dK − dV.
- **identity(K):** 0.5·[(Δlog p_S(K_S) − Δlog p_S(K_X)) + (Δlog p_X(K_X) − Δlog p_X(K_S))], where X is a third location absent from the story body. This double difference cancels any non-specific effect of a foreign key, such as weakening B. identity(V) is defined analogously.
- **CIs:** 95% core bootstrap (10,000 resamples).

**Row-restricted splice (localisation).** Each attention block runs twice per layer, once with the base key at the critical position and once with the source key, and output rows are spliced by a mask. Only a chosen token group "sees" the swapped key. This is exact: all rows reproduce the full swap and no rows reproduce the clean run, each to < 1e-4.

**Paper 1 frames (stage 2).**
- **Bases:** Paper 1's released rank-16 bases (learned pair-swap remap M, unfitted PCA P, intended transfer f*; seeds 101–103) are applied as h_B + (h_S − h_B)UᵀU at the output of 1-based block 4 over the event span. This is computed in BF16 exactly as Paper 1's engine, with transformers 5.9.0 and pinned model revisions.
- **Reproduction:** argmax agreement with Paper 1's saved outputs is 0.994–1.000.
- **Key-only exchange:** Paper 1's native design (keys at the critical token from 1-based block 6 on; queries and values evolve).
- **Metrics:**
  - φ = [m(M) − m(P)] / [m(T) − m(S)], with m = log p(T) − log p(S);
  - ψ = [m(P + K_M) − m(P)] / [m(M) − m(P)] (addition);
  - ρ = [m(M + K_P) − m(M)] / [m(P) − m(M)] (removal).
- **Population:** 96 native cores with B, S and T distinct.

**Models (10).**
- Qwen2.5-Instruct: 1.5B, 3B, 7B, 14B, 32B, 72B.
- Qwen3-8B.
- Mistral-7B-Instruct-v0.3 and Mistral-Small-24B-Instruct-2501.
- OLMo-2-7B-Instruct.

n = 150 cores per format (n = 40 at 1.5B on CPU for the early tests; the stage-1 GPU run re-ran 1.5B at n = 150).

**Statistics and preregistration.** Predictions were committed to the repository before each run, and the scoring scripts were committed before outputs were inspected. Four preregistrations (A–D) exist; their outcomes are in `docs/PREREGISTRATION.md`.

**Code audit.** An independent multi-agent code review found, and we fixed, several issues before the GPU runs:
- a KV-cache bug in the double-attention splice;
- the Qwen3 thinking-mode template;
- the Gemma system-role template;
- an unapplied competence filter;
- hand-computed statistics.

Worth one sentence in the appendix: it supports trust.

## Results to report (macros in brackets)
**R1. Keys carry identity when candidates are re-mentioned after the state** (`tab_natural.tex`, `fig_formats.pdf`, `fig_scale.pdf`).
- Across all 10 models, identity(K) is large for P1 and LETTER.
- POST is intermediate at ≥ 7B and about 0 at 1.5–3B.
- NONE is small (max `\noneMax` nats).
- BEFORE is ≤ `\beforeMax` in every model. This is the structural control: no reader after the state.
- Preregistered predictions 1 and 2 of B were met in 5/5 models.

**R2. Scale.** Qwen2.5 key share:
- `\shareQwenOnefive` (1.5B), `\shareQwenThree`, `\shareQwenSeven`, `\shareQwenFourteen` (14B), `\shareQwenThirtytwo` (32B), `\shareQwenSeventytwo` (72B).
- It rises to 14B and is non-monotonic after that.

Other models: Qwen3-8B `\shareQwenThreeEight`, Mistral-7B `\shareMistralSeven`, Mistral-24B `\shareMistralTwentyfour`, OLMo-2-7B `\shareOlmoSeven`. The 72B value agrees with Paper 1's fixed-value estimate `\sKQwenFV`, from a different estimand.

**R3. Localisation at 1.5B** (`fig_localisation.pdf`).
- Option-word rows alone recover `\locOptChoicewords` (P1) and `\locLetterChoicewords` (LETTER) of the full key effect.
- Question rows recover `\locOptQuestion`, story-tail rows `\locOptStorytail`, and the remaining rows `\locOptRestafterp`.
- Layers 4–15 of 28 carry the read, with one KV group dominant (stated in PREREGISTRATION/PROPOSAL).
- A mechanism probe agrees: the option word matching the key's identity attends to the state token.
- `\todo{}` for localisation at 7–14B (stage 3).

**R4. Paper 1 revisited** (`tab_frames.tex`, `fig_frames.pdf`).
- The learned remap's behaviour is readout-invariant: φ is `\phiOptMistral`–`\phiLetterMistral` for Mistral and about `\phiOptQwen` for Qwen across all formats; PCA source transfer is 1.00 in all formats.
- Key-only addition ψ is `\psiLetterQwen`/`\psiOptQwen` with options, `\psiPostQwen` with a re-mention sentence, `\psiNoneQwen` with none and `\psiBeforeQwen` with options before. Mistral shows the same pattern.
- Removal ρ behaves the same way.
- Conclusion: Paper 1's key result is a property of its readout. The edit is redundantly present in both channels.

**R5. Negative result.**
- The preregistered transfer law (free-form effect ≈ value-channel completeness κ_V, estimated from Paper 1's fixed-value clamping) failed for Mistral-24B: predicted 0.53 ± 0.15, observed `\phiNoneMistral`. It was marginal for Qwen-72B: predicted 0.73 ± 0.15, observed `\phiNoneQwen`.
- C2 (φ_P1 > φ_NONE) was null for Mistral and tiny for Qwen.
- Lesson: completeness under clamping does not predict how an intervention transfers across readouts.

**R6. Generality across tasks.** `\todo{}` stage 3: paint and schedule tasks.

## Discussion points
- **Practical rule:** a component-level causal claim about an intervention (or a circuit) should name its readout, and should be tested under at least one alternative readout. Multiple-choice evaluation formats, common in interpretability benchmarks such as MIB's MCQA and in safety evals, can route the same state through different components.
- **Attribution graphs** that freeze attention patterns would miss the key-channel share under re-mention readouts. This is an inference, not tested here; state it as such.
- **Relation to lookbacks:** our free-form case agrees with the address-in-key / payload-in-value picture (Prakash et al.; Oh & Demberg). The re-mention case adds an identity-matching route in which the state token's key carries content identity matched by later mentions.
- **Limitations:**
  - synthetic stories;
  - single-token states;
  - the post hoc history of the options-only refinement;
  - a non-monotonic scale curve with one family;
  - a residual NONE effect of about 1 nat at scale;
  - the clamp produces states no input produces (we mitigate with identity rows and exactness);
  - the mechanism of the residual NONE effect and of the POST/scale interaction is not established.
- **Thesis link:** this sharpens Paper 1's conclusion. Its key result holds under its readout but is not a readout-invariant property of the intervention.

## Notation (use exactly these; macros are defined in main.tex)
- Format arms: `\fmtOpt{}` (options after; Paper 1's format), `\fmtLetter{}` (lettered options), `\fmtPost{}` (re-mention sentence), `\fmtNone{}` (no re-mention), `\fmtBefore{}` (options before).
- Quantities: key share `$s_K$`; identity carried by keys `$\mathrm{ID}_K$` (values: `$\mathrm{ID}_V$`); remap effect `$\varphi$`; key-only addition `$\psi$`; key-only removal `$\rho$`; value-channel completeness `$\kappa_V$`.
- Runs: base B, source S, remapped target T = π(S), learned remap patch M, PCA patch P.
- Terms:
  - "key channel" / "value channel": the cached key / value of the state token, as read by later tokens.
  - "readout": the answer format.
  - "re-mention": any later token that names a candidate.

## Files
- Write your section to `paper/sections/<name>.tex`.
- Figures are in `paper/figures/` (`fig_scale.pdf`, `fig_formats.pdf`, `fig_frames.pdf`, `fig_localisation.pdf`).
- Tables are in `paper/tables/` (`tab_natural.tex`, `tab_frames.tex`); include them with `\input{tables/...}`.
- Label convention: `\label{sec:...}`, `\label{fig:...}`, `\label{tab:...}`, cross-referenced with `\cref`.
