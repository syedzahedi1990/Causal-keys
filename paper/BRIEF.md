# Writing brief v2: revision after stages 3 and 3b (ICML 2027 submission, 8-page main text)

## Hard rules (unchanged)
1. **Numbers.** Every number reported from our experiments must be a macro from `paper/numbers.tex`, or come from an auto-generated table in `paper/tables/`, or appear verbatim in `docs/PREREGISTRATION.md`. Never type a result number from memory.
2. **Citations.** Cite only keys in `paper/references.bib`. Descriptions of cited work must be supported by that work's abstract. Prior work by the authors is `anonymous2026fitted` and is always referred to in the third person, as "Anonymous (2026)".
3. **Anonymity.**
   - Never write "Paper 1", "P1", "our previous paper", "reviewer repository" or anything that links us to the prior work.
   - Use the macros `\fmtOpt` etc. for format names.
   - Remove the `% Paper 1's format` comments in `main.tex` if you touch it.
4. **Honesty.**
   - Separate preregistered (confirmatory) from post hoc results.
   - Report failed predictions: the transfer law C1/C2 and the C4 threshold.
   - Hedge claims we did not test, e.g. attribution graphs.
5. **Length.**
   - The main text, from the Introduction to the end of the Conclusion including figures and tables, must fit in **8 pages**.
   - The Impact Statement and references come after it.
   - Move detail to the appendix. Do not repeat the same number in more than two places in the main text.
6. **`\todo`.** Leave no `\todo` in the main text. All results are now in.

## Title
**Looked Up or Copied? Later Mentions Decide Which Attention Channel Carries an In-Context Value — and Where an Intervention Appears to Act**

## Core story (in one paragraph)
- A token that writes a value into context, such as the location in "the candle is moved to the shelf", can be read by later tokens in two ways:
  - through its **key**: a later token whose query matches it attends to it;
  - through its **value**: what is copied when it is attended to.
- We clamp the two channels separately at that token.
- **Finding 1.** The value's identity is **looked up through the key** whenever the candidate values are mentioned again *after* the writing token, by an option list, lettered options or, in models of 7B and up, even a neutral sentence. The re-mentioned words themselves are the readers. Without a later mention, the value is **copied**: keys carry no identity.
  - This holds in 10 models (1.5B–72B, four families) and on three tasks.
  - It survives an instruction-matched 2×2 design.
  - Position is causal: the same list or sentence placed before the writing token carries nothing.
  - The token's channels carry the value's **identity**, not the belief role around it.
- **Finding 2.** Revisiting a learned DAS intervention from Anonymous (2026) at 24B and 72B:
  - its behavioural effect is the same in every prompt format;
  - whether it is carried by the writing token's keys or values **flips** with the format (a crossover).
  - So a component-level explanation of an intervention ("the effect is carried by keys") is a property of the (intervention, prompt) pair, not of the intervention alone.
- **Finding 3** (negative): a preregistered transfer law based on clamping-estimated channel completeness failed.

## Terminology (fixes review item 9; use exactly)
- **Writing token:** the single token where the story states the value (the location word in the move event). It is called the "critical token" in Anonymous (2026); say so once.
- **Later mention:** any token after the writing token that names a candidate value.
- **Key channel / value channel:** the writing token's cached key (pre-RoPE `k_proj` output) and value (`v_proj` output), as read by later tokens.
- **Formats** (macros in `main.tex`):
  - `\fmtOpt` (options listed after the question, "Answer with exactly one choice"; the format of Anonymous 2026);
  - `\fmtListA` (the same list with "Answer with one word");
  - `\fmtLetter` (A–F options after);
  - `\fmtPost` (the neutral sentence "The room has a box, a basket, … and a closet." after the story);
  - `\fmtSentB` (the same sentence before the story);
  - `\fmtNone` (no candidates);
  - `\fmtBefore` (the options list before the story).
  - The instruction-matched 2×2 is {list, sentence} × {before, after} = `\fmtListA`, `\fmtBefore`, `\fmtPost`, `\fmtSentB`, all with the one-word instruction.
- **Self-clamp row:** the batched baseline in which the writing token is clamped to its own key and value.
- **Log-odds:**
  - natural factorial: m_SB = log p(S) − log p(B);
  - Anonymous-2026 frames: m_TS = log p(T) − log p(S).
- **Quantities:**
  - d_K, d_V, d_KV: effects of clamping the key, the value, or both to the source run's;
  - s_K = d̄_K / (d̄_K + d̄_V), the key share;
  - ID_K, ID_V: value identity carried by keys or values, the double difference against a third value X absent from the story;
  - φ, ψ_K, ρ_K, ψ_V, ρ_V for the intervention frames;
  - f_K, f_V: the fraction of the role-swap effect carried by the writing token's key or value.

## Setup facts (Methods, about 1.2 pages, self-contained)
- **Task:** the belief stories of Anonymous (2026) (template in the appendix). Six single-token locations. Base B and source S differ only at the writing token.
  - New tasks: **paint** ("Later, {agent} repainted the {thing} {colour}.", six colours) and **schedule** ("Later, {agent} moved the {thing} to {day}.", seven days).
- **Encoder:** system turn, user turn, generation prompt with thinking disabled, and an "Answer:" prefill. Full-vocabulary log-probabilities at the answer position.
- **Clamp:** for every layer ℓ ≥ ℓ₀, set the writing token's cached key and/or value to those of another run. Later tokens see a context token only through these, so clamping both from ℓ₀ = 0 reproduces the source run at all later positions; this is checked numerically. This is one sentence plus a footnote, not a lemma.
- **Identity double difference** (equation). It cancels non-specific effects of any foreign key, such as weakening B.
- **Row-restricted splice:** attention runs twice per layer and output rows are spliced by a mask, so that only a chosen group of later tokens sees the swapped key. It is exact (all rows = full swap, no rows = clean, < 1e-4).
- **Role-swap control:** the observer and non-observer names are swapped in the event, keeping the same location word. The belief answer changes from LOC to the initial location. f_C is the fraction of this change carried by the writing token's channel C.
- **Intervention of Anonymous (2026)** (self-contained, review item 8):
  - **M:** a rank-16 distributed alignment search (DAS) interchange intervention h_B + (h_S − h_B)UᵀU at the output of decoder block 4 over the whole move-event span. U was trained so that the patched model answers π(S) under the \fmtOpt readout, where π is a fixed pair-swap of the six locations.
  - **P:** an unfitted rank-16 PCA basis of source-minus-base differences, which transfers S.
  - We reuse the released bases (3 seeds each).
  - **Reproduction:** argmax agreement with the released outputs is 0.994–1.000.
  - **Key-only and value-only exchange:** the writing token's key (or value) from block 6 on is taken from the other run, as in Anonymous (2026). φ, ψ and ρ are defined with equations.
  - **Population:** 96 cores with B, S and T distinct.
- **Models:** Qwen2.5-Instruct 1.5B/3B/7B/14B/32B/72B, Qwen3-8B, Mistral-7B-Instruct-v0.3, Mistral-Small-24B-Instruct-2501, OLMo-2-7B-Instruct. n = 150 stories per format (60 for localisation).
- **Statistics:** 95% bootstrap over stories.
- **Preregistration:**
  - Five preregistrations (A–E), each committed to version control before its run, with scoring scripts committed before outputs were inspected.
  - The appendix includes a table classifying every main claim as confirmatory or post hoc.
  - An independent code audit before the GPU runs is mentioned in one appendix sentence.

## Results with macros
**R1. Later mentions decide** (Figure `fig_formats`, Table `tab_natural`, Table `tab_2x2`):
- **Natural factorial, 10 models** (preregistrations A, B, C):
  - ID_K is large for \fmtOpt and \fmtLetter in every model;
  - \fmtPost is intermediate from 7B up (about 0 at 1.5B and 3B, which was not predicted);
  - \fmtNone is at most `\noneMax` nats;
  - \fmtBefore is at most `\beforeMax`.
- **Instruction-matched 2×2** (preregistration E, 4 models):
  - ID_K(\fmtListA) is `\twoAfterMin`–`\twoAfterMax` nats;
  - ID_K(\fmtPost) is `\twoPostMin`–`\twoPostMax`;
  - \fmtBefore and \fmtSentB are at most `\twoBeforeMax`;
  - AFTER − BEFORE and POST − PRE are > 0 in 4/4 models.
  - Position, not the instruction or the list format, is causal.

**R2. Generality:**
- **Tasks** (preregistration D, `tab_tasks`):
  - paint: \fmtOpt ID_K `\taskPaintOptMin`–`\taskPaintOptMax` nats;
  - schedule: `\taskScheduleOptMin`–`\taskScheduleOptMax`;
  - \fmtBefore at most `\taskPaintBeforeMax` / `\taskScheduleBeforeMax`;
  - 5/5 models per task.
- **Scale** (`fig_scale`), Qwen2.5 key share: `\shareQwenOnefive`, `\shareQwenThree`, `\shareQwenSeven`, `\shareQwenFourteen`, `\shareQwenThirtytwo`, `\shareQwenSeventytwo`.
  - The rise to 14B and the dip after it are not an artefact of the software environment: re-runs gave `\shareEnvFourteen` and `\shareEnvThirtytwo`, identical to the originals (E3).
  - Other families: `\shareQwenThreeEight`, `\shareMistralSeven`, `\shareMistralTwentyfour`, `\shareOlmoSeven`.
- **Onset sweep** (`tab_onset`, appendix): key shares remain substantial with the clamp starting at about 0.3L in most models, so the read is not just the input embedding. OLMo-2 is the exception.

**R3. The later mentions are the readers** (`fig_localisation`, `tab_localisation`, preregistration D3):
- At 7–14B, option-word rows alone recover `\locScaleOptMin`–`\locScaleOptMax` of the key effect, and question rows `\locScaleQuestionMax`.
- In \fmtPost, the re-mention words recover `\locScalePostMin`–`\locScalePostMax`.
- At 1.5B: option words recover `\locOptChoicewords` (\fmtOpt) and `\locLetterChoicewords` (\fmtLetter); layers 4–15; one dominant KV group (`tab_windows`, appendix).

**R4. Identity, not belief role** (`tab_role`, preregistration E2):
- The role swap shifts the belief answer by many nats.
- The writing token's key and value each carry at most `\roleFracMax` of that shift, in 5 models including 72B.
- So the channels carry *which value*, while who observed it is computed elsewhere. This also answers the lexical-confound concern: the key read is an identity lookup, and we claim nothing more.

**R5. Where the intervention appears to act** (`fig_frames`, the headline crossover; `tab_frames`, appendix):
- φ is flat across formats: Mistral `\phiOptMistral`–`\phiBeforeMistral`, Qwen `\phiNoneQwen`–`\phiBeforeQwen`. Check actual ranges against the table.
- ψ_K vs ψ_V:
  - \fmtLetter: `\psiLetterQwen` vs `\psiVLetterQwen`;
  - \fmtOpt: `\psiOptQwen` vs `\psiVOptQwen`;
  - \fmtPost: `\psiPostQwen` vs `\psiVPostQwen`;
  - \fmtNone: `\psiNoneQwen` vs `\psiVNoneQwen`;
  - \fmtBefore: `\psiBeforeQwen` vs `\psiVBeforeQwen`.
  - Mistral shows the same pattern.
- The edit is present in both channels; which one "carries" it depends on which channel the format's readers use.
- The key result of Anonymous (2026) holds for its format, but it is not a property of the intervention alone.

**R6. Failed preregistered predictions (C1/C2; C4 threshold):**
- **C1/C2:** φ under \fmtNone was predicted to equal κ_V estimated from clamping (0.53 Mistral, 0.73 Qwen). Observed: `\phiNoneMistral` (fail) and `\phiNoneQwen` (at the edge of the margin). φ_\fmtOpt − φ_\fmtNone is null for Mistral and tiny for Qwen.
- **C4:** the key-share ≥ 0.80 threshold was met only at Qwen2.5-32B.

## Review items to fix (checklist)
1. **Length.** Fit 8 pages; cut repetition; compress Related Work and strip its numbers; a 5–6 sentence abstract with at most 4 numbers; add an Impact Statement.
2. **"Copied"/"both channels".** Now measured (R5, ψ_V; plus d_V and ID_V in `tab_decomposition`, appendix). State it with that evidence.
3. **Lexical confound.**
   - Say plainly that the key read is an identity lookup in the style of duplicate-token attention by later mentions, and cite `wang2022ioi` for duplicate-token heads.
   - Show it is not only the input embedding (onset sweep).
   - Show it is about identity, not belief role (R4).
4. **Readout vs position.** The 2×2 shows position is what matters, so frame the paper around *later mentions*, not "readout format" in general.
5. **ID_K scale.**
   - Report CIs (they are in the tables).
   - Describe \fmtBefore honestly: it is ≤ 0 in most models and reaches about −2 at Qwen2.5-72B (see `tab_natural`).
   - Do not headline "+0.01".
   - Report the interaction and d_K/d_KV in the appendix (`tab_decomposition`).
6. **Scale/environment.** Resolved by E3; say so.
7. **Localisation.** Now has 4 models with CIs in the tables. Title the subsection precisely.
8. **Self-contained description** of the Anonymous (2026) intervention (above); remove f* if unused.
9. **Terminology:** one glossary, used consistently (above).
10. **Figures:**
    - Fig 1 = `fig_schematic.pdf` (figure*, top of page 2 or 1).
    - Fig 2 = `fig_frames.pdf` (the crossover headline, figure*).
    - Fig 3 = `fig_scale.pdf`.
    - Fig 4 = `fig_localisation.pdf` (figure*, or column width with \resizebox).
    - Table 1 = `tab_2x2.tex` (main, compact).
    - Move `fig_formats`, `tab_natural`, `tab_frames` and the rest to the appendix if space requires.

## Files
- Sections: `paper/sections/{abstract,intro,method,results,related,discussion,impact,appendix}.tex`.
- `main.tex` order: abstract, intro, method (titled "Setup"), results (two `\section`s: "How later tokens read a written value" and "Where an intervention appears to act"), related, discussion (with a conclusion paragraph), impact statement, references, appendix.
- Tables: `paper/tables/*.tex`. Figures: `paper/figures/*.pdf`.
