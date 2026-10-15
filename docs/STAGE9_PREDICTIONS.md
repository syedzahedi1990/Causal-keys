# Stage 9 predictions: the listed candidates set the edit window

Dated 2026-10-14. **Frozen when committed before the stage-9 GPU run**, in the commit titled `Freeze stage-9 predictions`.
`scripts/gpu_stage9.sh` records that commit and refuses to run if this note, `docs/stage9/SPEC.md` or the code changed
after it. This is a short dated note, not a preregistration entry. Populations, rows, parsers and statistics are defined
in `docs/stage9/SPEC.md` (cited as S5, S7, ...).

**Claim under test.** At the token that writes a value, with the candidates listed after the context, the token's
effect on the answer is fixed early: by about a third of the depth in Mistral-7B and Llama-3.1-8B, about half in
Qwen2.5-7B. Overwriting all it holds from there on leaves the old answer, confidently. Overwriting only the layers
before does the reverse. Without a later list (no mention, list before, null sentence after) the late overwrite
changes the answer and the early one does not; a neutral sentence after keeps the late window open (Direction A). An edit reaches a listed answer only if it
delivers most of the natural key displacement, while free-form answers follow partial edits (Direction B).

## 1. Seen before this note (post hoc)

The pattern was found in stage 1-8 data after stage 8. Everything below is post hoc. The new cores (N9, E9), the early
windows, every window row on passages, every generated answer under a window, the dose and multiplier rows and
all-position steering are unseen. Numbers re-derived on 2026-10-14 from committed results; `analysis/stage9_pre.py`
(S15) reproduces them. Every measured number in this note is rounded half up from the recomputed value (an exact tie
such as 39/40 = 0.975 prints 0.98).

| # | Source (committed) | What was seen |
|---|---|---|
| 1 | stage 8b, 150 F cores, exact K/V patch of S from round(0.3 L), E-scored argmax (not generation), two-stage cluster bootstrap 95 % | AFTER answer stays B: Mistral 0.90 [0.83, 0.96], Llama 0.82 [0.74, 0.90], Phi-4 0.67, OLMo-2 0.61, Falcon3 0.25, Gemma-2-9B 0.19, Qwen-7B 0.09 [0.03, 0.17], Qwen-14B 0.01. NONE, BEFORE, POST-NULL, PRE flip 0.98-1.00; POST 0.87 (Mistral), 0.99-1.00 otherwise; full patch 0.98-1.00. Of the non-flips, B: 0.985 / 0.992 / 0.61 (Mistral / Llama / Qwen-7B). Median absolute p(B) of stale answers: Mistral 0.999, Llama 0.846 |
| 2 | stage 8c, 80 E cores, natural K/V patch from blocks 4, 8, 12, 16, E-argmax, targets S and X | P1 (options after, "exactly one choice"): Qwen 0.98 / 0.94 / 0.69 / 0.00, Mistral 1.00 / 0.36 / 0.01 / 0.00, Llama 1.00 / 0.68 / 0.01 / 0.00. LETTER: Qwen 0.86 / 0.73 / 0.30 / 0.00, Mistral 0.97 / 0.29 / 0.00 / 0.00, Llama 1.00 / 0.22 / 0.00 / 0.00. NONE 1.00 at all four. Value-only: NONE 0.84-0.99, P1 at most 0.01. Brackets for l_c(P1): Mistral (4, 8], Llama (8, 12], Qwen (12, 16] blocks; l_c(NONE) > 16 in all three |
| 3 | stage 8a explore, 185 E passages, single-channel rows, argmax over the five decision tokens | OPTA key from layer 0 / from 0.3 L: 0.65 / 0.19 (Qwen), 0.41 / 0.01 (Mistral), 0.25 / 0.04 (Llama). NOM value: 0.93 / 0.92, 0.88 / 0.87, 0.93 / 0.94 |
| 4 | stage 8c explore, block 7, 40 cores, generated answers | E1 flips NONE 1.00 vs P1 0.35 (Mistral), 0.95 (Qwen), 0.65 (Llama). E4b 0.98 vs 0.03 (Mistral), 0.99 vs 0.25 (Qwen). At Mistral 99-100 % of the P1 failures are B. Edit key share κ/σ vs transfer: Spearman 0.92 (9 edit x model pairs) |
| 5 | stage 8c dose rows (SK, SV), block 3 | Half key, full value: P1 flips 0.31 / 0.32 / 0.66 (Qwen / Mistral / Llama). Half value, full key: NONE flips 0.93 / 0.92 / 0.94. ψ_K(0.5) in P1 0.10 / 0.18 / 0.13 against ψ_V(0.5) in NONE 0.60 / 0.58 / 0.56. Per-core ψ_K(0.5) at Qwen: median 0.06 (10th-90th percentile 0.02-0.20). λ ranks: calibrated curve Spearman 0.862, log proj_k alone 0.855 (difference of the unrounded values 0.008, 95 % [−0.060, 0.090]) |
| 6 | stage 5 knockout | Cutting the answer's attention to the writing token keeps 1.00 of listed answers. Under NONE it sends 0.79-1.00 of answers to the initial or distractor location |

## 2. Predictions

All are read on point estimates against the thresholds below. A verdict per model is SUPPORTED (point and the whole
95 % interval on the predicted side), SUPPORTED AT THE POINT, NOT SUPPORTED, or NOT EVALUABLE (S10). "Each model" = each
of mistral7, llama8 and qwen7. l is the onset block. Grid: every second block; l_c = the first onset with late flip
< 0.5 (L if none). l_b = the first onset with late AFTER flip ≤ 0.2 on the 30 C cores, fixed before N9 is scored (S9.6).
The last column gives the in-hand analogue (from section 1), to show which predictions are at risk.

| ID | Prediction (generated answers; stories N9, passages E) | In-hand analogue |
|---|---|---|
| P1 | Stories: late flip ≥ 0.95 at every l ≤ 0.5 L in NONE, BEFORE and POST-NULL, each model. Passages: late flip ≥ 0.8 in NOM and OPTB at x = 0.15, 0.3, 0.45, each model | 0.98-1.00 at 0.3 L (1); NONE 1.00 to block 16 (2); NOM value 0.87-0.94 (3) |
| P2 | Stories: l_c(AFTER)/L and l_c(LETTER)/L ≤ 0.35 at Mistral and Llama and ≤ 0.55 at Qwen; [l_c(NONE) − l_c(AFTER)]/L ≥ 0.15, each model. Passages: OPTA late flip at round(0.3 L) ≤ 0.4 in at least 2 of 3 models | P1 brackets (2): Mistral (0.13, 0.25] L, Llama (0.25, 0.38] L, Qwen (0.43, 0.57] L, which straddles 0.55; l_c(NONE) > 0.5 L (Qwen > 0.57 L) |
| P3 | Stories, at l_b: early flip ≥ 0.7 in AFTER and in LETTER, ≤ 0.05 in NONE and in BEFORE, each model. Passages, early window to round(0.3 L): OPTA ≥ 0.5, NOM ≤ 0.1, OPTB ≤ 0.1, each model | none (no early row was ever run). OPTA key alone from 0: 0.25 at Llama |
| P4 | At l_b in AFTER: ≥ 0.8 of non-flipped answers are B, each model. At Mistral: median absolute p(B) of the B answers ≥ 0.8 | 0.985 / 0.992 / 0.61 and 0.999, at 0.3 L (1) |
| P5a | Single-site E1 tracks the natural late patch: abs(flip(E1) − flip(nat)) ≤ 0.1 in every format x block cell (NONE, AFTER, LETTER, POST x 3, 7, 11, 15), each model | largest gap 0.11 (Qwen, P1, block 11; E-argmax) |
| P5b | E4b fails with the list even where the window is open: transfer τ = flip(AFTER)/flip(NONE) ≤ 0.7 at every block where E4b's NONE flip ≥ 0.5, at Qwen and Mistral. Precondition: the natural patch's τ at block 3 ≥ 0.9 (else NOT EVALUABLE) | P1: τ(3) 0.59 / 0.56 (E-argmax); block 7: 0.25 / 0.03 (generation) |
| P5c | All-position steering (CAA, E1 at every position ≥ p): reported either way. It "transfers" if its AFTER flip ≥ 0.5 at block 11 or 15 where the natural patch's AFTER flip ≤ 0.2 | none |
| POST | Late flip ≥ 0.8 at every l ≤ 0.5 L, each model (a neutral later mention does not close the window) | Mistral 0.87 at 0.31 L; 0.88-0.89 at blocks 8-16 (2) |
| B1 | Block 3: joint interpolation at α = 0.5 flips ≤ 0.35 of AFTER answers and ≥ 0.9 of NONE answers, each model | half key, full value 0.31 / 0.32 / **0.66** (Llama at risk) |
| B2 | Block 3: E1 at 0.5x flips ≥ 0.9 of NONE answers and ≤ 0.5 of AFTER answers, each model | none |
| B3 | r_K (delivered fraction of the natural key displacement, S8.2) predicts AFTER transfer: AUC ≥ 0.8 over the E1 multipliers, E4b and the joint-α rows, in cells where the natural patch transfers (S9.7), each model | κ/σ vs transfer 0.92 (4); log proj_k 0.855 (5) |

## 3. Falsifiers (point estimates, any model)

| ID | Triggered when | What it would mean |
|---|---|---|
| F1 | early flip at l_b ≥ 0.2 in NONE or in BEFORE | the copy route also reads shallow layers: no crossover, only the one-sided window |
| F2 | [l_c(BEFORE) − l_c(AFTER)]/L < 0.1; passages: late flip OPTB − OPTA at round(0.3 L) < 0.1 | the multiple-choice answer space, not the list's position, closes the window: an evaluation warning |
| F3 | [l_c(NONE) − l_c(AFTER)]/L < 0.1 | no format gap in depth |
| B-F1 | joint α = 0.5 AFTER flip ≥ 0.6 at block 3 | no behavioural threshold |
| B-F2 | ψ_V(NONE, 0.5) − ψ_K(AFTER, 0.5) ≤ 0.1 at block 3 (E, nats) | the value curve is as convex as the key curve |

## 4. Controls in the design, and the alternative each removes

| Control | Alternative removed |
|---|---|
| Full patch KV@0 ≈ 1.00 in every format (stories and passages) | "listed answers are sticky" |
| BEFORE and OPTB (same answer space, list before the context) | "the multiple-choice answer space protects the answer" |
| POST-NULL (same length, no candidates) | "a longer prompt" |
| Competence filter, clean accuracy reported per format | "the model cannot do the listed task" |
| Both lexicons, per-core list orders (B before S vs S before B reported) | "a list-order or lexicon artefact" |
| Early ∪ late = full patch, exact in FP32 (S7.2) | "mixed-depth states are broken states" |
| The none row against the clean B run of the same item, computed in another batch (the second none row checks that one batch is deterministic) | BF16 noise as an effect |
| All-position CAA | "every steering method fails" (scope) |
| E4b at block 3, where the natural patch transfers | "remaps fail only because they act late" |
| Interpolation and logit-difference ψ beside nats | "log-probability saturation makes the value curve look concave" |

## 5. Primary analysis

- The deciding numbers are generated answers. Stories and edits use 16 tokens with the candidate stop and the letter
  parser. Passages use 48 tokens and the containment rule (S9.3): an entity counts when its normalised text occurs as
  whole words in the normalised first line, where normalising drops a possessive 's, removes Unicode punctuation and
  splits words at dashes and slashes, the same for every candidate. Each format uses its competent items (S9.4).
  E-scored argmax, strict match (the stage-8a rule) and all-items versions are sensitivity analyses, and so is the count
  of letter answers whose text names another candidate word. They are printed, and they do not decide.
- Intervals: 95 % two-stage cluster bootstrap, 10,000 resamples, seed 20261015. Clusters: (lexicon, base, source) for
  stories, (base, source) for edits, the article for passages. One index set per population is shared across formats,
  so contrasts are paired. l_c, ψ and the AUC are recomputed in every resample.
- Combination over models: a prediction is SUPPORTED if SUPPORTED in all three; SUPPORTED AT THE POINT if at least that
  in all three; NOT SUPPORTED if any model is NOT SUPPORTED (named); otherwise NOT EVALUABLE. P2-passages counts models at
  the point. There is no multiplicity correction, and every number is printed.
- The 71 R passages and the LETA format are held-out and optional checks. They decide nothing.

## 6. Reported whichever way, and the pre-written readings

Always reported:
- late and early flip curves for every format and model (the main figure), l_c, l_b;
- silent-failure shares and confidence, and the free-form/listed inconsistency;
- edit transfer bars (nat, E1, CAA, E4b) and the ψ curves in nats and logits;
- r_K, the AUC and the fixed rule "r_K ≥ 0.8";
- the passage table (E and R), and the controls of section 4.

Depths are stated per model, never as a universal depth.

- **P1, P2, P3 (stories) at least at the point, F1-F3 not triggered:** "When the candidates are listed after the context,
  a model has finished using a context token by l_c(AFTER)/L of its depth: overwriting everything the token holds from
  there on leaves X-Y % of answers unchanged, with confidence, though the same overwrite changes every free-form answer,
  and overwriting only the layers before it does the reverse." Add the edits clause if P5a-b hold.
- **P1, P2 hold, P3 fails or F1 triggers:** "Late edits fail silently with listed candidates; the early window also moves
  free-form answers, so the routes overlap in depth."
- **F2 triggers:** report only the evaluation warning ("intervention verdicts depend on the answer format"), in the
  appendix, with no mechanism claim.
- **P5c transfers:** state the claim for single-site edits, patching and KV-cache edits only.
- **B1-B3 hold and B-F1, B-F2 not triggered:** "an edit at a context token reaches a listed answer only if it delivers
  about 80 % or more of the natural key displacement; free-form answers follow partial edits." Otherwise, describe the
  convexity without a threshold claim. The failed channel-ratio law stays reported as failed.
