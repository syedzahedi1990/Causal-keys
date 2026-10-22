# Stage 9: outcome (recorded 2026-10-22)

The predictions are in `docs/STAGE9_PREDICTIONS.md` and the computations in `docs/stage9/SPEC.md`. Both were frozen
unchanged (sha256 `3e97d031...` and `ca9efe81...`) at `69763e9`. The results are in `results/gpu_stage9/`, and the
score is in `results/gpu_stage9/STAGE9_SCORE.txt`, written by the frozen scorer. A local re-score reproduces it
exactly, paths aside. `docs/stage9/RUN_LOG.md` lists the sessions.

## Runs

| Session | Date | Setting | Result |
|---|---|---|---|
| 0 | 2026-10-21 | freeze `45f5d8a` | stopped at the test gate, before any model loaded (RUN_LOG) |
| 1 | 2026-10-21 | freeze `69763e9`, `DEADLINE_H` 6.0 | every step except the edits of Mistral and Llama, which stopped at the deadline reserve |
| 2 | 2026-10-22 | same commit and results, `DEADLINE_H` 8 | redid the Mistral and Llama edits from the start and finished them |

The spec estimated 25 minutes for the edits per model; the run took about 100 at Mistral and Llama and about 32 at
Qwen. All 24 steps finished. The checks are OK: the FP32 gate passed, l_b was recomputed and read in order, T_check
is 0 everywhere, and there is one freeze record. The archive's sha256 is `3ed66fc4...`.

## Verdicts

| Line | Verdict | Mistral-7B | Llama-3.1-8B | Qwen2.5-7B |
|---|---|---|---|---|
| P1 (free form stays open) | NOT SUPPORTED | at the point | NOT SUPPORTED (passages) | SUPPORTED |
| P2 (listed candidates close the late window) | **SUPPORTED** | SUPPORTED | SUPPORTED | SUPPORTED (stories; passages NOT) |
| P3 (the crossover: the early window opens) | NOT SUPPORTED | NOT SUPPORTED | at the point | NOT SUPPORTED (passages) |
| P4 (failures are silent) | **SUPPORTED** | SUPPORTED | SUPPORTED | SUPPORTED |
| P5a (single-site E1 tracks the patch) | NOT SUPPORTED | at the point | at the point | NOT SUPPORTED (one cell, -0.106) |
| P5b (E4b fails with the list) | NOT SUPPORTED | NOT SUPPORTED (block 3: 0.703 vs 0.7) | (not in the line) | at the point |
| P5c (all-position steering, reported) | TRANSFERS | TRANSFERS | TRANSFERS | TRANSFERS |
| POST (scope) | NOT SUPPORTED | NOT SUPPORTED | at the point | SUPPORTED |
| B1, B2 (half-dose thresholds) | NOT SUPPORTED | NOT | NOT | NOT |
| B3 (r_K predicts transfer, AUC >= 0.8) | **SUPPORTED** | SUPPORTED | SUPPORTED | SUPPORTED |
| F2 (format dependence absent) | TRIGGERED | not | not | TRIGGERED on passages only |
| F1, F3, B-F1, B-F2 | not triggered | | | |

Totals over P1-P4, P5a, P5b, POST and B1-B3: 3 SUPPORTED, 7 NOT SUPPORTED, 0 NOT EVALUABLE. Several NOT SUPPORTED
lines miss by one cell near the threshold (P5a at Qwen, P5b at Mistral, P3s at Mistral). The score file lists every
cell.

## Key numbers (generated answers, 95 % two-stage cluster bootstrap)

- **The window, from stories (N9, 150 cores).** l_c is the last block at which a late K/V patch at the writing token
  still flips the answer.

  | Model | l_c(NONE)/L | l_c(AFTER)/L | l_c(LETTER)/L | (NONE - AFTER)/L |
  |---|---|---|---|---|
  | Mistral | 22/32 | 8/32 = 0.25 | 0.25 | 0.44 |
  | Llama | 26/32 | 10/32 = 0.31 | 0.19 | 0.50 |
  | Qwen | 26/28 | 14/28 = 0.50 | 0.36 | 0.43 |

  The BEFORE format (options before the story) stays open, like NONE: (l_c(BEFORE) - l_c(AFTER))/L is 0.31, 0.25 and
  0.36.
- **The early window at l_b, from stories.** The early patch flips 0.000 of the answers in NONE and in BEFORE in every
  model. It flips this share in AFTER and LETTER:

  | Model | AFTER | LETTER |
  |---|---|---|
  | Mistral | 0.64 | 0.73 |
  | Llama | 0.83 | 0.98 |
  | Qwen | 0.87 | 0.85 |
- **Silent failure.** Where the late patch fails in AFTER at l_b, the answer is the stale B answer in 98.9-100 % of
  cases. At Mistral the median p(B) of those answers is 0.998.
- **Passages (stage 8a's E set, 185 items).** OPTA's late flip at 0.3 L is 0.08 at Mistral and 0.28 at Llama, against
  OPTB's 0.99 and 0.95 (OPTB - OPTA = 0.91 and 0.67). At Qwen it is 0.94: the options do not close the window there,
  so F2p triggers. OPTA's early flip is 0.80 at Mistral and 0.52 at Llama; NOM and OPTB stay at 0.00-0.02.
- **Edits (E9, 80 cores).**
  - **E1 (single-site).** It tracks the natural patch within 0.1 in 47 of 48 format x block cells.
  - **E4b (trained DAS remap).** It flips the free-form answer in 0.64-1.00 of cores but transfers to AFTER at only
    tau 0.70 (block 3), 0.05-0.30 (block 7) and 0.00-0.03 (blocks 11 and 15).
  - **All-position steering (CAA).** Where the natural patch's AFTER flip is 0.000-0.013, CAA's is 0.96-1.00 at
    blocks 11 and 15, in all three models.
- **Key share predicts transfer.** r_K = proj_k predicts whether an edit transfers to AFTER with AUC 0.90, 0.94 and
  0.91 (n = 1568, 1380, 3122 units). The rule r_K >= 0.8 classifies cores with accuracy 0.82-0.86.
- **Dose.** Half-dose edits flip only 0.31-0.47 of the NONE answers. The convexity holds, but B1 and B2's thresholds
  (>= 0.9) fail.

## Pre-written readings the outcome selects (note section 6)

- **F2 triggers, on passages at Qwen.** The passages part is reported only as an evaluation warning ("intervention
  verdicts depend on the answer format"), with no mechanism claim.
- **P5c transfers.** The window claim is stated for single-site edits, patching and KV-cache edits only, not for
  all-position steering.
- **Direction B.** The convexity is described without a threshold claim. The failed channel-ratio law stays reported
  as failed.
