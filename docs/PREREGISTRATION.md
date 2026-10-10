# Preregistered predictions (committed before the corresponding runs)

## P-2026-10-03-A: fresh-seed CPU confirmation of the option-listing gate

**Context.** The option-listing version of claim C1 was formed *after* seeing the seed-0 data (results/format_factorial_v2). The neutral re-mention arm (POST) had no key effect, so the original "any re-mention" claim was revised. That revision is exploratory. This test uses new story cores (seed 1) and the corrected script (experiments/format_factorial.py at the commit that adds this file).

**Run:** `python experiments/format_factorial.py --model Qwen/Qwen2.5-1.5B-Instruct --n 40 --seed 1 --arms P1,NONE,BEFORE,POST,LETTER --out results/format_factorial_v3`

**Statistic.** identity(K) = 0.5[(Δlogp_S(K_S) − Δlogp_S(K_X)) + (Δlogp_X(K_X) − Δlogp_X(K_S))], with keys clamped from layer 0. All items are primary. CIs come from a core bootstrap.

**Predictions:**
1. identity(K) > 1.0 nats with a 95% CI excluding 0, in P1 and in LETTER.
2. identity(K) inside the equivalence margin ±1.0 nats (95% CI contained in [−1.0, +1.0]), in NONE, BEFORE and POST.
3. Key share (l0 = 0) in P1 lies in [0.25, 0.50].

**Reporting.** Each prediction is reported as met or not met, with no reinterpretation. If prediction 2 fails for an arm, C1 is weakened to "keys carry much more identity with listed options than without", tested as P1 − NONE with a CI.

## P-2026-10-03-B: GPU stage 1 (3–14B, notebooks/gpu_stage1_format_factorial.ipynb), out-of-sample models

**Predictions,** each judged per model on all items (n = 150):
1. identity(K) in P1 and LETTER is positive, with a CI excluding 0, in at least 4 of 5 open models (Qwen2.5-3B/7B, Qwen3-8B, Mistral-7B-v0.3, OLMo-2-7B).
2. identity(K)(P1) − identity(K)(NONE) > 0, with a CI excluding 0, in at least 4 of 5 models.
3. Exploratory, no directional claim: how the P1 key share (l0 = 0 and l0 = 0.0625·L) varies with size within Qwen2.5 (1.5B, 3B, 7B, 14B).

---

## Outcome of P-2026-10-03-A (run after the predictions were committed; results/format_factorial_v3)

Qwen2.5-1.5B-Instruct, seed 1 (new cores), n = 40 per arm, all items.

| Prediction | Observed | Verdict |
|---|---|---|
| 1. identity(K) > 1.0, CI excluding 0, in P1 and LETTER | P1 +2.85 [2.38, 3.33]; LETTER +4.10 [3.45, 4.77] | **met** |
| 2. identity(K) CI within ±1.0 in NONE, BEFORE, POST | NONE +0.10 [−0.01, 0.21]; BEFORE −0.20 [−0.28, −0.13]; POST −0.41 [−0.56, −0.26] | **met** |
| 3. Key share (l0 = 0) in P1 within [0.25, 0.50] | 0.38 [0.35, 0.41] | **met** |

**Notes.**
- BEFORE and POST are slightly but reliably negative. They sit inside the margin but are not exactly zero.
- The seed-0 LETTER interaction (+2.0 [0.3, 3.7]) did not replicate: seed 1 gives +0.6 [−1.0, 2.2].
- This confirms the post hoc option-listing claim in the **same model** on new data. Generalisation across models is P-2026-10-03-B, the GPU test.

---

## Outcome of P-2026-10-03-B (GPU stage 1; scored by analysis/stage1_prereg.py, committed before inspection at 004579c)

- **Setup:** A100 80GB, code at f706ef5, n = 150 per arm, all items, 95% core bootstrap. Full output: `results/gpu_stage1/PREREG_SCORE.txt`.
- **Prediction 1** (identity(K) > 0, CI excluding 0, in both P1 and LETTER): **met, 5/5 models.**
- **Prediction 2** (paired identity(K) for P1 − NONE > 0, CI excluding 0): **met, 5/5 models.**
- **Prediction 3** (exploratory): the Qwen2.5 P1 key share rises steadily with size. The depth-matched l0 gives identical values.

| Qwen2.5 | 1.5B | 3B | 7B | 14B |
|---|---|---|---|---|
| P1 key share | 0.37 [0.36, 0.39] | 0.69 [0.68, 0.71] | 0.82 [0.81, 0.83] | 0.91 [0.90, 0.91] |
| P1 identity(K), nats | +2.8 | +10.3 | +21.2 | +39.0 |

Other families: Qwen3-8B 0.84, Mistral-7B 0.92, OLMo-2-7B 0.73.

**Unpredicted observations** (reported, not reinterpreted):
- **POST** (a neutral sentence re-mentioning all six locations after the story, free-form answer) carries large key identity in the larger models:
  - Qwen2.5-7B +5.5, Qwen2.5-14B +12.1, Qwen3-8B +8.8, Mistral-7B +7.9, OLMo-2-7B +2.0 nats;
  - but about 0 at Qwen2.5-1.5B/3B (−0.47 and −0.04).

  So the post hoc "listed answer options only" refinement from 1.5B does **not** generalise. At scale, a plain re-mention after the state token also opens the key channel, at roughly a third to half the strength of an options list. This is closer to the original re-mention hypothesis H*.
- **BEFORE** (options before the story) is at or below 0 in every model (−0.45 to +0.01). This is the cleanest structural control: readers that precede the state token cannot read its key.
- **NONE** carries small positive identity(K) in larger models (+1.0 to +1.25 nats at 7–14B). That is outside the CPU-era ±1 margin, though under 5% of P1.

---

## P-2026-10-03-C: GPU stage 2, Paper 1's released bases at 24B/72B across formats, plus a natural factorial at 24–72B

**Committed before any stage-2 run.**

**Runs:**
- `scripts/gpu_stage2.sh` → `experiments/paper1_frames.py` (native story cores, Paper 1 encoder, transformers 5.9.0, pinned model revisions, released `original_1000` bases m3/pca/f_star, seeds 101–103).
- `experiments/format_factorial.py` at Mistral-Small-24B, Qwen2.5-32B and Qwen2.5-72B (n = 150).

**Scoring:** `analysis/stage2_score.py`, committed with this entry.
- Primary population: native cores with B, S and T distinct.
- Fits are averaged within core, then a core bootstrap is taken.
- φ_f = [m(M) − m(P)] / [m(T) − m(S)], where m = logp(T) − logp(S).
- ψ_f = [m(P+K_M) − m(P)] / [m(M) − m(P)].

**Gate (reproduction).** In P1, the argmax of M and of P must agree with Paper 1's saved native outputs on ≥ 0.95 of (core, fit) items, per model. If this fails, C1–C3 are reported as unreliable for that model.

**Predictions** (per model, Mistral-Small-24B and Qwen2.5-72B):
- **C1, transfer law.** φ_NONE(M) lies within ±0.15 of the remap's value completeness κ_V, which was estimated beforehand from Paper 1's fixed-value data: 0.73 for Qwen-72B and 0.53 for Mistral-24B.
- **C2.** φ_P1(M) − φ_NONE(M) > 0, with a paired bootstrap CI excluding 0. The learned remap transfers less to free-form answers than to the multiple-choice format it was fit in.
- **C3.** Key-only addition is much weaker without a post-state re-mention: ψ_NONE < 0.5 · ψ_P1.
- **C4, natural factorial.** P1 key share ≥ 0.80 at Mistral-Small-24B, Qwen2.5-32B and Qwen2.5-72B. BEFORE identity(K) ≤ 0.5 nats in all three.

**Exploratory, no prediction:** φ, ψ and ρ for BEFORE, POST and LETTER; PCA and f_star source-transfer rates per format.

---

## Outcome of P-2026-10-03-C (GPU stage 2; scored by analysis/stage2_score.py and analysis/stage1_prereg.py)

- **Run:** 2× A100 80GB, torch 2.11.0, transformers 5.9.0, code at 2b1eca7.
- **Output:** `results/gpu_stage2/STAGE2_FRAMES_SCORE.txt` and `STAGE2_FACTORIAL_SCORE.txt`.

**Gate (reproduction of Paper 1, P1 format, argmax token vs saved native outputs): passed.**
- Mistral-24B: M 0.994, P 1.000 (n = 360 each).
- Qwen-72B: M 1.000, P 1.000.

| Prediction | Mistral-Small-24B | Qwen2.5-72B |
|---|---|---|
| C1 φ_NONE(M) within κ_V ± 0.15 | 0.708 [0.681, 0.734] vs 0.53 ± 0.15 → **not met** | 0.861 [0.837, 0.884] vs 0.73 ± 0.15 → **met** (at the edge) |
| C2 φ_P1 − φ_NONE > 0 | −0.003 [−0.019, +0.013] → **not met** | +0.034 [+0.018, +0.049] → **met** (small) |
| C3 ψ_NONE < 0.5·ψ_P1 | 0.069 vs 0.528 → **met** | 0.052 vs 0.633 → **met** |

**C4, natural factorial.**
- P1 key share ≥ 0.80 → **not met** overall: Mistral-24B 0.78 [0.77, 0.79] no; Qwen2.5-32B 0.86 [0.84, 0.87] yes; Qwen2.5-72B 0.76 [0.75, 0.77] no.
- BEFORE identity(K) ≤ 0.5 → **met** in all three: −0.53, −0.42, −1.99.

**Exploratory results** (all formats, distinct cores, fits averaged):

| Format | φ(M) Mistral / Qwen | ψ (key addition) Mistral / Qwen | ρ (key removal) Mistral / Qwen |
|---|---|---|---|
| P1 | 0.71 / 0.90 | 0.53 / 0.63 | 0.76 / 0.79 |
| LETTER | 0.75 / 0.90 | 0.78 / 0.89 | 0.99 / 0.99 |
| POST | 0.72 / 0.87 | 0.20 / 0.21 | 0.45 / 0.34 |
| NONE | 0.71 / 0.86 | 0.07 / 0.05 | 0.18 / 0.11 |
| BEFORE | 0.74 / 0.91 | 0.00 / −0.03 | 0.01 / −0.02 |

**Reading** (stated after seeing the data, so not part of the preregistration):
- The learned remap's behavioural effect is essentially **readout-invariant**: φ is flat across formats, and PCA source transfer is 1.000 in every format.
- The **key-only exchange is strongly readout-dependent**: it is large only when candidates are listed after the state token and vanishes for free-form or options-before readouts.
- So Paper 1's key-exchange result reflects its multiple-choice readout. The remap is written redundantly into the key and value channels, and the channel that appears to "carry" it depends on which channel the readout reads.
- The κ_V-based transfer law (C1/C2) is not supported. The value-channel completeness estimated from fixed-value clamping underestimates how fully the remap is carried by values.

**Qwen2.5 P1 key share across scale:** 0.37 (1.5B), 0.69 (3B), 0.82 (7B), 0.91 (14B), 0.86 (32B), 0.76 (72B). It is non-monotonic beyond 14B.

---

## P-2026-10-03-D: GPU stage 3, generality to new tasks and localisation at 7–14B

**Committed before any stage-3 run.**

**Runs:** `scripts/gpu_stage3.sh`.
- `experiments/task_factorial.py` on two new state tasks (`ckeys/tasks.py`), n = 150, in five open 7–14B models (Qwen2.5-7B, Qwen2.5-14B, Qwen3-8B, Mistral-7B-v0.3, OLMo-2-7B):
  - **paint:** "Later, {agent} repainted the {thing} {colour}."
  - **schedule:** "Later, {agent} moved the {thing} to {day}."
- `experiments/row_restricted_keys.py` on the belief task with Paper 1's encoder, n = 60, arms P1/POST/LETTER, in Qwen2.5-7B, Qwen2.5-14B and Mistral-7B.

**Scoring:** `analysis/stage3_score.py`, committed with this entry.

**Predictions:**
- **D1, generality.** Per task, a model counts if identity(K) in P1 is > 0 with a CI excluding 0, *and* the paired P1 − NONE difference is > 0 with a CI excluding 0. Met if ≥ 4 of 5 models count, separately for paint and for schedule.
- **D2, structural control.** BEFORE identity(K) ≤ 0.5 nats in ≥ 9 of the 10 task × model cells.
- **D3, localisation at scale.**
  - In P1, choice-word rows recover ≥ 0.70 of the full key effect, while question rows and the remaining tail rows each recover ≤ 0.15, in all 3 models.
  - In POST, where the full key effect is positive, re-mention-word rows recover ≥ 0.50 in ≥ 2 of 3 models.

**Exploratory:** key shares on the new tasks; LETTER and POST identity; layer-window and KV-group localisation at 7–14B.

---

## P-2026-10-04-E: GPU stage 3b, reviewer-driven controls

**Committed before any stage-3b run.**

Two internal reviews of the draft raised four points:
1. A lexical/duplicate-token confound.
2. Format arms that differ in their instruction.
3. A scale curve confounded with the software environment.
4. "Written into both channels" asserted without a value-only test.

An exploratory CPU pilot was run before this entry (Qwen2.5-1.5B, n = 40; `results/role_factorial_cpu`). Swapping who observed the move, with the same location word, shifts the belief answer by 5–8 nats. Of that shift, the critical token's key and value carry only 0.00–0.05.

**Runs:** `scripts/gpu_stage3b.sh`. **Scoring:** `analysis/stage3b_score.py`, committed with this entry.

**Predictions:**
- **E1, instruction-matched 2×2.** Every arm uses the instruction "Answer with one word": AFTER (list after the story), BEFORE (list before), POST (sentence after), PRE (sentence before), plus NONE. Models: Qwen2.5-7B, Qwen2.5-14B, Mistral-7B and OLMo-2-7B.
  - (a) ID_K(AFTER) > 0 with a CI excluding 0, in 4/4 models.
  - (b) ID_K(BEFORE) ≤ 0.5 and ID_K(PRE) ≤ 0.5, each in ≥ 3/4 models.
  - (c) Paired ID_K(AFTER) − ID_K(BEFORE) > 0 in 4/4 models, and paired ID_K(POST) − ID_K(PRE) > 0 in ≥ 3/4 models, each with a CI excluding 0.
- **E2, role control (identity vs belief role).** For the direct question in P1 and NONE, the critical token's key and value each carry at most 0.15 of the role-swap effect (|f_K| ≤ 0.15 and |f_V| ≤ 0.15) in ≥ 3/4 models. The location token's channels carry the location's identity, not who observed the move. Qwen2.5-72B is exploratory.
- **E3, environment.** Each re-run must lie within 0.03 of the original P1 key share:
  - Qwen2.5-14B re-run under transformers 5.9.0, sharded over 2 GPUs: original 0.91.
  - Qwen2.5-32B re-run under transformers 5.18.0 on 1 GPU: original 0.86.

  If both hold, the drop beyond 14B is not an artefact of the software environment.
- **E4, value-only exchange in Paper 1's frames** (Mistral-24B, Qwen-72B). The design mirrors Paper 1's key-only exchange, applied to values instead. Predicted: ψ_V(NONE) ≥ 0.5 and ψ_V(NONE) > ψ_K(NONE) in both models. Where keys carry nothing, values carry the remap.

**Exploratory:** ψ_V and ρ_V in the other formats; the role control in LETTER and at 72B.

---

## Outcome of P-2026-10-03-D (GPU stage 3; scored by analysis/stage3_score.py, committed with the predictions)

- **Run:** 1× A100 80GB, transformers 5.18.0, code at b8815cb (after the D commit 0f5cbff).
- **Output:** `results/gpu_stage3/STAGE3_SCORE.txt`.

| Prediction | Result | Verdict |
|---|---|---|
| D1 paint: P1 ID_K > 0 and P1 − NONE > 0 (CIs excluding 0) | 5/5 models (P1 ID_K +6.5 to +42.7 nats) | **met** |
| D1 schedule: same | 5/5 models (P1 ID_K +4.5 to +28.5 nats) | **met** |
| D2 BEFORE ID_K ≤ 0.5 | 10/10 task × model cells (max +0.32) | **met** |
| D3 P1: choice-word rows ≥ 0.70; question and tail ≤ 0.15 | 3/3 models: choice words 0.92 / 0.94 / 0.98; question 0.00; tail ≤ 0.01 | **met** |
| D3 POST: re-mention-word rows ≥ 0.50 | 3/3 models: 0.67 / 0.72 / 0.73 | **met** |

---

## Outcome of P-2026-10-04-E (GPU stage 3b; scored by analysis/stage3b_score.py, committed with the predictions)

- **Run:** 2× A100 80GB, transformers 5.18.0 and 5.9.0 as designed, code at 3ca63f1 (the E commit).
- **Output:** `results/gpu_stage3b/STAGE3B_SCORE.txt`.

| Prediction | Result | Verdict |
|---|---|---|
| E1a ID_K(AFTER) > 0, 4/4 models | +8.7 to +35.8 nats, 4/4 | **met** |
| E1b ID_K(BEFORE) ≤ 0.5 and ID_K(PRE) ≤ 0.5 | 4/4 and 4/4 (max +0.01) | **met** |
| E1c AFTER − BEFORE > 0 (4/4); POST − PRE > 0 (≥ 3/4) | 4/4; 4/4 | **met** |
| E2 role control: \|f_K\|, \|f_V\| ≤ 0.15 in P1 and NONE | 4/4 and 4/4 (all fractions ≤ 0.02; also ≤ 0.01 at Qwen2.5-72B) | **met** |
| E3 environment: s_K within 0.03 of the original | Qwen2.5-14B under 5.9.0 on 2 GPUs: 0.91 (original 0.91); Qwen2.5-32B under 5.18.0 on 1 GPU: 0.86 (original 0.86) | **met** |
| E4 value-only exchange: ψ_V(NONE) ≥ 0.5 and > ψ_K(NONE) | Mistral-24B 0.80 vs 0.07; Qwen-72B 0.90 vs 0.05 | **met** |

**Exploratory E4 crossover** (ψ_K / ψ_V, Mistral-24B; Qwen-72B):

| Format | Mistral-24B | Qwen-72B |
|---|---|---|
| LETTER | 0.78 / 0.02 | 0.89 / 0.01 |
| P1 | 0.53 / 0.23 | 0.63 / 0.21 |
| POST | 0.20 / 0.52 | 0.21 / 0.68 |
| NONE | 0.07 / 0.80 | 0.05 / 0.90 |
| BEFORE | 0.00 / 0.97 | −0.03 / 1.00 |

The same learned intervention is carried by the critical token's keys or values depending on the readout, while its behavioural effect φ is constant.

---

## P-2026-10-05-F: GPU stage 4, does the fitting format decide which channel carries a learned remap? (paper v2 only)

**Final.** Fixed in the commit titled "Finalise preregistration F", together with the final scoring script `analysis/stage4_score.py` and the stage-4 code, before any stage-4 GPU run; the GPU script refuses to run on a draft entry or a modified tree. Results go into paper v2 only; v1 (`paper/versions/paper2_v1.pdf`, commit 4187b25) has no stage-4 content.

**Question.** In stage 3b, Paper 1's released remap M (fit with the options listed after the question) was carried by the writing token's keys under LETTER (ψ_K 0.78, ψ_V 0.02 at Mistral-Small-24B) and by its values under NONE (ψ_K 0.07, ψ_V 0.80), while φ stayed at 0.70–0.75. Two accounts:
- **H_read:** the readers in the evaluation format decide which channel appears to carry the edit; a remap fit with no later mention also writes into the keys, so it shows the same crossover.
- **H_fit:** the fitting format decides what the remap writes; a remap fit with no later mention writes only what the value channel reads, so it stays value-carried even under LETTER and transfers poorly there.

**Runs:** `scripts/gpu_stage4.sh` (Mistral-Small-24B-Instruct-2501 at revision 9527884be6e5616bdd54de542f9ae13384489724, BF16, transformers 5.9.0). **Scoring:** `analysis/stage4_score.py`, committed with this entry.
- **Fits.** Paper 1's released `gpu/train.py` and `gpu/behavior_engine.py`, unmodified, run through `experiments/refit_remap.py`, which replaces at import time: the prompt function (fit_none only); Paper 1's release-integrity check (only README.md may differ; the code and training data must match the release manifest); and Paper 1's model loader, by one that uses the same load arguments (BF16, sdpa, eval mode, no gradients, no cache, pad = eos, `fix_mistral_regex`) at the pinned revision, with Paper 1's exact Python, package and device pins waived. Every replacement is recorded with hashes in each run's FRAME.json. Recipe: rank 16, output of (1-based) block 4, every event-span token, objective m3 (pair-swap), six-way conditional cross-entropy at the reply start (no prefill, Paper 1's "original" interface), AdamW lr 1e-3, batch 1, one shuffled epoch of the 1000 training pairs, seeds 101–103, each run's own PCA basis as initializer and as control P.
  - **fit_p1** (same-code control): Paper 1's own format (options after the question, "Answer with exactly one choice.").
  - **fit_none**: identical up to the end of the story, then "Question: … / Answer with one word. / Answer:", with no candidates named.
- **Evaluation** as in stage 3b (`experiments/paper1_frames.py`): the 96 native cores with B, S and T distinct; LETTER, P1, POST, NONE, BEFORE; "Answer:" prefill; key-only and value-only exchanges from (1-based) block 6, each refit M paired with its own run's P; the released bases are re-run in the same job as an anchor. Fits are averaged within core; 95% core bootstrap, 10,000 resamples, ratio of means.

**Gates** (the predictions are interpreted only if all pass; a failed gate is reported as a failed fit or a failed control):
- **G1, same activations.** Each refit P spans the released P of the same seed: mean squared principal cosine ≥ 0.99 for every seed in both runs.
- **G2, the fits work.** φ ≥ 0.5 in the primary evaluation ("Answer:" prefill) under each run's fitting prompt: φ_none(NONE) and φ_p1(P1).
- **G3, the control reproduces the released crossover.** fit_p1 has ψ_K(LETTER) ≥ 0.5 and ψ_V(NONE) ≥ 0.5.

Thresholds on φ, ψ_K and ψ_V refer to the ratio-of-means point estimates; only F3 uses an interval. Every statistic is computed per run (fit_none, fit_p1) with the three seeds averaged within core.

**Predictions (H_read):**
- **F1, crossover without later mentions at fitting.** fit_none has ψ_K(LETTER) ≥ 0.5 and ψ_V(NONE) ≥ 0.5.
- **F2, transfer to lettered options.** fit_none has φ(LETTER) ≥ 0.5.
- **F3, no fit-format effect on the crossover.** D = [ψ_K(LETTER) − ψ_K(NONE)] for fit_none minus the same for fit_p1, paired over cores: its 95% CI lies within [−0.25, +0.25].
- **F4, the channel follows the natural read.** For each of fit_none and fit_p1, across the five formats, the key share ψ_K/(ψ_K + ψ_V) (from the point estimates) correlates with the identity key share s_ID = mean ID_K/(mean ID_K + mean ID_V) of the unpatched Mistral-Small-24B (stage-2 natural factorial, `results/gpu_stage2/format_factorial`) at Pearson r ≥ 0.9. F4 is met only if both runs meet it; if ψ_K + ψ_V < 0.2 in any format, that run's F4 is not evaluable and counts as not met.

H_fit would predict instead: F1 fails with ψ_V(LETTER) ≥ 0.5 for fit_none, F2 fails, and D ≤ −0.4.

**Exploratory:** evaluation without the "Answer:" prefill; ρ_K and ρ_V; principal cosines between fit_none, fit_p1 and the released M, against the between-seed baseline; loss curves; the f_star fits that `train.py` makes alongside m3.

---

## Outcome of P-2026-10-05-F (GPU stage 4; scored by analysis/stage4_score.py at the finalising commit 1ef8696, which the run used)

**Run.** 2× A100-SXM4-80GB, Python 3.12.14, torch 2.11.0+cu128, transformers 5.9.0, clean checkout of 1ef8696. The two fits ran in parallel (3,293 s each); the evaluation with and without the prefill took about 20 min each. No step failed. Paper 1's `load_training_bases` passed for both runs, and the tokenizer check found identical ids with and without `fix_mistral_regex` on all 3,600 evaluation prompts. Re-scoring the archive off the GPU box gives an identical `STAGE4_SCORE.txt` (apart from the path of Paper 1's repository).

**Gates: all met.**
- G1: mean squared principal cosine 1.0000 for all six refit P bases.
- G2: φ_none(NONE) 0.813, φ_p1(P1) 0.720.
- G3: fit_p1 ψ_K(LETTER) 0.776, ψ_V(NONE) 0.809.

**Predictions: all met.**
- F1: fit_none ψ_K(LETTER) 0.768 and ψ_V(NONE) 0.798.
- F2: fit_none φ(LETTER) 0.666.
- F3: D = −0.008 [−0.027, +0.009].
- F4: Pearson r = 0.957 (fit_none) and 0.978 (fit_p1).
- The H_fit pattern is not met (fit_none ψ_V(LETTER) 0.024).

**Reading.** H_read is supported: a remap fit with no later mention shows the same crossover as one fit with the options listed, so the format the remap is read in, not the format it was fit in, decides which channel appears to carry it.

**Exploratory.**
- The three remaps (released, fit_p1, fit_none) have nearly the same ψ_K and ψ_V in every format (within 0.10), although fit_none's subspace overlaps the released M (mean squared principal cosine 0.37–0.39) no more than two seeds of one run overlap each other (0.41–0.44). fit_p1 overlaps the released M at 0.71–0.79.
- The fitting format changes behaviour a little: fit_none transfers more without a later mention (φ 0.81 under NONE and POST, against 0.72–0.73 for fit_p1) and less with lettered options (0.67 against 0.77).
- Without the "Answer:" prefill the pattern is the same (D = +0.004 [−0.014, +0.021]).
- The same-code control reproduces the released M's frame quantities (φ, ψ_K, ψ_V, ρ_K, ρ_V) to within 0.03 in every format.

---

## P-2026-10-05-G: GPU stage 5, what a later mention must be, and do, to read the writing token's key (paper v3)

**Final.** Fixed in the commit titled "Finalise preregistration G", together with the final scoring script `analysis/stage5_score.py`, the stage-5 code and `scripts/gpu_stage5.sh`, before any stage-5 GPU run; the GPU script refuses to run on a draft entry, a modified tree, or code that differs from that commit. The draft was committed at 6e253bb; the changes since are clarifications found while building the scorer (listed in that commit) and the IOI seed change below. **First GPU attempt (eb0b5fd, 2026-10-06):** stopped in the unit tests before any model was loaded or any stimulus run (logs in `results/gpu_stage5_attempt1/`): check (v) of `tests/test_knockout.py::test_knockout_exact` compared batch rows with equal inputs at atol 1e-5, which a many-core CPU can exceed because batched GEMMs round rows by batch position (the same failure reproduces on a 4-core Xeon at 64 threads, 1.6e-5 nats; the rows are bitwise equal up to 48 threads). The entry was re-finalised with test-only changes: that check uses the 1e-4 of the other batched logit checks, and `tests/conftest.py` caps the tests at 16 CPU threads. No experiment, stimulus, scorer or threshold changed. Results go into paper v3.

**Context.** An external critique of v2 expects the objection "duplicate-token attention in a new setting". Five forward-pass experiments, (a)-(d) on the same 150 seed-0 story cores as stages 1 and 3b (the row splice and the form probe on the first 60 of them) and (e) on 200 fresh IOI cores (all contrasts paired by core; 10,000-resample core bootstrap, 95 % percentile intervals, ratios of means recomputed within each resample) ask what the duplicate-token account predicts and where it stops: (a) whether the re-mentioned words attend to the writing token where the key effect is zero (Qwen2.5-1.5B/3B under SENTENCE-AFTER); (b) whether cutting the re-mentions' attention to the writing token removes the key identity and whether the copy route then takes over with the answer preserved; (c) whether the key read is set by which candidates are re-mentioned rather than how many; (d) whether non-identical re-mentions (case, number, synonym, translation, multi-token wrapper) read the key; (e) whether a later list opens a key read on IOI, a value-copy task, and with which sign. Disclosed pilots. Before the draft was committed: two CPU pilots of (a) at Qwen2.5-0.5B (E 0.69-0.75, F/E 0.75-0.87, in-run ID_K −0.7 to −0.8 under SENTENCE-AFTER) already show the H_diss pattern of G2, so G2 is a confirmatory replication in sibling models; FP32 exactness tests of the knockout at 0.5B; tokenisation and span checks for (c) and (d) on 300 cores × 4 tokenizers (0 bad); a GPT-2 small competence pilot and a Qwen2.5-0.5B IOI clamp pilot (n = 12, AFTER ID_K −1.82 nats) that motivated the two-sided IOI tests. **After the draft and before this entry was finalised:** a full GPT-2 small IOI run on CPU (FP32, n = 200, seed-0 IOI cores; `pilots/ioi_gpt2_cpu`), scored with the stage-5 scorer, whose verdicts were seen while the code was built: Gate e passed for PLAIN, INLINE and INLINE_BEFORE; G18 (GPT-2 part), G21c, G22a and G22b met; ID_K(INLINE) −2.34 [−2.46, −2.23]. The IOI run of this stage therefore uses fresh cores (seed 1, `ckeys.ioi.SEED`) in every model; at GPT-2 small, G18, G21c and G22 are a replication of that disclosed pilot on fresh cores and are not counted as confirmatory evidence. No IOI run of any 7B model has been made, so the 7B-pair predictions (G18 at the pair, G19-G21b) are confirmatory.

**Runs:** `scripts/gpu_stage5.sh` (one A100-80GB, BF16, transformers 5.18.0, torch recorded; per-model loop with the HF cache cleaned between models; `TEST_MODE=1` runs every part at Qwen2.5-0.5B in FP32 on CPU). Per model: `experiments/format_factorial.py` with the standard arms plus the 13 subset arms of (c) and the 17 variant arms of (d) (one seed-0 file serves both parts), and the seed-1 factorial on the subset arms at the three primary models of (c); `experiments/attention_knockout.py`; `experiments/row_restricted_keys.py` on the subset arms (n = 60; also at OLMo-2-7B and, with EXTRA, Qwen3-8B, exploratory); `experiments/remention_attention.py` (eager attention, n = 150); the form probe of (d), `experiments/form_attention.py` (n = 60); `experiments/form_competence.py`; for (e), `experiments/ioi_factorial.py` at GPT-2 small, Qwen2.5-7B-Instruct and Mistral-7B (GPT-2 XL, Qwen2.5-7B base and Qwen2.5-14B exploratory), `experiments/ioi_attention.py` at GPT-2 small and the IOI row splice in AFTER at the 7B pair (n = 60), all on the seed-1 IOI cores. Results go to subdirectories of `results/gpu_stage5/` (attention, knockout, factorial, row_restricted, competence, form_attention, ioi, ioi_attention); each model's Hub revision is pinned in REVISIONS.txt on the first run. `IOI_EXACT=no` is a rerun-only switch that records, instead of asserting, GPT-2's FP32 exactness violations. **Scoring:** `analysis/stage5_score.py` (sections a-e; writes `results/gpu_stage5/STAGE5_SCORE.txt`), committed with this entry. Formats are named as in the paper, with code arms in parentheses: OPTIONS-AFTER (P1), LIST-AFTER (AFTER), LETTERS-AFTER (LETTER), SENTENCE-AFTER (POST), SENTENCE-BEFORE (PRE), LIST-BEFORE (BEFORE), NO-MENTION (NONE).

**Shared gate (Gate 0, reproduction; parts c and d).** Per model, from the stage-5 factorial on the same cores: skipped_items = 0; |ID_K − stage 3b| ≤ max(0.5 nat, 0.10 × |reference|) for SENTENCE-AFTER (5.52 / 12.08 / 7.87 / 2.01 at Qwen2.5-7B / 14B / Mistral-7B / OLMo-2-7B), NO-MENTION (1.04 / 1.05 / 0.55 / 0.38) and LIST-AFTER (20.60 / 35.81 / 15.55 / 8.65); lower CI bounds of ID_K(SENTENCE-AFTER) and ID_K(LIST-AFTER) > 0. An exploratory model without a stage-3b reference (Qwen3-8B) is checked on the remaining conditions only. A model failing Gate 0 is not evaluable in (c) and (d). In (c) "not evaluable" counts as not met in any k/k line; in (d) the verdict rule of (d) decides. Throughout, a line with nothing evaluable is NOT EVALUABLE.

### (a) Do the re-mentions attend to the writing token where the key effect is zero?

Models Qwen2.5-1.5B, 3B, 7B, 14B, eager attention, SENTENCE-AFTER and OPTIONS-AFTER (token-identical to LIST-AFTER through the last option word). Odd-indexed cores select heads, even-indexed cores evaluate. D_i(l,h) = A^B[r_b → p] − mean over the no-prior-mention candidates N_i of A^B[r_j → p]; H* = top-3 heads by mean D on the selection half; E = mean over evaluation cores and H* of D; F = mean gain of A[r_s → p] under the K_S clamp at H*; G = the hop-2 analogue of E with its own top-3 heads H2*, selected on the selection half by mean D2 = A^B[ans → r_b] − mean over N_i of A^B[ans → r_j], and Q likewise; in-run ID_K on lowercase and capitalised candidate ids, with the emitted casing per cell.

**Gate a (per model):** lowercase ID_K(OPTIONS-AFTER) > 1.0 nats with CI excluding 0; at 7B/14B ID_K(SENTENCE-AFTER) > 3.0 with CI excluding 0; at 1.5B/3B its CI within [−1.0, +1.0]; in cells whose emitted casing is capitalised, the capitalised ID_K is > 0 with CI excluding 0 in the positive cells (OPTIONS-AFTER; SENTENCE-AFTER at 7B/14B) and has its CI within [−1.0, +1.0] in the null cells (SENTENCE-AFTER at 1.5B/3B); E(OPTIONS-AFTER) ≥ 0.30 with lower bound > 0.15 and F/E ≥ 0.5. A model failing is gated out (its attention statistics exploratory). The G2 call is made over the gated-in small models (2/2 full; 1/1 provisional; this applies to G2 only).

**Predictions:**
- **G1, anchor.** At Qwen2.5-7B and 14B under SENTENCE-AFTER: E ≥ 0.20 with lower bound > 0.10, and F/E ≥ 0.5, at both anchors; an anchor gated out or missing counts as not met; with no anchor gated in, G1 is not evaluable.
- **G2, dissociation.** At the gated-in small models under SENTENCE-AFTER: **H_diss** is met if E ≥ 0.20 (lower bound > 0.10) and F/E ≥ 0.5 in every one; **H_track** is met if the upper bound of E < 0.10 in every one; anything else is "mixed" and no account is declared. G2 is met only under H_diss; H_track is the stated alternative: if it holds it is declared, G2 is scored not met, and G3 is judged under it. An H_diss call is qualified "embedding-level heads only" if all H*(SENTENCE-AFTER) lie in layers < 2 or below the lowest layer of H*(OPTIONS-AFTER) in that model; the Jaccard overlap of the two H* is reported beside the call. The non-specific d_K is reported beside ID_K in every cell.
- **G3, magnitude.** Under H_diss: R_A = E(SENTENCE-AFTER)/E(OPTIONS-AFTER) ≥ 0.5 and E(m) ≥ 0.5 × min over gated-in anchors of E(anchor, SENTENCE-AFTER) (story-paired: on the evaluation cores shared by the small model and every gated-in anchor), at each gated-in small model. Under H_track: R_A ≤ 0.10 at each.
- **G4, hop 2.** (a) At both anchors under SENTENCE-AFTER, G ≥ 0.10 with lower bound > 0.05 (an anchor gated out or missing counts as not met; with no anchor gated in, not evaluable); if instead G's upper bound < 0.10 at both, the hop-2 measure is unsuitable and (b) is not evaluable. (b) Under H_diss and (a): Q(m) = G(m)/min(G(7B), G(14B)) ≤ 0.5 with upper bound < 1.0 at each gated-in small model; (b) is not applicable, and left out of G4, when G2 declares no H_diss (H_track, mixed, or no gated-in small model) or when (a) is not met.

Alternative: H_track (the heads are context-selective at small scale and the key route tracks the attention pattern); if G4b fails with H_diss met, the difference is not an attention-pattern property at either hop and item 3 is pointed at OV content.

### (b) Blocking the lookup route

Models Qwen2.5-7B, 14B, Mistral-7B; arms LIST-AFTER, OPTIONS-AFTER, SENTENCE-AFTER, NO-MENTION. A knockout sets chosen (row, column) attention weights to exactly 0 in every layer and head, renormalising the row. Masks: M0 none; M1 R_cand × {p} (the six candidate-word rows forbidden from the writing token); M2 R_cand × C_init (control column: the pre-p occurrences of the initial-location word); M3 {a} × {p} (answer position only); M4 (R_cand ∪ {a}) × {p}; M8 all rows after p × {p}. Every effect is measured against the self-clamp row under the same mask (8 clamp rows × n_masks in one batch, duplicate self-clamp row as the noise floor). r_K(M) = mean ID_K^M / mean ID_K^M0; q_V^M = mean ID_V^M / mean d_KV^M (span-normalised value identity; every threshold on q_V uses the in-run q_V^M0(NO-MENTION); the stage-3b values 0.391 / 0.335 / 0.433 are reported only); on_B = full-vocabulary top-1 on target (lowercase, single-token capitalised, or first piece of the capitalised form); loc_mass = probability on the location tokens.

**Gate b (sanity, every arm and model):** under M8, |mean ID_K| ≤ 0.1 and |mean ID_V| ≤ 0.1 nats, acc_B ≤ 0.5 and on_B ≤ 0.5. Failure means the mask does not reach every path from p; the part is uninterpretable until fixed and re-run.

**Predictions:**
- **G5, necessity of the candidate-word edges.** r_K(M1) ≤ 0.20 with CI upper bound ≤ 0.25 under LIST-AFTER and under OPTIONS-AFTER, in 3/3 models; under SENTENCE-AFTER r_K(M1) ≤ 0.40 in ≥ 2/3; G5 needs both halves. (The one result the duplicate-token account also predicts; reported as the removal half of D3.)
- **G6, matched control column.** Under LIST-AFTER: r_K(M2) ∈ [0.80, 1.20]; |mean ID_V^M2 − mean ID_V^M0| ≤ 0.15 × mean ID_V^M0(NO-MENTION); paired ID_K^M1 − ID_K^M2 < 0 with CI excluding 0; each in 3/3.
- **G7, the copy takes over and the answer stays (H_redundant).** Under LIST-AFTER and OPTIONS-AFTER with M1: (a) q_V^M1 ≥ 0.5 × q_V^M0(NO-MENTION) and paired ID_V^M1 − ID_V^M0 > 0 with CI excluding 0; a model meets G7a if it passes in both formats, and G7a needs ≥ 2/3 such models; (b) acc_B, acc_S, on_B, on_S ≥ 0.90 and mean loc_mass^M1 / mean loc_mass^M0 ≥ 0.50 with lower bound ≥ 0.40, in 3/3 in both formats, and span^M1 ≥ 0.5 × span^M0 in ≥ 2/3. **H_replaced** (the pattern that would change claim 1) is met if the paired ID_V difference has upper bound ≤ 1.0 nat and q_V^M1 ≤ 0.5 × q_V^M0(NO-MENTION) (in-run) in ≥ 2/3, in both list formats, with on_B^M1 ≤ 0.60 or a loc_mass ratio ≤ 0.50 in ≥ 2/3. Neither = partial takeover, reported as such.
- **G8, routes at the answer position.** (a) Under NO-MENTION with M3: mean ID_V^M3 ≤ 0.60 × mean ID_V^M0 and the paired difference < 0 with CI excluding 0, in ≥ 2/3. (b) Under LIST-AFTER with M3: r_K(M3) ≥ 0.80, acc_B and on_B ≥ 0.90, in 3/3 (the key route is two-hop). (c) Under LIST-AFTER and OPTIONS-AFTER, M4 against M1: mean ID_V^M4 ≤ 0.60 × mean ID_V^M1 with the paired difference < 0 and CI excluding 0, per format, in ≥ 2/3 models, a model counting only if it meets G7a (in both formats).

### (c) Membership, dose and reader rows

Models Qwen2.5-7B, 14B, Mistral-7B (Qwen3-8B, OLMo-2-7B exploratory). Arms with "Answer with one word": sentence family S2 {S, X}, S3 {B, S, X}, S3out {B, o1, o2}, S3half, S4, S4out, S6 (= SENTENCE-AFTER byte for byte); list family L2, L3, L3out, L4, L4out, L6 (= LIST-AFTER); NO-MENTION. Named words in canonical order. r_k = mean ID_K(arm_k)/mean ID_K(arm_6); R_V = [ID_V(L3out) − ID_V(L3)] / [ID_V(NO-MENTION) − ID_V(L3)]. Row-restricted splice (n = 60, first 60 cores) on S2, S3, L2, L3 (S3out, L3out, S6, L6 exploratory): f_words = mean d(mention_words)/mean d(all).

**Predictions (seed 0; each in 3/3 primary models after Gate 0):**
- **G9, membership.** Per family F ∈ {S, L}, scored separately: (a) paired ID_K(F3) − ID_K(NO-MENTION) > 0 with CI excluding 0, and ID_K(F3)/ID_K(F6) ≥ 0.5; (b) paired ID_K(F3) − ID_K(F3out) > 0 with CI excluding 0, and the upper bound of paired ID_K(F3out) − ID_K(NO-MENTION) ≤ 1.0 nat.
- **G10, proportionality refuted.** Per family: the lower 95 % bound of r_2 exceeds 1/3 and that of r_3 exceeds 1/2 (what r_k = k/6 would give). The ladder r_2, r_3, r_4 is reported against 0.5 descriptively.
- **G11, the copy carries the identity again when the readers leave the list.** Paired ID_V(L3out) − ID_V(L3) > 0 with CI excluding 0 and R_V ≥ 0.5; evaluable only if mean ID_V(NO-MENTION) − mean ID_V(L3) ≥ 2.0 nats.
- **G12, reader rows at k = 2 and 3.** f_words ≥ 0.5 with lower bound > 0.25 in S2, S3, L2 and L3 (12/12 arm-model cells); validity per cell |mean d(none)| ≤ 0.01 and mean d(all) ≥ 2.0 nats.

Alternative (graded list-likeness): OUT ≈ IN ≈ ½ × k6, r_k ≈ k/6, readers not in the named rows. The seed-1 replication (same rules, three models, 14 arms) is reported as "replicated in k/3" and does not alter the verdicts; there is no seed-1 stage-3b reference, so a model is evaluable at seed 1 if skipped_items = 0 and the lower CI bounds of ID_K(S6) and ID_K(L6) are > 0.

### (d) Non-identical re-mentions: token-level or concept-level?

Models Qwen2.5-7B, 14B, Mistral-7B, OLMo-2-7B. Sentence arms replacing the SENTENCE-AFTER sentence: POST_THE, POST_MODIF, POST_TITLE, POST_UPPER, POST_PLURAL, POST_SYN, POST_FRMIX, POST_DEMIX (frame-matched translations), POST_FR, POST_DE, and the structure-matched floors POST_OTHER, POST_FR_OTHER, POST_DE_OTHER; list family AFTER_SYN/FR/DE/OTHER secondary. r_K(v) = [ID_K(v) − ID_K(FLOOR_v)] / [ID_K(ANCHOR_v) − ID_K(FLOOR_v)] and r_K^any(v) = [ID_K^any(v) − ID_K(FLOOR_v)] / [ID_K(ANCHOR_v) − ID_K(FLOOR_v)], with floor and anchor scored on the English tokens; rho_s(v) = s_ID(v)/s_ID(ANCHOR_v); ANCHOR_v is SENTENCE-AFTER for the sentence arms and LIST-AFTER for the secondary list family; the "any" versions score the answer in the variant's own form as well; a_v = A_v/A_POST from the attention probe (double difference of span-summed attention to p under K_S vs K_X, n = 60). Verdict rule: a prediction is met if ≥ 3 evaluable models meet it, not met if ≥ 3 are evaluable and fewer than 3 meet it, otherwise not evaluable.

**Gates:** Gate 0; Gate d1 (per model × arm): clean B and S accuracy ≥ 0.95, and a variant cell is evaluable only if v, FLOOR_v and ANCHOR_v each pass d1; Gate d2 (competence, per model × form family): ≥ 5/6 words map correctly in the six-way prompt, with the per-word gated subset as the primary population of SYN/FRMIX/DEMIX/FR/DE cells for every statistic of the cell, a_v included; Gate d3: A_POST > 0 with CI excluding 0 (else G16's attention part is not evaluable in that model).

**Predictions:**
- **G13, wrappers keep the read.** r_K(POST_THE) ≥ 0.75 with lower bound > 0.5; r_K(POST_MODIF) ≥ 0.75 with lower bound > 0.5 (scored separately). If THE fails, G14-G17 are confounded with the frame and reported as such.
- **G14, an exact repeat reads most.** For each of TITLE, UPPER, PLURAL, SYN, FRMIX, DEMIX: r_K(v) ≤ 0.75 with upper bound < 1.0 (six sub-verdicts).
- **G15, token-level on the paper's measure.** For each of SYN, FRMIX, DEMIX: r_K(v) ≤ 1/3 with upper bound < 0.5, and rho_s(v) ≤ 0.5 with upper bound < 0.75.
- **G16, disambiguation (symmetric cut-points).** For each of SYN, FRMIX, DEMIX, on the collision-excluded population: **token pattern** = r_K^any(v) ≤ 1/3 (upper < 0.5) and a_v ≤ 1/3 (upper < 0.5); **concept pattern** = r_K^any(v) ≥ 2/3 (lower > 0.5), rho_s^any(v) ≥ 0.5 (lower > 0.25) and a_v ≥ 2/3 (lower > 0.5); otherwise graded. The author predicts the token pattern. Decision table fixed now ("token on both" = the token pattern on r_K^any and on a_v within the same model, in ≥ 3 models): G15 met + token on both = token-level lookup; concept on a_v with (concept on r^any or G15 not met) = concept-level lookup; concept on a_v with G15 met and token on r^any = lookup without readout.
- **G17, value compensation where the read is lost.** For FRMIX and DEMIX, only in cells where G15 is met and in models with a stage-3b gap ID_V(NO-MENTION) − ID_V(SENTENCE-AFTER) > 2 nats (Qwen2.5-7B, Mistral-7B, OLMo-2-7B): paired ID_V(v) − ID_V(SENTENCE-AFTER) > 0 with CI excluding 0; met per v if ≥ 2 evaluable cells meet it; not evaluable with fewer than 2 evaluable cells.

Alternative: concept-level lookup (Feucht et al.'s concept route), which refutes the strict duplicate-token reading.

### (e) IOI: does a later list open a key read of the IO name, and with which sign?

n = 200 cores (ckeys/ioi.py, seed 1 = `ckeys.ioi.SEED`, fresh cores, see the pilot disclosure above; facts in `results/gpu_stage5/ioi/ENTRY_FACTS.txt`; 15 Wang et al. templates, ABBA/BABA alternating, four names IO_B, IO_S, IO_X, S). Arms: PLAIN; AFTER (chat-wrapped "Choices:" after the sentence); BEFORE; QUESTION (no candidates); INLINE and INLINE_BEFORE (GPT-2 small, a spaced parenthetical). Clamps as in the paper at the IO mention p (GPT-2: the k/v slices of c_attn); FP32 exactness asserted for GPT-2, BF16 floors recorded at 7B/14B. f_K = mean ID_K/mean ID_KV, f_V likewise; signed contrasts fix the sign of the manipulated arm's point estimate before resampling.

**Gate e (per model × arm, clean B run):** two-way accuracy ≥ 0.75, four-way ≥ 0.50, mean LD > 0 with CI excluding 0; a failing cell is not evaluable.

**Predictions:**
- **G18, no key read in plain IOI.** f_K(PLAIN) ∈ [−0.10, +0.10] with CI within [−0.20, +0.20] in 3/3 of GPT-2 small, Qwen2.5-7B-Instruct, Mistral-7B (precondition ID_KV ≥ 3 nats; a model failing the precondition or Gate e is not evaluable and counts as not met). The GPT-2 part replicates the disclosed pilot.
- **G19, a later list opens a key read.** At the two 7B instruct models: (a) the CI of mean ID_K(AFTER) excludes 0, |mean| ≥ 2.0 nats, and the signed contrast sgn × [ID_K(AFTER) − ID_K(QUESTION)] > 0 with CI excluding 0, in 2/2; (b) the sign is positive (selection) in 2/2, evaluated where (a) holds.
- **G20, the lookup replaces the copy.** Given G19 in a model: f_V(AFTER) ≤ 0.50, f_V(QUESTION) ≥ 0.75, paired f_V(QUESTION) − f_V(AFTER) > 0 with CI excluding 0, and s_ID(AFTER) ≥ 0.50, in 2/2.
- **G21, position and wrapper controls.** (a) |mean ID_K(BEFORE)| ≤ 0.5 nats and sgn × [ID_K(AFTER) − ID_K(BEFORE)] > 0 with CI excluding 0, 2/2, the contrast evaluated in a model only where G19a holds (else not evaluable); (b) |f_K(QUESTION)| ≤ 0.15 and |mean ID_K(QUESTION)| ≤ 0.20 × |mean ID_K(AFTER)|, 2/2; (c) GPT-2 small |mean ID_K(INLINE_BEFORE)| ≤ 0.5 (replication of the disclosed pilot).
- **G22, an in-sentence re-mention in GPT-2 small (replication of the disclosed pilot).** (a) The CI of mean ID_K(INLINE) excludes 0, |mean| ≥ 0.5 nats, the signed contrast against PLAIN has point estimate ≥ 0.5 nats with CI excluding 0 and against INLINE_BEFORE > 0 with CI excluding 0; (b) mean ID_K(INLINE) < 0 with CI excluding 0 (the canonical circuit inhibits a repeated name).

Alternative: an inhibitory read at 7B (G19b fails with G19a met; reported as "inhibitory key read", G20 then descriptive) or an additive mixture (f_V(AFTER) ≥ 0.5).

**Exploratory (all parts):** casing tables and the d_K-F_b co-variation (a); layer profiles, KV groups, Jaccard and cross-format head evaluation (a); M5/M6/M7/Mq/Mpost/M2b masks, the emission-position variant of M3 under NO-MENTION, top-5 distributions, SENTENCE-AFTER takeover, Qwen2.5-1.5B/3B and OLMo-2-7B if budget remains (b); S3half, B's membership at k = 4, OUT residual stratified by story presence, k-trend shape, exploratory models, splice consistency (c); the r_K ordering across all variants, frame effect FR vs FRMIX, per-pair SYN/DE breakdown, form-scored ID_K, deeper onsets, the list family, per-layer and per-head probe profiles (d); sign across every model including Qwen2.5-7B base, 14B and GPT-2 XL, ABBA/BABA split, the row splice in AFTER (n = 60), the GPT-2 name-mover/duplicate-head attention probe, onset rows at 0.3 L, four-way renormalised measures (e); the agreement of every re-run cell with stages 1/3b.

---

## Outcome of P-2026-10-05-G (GPU stage 5; scored by analysis/stage5_score.py at the finalising commit 51e105e, which the run used)

**Run.**
- **Hardware and software:** one A100-SXM4-80GB, Python 3.12.14, torch 2.11.0+cu128, transformers 5.18.0, numpy 2.5.3. The run used a clean checkout of 51e105e (`COMMIT.txt`) with `PARTS=a,b,c,d,e`, `EXTRA=0` and `TEST_MODE=0`, started 2026-10-06T21:25Z.
- **Precision and attention:** BF16 throughout, except GPT-2 small and GPT-2 XL, which ran in FP32 with exactness asserted (0 violations). Part (a) and the form probe used eager attention; every other part used sdpa.
- **Provenance and population:**
  - One Hub revision per model, nine models (`REVISIONS.txt`), and every results file carries commit 51e105e.
  - Every file has the preregistered n (150, 60 or 200 per arm) with the same cores in every arm and across the parts that share a model, and `skipped_items` is 0 wherever it is recorded.
  - The scorer prints "provenance OK; population OK".
- **What failed:** one step failed: the exploratory GPT-2 small name-mover/duplicate-head attention probe (`experiments/ioi_attention.py`). It hit a device mismatch: an index tensor stayed on the CPU while the weights were on cuda:0 (`log_ioi_attention_gpt2.txt`, `FAILED.txt`). No gate or prediction uses this probe. It was re-run afterwards on CPU (see Deviations).
- **Scope of the run:** `EXTRA=0` means the exploratory Qwen3-8B cells (part (c) factorial and splice) were not run. No prediction uses them.
- **Re-score:** re-scoring the archive off the box with the committed scorer at 51e105e reproduces `STAGE5_SCORE.txt` byte for byte when run from a checkout root with the default relative `--root`. With an absolute `--root`, only the printed root path differs, in two header lines.
- **Independent checks:** five independent recomputations from the raw files, none of which imports the scorer, reproduce every gate value, point estimate and verdict (details under Deviations).
- **Summary line:** 12 MET, 10 NOT MET, 0 NOT EVALUABLE of 22.

Numbers below are from `STAGE5_SCORE.txt` unless marked *(recomputed)*, which means recomputed from the raw files and not printed in the score file; "our arithmetic" marks a difference or ratio of printed score-file values. Model order in triples is Qwen2.5-7B / Qwen2.5-14B / Mistral-7B, and in quadruples Qwen2.5-7B / 14B / Mistral-7B / OLMo-2-7B (all Instruct), unless stated.

### Gates

- **Gate 0 (reproduction of stage 3b; parts c, d): passed 4/4.**
  - ID_K for SENTENCE-AFTER / NO-MENTION / LIST-AFTER: 5.52 / 1.04 / 20.60 (Qwen2.5-7B), 12.08 / 1.05 / 35.81 (14B), 7.87 / 0.55 / 15.55 (Mistral-7B), 2.01 / 0.38 / 8.65 (OLMo-2-7B). Every value equals the reference at the printed precision.
  - Per core, these cells are bitwise identical to the stage-3b raw files *(recomputed: largest per-core difference 0.0 over 150 cores × 3 arms × 4 models)*.
- **Gate a (part a): all four models gated in.**
  - Lowercase ID_K(OPTIONS-AFTER) is +19.39 / +38.95 / +2.63 / +10.37 (Qwen2.5-7B / 14B / 1.5B / 3B), every CI excluding 0.
  - ID_K(SENTENCE-AFTER) is +4.50 [+4.18, +4.82] at 7B and +12.05 [+11.44, +12.67] at 14B. At 1.5B it is −0.53 [−0.64, −0.41] and at 3B −0.02 [−0.40, +0.37], both inside [−1, +1].
  - In the cells where the emitted casing is capitalised, the capitalised ID_K is +4.73 (7B SENTENCE-AFTER), +13.48 (14B) and +0.04 [−0.35, +0.44] (3B).
  - E(OPTIONS-AFTER) is 0.887 / 0.866 / 0.750 / 0.832, and F/E(OPTIONS-AFTER) is 1.00 / 0.99 / 1.04 / 1.03.
- **Gate b (part b, M8 in every arm): met 3/3.**
  - Under M8, mean ID_K = mean ID_V = 0.000 in all 12 arm × model cells, and the within-batch duplicate-row floor is 0.000.
  - acc_B and on_B under M8 are 0.00 at Qwen2.5-7B, 0.00–0.15 at 14B and 0.04–0.19 at Mistral-7B.
- **Gates d1–d3 (part d).**
  - d1 failed only in the secondary list arms AFTER_FR and AFTER_DE, at Qwen2.5-7B and Mistral-7B; for example, Qwen2.5-7B AFTER_DE has clean B accuracy 0.727. Those secondary cells are not evaluable; every primary sentence cell passed.
  - d2 passed in every model and form family: DE 5/6 in all four models, SYN 5/6 at Mistral-7B, FR 5/6 at OLMo-2-7B, all others 6/6.
  - d3 passed 4/4.
- **Gate e (part e, per model × arm).**
  - Qwen2.5-7B-Instruct passed in all six arms.
  - Mistral-7B failed AFTER (two-way 0.59, four-way 0.42), BEFORE (four-way 0.27) and INLINE_BEFORE (four-way 0.32). Mistral-7B is therefore not evaluable in G19–G21, which counts as not met.
  - GPT-2 small passed PLAIN, INLINE and INLINE_BEFORE and failed AFTER, BEFORE and QUESTION, the same pattern as in the disclosed pilot.
  - Exploratory models: Qwen2.5-7B base failed QUESTION, Qwen2.5-14B-Instruct failed INLINE_BEFORE, and GPT-2 XL failed AFTER, BEFORE and QUESTION.

### Predictions

**Part (a), re-mention attention (SENTENCE-AFTER unless stated).**

| Prediction | Observed | Verdict |
|---|---|---|
| G1 anchors: E ≥ 0.20 (lower > 0.10), F/E ≥ 0.5 at 7B and 14B | E 0.869 [0.852, 0.885], F/E 1.00 (7B); E 0.915 [0.908, 0.921], F/E 0.98 (14B) | **met** (2/2) |
| G2 dissociation at 1.5B/3B: H_diss | E 0.789 [0.778, 0.802], F/E 1.03 (1.5B); E 0.879 [0.866, 0.892], F/E 0.99 (3B). Not embedding-level: H* layers 8, 12, 6 and 17, 22, 19. Jaccard(H*SENTENCE-AFTER, H*OPTIONS-AFTER) 1.00 / 0.50. d_K beside ID_K: +0.31 [+0.09, +0.54] (1.5B) and +2.89 [+2.16, +3.64] (3B) | **met** (H_diss, 2/2) |
| G3 magnitude under H_diss: R_A ≥ 0.5 and E(m) ≥ 0.5 × min anchor E | R_A 1.05 [1.04, 1.07] / 1.06 [1.03, 1.08]; E(m)/min anchor E 0.91 [0.88, 0.94] / 1.01 [1.00, 1.03] (1.5B / 3B) | **met** (2/2) |
| G4 hop 2: (a) G ≥ 0.10 (lower > 0.05) at the anchors; (b) Q ≤ 0.5 (upper < 1.0) at 1.5B/3B | (a) G 0.255 [0.222, 0.288] (7B), 0.329 [0.304, 0.353] (14B); (b) Q 0.34 [0.29, 0.39] (G 0.086), 0.44 [0.37, 0.52] (G 0.113) | **met** ((a) 2/2, (b) 2/2) |

G2 replicates, in sibling models, the H_diss pattern of the disclosed Qwen2.5-0.5B CPU pilots. G4 is an attention measure; it does not intervene on hop 2.

**Part (b), attention knockout (Qwen2.5-7B / 14B / Mistral-7B).**

| Prediction | Observed | Verdict |
|---|---|---|
| G5 r_K(M1) ≤ 0.20 (upper ≤ 0.25) in LIST-AFTER and OPTIONS-AFTER, 3/3; ≤ 0.40 in SENTENCE-AFTER, ≥ 2/3 | LIST-AFTER +0.001 / −0.026 / −0.007; OPTIONS-AFTER +0.003 / −0.028 / −0.009 (every upper bound ≤ 0.007); SENTENCE-AFTER −0.105 / −0.036 / +0.001 | **met** (list 3/3, sentence 3/3) |
| G6 matched control column (LIST-AFTER, M2) | r_K(M2) 1.094 / 0.997 / 0.990; \|ΔID_V\| 0.785 / 0.795 / 0.255 against limits 2.494 / 2.867 / 2.046; paired ID_K M1 − M2 −22.54 / −36.56 / −15.50, all CIs below 0 | **met** (3/3) |
| G7 H_redundant: (a) copy takes over, ≥ 2/3 models in both formats; (b) answer preserved, 3/3 | (a) 3/3: q_V^M1 0.390 / 0.440 / 0.448 (LIST-AFTER) and 0.377 / 0.427 / 0.422 (OPTIONS-AFTER) against q_V^M0(NO-MENTION) 0.391 / 0.335 / 0.433; paired ID_V M1 − M0 +5.95 / +23.09 / +9.22 and +5.02 / +24.23 / +7.19, all CIs above 0. (b) 1/3: acc_B / acc_S / on_B / on_S under M1 are 0.37 / 0.40 / 0.37 / 0.40 and 0.30 / 0.31 / 0.31 / 0.31 (Qwen2.5-7B), 1.00 / 0.99 / 1.00 / 0.99 and 0.99 / 0.99 / 0.99 / 0.99 (14B), 0.86 / 0.83 / 0.86 / 0.83 and 0.73 / 0.74 / 0.73 / 0.74 (Mistral-7B); loc_mass ratio ≥ 0.956 everywhere; span ratio ≥ 0.5 in 3/3 | **not met**. H_replaced **not met** (ID_V/q_V pattern 0/3, behaviour pattern 1/3). Preregistered outcome: **partial takeover** |
| G8 routes at the answer position: (a) NO-MENTION M3; (b) LIST-AFTER M3; (c) M4 against M1 | (a) ID_V^M3/ID_V^M0 0.134 / 0.370 / 0.417, paired −14.40 / −12.05 / −7.95; (b) r_K(M3) 1.043 / 1.027 / 1.010 with acc_B = on_B = 1.00; (c) ID_V^M4/ID_V^M1 0.134 / 0.465 / 0.570 (LIST-AFTER) and 0.111 / 0.411 / 0.517 (OPTIONS-AFTER), all paired CIs below 0 | **met** ((a) 3/3, (b) 3/3, (c) 3/3 in each format) |

**Part (c), membership, dose and reader rows (Qwen2.5-7B / 14B / Mistral-7B; seed 0, seed-1 replication beside).**

| Prediction | Observed | Verdict |
|---|---|---|
| G9 membership, per family | Sentence family: S3 − NO-MENTION +2.44 / +8.47 / +4.70; S3/S6 0.63 / 0.79 / 0.67; S3 − S3out +2.76 / +8.77 / +4.91; S3out − NO-MENTION −0.32 [−0.36, −0.27] / −0.30 [−0.38, −0.22] / −0.21 [−0.26, −0.17]. List family: L3 − NO-MENTION +20.66 / +34.78 / +15.11; L3/L6 1.05 / 1.00 / 1.01; L3 − L3out +21.01 / +34.91 / +15.35; L3out − NO-MENTION −0.35 / −0.13 / −0.25 | **met** (sentence 3/3, list 3/3; seed 1 replicated 3/3 in each family) |
| G10 proportionality refuted (lower bound of r_2 > 1/3 and of r_3 > 1/2), per family | Sentence family: r_2 0.28 [0.23, 0.32] / 0.58 [0.55, 0.61] / 0.46 [0.43, 0.50]; r_3 0.63 / 0.79 / 0.67; ladder r_4 0.83 / 0.90 / 0.78. List family: r_2 0.96 / 0.98 / 0.98, r_3 1.05 / 1.00 / 1.01 | **not met**: sentence 2/3 (fails at Qwen2.5-7B, also at seed 1: r_2 0.32 [0.28, 0.37]); list **met** 3/3 (seed 1 3/3) |
| G11 the copy returns when the readers leave the list: ID_V(L3out) − ID_V(L3) > 0 and R_V ≥ 0.5 | +1.43 [+1.14, +1.72] / +0.52 [+0.28, +0.76] / −0.83 [−1.24, −0.46]; R_V 0.11 / 0.03 / −0.08; evaluability gaps 12.53 / 16.16 / 10.35 nats | **not met** (0/3; seed 1 0/3, R_V 0.10 / 0.03 / −0.08) |
| G12 reader rows: f_words ≥ 0.5 (lower > 0.25) in S2, S3, L2, L3 | S2 0.40 [0.32, 0.46] / 0.38 [0.33, 0.42] / 0.39 [0.33, 0.45]; S3 0.68 / 0.72 / 0.72; L2 0.91 / 0.94 / 0.92; L3 0.94 / 0.95 / 0.95; all 12 cells valid (d(none) 0.000, d(all) ≥ 8.14 nats) | **not met** (9/12; S2 fails in all three models) |

**Part (d), non-identical re-mentions (Qwen2.5-7B / 14B / Mistral-7B / OLMo-2-7B; met if ≥ 3 evaluable models meet).**

| Prediction | Observed | Verdict |
|---|---|---|
| G13 wrappers: r_K(THE), r_K(MODIF) ≥ 0.75 (lower > 0.5), scored separately | THE 1.09 / 0.99 / 1.06 / 0.96; MODIF 0.32 / 0.64 / 0.64 / 0.55 | **not met** (THE **met** 4/4; MODIF **not met** 0/4) |
| G14 an exact repeat reads most: r_K(v) ≤ 0.75 (upper < 1.0), six sub-verdicts | TITLE 0.94 / 1.09 / 1.07 / 0.87 and UPPER 1.08 / 1.14 / 1.01 / 0.83 (0/4 each); PLURAL 0.33 / 0.37 / 0.49 / 0.11, SYN −0.03 / 0.00 / 0.06 / −0.13, FRMIX 0.15 / 0.10 / 0.07 / 0.09, DEMIX −0.01 / 0.02 / 0.02 / 0.00 (4/4 each) | **not met** (4/6 sub-verdicts met; TITLE and UPPER not met) |
| G15 token-level on the paper's measure: r_K ≤ 1/3 (upper < 0.5) and rho_s ≤ 0.5 (upper < 0.75) for SYN, FRMIX, DEMIX | r_K as in G14; rho_s: SYN 0.19 / 0.14 / 0.16 / 0.09, FRMIX 0.33 / 0.29 / 0.17 / 0.30, DEMIX 0.19 / 0.20 / 0.11 / 0.22 | **met** (each sub-verdict 4/4) |
| G16 token pattern on r_K^any and on a_v within a model, ≥ 3 models, per variant | SYN 1/4: r^any token in 4/4 (0.09 / 0.09 / 0.26 / −0.13); a_v 0.41 / 0.41 / 0.53 / 0.21, graded except OLMo-2-7B (token). FRMIX 0/4: concept pattern on both measures at Qwen2.5-7B (r^any 0.95, a_v 0.86) and 14B (1.06, 0.90), graded at Mistral-7B (0.56, 0.55) and OLMo-2-7B (0.44, 0.39). DEMIX 2/4 (collision-excluded): concept pattern on r^any and graded on a_v at Qwen2.5-7B (0.69, 0.66) and 14B (1.32, 0.65); token pattern on both at Mistral-7B (0.31, 0.33) and OLMo-2-7B (−0.13, 0.09) | **not met** (SYN, FRMIX, DEMIX each not met). Decision table: none of the three preregistered outcomes holds for any variant (graded) |
| G17 value compensation: paired ID_V(v) − ID_V(SENTENCE-AFTER) > 0 in ≥ 2 of the evaluable cells (Qwen2.5-7B, Mistral-7B, OLMo-2-7B) | FRMIX +2.58 / +1.37 / +0.31 [+0.08, +0.53]; DEMIX +2.02 / +1.13 / +0.39 [+0.12, +0.65]. Qwen2.5-14B is excluded by the entry's gap rule: it is not among the models with a stage-3b gap ID_V(NO-MENTION) − ID_V(SENTENCE-AFTER) > 2 nats (in the stage-5 factorial the gap is 19.12 − 18.46 = +0.66, our arithmetic on the score-file arm table). Its paired differences, printed as not evaluable, are −2.53 and −3.08 | **met** (FRMIX 3/3, DEMIX 3/3) |

THE is met, so G14–G17 are not confounded with the frame.

**Part (e), IOI (seed-1 cores, n = 200).**

| Prediction | Observed | Verdict |
|---|---|---|
| G18 f_K(PLAIN) ∈ [−0.10, +0.10], CI within [−0.20, +0.20], 3/3 | GPT-2 small −0.02 [−0.02, −0.01], Qwen2.5-7B-Instruct +0.02 [+0.01, +0.03], Mistral-7B +0.00 [−0.00, +0.01]; ID_KV 8.61 / 10.33 / 7.94 | **met** (3/3) |
| G19 a later list opens a key read: (a) existence, (b) positive sign, 2/2 | Qwen2.5-7B-Instruct: ID_K(AFTER) +7.05 [+6.26, +7.81], contrast against QUESTION +6.44 [+5.65, +7.21], positive. Mistral-7B: not evaluable (AFTER failed Gate e) | **not met** ((a) 1/2, (b) 1/2) |
| G20 the lookup replaces the copy, 2/2 | Qwen2.5-7B-Instruct: f_V(AFTER) 0.45 [0.41, 0.48] ≤ 0.50 (passes); f_V(QUESTION) 0.69 [0.67, 0.70] < 0.75 (fails); paired f_V(QUESTION) − f_V(AFTER) +0.24 [+0.20, +0.28] (passes); s_ID(AFTER) 0.57 (passes). Mistral-7B not evaluable | **not met** (0/2) |
| G21 controls: (a) BEFORE, (b) QUESTION, (c) GPT-2 small INLINE_BEFORE | (a) Qwen2.5-7B-Instruct ID_K(BEFORE) −2.02 [−2.18, −1.86], \|mean\| > 0.5 (the AFTER − BEFORE contrast +9.07 [+8.27, +9.85] passes); Mistral-7B not evaluable: 0/2. (b) Qwen2.5-7B-Instruct \|f_K(QUESTION)\| 0.02, \|ID_K(QUESTION)\|/\|ID_K(AFTER)\| 0.09 [0.06, 0.11]: met; Mistral-7B not evaluable: 1/2. (c) ID_K(INLINE_BEFORE) −0.25 [−0.29, −0.21]: met | **not met** ((a) not met, (b) not met, (c) met) |
| G22 GPT-2 small in-sentence re-mention: (a) key read, (b) inhibitory | ID_K(INLINE) −2.35 [−2.47, −2.23]; signed contrast against PLAIN +2.20 [+2.09, +2.32] and against INLINE_BEFORE +2.10 [+1.99, +2.22] | **met** ((a), (b)) |

Following the entry, the GPT-2 small parts are replications of the disclosed CPU pilot on fresh cores and are not counted as confirmatory evidence. These are the GPT-2 line of G18, G21c and G22. The pilot gave ID_K(INLINE) −2.34 [−2.46, −2.23]. The confirmatory IOI evidence is the 7B-pair part of G18 (met) and G19–G21a/b (not met).

**Status of the preregistered alternatives.**
- **(a) H_track:** not declared. G2 is met under H_diss, and G4b is met (Q 0.34 / 0.44), so the entry's fallback ("if G4b fails with H_diss met, … OV content") does not apply.
- **(b) H_replaced:** not met (see G7). Neither H_redundant nor H_replaced holds; the preregistered outcome is partial takeover.
- **(c) Graded list-likeness (OUT ≈ IN ≈ ½ × k6, r_k ≈ k/6, readers not in the named rows):** not supported overall. OUT lies below NO-MENTION in both families (S3out − NO-MENTION −0.32 / −0.30 / −0.21, L3out − NO-MENTION −0.35 / −0.13 / −0.25), not near ½ × k6, and the list ladder is complete at k = 2 (r_2 0.96–0.98). Two features match it in part: in sentences at Qwen2.5-7B, r_2 0.28 [0.23, 0.32] is close to 2/6; and in S2 the rows after the question carry 0.46 / 0.49 / 0.38 of the splice effect d.
- **(d) Concept-level lookup:** not established by the decision table, which needs the concept pattern on a_v in ≥ 3 models. It holds in two models, and for one variant only (FRMIX at Qwen2.5-7B and 14B; see G16).
- **(e) Inhibitory key read at 7B, or additive mixture:** neither holds at Qwen2.5-7B-Instruct. G19b is met there (positive sign), and f_V(AFTER) is 0.45 [0.41, 0.48] < 0.5. Mistral-7B is not evaluable.

### Deviations and disclosures

- **GPT-2 attention probe re-run on CPU (exploratory, no prediction).** The failed probe was re-run with the same arguments: gpt2 at revision 607a30d, n = 200, seed-1 IOI cores, FP32, eager attention, code of 51e105e, skipped_items 0. Output: `results/gpu_stage5_cpu_probe/`.
  - The software differs from the box: torch 2.14.1+cpu and Python 3.11.15, against torch 2.11.0+cu128 and Python 3.12.14. transformers is 5.18.0 in both.
  - Its summary lines agree with its stored attention matrices *(recomputed)*.
- **GPT-2 replication status.** On fresh cores, every GPT-2 small result of the disclosed pilot recurs:
  - Gate e passes in PLAIN, INLINE and INLINE_BEFORE only.
  - G18 (GPT-2 line), G21c, G22a and G22b are met.
  - ID_K(INLINE) is within 0.01 nats of the pilot.
- **Headline lines combine sub-verdicts (scorer rule at 51e105e, documented in the `stage5_score.py` docstring).** A prediction with sub-verdicts is printed MET only when every sub-verdict is met. The entry scores G9 and G10 per family and G13 and G14 as separate sub-verdicts, so this record states them:
  - G10: list family met 3/3, sentence family not met 2/3.
  - G13: THE met, MODIF not met.
  - G14: PLURAL, SYN, FRMIX and DEMIX met; TITLE and UPPER not met.

  The count of 12 MET / 10 NOT MET uses the combined lines. No verdict is changed.
- **G16 label (scorer wording, not the entry's).** For a variant matching none of the three preregistered outcomes, the scorer prints "graded (per-model profile reported verbatim, phrased as near-token-level)".
  - The entry says only "otherwise graded", and "near-token-level" appears nowhere in it.
  - The phrase does not describe FRMIX, where Qwen2.5-7B and 14B show the concept pattern on both measures, or DEMIX at the Qwen models. This record reports the per-model profiles and does not adopt the phrase.
  - The scorer also has a fourth branch ("reported as is", for a token pattern on a_v with a concept pattern on r^any) that the entry does not list. It did not fire.
- **Part (a), points the entry leaves open (no verdict effect).**
  - "Emitted casing" is taken to be the casing with the larger clean-B candidate mass. It matters only at Qwen2.5-1.5B under SENTENCE-AFTER (lowercase 0.65, capitalised 0.33), where both casings pass the gate.
  - F, F_b and the K_X mirror are computed on the evaluation half at H*, while the key effects are bootstrapped over all 150 cores.
  - The Gate a key effects come from the eager single-sequence run. At Qwen2.5-7B they are lower than the sdpa values on the same cores, by about 1.0 nat under SENTENCE-AFTER and 1.8 under OPTIONS-AFTER, with non-overlapping CIs: SENTENCE-AFTER +4.50 against +5.52 (stage-5 part (c) factorial and stage 1), OPTIONS-AFTER +19.39 against +21.16 (stage 1, printed beside it; the stage-5 part (b) knockout M0 gives +21.17). The other three models agree with the stage-1 sdpa values within 0.2 nats. The gate is not near its bound (4.50 > 3.0).
- **Part (b), points the entry leaves open (no verdict effect).**
  - The H_replaced behaviour clause is applied per model in both list formats.
  - The q_V thresholds are applied to point estimates.
  - The G6 \|ΔID_V\| criterion uses cell means, which equal the paired means because every arm has the same cores.
  - Either reading of each point gives the same verdicts.
- **Part (e), the 7B pair rests on one evaluable model.** Mistral-7B's AFTER, BEFORE and INLINE_BEFORE cells failed Gate e. Its not-evaluable cells count as not met in the 2/2 lines, so the NOT MET verdicts of G19, G20 and G21a/b reflect Qwen2.5-7B-Instruct alone (G19a/b and G21b met there; G20 and G21a not met there) together with Mistral-7B's gate failure.
- **Gate 0 checks determinism, not sampling agreement.** On the same stack and GPU type, the reproduction cells are bitwise identical to stage 3b *(recomputed)*.
- **Independent verification.** Five recomputations from the raw files, without the scorer, gave the following *(recomputed)*:
  - Every gate and verdict and every printed point estimate agree.
  - In part (b), the scorer's own bootstrap index set reproduces all 63 cell lines byte for byte.
  - With independent seeds, CI bounds move by at most about 0.05 nats (0.048 in part (b); 0.039 over 600 cell statistics in part (d)) and by up to 0.01 in ratios (for example the 3B Q upper bound, 0.52 against 0.51). No verdict lies within that distance of its bound.
  - The exceptions are ratio CIs with denominators near zero, such as f_K and f_V for Mistral-7B INLINE_BEFORE and GPT-2 XL AFTER. These enter no verdict.
  - The scorer files are unchanged between 51e105e and the results commit.

### Unpredicted observations (reported, not reinterpreted)

**Part (a).**
- **Accuracy at the small models.** Under SENTENCE-AFTER, clean accuracy is low *(recomputed, argmax over the 12 candidate ids)*: B 0.76 / S 0.79 at 1.5B and 0.82 / 0.82 at 3B. Almost every wrong answer is the initial location.
- **3B on competent cores.** Restricted to cores answered correctly in both B and S, 3B SENTENCE-AFTER ID_K (capitalised) is +0.71 [+0.29, +1.14] *(recomputed; not preregistered)*. At 1.5B it stays at −0.51 [−0.65, −0.37].
- **Non-specific key effect at 3B.** Under SENTENCE-AFTER, 3B has a non-specific key effect without identity: d_K +2.89 (lowercase), +3.03 (capitalised). Its co-variation with F_b is weakly positive: Pearson +0.25 [+0.07, +0.41].
- **Hop 2 under the key clamp.** At 1.5B the hop-2 attention does not follow the clamped key: d(ans→r_s) +0.015, d(ans→r_b) +0.028. At 3B it partly follows (+0.127 / −0.105), and at 7B and 14B it follows (+0.343 / −0.243 and +0.321 / −0.213).
- **What the answer row attends to.** Under SENTENCE-AFTER at 7B and 14B, the answer row at the hop-2 heads attends mostly to the writing token itself (0.534, 0.468), not to the re-mentions. Under OPTIONS-AFTER its top-5 columns are all re-mention words.
- **Head-set overlap.** H*(SENTENCE-AFTER) and H*(OPTIONS-AFTER) are identical at 7B and 1.5B (Jaccard 1.00) but overlap little at 14B (0.20). At 14B each format's heads still work on the other format (cross-format E 0.768, 0.734).

**Part (b).**
- **Which answer replaces B under M1.** At Qwen2.5-7B, the self-clamp row's restricted argmax is the initial location in 0.63 (LIST-AFTER) and 0.69 (OPTIONS-AFTER) of cores; at Mistral-7B it is 0.14 and 0.24, and at 14B 0.00 *(recomputed, descriptive)*.
  - Under SENTENCE-AFTER with M1, the answer stays B in 1.00 / 1.00 / 0.99 of cores.
  - The loc_mass criterion of G7b cannot see the initial-location answers, because they are still location words.
- **M3 and M7 under SENTENCE-AFTER (exploratory).** Cutting only the answer position's attention to the writing token raises the key read: r_K(M3) 1.66 / 1.80 / 1.39. M7 (tail rows) gives 1.18 / 1.94 / 1.46.
- **Row specificity (exploratory).** Cutting only the rows of the three clamped candidates (M5) removes the key read as completely as M1: r_K −0.008 / −0.025 / −0.007 under LIST-AFTER. Cutting the other three candidate rows (M6) leaves it intact: 0.961 / 1.003 / 1.003.
- **Copy takeover under SENTENCE-AFTER (exploratory).** q_V(M1) is 0.423 / 0.443 / 0.460, against q_V(NO-MENTION) 0.391 / 0.335 / 0.433.
- **Initial-location column at Qwen2.5-7B.** Cutting this column (M2) slightly strengthens the key read: r_K 1.094, and 1.137 on the 40 cores with two such columns.

**Part (c).**
- **F3out falls below NO-MENTION.** Naming B with two other candidates instead of S and X pushes ID_K below NO-MENTION, although the written word B is re-mentioned: S3out − NO-MENTION −0.32 / −0.30 / −0.21, L3out − NO-MENTION −0.35 / −0.13 / −0.25, mostly with CIs excluding 0.
- **Sentence ladder.** In sentences every added candidate raises ID_K. At Qwen2.5-7B the paired steps are S3 − S2 +1.96 (adds B), S4 − S3 +1.11 (adds one non-matching candidate) and S6 − S4 +0.93 (adds two), each CI excluding 0. In lists the read is complete at k = 2.
- **Value identity when B is not named.** In lists that do not name the written word B, the key read is at full strength while the value identity stays near or above NO-MENTION.
  - ID_V(L2) is 18.63 / 29.48 / 16.24 and ID_V(L4out) 14.90 / 30.76 / 13.68, against NO-MENTION 16.60 / 19.12 / 13.65 and L3 4.06 / 2.97 / 3.29.
  - Paired L4out − L4 is +10.19 / +26.57 / +10.06 *(recomputed)*.
  - The same holds at seed 1, for example 14B L4out 30.88 against NO-MENTION 18.84.
- **Readers in S2.** In S2 the rows after the question carry 0.46 / 0.49 / 0.38 of the splice effect d (the total key-swap effect, d_K, not ID_K), against 0.10–0.13 in S3. In S2 most of d_K is not identity-specific: ID_K/d_K is 0.14 / 0.29 / 0.38, against 0.44 / 0.56 / 0.61 in S3 (our arithmetic on the score-file arm tables, 150 cores).
- **Readout mass in the sentence family.** Under NO-MENTION and every sentence arm, the clean-B probability mass on the six lowercase candidates is 0.00–0.03 at the three primary models, so the sentence-family contrasts are scored on tokens the model is not about to emit (as in stage 3b). In lists the mass is 0.21–1.00.
- **Unnamed B is still chosen.** At Qwen2.5-7B the model chooses B in 1.00 of L2 cores, although B is not listed, and in 0.71 of L4out cores (Mistral-7B 0.81).

**Part (d).**
- **Case variants.** A capitalised or all-caps re-mention reads the key nearly as fully as the exact repeat, although it has different token ids: r_K 0.83–1.14. The CI lies below 1 for OLMo-2-7B TITLE and UPPER (0.87, 0.83) and Qwen2.5-7B TITLE (0.94); the other five of the eight cells are at or above 1.
- **Attention without the read.** A multi-token wrapper (MODIF) attends to the writing token about as much as the exact repeat (span-summed a_v 1.22 / 1.24 / 1.08 / 0.94). The plural attends at 0.87–0.89 of the exact repeat (1.40 at Mistral-7B). Both read the key only partly (r_K 0.32–0.64 and 0.11–0.49). The values above 1 come from summing attention over multi-token spans. With the last token only, MODIF is 0.95 / 0.95 / 0.84 / 0.79 (span 2.33–2.50 tokens) and PLURAL 0.87 / 0.88 / 0.62 / 0.89 (1.83 tokens at Mistral-7B, one token elsewhere) *(recomputed)*.
- **Translations at Qwen.** Scored in the variant's own form, the French-mixed re-mention at the Qwen models reads the key fully: r_K^form 0.99 / 1.28 (r_K^any 0.95 / 1.06). The raw rise of log p(S) under the K_S clamp does not separate the variants (it is as large for SYN, which has almost no read: r_K^any 0.09 / 0.09), so only these identity contrasts are reported.
- **Size of the G17 compensation.** The value compensation of G17 is no larger than that produced by the structure-matched floor sentence: ID_V(POST_OTHER) − ID_V(SENTENCE-AFTER) is +2.71 / +2.51 / +0.54 at Qwen2.5-7B / Mistral-7B / OLMo-2-7B *(recomputed)*, against FRMIX +2.58 / +1.37 / +0.31.
- **Span-summed attention.** The a_v of the multi-token forms depends on summing attention over the span. With the last token only, FRMIX a_v is 0.55 / 0.60 / 0.45 / 0.29 *(recomputed)*.

**Part (e).**
- **BEFORE gives a negative key read.** It is negative in every gated-in BEFORE cell: −2.02 (Qwen2.5-7B-Instruct), −3.83 (14B), −0.95 (Qwen2.5-7B base). The cells that failed Gate e are negative too: Mistral-7B −0.96, GPT-2 small −0.05, GPT-2 XL −0.57.
- **The sign of the AFTER read varies by model.**
  - Positive at Qwen2.5-7B-Instruct (+7.05) and 14B-Instruct (+10.24; the G19 criteria and G21b are met there, exploratory). The 14B model fails G20 on f_V(QUESTION) = 0.68, as the 7B model does.
  - Absent at Qwen2.5-7B base: −0.03 [−0.15, +0.08]. This is a weak null: the total K+V effect there is only +1.72 nats (ID_KV), against +11.96 at the instruct model.
  - Negative in the gate-failed cells: Mistral-7B −1.98 [−2.38, −1.58], GPT-2 small −1.77, GPT-2 XL −1.87.
- **INLINE is inhibitory in every model.** An in-sentence re-mention gives a negative key read in all six models, with Gate e passed in every INLINE cell: −2.33 to −3.29 nats.
- **Row splice (AFTER, n = 60, exploratory).** The listed names carry the whole key effect: options rows 1.05 at Qwen2.5-7B-Instruct and 1.07 at Mistral-7B. The rows of the two clamped names alone carry 0.27 at Qwen2.5-7B-Instruct and 0.84 at Mistral-7B, whose AFTER cell failed Gate e.
- **GPT-2 small probe (CPU re-run, exploratory).**
  - Name movers 9.9 and 9.6 attend from END to the IO mention with 0.767 and 0.674 in PLAIN, and 0.514 and 0.443 in INLINE.
  - In INLINE, duplicate heads 3.0 and 0.1 attend from the listed copy of the IO name to the IO mention (0.656, 0.496).
  - Under the K_S clamp this attention moves to the listed swapped name's row (0.662, 0.496), and the IO row falls to 0.004 and 0.001.

---

## P-2026-10-05-H: GPU stage 6, the reader heads and the second hop; the exchange on Prakash et al.'s intervention and the out-of-sample law (paper v3)

**Final.** Fixed in the commit titled "Finalise preregistration H", together with the scoring script `analysis/stage6_score.py` (with `analysis/stage6_parts/`), the stage-6 code and `scripts/gpu_stage6.sh`, before any stage-6 GPU run and before any stage-5 output was inspected (nothing here depends on them); the GPU script refuses a draft entry, a modified tree or code that differs from that commit, and runs the FP32 unit tests before loading any model. The draft was committed at 6e253bb; the changes since are clarifications and decisions made while the scorer was built and reviewed (listed in that commit). No model output beyond the disclosed pilots below and the plumbing runs of `TEST_MODE` (Qwen2.5-0.5B, FP32, two stories or pairs, every gate failing or not evaluable) was seen.

**Context.** The claims table lists "second hop" and "head identity" as hypothesis, not tested (critique item 3), and the exchange has only been applied to our own predecessor's intervention (item 4) with an in-sample r = 0.98 between the remap's key share and s_ID (item 5). Part (a) names the heads that read the writing token's key at Qwen2.5-7B and Mistral-7B, tests whether they are canonical duplicate-token heads, ablates them, and gives only the answer position the base K/V of the option words. Part (b) runs the key-only / value-only exchange on Prakash et al.'s (2026) reversed-sentence binding swap (BIND) and on an identity edit at the same positions (ID), on their stories, word lists, raw wrapper and seed-10 pool, at Qwen2.5-14B, and tests the law kappa ≈ s_ID depth-matched. Disclosed pilots: (a) Qwen2.5-0.5B FP32 (a3-ranked top heads are duplicate heads, k80 = 12 of 336, ablation of 16 heads ID_K 1.69 → −0.09, answer-only hop r_ans(KV) 0.37 and 0.04 on two stories); (b) 0.5B prototype of the exchange (exact, B+KV_M = M to 0.0; psi_K = +1.54, psi_V = −0.61 under NO-MENTION, which the evaluability rule must mark not evaluable); the release's own 14B per-layer IIA (1.00 at blocks 28-34), which predicts l* = 27-28.

**Runs:** `scripts/gpu_stage6.sh`: `experiments/stage6_heads.py` (Qwen2.5-7B, Mistral-7B, BF16, eager attention, use_cache=False; ranking set R = make_cores(60, Random(0)), evaluation set E = make_cores(60, Random(1)); OPTIONS-AFTER confirmatory, SENTENCE-AFTER reported) and `experiments/prakash_swap.py` (Qwen2.5-14B, BF16, sdpa; `MODEL=llama70` optional on 2 GPUs, with the three confirmatory formats, the NO-MENTION sweeps, the exchange and the preregistered onsets only). Prakash et al.'s release is fetched at commit 0579347 at run time (nothing of it is copied into the repository); the sha256 of every file used and of the seed-10 pool are asserted. **Scoring:** `analysis/stage6_score.py` (writes `results/gpu_stage6/STAGE6_SCORE.txt`; outputs of `TEST_MODE` carry the tag TEST_), committed with this entry. **Statistics:** means over the evaluation stories E (a) or the population pairs (b); 95 % percentile intervals from 10,000 bootstrap resamples (seed 20261005, one fixed index set per n), every ratio of means and paired difference recomputed within each resample; verdicts (MET, NOT MET, NOT EVALUABLE) use the point estimates and the bounds named.

### (a) Which heads read the key, and does the answer read the re-mentions directly?

a3(l,h) = ½[(A^{K_S} − A^B)[rowS → p] + (A^B − A^{K_S})[rowB → p]] (attention change under the key clamp, ranked on R only); HeadSplice lets chosen heads see K_S in the option rows G; R(k) = mean[m(top-k) − m(none)] / mean d_G with d_G the option-row ceiling; KO(k) = 1 − the same with the top-k heads blinded; k* = ceil(0.05 × n_heads) = 40 (Qwen2.5-7B) / 52 (Mistral-7B); mean-ablation replaces a head's output at the six option rows by its position-matched mean over R; duplicate score D, induction score I on 100 random repeated-token sequences, task-side T_dup from the other repeated location words; HopSplice gives only the answer position (row T−1) the base run's K and/or V of rows G while K_S is clamped at p: r_ans(C) = 1 − [m(row C) − m(ID)]/[m(K_S) − m(ID)], r_other for all other later rows, r_all for all rows. Every batched quantity is differenced against an in-batch reference row. The k grid is {1, 2, 3, 5, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128} ∪ {k*}; the random sets are the first k* heads of three fixed random permutations of all heads (numpy default_rng(2)), shared by H2 and H3 (c); the active-at-G set is the k* heads with the largest mean o_proj-input norm at the option rows over R, excluding the top 2k* by a3. T_dup is the attention from each option row G[j] to the earlier occurrences of word j other than p in the clean base run, summed over occurrences and averaged over the repeated words, measured on E (held out from the a3 ranking).

**Gates:** **Gate a1** (FP32, CPU, Qwen2.5-0.5B, `tests/test_head_splice.py`, seven tests, 1e-4 in the logits): HeadSplice all-heads == full key clamp, empty == clean, all heads in rows G == RowSplice(G), group slices == RowSplice(group), batched grid == single runs, self-ablation == clean; HopSplice all-rows == plain K/V clamp of rows G, no-overwrite == clean/full, answer-only with the K_S-run cache == full, batched == single. **Gate a2** (BF16 floor, per model and format; rows of the a3 sufficiency batch): mean |m(none row) − m(clean pass)| and mean |m(all_T row) − m(full clamp pass)| ≤ max(0.5 nats, 0.02 × mean d_full), where all_T is the in-batch row in which every head sees K_S in every row (the all-heads row in rows G equals RowSplice(G), not the full clamp, and its gap is printed as a raw floor); the hop exactness row within 0.1 nats of the K_S row in every story. **Gate a3** (D3 replicates on E): mean d_full ≥ 10 nats under OPTIONS-AFTER and > 0 under SENTENCE-AFTER; mean d_G / mean d_full ≥ 0.8 under OPTIONS-AFTER, both models. **Evaluability (a):** a model is not evaluable in H1–H5 if its results are missing, its Gate a3 fails or its Gate a2 under OPTIONS-AFTER fails; H5's SENTENCE-AFTER line and H6 also need Gate a2 under SENTENCE-AFTER; Gate a1 failing or not run makes H1–H6 not evaluable. A prediction over both models is MET if met in both, NOT MET if not met in at least one evaluable model, and NOT EVALUABLE otherwise; the same rule combines the parts of H2 and H3 within a model.

**Predictions (OPTIONS-AFTER, both models unless stated):**
- **H1, sparsity.** With the top-k* heads by a3 seeing K_S in rows G and every other head the base key: R(k*) ≥ 0.8 with lower bound ≥ 0.7. (k80 for the a3, single-head f+ and leave-one-out d− rankings reported.)
- **H2, necessity and specificity.** KO(k*) ≥ 0.8; three random sets of k* heads give R_rand(k*) ≤ 0.25 and KO_rand(k*) ≤ 0.25 (means over draws).
- **H3, ablation.** Mean-ablation of the top-k* heads at the six option rows in every row of the 13-row ID_K/ID_V batch and its clean passes: (a) rho_K = mean ID_K(abl)/mean ID_K(none) ≤ 0.5 with upper bound ≤ 0.6; (b) the probability mass on the six location candidates in the clean B pass under ablation ≥ 0.9 in ≥ 80 % of stories; (c) the random sets and the size-matched active-at-G control set each leave rho_K ≥ 0.75; (d) paired ID_V(abl) − ID_V(none) > 0 with CI excluding 0 and mean ≥ 0.25 × [ID_V(NO-MENTION) − ID_V(OPTIONS-AFTER)] from stage 1 (2.7 nats at Qwen2.5-7B, 2.8 at Mistral-7B). The base-argmax rate (the fraction of E stories whose argmax over the six candidates in the clean B pass under ablation is B) ≥ 0.8 is predicted only if (d) is met. H3 is met only if (a)–(d) hold and, (d) being met, the base-argmax rate ≥ 0.8; (c) applies to each random set and to the active-at-G set.
- **H4, canonical duplicate-token heads.** Over the causal set C = top-k_C by a3 (k_C = min(k80, k*); k80 = the smallest k on the grid with R(k) ≥ 0.8, k* if there is none): (i) median D ≥ 0.2 and median I ≤ 0.1; (ii) median T_dup ≥ 0.2 (medians are point estimates; their CIs come from sequence and story bootstraps). Secondary: |top-10(a3) ∩ top-10(D)| ≥ 3 (hypergeometric P = 1.7e-4 at 784 heads, 7.8e-5 at 1024), Spearman rho(a3, D) and rho(a3, T_dup).
- **H5, the second hop at the answer.** Met when r_ans(KV) ≥ 0.5 with lower bound ≥ 0.4, r_ans(KV) > r_other (paired, CI excluding 0) and r_all(KV) ≥ 0.8 (consistency); r_ans(K) > r_ans(V) (secondary) and the strong version r_ans(KV) ≥ 0.7 are reported, not scored. SENTENCE-AFTER reported against r_ans(KV) ≥ 0.35 and r_all ≥ 0.5.
- **H6, the same readers for list and sentence.** |top-20(a3, OPTIONS-AFTER) ∩ top-20(a3, SENTENCE-AFTER)| ≥ 10 per model (null P = 1.3e-12 at 784 heads, 9.3e-14 at 1024).

Alternative: the read is distributed beyond 5 % of heads (H1/H2 fail, curves reported); the readers are task-tuned lookup heads rather than general duplicate heads (H4 (i) fails); the flag reaches the answer through the instruction/template rows, a three-hop route (H5 fails with r_all ≥ 0.8); the two routes do not trade off at the head level (H3d fails, as at 0.5B).

### (b) The exchange on Prakash et al.'s intervention; the law out of sample

Qwen2.5-14B-Instruct; their template-2 stories, raw prompt (no chat template), pool of 320 pairs under random.seed(10) (sha256 asserted); primary population = the first 150 pairs whose clean and counterfactual prompts the model answers correctly under NO-MENTION (every passing pair if fewer than 150 pass, reported as a deviation). Formats (readout only): NO-MENTION (their question), QNAMES (the question names the four candidates s1, s2, S, X), OPTIONS-AFTER ("Choices: ... Answer with exactly one choice."), LETTERS-AFTER exploratory. BIND: at the output of block l* set the clean residual at [166,167,154,155] to the counterfactual's at [154,155,166,167]; ID: at block l*_ID set the queried state span to a donor story's with s_q → S. l* and l*_ID = the earliest layer maximising IIA on the NO-MENTION sweep (ties within 0.01 → earliest). m = log p(target) − log p(s_q) at the answer position (target: the other original state for BIND, S for ID; the letters of those words under LETTERS-AFTER); S and X are two drinks absent from the story, named in the candidate lists, with S the ID donor's replacement for s_q; IIA = the fraction of pairs whose patched argmax, decoded lower-case and stripped, is the target word. Exchange rows r0 = B+KV_B (self), r1 = M, r2 = B+K_M, r3 = B+V_M, r4 = B+KV_M, r5 = M+K_B, r6 = M+V_B, K/V of the patched positions from block l_patch+1 on; psi_K = [m(r2) − m(r0)]/[m(r1) − m(r0)], psi_V likewise with r3, kappa = psi_K/(psi_K + psi_V); kappa_w (state words only) secondary. Natural clamp on the same clean stories at the queried state word, onsets l0 ∈ {l*+1, l*_ID+1, 0, 3, 14} plus an exploratory sweep, giving s_ID(f, l0) = mean ID_K / (mean ID_K + mean ID_V) of the K/V clamp at p from block l0 on, against the self-clamp row, defined when mean ID_K + mean ID_V > 0. The onsets 0, 3 and 14 are absolute block indices, also at 70B.

**Gates (per format and arm):** **Gate b0** exactness, per format, arm and depth: the means over pairs of |m(r4) − m(r1)| and of |m(r0) − m(unbatched B)| ≤ 0.3 nats each (≤ 1e-3 in the FP32 TEST_MODE). **Gate b1** reproduction: IIA(l*) ≥ 0.7 under NO-MENTION for BIND; IIA_ID(l*_ID) ≥ 0.7 for ID. **Gate b2** effect size: Phi = mean[m(r1) − m(r0)] ≥ 3 nats. **kappa evaluability:** psi_K + psi_V ≥ 0.5 and psi_K, psi_V ≥ −0.1 (point estimates; resamples failing the rule are dropped, and a CI with more than 5 % dropped resamples does not exclude 0, so the criterion is not met); otherwise psi_K, psi_V and the interaction (1 − psi_K − psi_V) are reported and the cell is "interaction-carried" if the interaction ≥ 0.5. **Evaluable f:** a format is evaluable in a cell when Gates b0 and b2 pass in it, the kappa rule holds, and the arm's edit reproduces at the cell's depth on the NO-MENTION sweep, for every format (IIA(l*) ≥ 0.7 for BIND; IIA_ID(l*_ID) ≥ 0.7 for ID at l*_ID; IIA_ID(l*) ≥ 0.7 for ID at l*); H7 and H11 also need s_ID(f, ·) defined. H8 needs Gates b0–b2 but not the kappa rule. **Gate b3** (interpretation of H10 as a dissociation): ID_K(OPTIONS-AFTER, l0 = l*+1) > 0 with CI excluding 0 and s_ID(OPTIONS-AFTER, l*+1) ≥ 0.5, and, the ID cell at depth l* under OPTIONS-AFTER being evaluable (so IIA_ID(l*) ≥ 0.7), kappa_ID there ≥ 0.5; otherwise Gate b3 fails (not evaluable if either cell is missing).

**Predictions:**
- **H7, the law on the binding swap, depth-matched.** For every evaluable f ∈ {NO-MENTION, QNAMES, OPTIONS-AFTER}, with NO-MENTION and OPTIONS-AFTER evaluable: |kappa(f) − s_ID(f, l*+1)| ≤ 0.25 (the paper's in-sample maximum deviation is 0.18), and Pearson r(kappa, s_ID(·, l*+1)) ≥ 0.9 over the three (the r part needs all three formats evaluable; otherwise it counts as not met and H7 is NOT MET). Not evaluable unless NO-MENTION and OPTIONS-AFTER are evaluable. The same against s_ID(f, 0) is secondary.
- **H8, shared prediction under NO-MENTION.** psi_V ≥ 0.5 and psi_K ≤ 0.25, with the CI of psi_V − psi_K excluding 0.
- **H9, crossover on their intervention (H_read).** kappa(OPTIONS-AFTER) − kappa(NO-MENTION) ≥ 0.4 with the paired CI excluding 0; kappa(QNAMES) between the two point estimates if evaluable.
- **H10, the rival (H_binding).** kappa(f) ≤ 0.25 for every evaluable f ∈ {NO-MENTION, QNAMES, OPTIONS-AFTER}, with NO-MENTION and OPTIONS-AFTER evaluable (otherwise not evaluable), stated on kappa alone (incompatible with H7/H9 whenever s_ID(OPTIONS-AFTER, l*+1) ≥ 0.5). "Flat-high" (kappa ≥ 0.75 in every evaluable f) is reported beside H10, without a verdict. Read as "keys carry identity, values carry binding" only if Gate b3 passes; otherwise "not evaluable as a dissociation at this depth".
- **H11, positive control: the law on an identity edit on their material.** Part 1 (own depth; needs Gate b1 for ID and NO-MENTION and OPTIONS-AFTER evaluable): for every evaluable f, |kappa_ID(f) − s_ID(f, l*_ID+1)| ≤ 0.25, and kappa_ID(OPTIONS-AFTER) − kappa_ID(NO-MENTION) ≥ 0.4 with CI excluding 0. Part 2 (the binding depth, evaluable if IIA_ID(l*) ≥ 0.7): |kappa_ID(f) − s_ID(f, l*+1)| ≤ 0.25 for every evaluable f, kappa_ID(f) here being the ID exchange at depth l*. Met if Part 1 holds and Part 2 holds or is not evaluable.
- **H12, optional Llama-3-70B.** If run (BIND sweep over blocks 25..44, ID sweep over the even blocks 0..78 and 25..44; l* and l*_ID chosen within them): Gate b1 for BIND with l* ∈ 30..40, and the 14B verdicts of H8, H11 Part 1 and whichever of {H7, H9} or H10 was met are reproduced at the same thresholds and gates: each of these 14B verdicts that is MET or NOT MET recurs at 70B with the same verdict; a verdict NOT EVALUABLE at 14B is left out, and H12 is not evaluable if none remains or the 14B results are absent.

Alternative: H10 is the explicit rival of H7/H9; a flat-high kappa (address read by the answer position) and interaction-carried cells are reported as such; failure of H11 means the in-sample law does not transfer to their stories and wrapper even for an identity edit.

**Exploratory:** f+/d− rankings and grids, layer profile, split-half reliability of a3, zero-ablation and top-10/20, K-only/V-only hop rows and row restrictions, previous-token scores, optional Qwen2.5-14B (a); IIA/Phi sweeps under every format, FP32 re-check at l* ± 2, words-only and periods-only exchanges, rho_K/rho_V and the interaction, QNAMES2, LETTERS-AFTER as a fourth point, the full onset sweep of s_ID on their stories, a multi-position row splice if kappa(OPTIONS-AFTER) ≥ 0.3, per-pair scatter, the overlap with their 80 validation pairs (b).

---

## Outcome of P-2026-10-05-H (GPU stage 6; scored by analysis/stage6_score.py at the finalising commit cc3a3e0, which the run used)

**Run.**
- **Hardware and software:** one A100-SXM4-80GB, Python 3.12.14, torch 2.11.0+cu128, transformers 5.18.0, numpy 2.5.3. The run used a clean checkout of cc3a3e0 (`COMMIT.txt`: only `results/gpu_stage6/` untracked) with `PART=all` and `TEST_MODE=0`, started 2026-10-07T10:57Z. The score was written on the box at 17:13Z.
- **Order:** the FP32 unit tests of stage 6 ran first (10:57Z), followed by the earlier stages' regression tests (`log_pytest.txt`, 11:01Z, 10 passed). Part (a) followed: Qwen2.5-7B-Instruct (6964 s), then Mistral-7B-Instruct-v0.3 (9765 s). Part (b) came last, at Qwen2.5-14B-Instruct: preflight, LM filter, sweeps, exchange and clamps, and then the exploratory FP32 re-check sweeps.
- **Precision and attention:**
  - Part (a) used BF16 with eager attention.
  - Part (b) used BF16 with sdpa.
  - The exception is the two exploratory re-check sweeps (`sweep_*_fp32.json`), which ran in FP32 with sdpa.
- **Prakash et al.'s release:** fetched at run time at 0579347e3c. All six file hashes and the sha256 of the seed-10 pool (4451da1a…, n = 320) were checked (`RELEASE.txt`).
- **Provenance and population:**
  - One Hub revision per model, for three models (`REVISIONS.txt`). Every results file carries commit cc3a3e0.
  - Part (a): in both arms and both models, R = make_cores(60, Random(0)) and E = make_cores(60, Random(1)). The duplicate scores use 100 sequences.
  - Part (b): the LM filter passed 315 of the 320 pairs. The population is the first 150 passing pairs in pool order. All 13 exchange cells and all 38 clamp cells are over this population.
  - The scorer prints "provenance OK; population OK".
- **What failed:** no step failed.
- **Not run:** the optional Llama-3-70B run (H12) was not made.
  - Reason: the plan agreed before the stage-6 run (budget) was to make the 70B run only if Qwen2.5-14B failed to reproduce Prakash et al.'s edit, i.e. failed Gate b1. Gate b1 passed (IIA 0.993 at l* = 28), and the run was not made. It needs a separate 2×80GB box and a Hugging Face token for the gated weights (`docs/GPU_RUNBOOK.md`); the main run had `HF_TOKEN` unset (`ENV.txt`).
  - The entry and the runbook allow it to be run later and scored against the fixed 14B results. Because none of H7, H9 and H10 was met at 14B, it would test Gate b1 (l* in 30–40), H8 and H11 Part 1 only. Paper v3 reports H12 as not run.
- **Re-score:** re-scoring the archive off the box with the committed scorer at cc3a3e0 reproduces `STAGE6_SCORE.txt` byte for byte (default `--root`, run from a checkout root).
- **What changed between cc3a3e0 and the results commit (f24586a).**
  - No scorer, experiment, test or `ckeys/` file changed. The only code change is one line of `scripts/make_anonymous_release.py`, which adds `docs/V3_STAGE5_IMPLICATIONS.md` to the release's exclusion list.
  - The other changes are documentation and repository settings: the stage-5 outcome record G (+191 lines, insertions only, placed above entry H in `docs/PREREGISTRATION.md`), the stage-5 memo `docs/V3_STAGE5_IMPLICATIONS.md`, a one-line timing edit in `docs/GPU_RUNBOOK.md` and one `.gitignore` line.
  - Entry H itself is byte-identical to cc3a3e0: the section from "## P-2026-10-05-H" to the end of the file has the same SHA-256 (1c01221a…) in both versions.
- **Timeline relative to stage 5.**
  - H was finalised at cc3a3e0, committed 2026-10-07T01:16Z. The entry and the commit message state that no stage-5 output had been inspected.
  - The stage-5 GPU run had started at 2026-10-06T21:25Z (`results/gpu_stage5/COMMIT.txt`). Its results were first committed at 08:41Z (88bb877) and its outcome at 09:37Z (e24e928), both after cc3a3e0.
  - The stage-6 run started at 10:57Z on cc3a3e0, unchanged.
- **Independent checks:** four recomputations from the raw files reproduce every gate value, point estimate and verdict (details under Deviations). None of them imports the scorer.
- **Summary line:** 6 MET (H1–H5, H11), 4 NOT MET (H7–H10), 1 NOT EVALUABLE (H6), 1 NOT RUN (H12) of 12.

Numbers below are from `STAGE6_SCORE.txt` unless marked. *(recomputed)* means recomputed from the raw files and not printed in the score file. "Our arithmetic" marks a difference or ratio of printed score-file values. In part (a), pairs are Qwen2.5-7B-Instruct / Mistral-7B-Instruct-v0.3. Part (b) is Qwen2.5-14B-Instruct, and its format triples are NO-MENTION / QNAMES / OPTIONS-AFTER unless stated. Formats are named as in the paper; the code arms are OPTIONS-AFTER (P1) and SENTENCE-AFTER (POST).

### Gates

- **Gate a1 (FP32 exactness): met.**
  - `tests/test_head_splice.py`: 7 passed, 0 failed in `log_pytest_stage6.txt`. This is the only run recorded in that log (2026-10-07T10:57:32Z, commit cc3a3e0).
  - The same run passed 29 tests and skipped 1, the Llama-3 tokenizer test of `tests/test_prakash.py`.
- **Gate a2 (BF16 floor, per model and format).** Each line gives |none − clean| and |all_T − full| against the bound, then the hop exactness row (max |exact − K_S|) against 0.1 nats.

  | Model | Format | \|none − clean\| | \|all_T − full\| | Bound | Hop exactness | Verdict |
  |---|---|---|---|---|---|---|
  | Qwen2.5-7B | OPTIONS-AFTER | 0.606 | 0.669 | 0.719 | 0.0000 | **met** |
  | Qwen2.5-7B | SENTENCE-AFTER | 0.770 | 0.489 | 0.500 | 0.0000 | **not met** (\|none − clean\| exceeds the bound) |
  | Mistral-7B | OPTIONS-AFTER | 0.079 | 0.162 | 0.529 | 0.0000 | **met** |
  | Mistral-7B | SENTENCE-AFTER | 0.069 | 0.224 | 0.500 | 0.0000 | **met** |

  - The bound is max(0.5, 0.02 × mean d_full). For Qwen2.5-7B SENTENCE-AFTER, 0.02 × 10.438 = 0.209 (our arithmetic), so the 0.5-nat floor applies.
  - The raw floor |all_G − full| is printed beside the gate, as the entry requires. It is 3.000 and 4.624 at Qwen2.5-7B, and 1.097 and 3.743 at Mistral-7B (OPTIONS-AFTER, SENTENCE-AFTER).
- **Gate a3 (D3 replicates on E): met in both models.**
  - Mean d_full under OPTIONS-AFTER is +35.969 [+34.591, +37.333] and +26.469 [+24.548, +28.259], each ≥ 10.
  - Under SENTENCE-AFTER it is +10.438 [+9.462, +11.475] and +12.254 [+10.716, +13.799], each > 0.
  - d_G/d_full under OPTIONS-AFTER is 0.926 [0.905, 0.946] and 0.982 [0.969, 0.997], each ≥ 0.8.
- **Evaluability, part (a).**
  - Both models are evaluable in H1–H5.
  - H5's SENTENCE-AFTER line and H6 need Gate a2 under SENTENCE-AFTER. They are therefore not evaluable at Qwen2.5-7B, and by the entry's rule for combining the two models they are NOT EVALUABLE overall.
- **Gate b1 (reproduction, NO-MENTION sweeps): met.**
  - BIND: IIA(l* = 28) = 0.993. The IIA is 1.00 at blocks 29–34, and the tie rule (within 0.01, earliest) selects 28.
  - ID: IIA_ID(l*_ID = 0) = 1.000, and IIA_ID(l*) = IIA_ID(28) = 1.000, so H11 Part 2 and Gate b3(b) are evaluable.
  - l* and l*_ID re-derived from the sweep files equal `lstar.json`.
- **Gate b0 (exactness, 13 cells): met in every cell.** mean|m(r4) − m(r1)| is 0.099–0.194 and mean|m(r0) − m(unbatched B)| is 0.141–0.203, against 0.3 nats.
- **Gate b2 (effect size Phi ≥ 3 nats): met in every cell except the two LETTERS-AFTER cells at depth 28.**

  | Cell | NO-MENTION | QNAMES | OPTIONS-AFTER | LETTERS-AFTER | QNAMES2 |
  |---|---|---|---|---|---|
  | BIND@28 | +31.616 | +5.814 | +6.523 | +0.225 (**not met**) | +7.285 |
  | ID@0 | +46.005 | +39.606 | +54.550 | +38.255 | – |
  | ID@28 | +45.797 | +9.321 | +10.885 | +0.034 (**not met**) | – |

- **kappa evaluability rule: holds in every cell except ID@28 LETTERS-AFTER.** That cell has psi_K −0.171, psi_V +3.024 and interaction −1.854, so it is not interaction-carried. In every cell where the rule holds, 0.0 % of resamples are dropped, except BIND@28 LETTERS-AFTER (0.2 %).
- **Gate b3 (H10 as a dissociation): not met.**
  - (a) ID_K(OPTIONS-AFTER, l*+1 = 29) = −0.092 [−0.123, −0.063], and s_ID = −0.015. The gate needs ID_K > 0 with the CI excluding 0, and s_ID ≥ 0.5.
  - (b) kappa_ID at depth 28 under OPTIONS-AFTER = +0.006 [−0.003, +0.013], against ≥ 0.5. The cell itself is evaluable.

### Predictions

**Part (a), reader heads and the second hop (OPTIONS-AFTER; Qwen2.5-7B / Mistral-7B; k* = 40 / 52).**

| Prediction | Observed | Verdict |
|---|---|---|
| H1 sparsity: R(k*) ≥ 0.8, lower bound ≥ 0.7 | R(40) +0.967 [+0.952, +0.982]; R(52) +0.942 [+0.919, +0.961]. k80: a3 20 / 16; f+ none / none; d− none / 12 | **met** (2/2) |
| H2 necessity and specificity: KO(k*) ≥ 0.8; R_rand, KO_rand ≤ 0.25 (means over 3 draws) | KO +0.977 [+0.966, +0.987] / +0.971 [+0.954, +0.984]; R_rand +0.004 / +0.007; KO_rand +0.011 / +0.010 | **met** (2/2) |
| H3 ablation: (a) rho_K ≤ 0.5, upper ≤ 0.6; (b) clean-B candidate mass ≥ 0.9 in ≥ 80 % of stories; (c) every random set and the active-at-G set rho_K ≥ 0.75; (d) dV > 0 (CI excl. 0) and ≥ floor; base-argmax rate ≥ 0.8 | (a) 0.238 [0.205, 0.270] / 0.059 [0.040, 0.079]; (b) 1.00 / 0.97 of stories; (c) random sets 1.002, 0.978, 0.953, active set 1.018 / random sets 1.002, 1.009, 1.006, active set 1.009; (d) +6.907 [+6.264, +7.561] against floor 2.7 / +10.861 [+9.995, +11.677] against 2.8; base-argmax 0.87 / 1.00 | **met** (2/2) |
| H4 canonical duplicate-token heads, over C = top-k_C by a3, k_C = min(k80, k*) = 20 / 16: (i) median D ≥ 0.2 and median I ≤ 0.1; (ii) median T_dup ≥ 0.2 | (i) D +0.370 [+0.361, +0.378] / +0.232 [+0.227, +0.238]; I +0.011 / +0.022. (ii) T_dup +0.459 [+0.445, +0.475] / +0.433 [+0.407, +0.446] (control words T_ctrl 0.027 / 0.065). Secondary: \|top-10(a3) ∩ top-10(D)\| 6 (P = 1.4e-10) / 5 (P = 6.7e-09); Spearman(a3, D) +0.185 / +0.230; Spearman(a3, T_dup) +0.146 / +0.251 | **met** (2/2) |
| H5 second hop: r_ans(KV) ≥ 0.5 (lower ≥ 0.4); r_ans − r_other > 0 (CI excl. 0); r_all(KV) ≥ 0.8 | r_ans(KV) +0.899 [+0.886, +0.913] / +0.859 [+0.845, +0.872]; r_other +0.118 / +0.165; r_ans − r_other +0.782 [+0.749, +0.813] / +0.694 [+0.671, +0.714]; r_all(KV) +0.992 / +0.992. Secondary (reported, not scored): r_ans(K) 0.797 / 0.683 against r_ans(V) 0.080 / 0.023, K − V +0.717 / +0.660 (CIs exclude 0); the strong version r_ans(KV) ≥ 0.7 holds in both | **met** (2/2) |
| H5, SENTENCE-AFTER line (reported against r_ans(KV) ≥ 0.35 and r_all ≥ 0.5) | Qwen2.5-7B r_ans(KV) +0.788 [+0.751, +0.824], r_all +0.841 (computed met; not evaluable, Gate a2 under SENTENCE-AFTER failed); Mistral-7B +0.780 [+0.761, +0.799], r_all +0.939 (met) | **not evaluable** (reported line; it does not enter the H5 verdict) |
| H6 the same readers: \|top-20(a3, OPTIONS-AFTER) ∩ top-20(a3, SENTENCE-AFTER)\| ≥ 10 per model | 13 (P = 9.2e-19; computed met, not evaluable: Gate a2 under SENTENCE-AFTER failed) / 14 (P = 9.9e-23; met) | **not evaluable** |

**Part (b), the exchange on Prakash et al.'s intervention (Qwen2.5-14B-Instruct, n = 150; l* = 28, l*_ID = 0).**

Evaluable formats: BIND@28 and ID@0 are evaluable under NO-MENTION, QNAMES and OPTIONS-AFTER. ID@28 is evaluable under those three formats. Each LETTERS-AFTER cell at depth 28 fails Gate b2, and ID@28 LETTERS-AFTER also fails the kappa rule. LETTERS-AFTER is exploratory in every cell.

| Prediction | Observed | Verdict |
|---|---|---|
| H7 the law on the binding swap, depth-matched: \|kappa(f) − s_ID(f, 29)\| ≤ 0.25 for every evaluable f, and Pearson r(kappa, s_ID(·, 29)) ≥ 0.9 | kappa +0.618 / +0.906 / +0.864 against s_ID(f, 29) +0.017 / −0.028 / −0.015; gaps 0.601 / 0.934 / 0.879; r = −0.986. Secondary, against s_ID(f, 0): gaps 0.600 / 0.181 / 0.163 | **not met** |
| H8 shared prediction under NO-MENTION: psi_V ≥ 0.5, psi_K ≤ 0.25, CI of psi_V − psi_K excluding 0 | psi_V +0.435, psi_K +0.703; psi_V − psi_K −0.268 [−0.317, −0.219] (the CI excludes 0, with the opposite sign) | **not met** |
| H9 crossover (H_read): kappa(OPTIONS-AFTER) − kappa(NO-MENTION) ≥ 0.4 (paired CI excl. 0); kappa(QNAMES) between the two | +0.247 [+0.211, +0.281], 0.0 % dropped; kappa(QNAMES) +0.906 is not between +0.618 and +0.864 | **not met** |
| H10 the rival (H_binding): kappa(f) ≤ 0.25 for every evaluable f | +0.618 / +0.906 / +0.864. Flat-high (kappa ≥ 0.75 in every evaluable f) does not hold either (NO-MENTION 0.618). Gate b3 not passed: "not evaluable as a dissociation at this depth" | **not met** |
| H11 positive control, identity edit. Part 1 (own depth, s_ID(f, 1)): every gap ≤ 0.25 and kappa_ID(OPTIONS-AFTER) − kappa_ID(NO-MENTION) ≥ 0.4 (CI excl. 0). Part 2 (depth 28, s_ID(f, 29)): every gap ≤ 0.25 | Part 1: kappa_ID +0.131 / +0.773 / +0.779 against s_ID +0.017 / +0.724 / +0.702, gaps 0.114 / 0.048 / 0.077; difference +0.648 [+0.630, +0.666], 0.0 % dropped. Part 2: kappa_ID +0.129 / +0.007 / +0.006 against s_ID +0.017 / −0.028 / −0.015, gaps 0.113 / 0.036 / 0.020 | **met** (Part 1 met, Part 2 met) |
| H12 optional Llama-3-70B | No Llama-3-70B results | **not run** |

**Status of the preregistered alternatives.**
- **(a) The read is distributed beyond 5 % of heads (H1/H2 fail):** not supported. H1 and H2 are met with large margins: R(k*) 0.967 / 0.942, KO(k*) 0.977 / 0.971. Random sets of equal size give R_rand 0.004 / 0.007 and KO_rand 0.011 / 0.010 (means over three draws); no single draw exceeds 0.02 *(recomputed: KO per draw at most 0.020 / 0.013)*. The set is not reducible to single heads, though; see the unpredicted observations.
- **(a) Task-tuned lookup heads rather than general duplicate heads (H4 (i) fails):** not supported. H4 (i) is met in both models: median D 0.370 / 0.232, median I 0.011 / 0.022. At Mistral-7B the margin over 0.2 is 0.032 (our arithmetic), and it depends on the size of C (see Deviations).
- **(a) Three-hop route through the instruction and template rows (H5 fails with r_all ≥ 0.8):** not supported. r_ans(KV) is 0.899 / 0.859 and r_other 0.118 / 0.165.
- **(a) No head-level trade-off (H3d fails, as at 0.5B):** not supported. dV is +6.907 / +10.861 with CIs excluding 0, and the base-argmax rate is 0.87 / 1.00.
- **(b) H10 as the rival (binding in values, "keys carry identity, values carry binding"):** not met. kappa is 0.618 / 0.906 / 0.864, against ≤ 0.25. Gate b3 also fails, so the dissociation reading is not available either way.
- **(b) Flat-high kappa (address read by the answer position):** not observed as defined. kappa ≥ 0.75 under QNAMES and OPTIONS-AFTER, but 0.618 under NO-MENTION.
- **(b) Interaction-carried cells:** none. No evaluable cell fails the kappa rule. The one cell that fails the rule (ID@28 LETTERS-AFTER) has interaction −1.854, below the 0.5 that would make it interaction-carried.
- **(b) H11 failure ("the in-sample law does not transfer to their stories and wrapper even for an identity edit"):** does not apply, because H11 is met. Its limits are stated under Deviations.

### Deviations and disclosures

- **No deviation from the entry in the run.** Every preregistered cell and population is present, every gate was evaluated as written, and no verdict is changed here.
- **Gate a2 compares batched rows with single passes (design, not a scorer bug).** The Qwen2.5-7B SENTENCE-AFTER failure is a BF16 batch-shape offset, not a splice error, on the following evidence *(recomputed)*:
  - Within each batch, every none row of the six sufficiency batches is identical (mean difference 0.000).
  - The single-pass clean B run of the ablation batch equals the single clean pass exactly.
  - Under SENTENCE-AFTER, the reference rows of the knockout batch (B = 17) and the hop batch (B = 8) match each other but differ from the B = 18 sufficiency rows by 0.624 nats. Under OPTIONS-AFTER they coincide (0.000).
  - Per story under SENTENCE-AFTER, |none − clean| has median 0.625 and maximum 3.125, with 31/60 stories above 0.5 and 18/60 above 1.0. The signed mean is +0.049.
  - The hop exactness row is 0.0000 in every story and cell.

  Every scored quantity is a difference against an in-batch reference row. The H6 quantity uses the single-pass a3 rankings on R. The rule nevertheless ties H5's SENTENCE-AFTER line and H6 to this gate, and they stay NOT EVALUABLE.
- **Qwen2.5-7B's OPTIONS-AFTER pass of Gate a2 is narrow.**
  - |all_T − full| is 0.669 against a bound of 0.719, and the bound exceeds the 0.5-nat floor only because it scales with d_full (35.969).
  - The bootstrap interval of that mean is [0.534, 0.816] *(recomputed)*, which crosses the bound. The verdict uses the point estimate, as the entry specifies.
- **The FP32 re-check (exploratory) is not in the score file.** The scorer skips labelled files by design.
  - Values *(recomputed)*, BIND IIA FP32 / BF16 at blocks 26–30: 0.747 / 0.747, 0.880 / 0.887, 0.993 / 0.993, 1.000 / 1.000, 1.000 / 1.000. The tie rule on the FP32 blocks also selects 28.
  - Per-pair agreement at block 28 is 0.987: a different single pair fails in FP32. Phi differs by about 0.13 nats.
  - ID: IIA 1.000 at blocks 0–2 in both precisions.
  - The command line in `log_fp32_Qwen2.5-14B-Instruct.txt` passes `--dtype` twice (bfloat16, then float32). The last value applies, and both files record torch.float32.
- **Null probabilities.**
  - The entry quotes hypergeometric P at the threshold: 1.7e-4 / 7.8e-5 for overlap ≥ 3 of the top-10 (H4), and 1.3e-12 / 9.3e-14 for overlap ≥ 10 of the top-20 (H6). These reproduce *(recomputed)*.
  - The scorer prints P(overlap ≥ observed) instead: 1.4e-10 / 6.7e-09 (H4) and 9.2e-19 / 9.9e-23 (H6).
  - Both are correct; they differ in conditioning.
- **H3 (d) floors.** The entry and the scorer use 2.7 and 2.8 nats. Recomputed from the stage-1 summaries, the floors are 2.710 and 2.838 *(recomputed)*. Both are far below the observed dV.
- **H4 (ii) averaging.** The scorer pools T_dup over (story, repeated word) pairs as a ratio of means, which is within the entry's wording. A per-story average gives 0.468 / 0.430, against 0.459 / 0.433 *(recomputed)*.
- **H5 paired condition.** "r_ans(KV) > r_other (paired, CI excluding 0)" is checked as the lower bound of the paired difference > 0. This is equivalent here.
- **Duplicate-score sequences at Qwen2.5-7B have no BOS token.** The tokenizer has none, and the file records `bos: False`, so the first repeated query's duplicate key is position 0, which is also the attention sink. This can raise a head's D by at most 1/30 = 0.033 (our arithmetic), which cannot move the median of 0.370 below 0.2. Mistral-7B's sequences have a BOS token.
- **Format evaluability in part (b) takes the arm's reproduction from the NO-MENTION sweep for every format, as the entry specifies.**
  - As a result, BIND@28 and ID@28 under QNAMES and OPTIONS-AFTER are evaluable, although the edit flips 0/150 answers there.
  - In the exchange cells the M row's argmax among the roles is s_q in 150/150 pairs for BIND under OPTIONS-AFTER and QNAMES *(recomputed)*.
  - These kappas therefore decompose a shift of the logit margin (Phi 6.5 and 5.8 nats against base margins m(B) of −27.69 and −20.06 *(recomputed)*), not a reproduced swap.
- **kappa and s_ID are different statistics (design).**
  - kappa is one-sided: m = log p(target) − log p(s_q), with K/V exchanged at the state word and its following punctuation token.
  - s_ID is the symmetric S-versus-X identity contrast at the word only.
  - With l*_ID = 0, the ID@0 exchange and the natural clamp from block 1 are nearly the same intervention. Their one-sided kappas agree to within 0.001 (NO-MENTION), 0.029 (QNAMES), 0.044 (OPTIONS-AFTER) and 0.012 (LETTERS-AFTER) *(recomputed)*.
  - The H11 Part 1 gaps (0.114 / 0.048 / 0.077) split between the two differences as follows, using the one-sided clamp kappas 0.132 / 0.744 / 0.735 *(recomputed; the split is our arithmetic)*:
    - NO-MENTION: almost all of the gap is the difference between the statistics (0.115), with the exchange 0.001 below the clamp.
    - QNAMES: 0.029 is exchange versus clamp and 0.020 the difference between the statistics.
    - OPTIONS-AFTER: 0.044 is exchange versus clamp and 0.033 the difference between the statistics.
  - For H7, the one-sided clamp analogue at l0 = 29 is 0.131 / 0.012 / 0.005 *(recomputed)*, which would still leave gaps of 0.49 or more. No verdict depends on the difference between the statistics.
- **l*_ID = 0 comes from the earliest-maximum rule on a flat curve.** The ID NO-MENTION sweep is 1.00 at every block 0–35 (Phi 46.0 to 41.2), so the rule picks block 0 by default, not by localisation. H11 Part 1 thus compares two near-identical interventions, and Part 2 compares cells whose kappa and s_ID are both near 0 under QNAMES and OPTIONS-AFTER.
- **l* = 28 rests on one pair.** IIA(28) = 149/150, against the tie threshold of 0.990; pair 60 is the only failure. One more failure would have made l* = 29. The FP32 re-check selects 28 as well.
- **Disclosed pilots (entry H) and how the outcome compares.**
  - Pilot (a), Qwen2.5-0.5B FP32: the a3-ranked top heads were duplicate heads, with k80 = 12 of 336. At 7B, H4 is met, which is consistent; k80 is 20 of 784 and 16 of 1024.
  - The same pilot's ablation of 16 heads took ID_K from 1.69 to −0.09 with no head-level trade-off (the entry's H3 alternative, "as at 0.5B"). Its answer-only hop gave r_ans(KV) 0.37 and 0.04 on two stories. At 7B both differ from the pilot: H3 (d) is met (dV +6.907 / +10.861) and H5 is met (r_ans(KV) 0.899 / 0.859).
  - Pilot (b), the 0.5B prototype of the exchange: exact (B + KV_M = M to 0.0), with psi_K +1.54 and psi_V −0.61 under NO-MENTION, which the evaluability rule marks not evaluable. At 14B, Gate b0 is met in all 13 cells and the rule holds in every NO-MENTION cell.
  - The release's own 14B per-layer IIA (1.00 at blocks 28–34) predicted l* = 27–28, and l* is 28. The per-layer replication table under Part (b) therefore compares against material seen before finalisation. It is a reproduction check, not an independent prediction.
- **Validation-split overlap.** The scorer computes "70/80" from our filter's passing list. Prakash et al.'s own filter could select slightly different pairs, so the figure is approximate.
- **Cosmetic.** For the cell that fails the kappa rule, the scorer does not print the dropped-resample fraction (ID@28 LETTERS-AFTER: 56.6 % *(recomputed)*). The cell enters no verdict.
- **Independent verification** *(recomputed)*:
  - Four recomputations from the raw files, none importing the scorer: gates a2/a3 with H1–H3, H5s and H6; gates a2/a3 with k80, H4–H6 and the exploratory part (a) lines; the part (b) sweeps, exchanges, gates and H7–H11; and the part (b) natural clamps and the law. Every gate value, point estimate and verdict agrees.
  - With independent seeds and 2000–4000 resamples, CI bounds agree with the scorer's 10,000-resample bounds to within about 0.01 on ratio scales and about 0.05 nats on nat-scale quantities. Examples: BIND@28 LETTERS-AFTER kappa upper 0.846 against 0.854; BIND@28 NO-MENTION Phi [+30.235, +32.979] against [+30.193, +32.985]; Gate a3 d_full at Qwen2.5-7B [34.574, 37.356] against [34.591, 37.333].
  - The exception is s_ID(LETTERS-AFTER, 29), whose denominator is near zero: lower bound −2.202 against −2.155. That cell enters no verdict.
  - Among the MET criteria, the narrowest bound-based margins are the H1 lower bounds (0.952 / 0.919 against 0.7, margins 0.252 / 0.219) and the H3 (a) upper bound at Qwen2.5-7B (0.270 against 0.6). On the NOT MET side, the H8 upper bound of psi_V − psi_K is −0.219 against 0. No bound is within 0.2 of its threshold (margins are our arithmetic).
  - The narrow margins are point estimates: Qwen2.5-7B's Gate a2 OPTIONS-AFTER pass (0.669 against 0.719; see above), H4 (i) at Mistral-7B (median D 0.232 against 0.2) and H3's base-argmax rate at Qwen2.5-7B (0.87 against 0.8). Qwen2.5-7B's SENTENCE-AFTER |all_T − full| (0.489 against 0.5) is also narrow, but that gate failed on its other line.
  - The SHA-256 hashes of all 37 files listed in `MANIFEST.sha256` verify.
  - Off the box, the committed scorer reproduces the score file byte for byte.

### Unpredicted observations (reported, not reinterpreted)

**Part (a).**
- **Single-head rankings.** Rankings by single-head causal effect do not find the set.
  - At Qwen2.5-7B, f+ reaches R(128) = 0.06 and d− reaches R(128) = 0.58.
  - At Mistral-7B, f+ reaches R(128) = 0.66, while d− reaches k80 = 12.
  - The a3 curve rises steeply: Qwen2.5-7B R(8) 0.16, R(12) 0.77.
  - At Mistral-7B, blinding the top 3 heads removes 0.83 (KO(3)), while those 3 alone recover 0.08 (R(3)).
- **Layer profile.** Letting all heads of one layer read K_S recovers at most 0.05 of d_G under OPTIONS-AFTER (Qwen2.5-7B layer 15, Mistral-7B layer 12). The single-layer shares sum to 0.163 / 0.094 *(recomputed)*. The top-20 / top-16 heads span 12 / 10 layers *(recomputed)*.
- **H4 depends on the size of the head set.**
  - Median D over the top-k* set is 0.179 / 0.065 *(recomputed)*. At Mistral-7B it is 0.156 over the top-20 and 0.294 over the top-12 *(recomputed)*.
  - Only 13 of the 20 heads in C (Qwen2.5-7B) and 9 of the 16 (Mistral-7B) have D ≥ 0.2. Over the whole model, 26 and 11 heads have D ≥ 0.2, and the whole-model median D is 0.028 / 0.007 *(recomputed)*.
  - At Qwen2.5-7B, R(16) = 0.792, just under 0.8, so k_C = 20. With C = top-16 the median D would also be 0.370 *(recomputed)*.
- **The copy after ablation.** At Mistral-7B, ID_V under ablation of the top-k* is 2.593 + 10.861 = 13.45 (our arithmetic). This is close to the stage-1 NO-MENTION ID_V of 13.65, a cross-stage comparison on different stories.
- **Answer preservation, compared with stage 5.**
  - At Qwen2.5-7B the base-argmax rate under head ablation is 0.87, against 0.97 under no ablation. The stage-5 edge knockout M1 (G7b) gave acc_B 0.37 / 0.30 (LIST-AFTER / OPTIONS-AFTER).
  - The interventions and populations differ: mean-ablation at six option rows leaves rho_K 0.238, while the knockout leaves r_K(M1) ≈ 0.
  - Zero-ablation of the top-k* (exploratory) gives rho_K 0.081, dV +11.315 and a base-argmax rate of 1.00 at Qwen2.5-7B, and rho_K 0.057 and a rate of 0.95 at Mistral-7B.
- **The second hop in detail (OPTIONS-AFTER).**
  - The routes add up: r_ans + r_other = 1.017 / 1.024 against r_all 0.992 / 0.992.
  - Restricted to single option rows, the source-word row gives r_ans 0.605 / 0.557 and the base-word row 0.414 / 0.391; the separator rows give 0.049 / 0.016.
  - Under SENTENCE-AFTER, r_ans(K) is 0.875 / 0.732 and r_ans(V) −0.046 / 0.052.
- **SENTENCE-AFTER (reported, not scored).**
  - At Qwen2.5-7B, R(k) and KO(k) exceed 1 (R up to 1.06 for k = 16–48).
  - At Qwen2.5-7B, mean-ablation of the top-k* drives rho_K negative: −0.080 [−0.123, −0.044].
  - One random set changes rho_K substantially: 0.655 at Qwen2.5-7B and 1.289 at Mistral-7B.
  - d_G/d_full is 0.572 / 0.728 *(recomputed)*, against 0.926 / 0.982 under OPTIONS-AFTER.
  - Clean-B candidate mass ≥ 0.9 holds in 0/60 stories in every ablation condition, as expected for the format.
- **Rank stability.**
  - Split-half Spearman of a3 (R against E) is 0.973 / 0.956 under OPTIONS-AFTER and 0.985 / 0.981 under SENTENCE-AFTER.
  - Across formats, the top-10 a3 heads overlap 7 / 9, and the full a3 grids correlate at Spearman 0.520 / 0.573 *(recomputed)*.
- **Previous-token scores.** The median previous-token score over C is 0.017 / 0.004, and the maximum 0.083 / 0.049 *(recomputed)*.

**Part (b).**
- **The intervention reproduces.** The BIND swap reproduces Prakash et al.'s released per-layer IIA for Qwen2.5-14B on their pool and wrapper. All values below are *(recomputed)*.

  | Block | Theirs (80 validation pairs) | Ours (150 pairs) | Ours, on the 70 pairs shared with their validation split |
  |---|---|---|---|
  | 24 | 0.69 | 0.720 | 0.700 |
  | 26 | 0.72 | 0.747 | 0.729 |
  | 28 | 1.00 | 0.993 | 1.000 |
  | 30–34 | 1.00 | 1.000 | 1.000 |
  | 36 | 0.06 | 0.080 | 0.057 |

- **The binding swap reproduces only under NO-MENTION.**
  - Under OPTIONS-AFTER, QNAMES and LETTERS-AFTER, its IIA is 0.00 at every swept block 18–40 (at most 0.01 under QNAMES).
  - Phi peaks at 6.5 (OPTIONS-AFTER, block 28), 6.1 (QNAMES, block 31) and 2.8 (LETTERS-AFTER, block 23).
- **The identity edit stops flipping the answer from about block 20 when candidates are named after the question.**
  - IIA_ID under OPTIONS-AFTER is 1.00 through block 11, 0.89 at block 14, 0.66 at 16, 0.17 at 19, and 0.00 from block 20.
  - Under QNAMES it is 0.90 at block 0 and 0.00 from block 20.
  - Under NO-MENTION it is 1.00 through block 35.
- **The identity key route closes with depth under OPTIONS-AFTER.** s_ID(OPTIONS-AFTER, l0) is 0.702 at l0 ≤ 3, 0.587 at 14, 0.218 at 18, 0.041 at 24 and −0.014 at 27. Under NO-MENTION it stays at 0.017–0.036 at every onset.
- **Binding swap against identity edit at the same positions and depth (block 28; exploratory, no verdict).**
  - BIND: psi_K +0.703 / +1.020 / +1.025, psi_V +0.435 / +0.106 / +0.161.
  - ID@28: psi_K +0.129 / +0.006 / +0.005, psi_V +0.872 / +0.827 / +0.811.
  - QNAMES2 (BIND): psi_K +1.016, psi_V −0.014.
- **Punctuation positions under NO-MENTION.** For BIND, the words-only kappa_w is 0.779 against kappa 0.618; words-only psi_V is 0.238 against the full 0.435 *(recomputed)*. Under OPTIONS-AFTER and QNAMES, kappa_w is within 0.013 of kappa.
- **Split by question.** BIND NO-MENTION gives psi_K 0.648 and psi_V 0.464 for q = 0 (n = 71), and 0.772 and 0.400 for q = 1 (n = 79).
- **The H7 correlation.** r = −0.986 is computed over s_ID(·, 29) values that span 0.045 around zero (+0.017, −0.028, −0.015).
- **s_ID(LETTERS-AFTER, 29) is undetermined.** It is +0.098 with CI [−2.155, +1.750], because its denominator ID_K + ID_V is 0.021 nats (our arithmetic).
- **Per-pair Gate b0 noise.** Single pairs reach 1.5 nats (BIND NO-MENTION, |m(r0) − m(B)|), although every cell mean is ≤ 0.203 *(recomputed)*.

---

## P-2026-10-08-I: GPU stage 7, blocking the reader heads while applying the released remap at Mistral-Small-24B (paper v3)

**Final.** Fixed in the commit titled "Finalise preregistration I", together with the scoring script `analysis/stage7_score.py` (with `analysis/stage7_parts/`), the stage-7 code (`ckeys/readerblind.py`, `experiments/stage7_link.py`) and `scripts/gpu_stage7.sh`, before any stage-7 GPU run; the GPU script refuses a draft entry, a modified tree or code that differs from that commit, and runs the FP32 unit tests (Gate I-G0) before it loads any model. Results go into the revision of paper v3; `paper/versions/paper2_v3.pdf` is not changed. The draft was committed at 7a95c53 and b8ea647; the changes since are clarifications found in review (Gate I-G0 checks 7 and 9 now state which positions each blocking leaves unchanged, since B_x changes the key that position p sees) and the handling of a deadline stub on a rerun. **Seen before finalisation:** every committed 24B number quoted below (stages 2, 3b and 4), the 7B head results of stage 6, and a Monte-Carlo power check run on the committed stage-3b per-core rows (see Power). No head-level output of any model at 24B has been seen. The only runs of the stage-7 code were the unit tests (full suite 126 passed, 17 skipped without a copy of the predecessor's release; 43 passed with it), `TEST_MODE` plumbing runs (Qwen2.5-0.5B, FP32, random bases, two stories) and code-review checks on the same model and on a tiny random-weight Mistral; no trained model larger than 0.5B. In the last `TEST_MODE` run every prediction was NOT EVALUABLE (I-G1 (a) has no stage-3b reference for that model, and I-G3 (1) fails because random bases give D ≈ 0), with provenance and population checks OK and I-G0 MET; no other pilot was run.

**Context.** An external review of paper v3 notes that its two halves are linked only by inference. The reader heads (preregistration H, part (a): about 5 % of heads, ranked by a3, read the writing token's key at the re-mentioned option words; H1–H5 met) were shown at Qwen2.5-7B and Mistral-7B. The intervention result (the released learned remap M of our predecessor, Anonymous (2026), is carried by the writing token's key or value depending on the readout format; E4 and the refit F) was shown at Mistral-Small-24B and Qwen2.5-72B. The paper says "That the readers are the same at 24B and 72B is an inference" (`results_intervention.tex`). This stage tests the link directly at Mistral-Small-24B. Step 1 finds the reader heads there. Step 2 applies M and its PCA control P exactly as in stage 3b and blocks those heads at the option words. **Prediction (H_link):** with the readers blocked, the remap's exchanged key no longer reaches the answer; under the formats with later mentions M's key share falls toward its NO-MENTION value, the value channel carries more of M's effect in nats, and M's behavioural effect is roughly kept; random head sets of the same size change nothing. Qwen2.5-72B is not run (two GPUs, 145 GB of weights); there the link stays an inference.

**Reference numbers (Mistral-Small-24B, stage 3b, `results/gpu_stage3b/STAGE3B_SCORE.txt`, n = 96 cores × 3 seeds).** φ, ψ_K and ψ_V are from the score file; κ = ψ_K/(ψ_K + ψ_V) and the nat-scale means are *(recomputed)* with the frames scorer's per-core rows (10,000-resample core bootstrap). D = mean m(M) − m(P), K = mean m(P + K_M) − m(P), V = mean m(P + V_M) − m(P), all in nats; the non-additive share is 1 − ψ_K − ψ_V.

| Format (code arm) | φ | ψ_K | ψ_V | κ | D | K | V | non-additive |
|---|---|---|---|---|---|---|---|---|
| OPTIONS-AFTER (P1) | 0.705 [0.677, 0.732] | 0.528 [0.489, 0.565] | 0.229 [0.216, 0.243] | 0.697 [0.680, 0.713] | 13.70 | 7.24 | 3.14 | 0.242 |
| LETTERS-AFTER (LETTER) | 0.749 [0.712, 0.784] | 0.777 [0.735, 0.817] | 0.018 [0.013, 0.023] | 0.977 [0.970, 0.984] | 12.87 | 10.00 | 0.23 | 0.205 |
| SENTENCE-AFTER (POST) | 0.719 [0.691, 0.746] | 0.196 [0.160, 0.231] | 0.523 [0.503, 0.543] | 0.272 [0.233, 0.310] | 12.17 | 2.38 | 6.36 | 0.282 |
| NO-MENTION (NONE) | 0.708 [0.681, 0.733] | 0.069 [0.058, 0.080] | 0.803 [0.777, 0.827] | 0.079 [0.067, 0.091] | 14.34 | 0.99 | 11.51 | 0.128 |
| LIST-BEFORE (BEFORE) | 0.738 [0.712, 0.763] | −0.001 [−0.012, 0.010] | 0.971 [0.959, 0.984] | −0.001 [−0.013, 0.010] | 12.13 | −0.02 | 11.78 | 0.030 |

Stage 4's same-stack anchor (released bases, transformers 5.9.0) reproduced every φ, ψ_K and ψ_V to 0.001. In the stage-2 natural factorial at 24B (n = 150), the natural key clamp d_K = m(K_S) − m(ID) is 17.10 / 17.04 / 3.07 nats from 0-based layer 0 / 2 / 12 under OPTIONS-AFTER, 18.70 / 18.54 / 2.92 under LETTERS-AFTER, 8.15 / 8.43 / 0.86 under SENTENCE-AFTER and 2.00 at every onset under NO-MENTION *(recomputed)*. Most of the 24B key read therefore lies in layers 2–11; a NO-MENTION-sized part (about 2 nats) sits in layers ≥ 12 and needs no later mention. The remap shows the same: its key-only effect is 0.99 nats under NO-MENTION against 7.24 / 10.00 under OPTIONS-AFTER / LETTERS-AFTER, so if that route persists with later mentions, about 0.14 / 0.10 of M's key-only effect is read outside the option rows (our arithmetic).

**Runs:** `scripts/gpu_stage7.sh` on one A100-80GB with transformers 5.18.0. The model is Mistral-Small-24B-Instruct-2501 at revision 9527884be6e5616bdd54de542f9ae13384489724 (that of stages 2–4), in BF16, with use_cache=False. The ranking passes use eager attention; every evaluation pass uses sdpa, as in stage 3b. The tokenizer check of `experiments/paper1_frames.py` (fix_mistral_regex installed; ids identical with and without it) runs on every ranking and evaluation prompt. Our predecessor's reviewer repository is checked before any model is loaded: the release manifest with `refit_remap.tolerant_verify`, which also pins the sha256 of `RELEASE.json` itself (2dea297d…, `refit_remap.PINNED`, as recorded in `results/gpu_stage4/preflight/INTEGRITY.json`); the nine Mistral `original_1000` bases against the sha256 recorded in the provenance of `results/gpu_stage4/frames/mistral.json`; and `native_story_120.json` against the sha256 listed in that manifest (22af9f5b…, hard-coded in the stage-7 code, which a unit test checks against the manifest). The steps, each kept or redone as in stage 6:
1. preflight (tokenizer only; it stops the run if any ranking story or evaluation core has B, S (or T) encodings of different lengths, which rank cannot use and the population check would reject);
2. rank (step 1, phase 1, eager);
3. family: `experiments/paper1_frames.py --model mistral`, unchanged, all five formats (the reproduction of stage 3b);
4. gate (step 1, phase 2, sdpa);
5. link (step 2, sdpa);
6. remap ranking (exploratory, eager, last).

**Scoring:** `analysis/stage7_score.py` writes `STAGE7_SCORE.txt`. Outputs of `TEST_MODE` carry the tag TEST_ and go to a scratch directory outside the repository. **Statistics:** means over the 96 cores, seeds averaged within core, as in stages 2–4. Intervals are 95 % percentiles of 10,000 core-bootstrap resamples from one fixed index set (seed 20261008). The formats share the cores and are resampled jointly. Every ratio of means, κ, closure and t is recomputed within each resample. Verdicts (MET, NOT MET, NOT EVALUABLE) use the point estimates and the bounds named. Evaluability is decided only by the code gate and by gates on unblocked rows (I-G0 to I-G3); a quantity that becomes undefined because of the blocking itself counts against the prediction (NOT MET, with the reason printed), never as NOT EVALUABLE.

### Definitions

**Material.** E is the 96 cores of our predecessor's `native_story_120.json` with B, S and T distinct (stages 2–4). M is the released `original_1000` m3 basis and P the pca basis, seeds 101–103, each M paired with the P of its own seed. M and P patch the output of 0-based layer 3 (1-based block 4, `FIT_LAYER0`) over the event span. The writing token p is the critical location word; its key or value is exchanged from 0-based layer 5 (1-based block 6, `FIRST_EXCHANGE0`). Prompts use the "Answer:" prefill. m = log p(T) − log p(S) at the answer position over the full vocabulary; under LETTERS-AFTER the letters of T and S are used. φ, ψ_K, ψ_V, ρ_K and ρ_V are as in stage 3b (`analysis/stage3b_score.py`).

The ranking set R is make_cores(60, Random(0)), the ranking stories of stage 6. It is disjoint from E: 0 of the 60 share the eight story fields, or object, initial and distractor locations, B and S, with any E core *(checked)*. 36 of the 96 E cores have the distractor location equal to B or S, which make_cores excludes; the head gate (I-G2) checks that the ranking transfers.

**Option rows G_f.** These are the six rows of the location words in the re-mention span after p, in canonical order:
- OPTIONS-AFTER: the "Choices:" list;
- LETTERS-AFTER: the location words of the lettered list (the letter rows are exploratory);
- SENTENCE-AFTER: the room sentence.

Under NO-MENTION, G is empty and every blocking is the identity. Under LIST-BEFORE, the list rows precede p, so the blinding and the knockout there are a structural check (I-G1 (d)).

**What the remap changes before layer 5.** M patches the output of layer 3, so from layer 4 on the M and P runs differ at p and at the other event-span tokens; in native core 0 eight patched tokens follow p *(checked)*. The exchange rows move only p's key or value from layer 5, so no head below layer 5 ever sees M's *exchanged* key. Heads below layer 5 are not blind to the remap, though: in the M row, the layer-4 heads at G see M's own key and value at p, and every head from layer 4 on can read the patched span tokens after p. These reads are part of D and of the non-additive share (0.242 / 0.205 under OPTIONS-AFTER / LETTERS-AFTER, against 0.128 under NO-MENTION), not of K or V. Operation A+ below covers layer 4; mean-ablation of a head set covers every column those heads read.

**Step 1, the readers at 24B.** The natural clamp sets K_S at p from layer 5 on (the exchange onset). a3(l, h) is defined as in H and computed on R_f per format, with eager attention.
- Heads in layers 0–4 have a3 = 0 by construction and are not eligible, which leaves 35 × 32 = 1120 eligible heads. At 7B, 7 of the top 40 and 2 of the top 52 heads lay in layers 0–4 *(recomputed)*, so a layer-0 ranking could fill H* with heads the exchange cannot reach.
- H*_f is the top-k* heads by mean a3 over R_f, with k* = ceil(0.05 × 1280) = 64. This is H's 5 % of all heads (5.7 % of the eligible).
- Random sets: the first 64 heads of three fixed permutations of the eligible heads (numpy default_rng(2)), the same three in every format. The active-at-G control is H's: the 64 eligible heads with the largest mean o_proj-input norm at G_f over R_f, outside the top 128 by a3.
- On E (sdpa), d_full, d_G, R(k) and KO(k) are computed as in H, with K_S from layer 5, over k ∈ {8, 16, 32, 64, 128}. The all_G and all_T rows cover every head of layers 5–39. The natural clamp from layer 5 is also run under NO-MENTION on E, giving d_full(NONE), so the mention-specific ratio d_G / [d_full − d_full(NONE)] can be printed beside d_G / d_full.
- One set is ranked per format, because the re-mention rows differ by format. The overlap of the sets and the cross-applied OPTIONS-AFTER set are exploratory.

**Step 2, operation (i): exchange blinding B_x(S)** (I1–I3). HeadSplice (`ckeys/headsplice.py`, unchanged) on the P_s-patched run. Every head in every row sees M_s's key at p in layers ≥ 5, except the heads of S in the rows G, which see P_s's own key K_P. The key at p seen outside the splice is pinned to the host's captured key from layer 5 on, so these heads see exactly K_P, not a key that has evolved under p's own read of K_M. The heads of S in the rows G therefore do not read the exchanged key directly. They still read p's value and the rows between p and G, which have read K_M (p's own row and every row outside G see it from layer 5), so g_K and KO_x are shares of the direct (one-hop) read at G, as d_G and KO were in H.
- B_x(∅) is the frames' key-only exchange P + K_M. B_x(all@G): only the rows outside G see K_M.
- With a(S) = m(B_x(S)) − m(P):
  - g_K = 1 − mean a(all@G) / mean a(∅), the share of M's key-only effect read directly at G; the mention-specific g̃_K = [mean a(∅) − mean a(all@G)] / [mean a(∅) − K(NONE)], with K(NONE) the in-run NO-MENTION key-only effect, is printed beside it;
  - KO_x(S) = [mean a(∅) − mean a(S)] / [mean a(∅) − mean a(all@G)], the share of the direct G-row read that passes through S (H's KO for the remap);
  - r_K(S) = mean a(S) / mean a(∅) and ψ_K^x(S) = mean a(S) / D are printed.
- B_x leaves M and P unchanged, so behaviour is unchanged by construction.

**Step 2, operation (ii): reader ablation A(S)** (I4–I7). In every row of the family batch (per seed P, M, P + K_M, P + V_M, M + K_P and M + V_P; then the natural B, S and T), the output of each head of S in the rows G (its o_proj-input slice) is replaced by its mean over R at the same option word, taken from the natural B runs in the frames encoding. This is H3's operation (HeadSplice "ablate"), applied in the remap's rows; at 7B it kept the base answer in 0.87 and 1.00 of stories. It removes everything those heads read at G: p's key and value, the patched span tokens after p, and every other column. A+(S) is A(S ∪ L4), with L4 the 32 heads of layer 4 at G. Positions ≤ p compute identically under A and A+, so the exchanged K/V tables are unaffected.

**Step 2, operation (iii): reader knockout N(S)** (reported contrast, no verdict). In every row of the same family batch, the heads of S cannot attend from the rows G to p: their attention weight on column p in the rows G is set to exactly 0 and the row renormalised, which is the stage-5 knockout restricted to those heads. Every other head, row and edge is unchanged. Its prior is G7: the all-heads knockout M1 lost the base answer at Qwen2.5-7B (acc_B 0.37 / 0.30 under LIST-AFTER / OPTIONS-AFTER; the answer moved to the initial location in 0.63 / 0.69 of cores) and kept it at Qwen2.5-14B; the paper does not reconcile this with H3. N's lines are therefore reported two-sided.

**Step-2 statistics** (for O ∈ {A, A+, N}, set S, format f; the ∅ condition is the unblocked batch):
- D^O(S) = mean[m(M^O) − m(P^O)], K^O(S) = mean[m(P^O + K_M) − m(P^O)], V^O(S) = mean[m(P^O + V_M) − m(P^O)], all in nats, with K^∅, V^∅ and D^∅ from the ∅ batch.
- Transfer kept: t(S) = D^O(S) / D^∅. φ^O on the blocked natural rows is secondary.
- Shares on the unblocked scale: ψ̃_K(S) = K^O(S) / D^∅ and ψ̃_V(S) = V^O(S) / D^∅. Shares on the blocked scale: ψ_K^O = K^O / D^O and ψ_V^O = V^O / D^O, printed beside them, with the non-additive share 1 − ψ_K^O − ψ_V^O.
- Key share κ^O(S) = K^O / (K^O + V^O), subject to the κ rule. Closure c_κ(f) = [κ^∅(f) − κ^O(f)] / [κ^∅(f) − κ^∅(NONE)].
- Key closure on the unblocked scale: c̃_K(f) = [ψ_K^∅(f) − ψ̃_K(f)] / [ψ_K^∅(f) − ψ_K^∅(NONE)].
- Value gain: Δ_V(S) = V^O(S) − V^∅, paired by core, in nats. Floor: F_V(f) = 0.25 × [V^∅(NONE) − V^∅(f)], in-run (2.09 nats under OPTIONS-AFTER and 2.82 under LETTERS-AFTER at the stage-3b values). The ratio Δ_V / (K^∅ − K^O), the share of the lost key effect that reappears in the values, is printed.
- Argmax rates over the six candidates (or letters), for the M, P and natural B rows under ∅, A(H*), A+(H*) and N(H*): M's T-rate, P's S-rate, B's B-rate, and in each row the rate on the initial location, so that a fall in t caused by answers moving to the initial location is visible.
- κ rule (H's, adapted): κ is defined when ψ_K + ψ_V ≥ 0.3 and both are ≥ −0.1 on the scale of its own condition, and that condition's D ≥ 3 nats. Within the bootstrap the rule is applied to every κ that enters a statistic (κ^∅(f), κ^∅(NONE) and κ^O(f)); resamples that fail are dropped. Resamples in which κ^∅(f) or κ^∅(NONE) fails the rule are counted separately: if they exceed 5 %, I4 is NOT EVALUABLE in f, because the failure is in unblocked rows. Otherwise a CI does not meet a bound when more than 5 % of the resamples are dropped because κ^O fails while both unblocked κ are defined.

### Gates

**I-G0, exactness** (FP32, CPU, Qwen2.5-0.5B, random bases; `tests/test_stage7_link.py`, 1e-4 nats in the logits). Every test must pass and none may be skipped. The checks:
1. B_x(∅), A(∅) and N(∅) equal the unblocked rows.
2. HeadSplice with every head in every row from layer 5, given M's key on the P run, equals the P + K_M row of `paper1_frames.run_family`; with every head in the rows G only, it equals RowSplice(G) with K_M; B_x(all@G) equals P + K_M with RowSplice(G) giving the rows G the K_P table from layer 5 (the key pin).
3. A(S) with each head's own o_proj input as its "mean" equals the unablated row; A+ masks exactly the 32 heads of layer 4 and S at G.
4. N(all heads) with the patch and exchange hooks active equals the stage-5 4D knockout mask on G × {p}; a null knockout (second pass, no edge removed) equals the unblocked run.
5. A mixed batch equals its rows run singly, for B_x, A and N.
6. The stage-7 natural passes and patches equal `run_core`'s.
7. Every position < p is identical under every B_x, A and N, and position p under every A and N; under B_x, position p equals the row in which p sees the same key (x_all for the x_ rows, P for the s_ rows).
8. The pinned stories file hash equals the manifest entry; the base hashes equal the stage-4 provenance.
9. The run's mean-ablation path (the ranking pass's capture of the o_proj inputs at G, the means MU, the mean table, A) leaves the natural B row unchanged when the means are that story's own values, and zero means change it. The hooks also hold at the 24B head geometry (heads × head_dim ≠ hidden size) on a tiny random-weight Mistral: this path, B_x(all) = P + K_M, A with each head's own input, N(all heads) = the 4D knockout, and positions ≤ p unchanged under A and N.

I-G0 failing makes I1–I7 NOT EVALUABLE.

**I-G1, reproduction and floors:**
- (a) The family run reproduces the stage-3b table above: |Δφ|, |Δψ_K| and |Δψ_V| ≤ 0.03 in each of the five formats (E3's tolerance; the change is from transformers 5.9.0 on 2 GPUs to 5.18.0 on 1).
- (b) The in-run ∅ rows of the link batches reproduce the family's ψ_K and ψ_V within 0.03 and its D within 3 %; mean a(∅) / mean[m(M) − m(P)] of the B_x batch (its own M and P rows) reproduces the family's ψ_K within 0.03.
- (c) Knockout floor (gates the N lines only): the null knockout gives |κ^null − κ^∅|, |ψ̃_V^null − ψ_V^∅| and |t(null) − 1| ≤ 0.02. The κ part is waived when κ is undefined under the κ rule in both the null and the unblocked batch; it fails when κ is undefined in only one of them.
- (d) Structural check on the box: under LIST-BEFORE, every B_x and N row blocked at the list rows with H*_OPTIONS-AFTER is within 0.1 nats of its unblocked row in every core (H's tolerance for its on-box exactness row); the maximum difference is printed. Failing (d) makes I1–I7 NOT EVALUABLE: the blocking code then acts on rows it cannot reach.

**I-G2, the readers at 24B** (per format, on E; point estimates):
- (a) mean d_full ≥ 3 nats with CI excluding 0 (SENTENCE-AFTER: > 0).
- (b) d_G / d_full ≥ 0.7 under OPTIONS-AFTER and LETTERS-AFTER (stage 6, OPTIONS-AFTER only: 0.926 at Qwen2.5-7B / 0.982 at Mistral-7B; LETTERS-AFTER was not run at 7B), and ≥ 0.5 under SENTENCE-AFTER (Qwen2.5-7B / Mistral-7B: 0.57 / 0.73). The threshold is 0.7, not H's 0.8, because a NO-MENTION-sized key read (about 2 nats, stage 2) sits outside the option rows at 24B; d_G / [d_full − d_full(NONE)] is printed.
- (c) R(k*) ≥ 0.7 and KO(k*) ≥ 0.8, with the means over the three random sets R_rand(k*) ≤ 0.25 and KO_rand(k*) ≤ 0.25 (stage 6, OPTIONS-AFTER, Qwen2.5-7B / Mistral-7B: R 0.967 / 0.942, KO 0.977 / 0.971, random sets ≤ 0.011).
- (d) BF16 floor of the HeadSplice rows, H's Gate a2: mean |m(none) − m(clean pass)| and mean |m(all_T) − m(full clamp pass)| ≤ max(0.5 nats, 0.02 × mean d_full).

Its outcome is also printed as "H1/H2 pattern at 24B: replicated / not replicated", without a verdict count.

**I-G3, the remap acts** (per format, ∅ condition): (first clause) D^∅ ≥ 3 nats; (second clause, for KO_x only) mean a(∅) − mean a(all@G) ≥ 1 nat.

**I-G4, reading I5 and I6 as a value takeover** (OPTIONS-AFTER; not a verdict gate, used like H's Gate b3): under A+(H*), t ≥ 0.6 and Δ_V > 0 with CI excluding 0. If I5 and I6 are met but I-G4 fails, the kept behaviour is attributed to unblocked layer-4 or span reads, not to the value copy.

**Evaluability.**
- I-G0 or I-G1 (d) failing makes I1–I7 NOT EVALUABLE in every format.
- I1 in format f needs I-G1 (a, b) and the first clause of I-G3 in f. It does not need I-G2, because g_K does not use H*.
- I2 and I3 need, in addition, I-G2 (a–d) and the second clause of I-G3 in f.
- I4–I7 need I-G1 (a, b), I-G2 (a–d) and the first clause of I-G3 in f. I4 and I5 use NO-MENTION as well, so they also need NO-MENTION to pass I-G1 (a, b) and the first clause of I-G3, and I4 under OPTIONS-AFTER needs κ^∅(NONE) and κ^∅(OPTIONS-AFTER) defined by the κ rule at the point estimates and, jointly, in at least 95 % of the resamples; otherwise they are NOT EVALUABLE.
- In an evaluable format, a κ^O that is undefined after blocking, or a CI with more than 5 % of resamples dropped by κ^O, means the criterion is NOT MET; the scorer prints the reason (for example "κ undefined after blocking: behaviour lost" or "interaction-carried").
- A prediction over the two confirmatory formats is MET if it is met in both, NOT MET if it is not met in at least one evaluable format, and NOT EVALUABLE otherwise. I5 and I6 are scored under OPTIONS-AFTER alone. The same rule combines the sets of I3 and I7.

### Predictions

The confirmatory formats are OPTIONS-AFTER and LETTERS-AFTER. SENTENCE-AFTER is reported against the same thresholds (I1: ≥ 0.5) without a verdict, because its power is low (see Power): the scorer says whether each threshold is met and whether SENTENCE-AFTER passes its own gates, with no MET or NOT MET.
- **I1, the remap's key is read at the option words** (D3 for the remap). g_K ≥ 0.7 with lower bound ≥ 0.6. g̃_K is printed beside it.
- **I2, through the natural readers** (H2 for the remap). KO_x(H*_f) ≥ 0.8 with lower bound ≥ 0.7. r_K(H*) and ψ_K^x(H*) are printed beside it, against ψ_K(NO-MENTION).
- **I3, specificity of the route.** KO_x ≤ 0.25 for each of the three random sets and for the active-at-G set.
- **I4, the key share falls toward NO-MENTION under A(H*).** OPTIONS-AFTER: c_κ ≥ 0.5 with lower bound ≥ 0.3 (κ^{A(H*)} ≤ 0.39 at the stage-3b values). LETTERS-AFTER: c̃_K ≥ 0.5 with lower bound ≥ 0.3; κ^{A(H*)} and c_κ are printed there two-sided, because κ is defined under LETTERS-AFTER only if the values take over (ψ_V^∅ = 0.018). I4 is a consistency check, not independent evidence: under OPTIONS-AFTER, c_κ ≥ 0.5 follows from a removed key fraction g_K × KO_x ≥ 0.73 even with the value effect fixed in nats, and c̃_K is close to g_K × KO_x × ψ_K^∅ / (ψ_K^∅ − ψ_K^∅(NONE)) whenever the ablation removes what the blinding removes. It confirms that the ablation removes M's key read as the blinding does, so that I5 and I6 are measured under an operation that blocks the readers.
- **I5, the value channel carries more of the remap** (OPTIONS-AFTER; the form of H3 (d)). Δ_V(H*) > 0 with CI excluding 0, and Δ_V(H*) ≥ F_V (about 2.1 nats). ψ̃_V, ψ_V^{A(H*)} and Δ_V / (K^∅ − K^A) are printed.
- **I6, the behaviour is kept** (OPTIONS-AFTER). t(H*) ≥ 0.75 with lower bound ≥ 0.6 under A(H*). The argmax rates, including the initial-location rate, are printed beside it. Basis: H3, the same operation, kept the base answer in 0.87 and 1.00 of stories at 7B.

  The LETTERS-AFTER lines of I5 and I6 are reported two-sided, without a verdict: a lettered answer may need the readers to map the copied location to its letter, and nothing measured so far predicts which way it goes.
- **I7, random sets change nothing.** For each random set and the active-at-G set under A, in both formats: |κ^S − κ^∅| ≤ 0.10, |ψ̃_V(S) − ψ_V^∅| ≤ 0.10 and |t(S) − 1| ≤ 0.10.

**Reported contrast, the reader knockout N(H*)** (two-sided, no verdict; needs I-G1 (c)): c_κ, c̃_K, Δ_V, t, κ^N and the argmax rates, beside the same lines under A. If t^N falls while t^A holds, the stage-5 / H3 difference recurs at the head level for the remap; this is read from the two t values and their intervals, which the scorer prints, and no threshold is set for it.

**Expected values** (the author's, not thresholds):

| Quantity | OPTIONS-AFTER | LETTERS-AFTER |
|---|---|---|
| g_K | ≈ 0.85 | ≈ 0.9 |
| KO_x(H*) | 0.9–0.95 | 0.9–0.95 |
| c̃_K | ≈ 0.9 | ≈ 0.9 |
| κ^{A(H*)} | 0.15–0.30 | (not stated) |
| Δ_V(H*) | +3 to +6 nats | (not stated) |
| t(H*) under A | 0.8–1.0 | (not stated) |

### Alternatives and what each would mean for the paper

| Outcome | Reading | Consequence for the paper |
|---|---|---|
| I1–I7 met, I-G4 passed | H_link with redundant writing | The inference sentence is replaced, for 24B only: ablating the 5 % reader heads at the option words removes M's key-only effect, moves its key share toward NO-MENTION, adds to what M's values carry and keeps the behaviour; random sets do not. 72B remains an inference. |
| I1–I7 met, I-G4 failed | The readers are the conduit, but the behaviour is kept by unblocked early or span reads | The head-level link is stated; "the values take over" is not. |
| I1–I4 and I7 met; I5 met, I6 not met | Partial takeover (G7's term) | The value channel carries more of M's effect without its readers, but not enough to keep the behaviour. |
| I1–I4 and I7 met; I5 and I6 not met | The readers are the conduit, and the copy does not take over | "Written redundantly into both channels" is qualified at the head level for OPTIONS-AFTER: the format-level crossover stands, but without its readers the remap is largely lost. |
| I4 not met under OPTIONS-AFTER because κ^{A(H*)} is undefined | Neither channel alone carries the remap once its readers are ablated, or its behaviour is lost | Counted as NOT MET; ψ̃_K, ψ̃_V and the non-additive share are reported. |
| I1 met, I2 not met (I-G2 passed) | H_other: M's key is read at the option words, but not mainly by the natural readers | The "same readers" inference fails at 24B. The exploratory remap ranking says which heads read M's key and how far they overlap with H*. |
| I1 not met | H_not_rows: M's key effect is not read directly at the option rows (it is read elsewhere, e.g. by the answer position directly, or reaches G only through other rows) | The reader-head account, as a direct read at the option words, does not explain the remap's key share at 24B. |
| I2 met, I4 not met | Backup readers: blinding H* stops the exchanged key, but ablating H* does not remove the read | Reported as compensation by other heads; the exploratory all-heads-at-G knockout is the check. |
| I3 or I7 not met | Non-specific: any 64 heads disturb the read | No head-level link is claimed. |
| I-G2 fails in a format | The 24B readers there are not a sparse 5 % set at the option rows (H1/H2 do not replicate at 24B) | I2–I7 are untestable as preregistered in that format; I1 stands; the curves are reported, and the paper says so. |
| I-G1 (a) fails in a format | The remap does not reproduce on this stack (transformers 5.18.0, one GPU) | That format is not evaluable; the stage-3b and stage-4 numbers stand as published, and stage 7 is reported as a failed reproduction there. |
| I-G0 or I-G1 (d) fails | The blocking code is not exact, in FP32 or on the box | Nothing is evaluable; the run is reported as such. |

### Power

The power check (`stage7/power2.py`, kept with the stage-7 build notes) uses the stage-3b 24B frames per-core rows (n = 96): 300 simulated replicates per cell, 1,000 bootstrap resamples each, and the decision rules above. In the worst-case noise model each blocked per-core quantity gets independent normal noise with the full per-core SD of its unblocked counterpart, i.e. no positive within-core correlation; in the 7B-like model the residual SD is 5 % of the mean numerator (at 7B, the KO numerator's per-story SD was 1.4–1.5 nats against a denominator of 26–33 nats). The table gives the probability that each prediction is met when the true value is as stated.

| Prediction | True value | OPTIONS-AFTER (worst / 7B-like) | LETTERS-AFTER (worst / 7B-like) |
|---|---|---|---|
| I1 (g_K) | 0.75 | 0.87 / 1.00 | 0.90 / 1.00 |
| I1 (g_K) | 0.85 | 1.00 / 1.00 | 1.00 / 1.00 |
| I2 (KO_x; true g_K 0.85) | 0.85 | 0.64 / 1.00 | 0.73 / 1.00 |
| I2 | 0.90 | 0.86 / 1.00 | 0.96 / 1.00 |
| I2 | 0.95 | 0.97 / 1.00 | 0.99 / 1.00 |
| I6 alone (t) | 0.80 | 0.98 | 0.98 |
| I6 alone (t) | 0.85 | 1.00 | 1.00 |

For I4–I6 the simulation removes a fraction f = 0.85 × 0.95 = 0.81 of M's key-only effect under A(H*) and lets a fraction c of the removed key effect reappear in the value-only effect (c = 0: no takeover, value effect fixed in nats). Worst-case noise; mean point estimates in parentheses:

| Scenario | True t | P(κ defined) | P(I4 met) | P(I5 met) | P(I6 met) |
|---|---|---|---|---|---|
| OPTIONS-AFTER, c = 0 | 0.57 | 1.00 | 0.95 (c_κ 0.64) | 0.00 (Δ_V 0.00) | 0.00 |
| OPTIONS-AFTER, c = 0.5 | 0.79 | 1.00 | 1.00 (c_κ 0.83) | 1.00 (Δ_V +2.92) | 0.94 |
| OPTIONS-AFTER, c = 1 | 1.00 | 1.00 | 1.00 (c_κ 0.91) | 1.00 (Δ_V +5.84) | 1.00 |
| LETTERS-AFTER, c = 0 | 0.37 | 0.96 | 1.00 (c̃_K 0.89; c_κ 0.10) | (two-sided) | (two-sided) |
| LETTERS-AFTER, c = 0.5 | 0.69 | 1.00 | 1.00 (c̃_K 0.89; c_κ 0.74) | (two-sided) | (two-sided) |
| LETTERS-AFTER, c = 1 | 1.00 | 1.00 | 1.00 (c̃_K 0.89; c_κ 0.88) | (two-sided) | (two-sided) |

So I4 is met with or without a takeover, as stated under I4, and is a consistency check. I5 and I6 separate the scenarios: neither is met without a takeover, and both are met at half or full takeover (I6 at 0.94 for c = 0.5). Under LETTERS-AFTER, c_κ would fail without a takeover, and κ approaches the rule's limit as f rises; this is why c̃_K is scored there. The I7 margins of 0.10 are more than 5 SE of κ, ψ̃_V and t. SENTENCE-AFTER (K 2.38 nats) gives P(I1 met) 0.69 at a true g_K of 0.85 and P(I2 met) 0.28 at a true KO_x of 0.9 under the worst case, which is why it is reported and not scored.

### Compute

| Part | Row-forwards |
|---|---|
| Rank (eager; 3 formats × 60 stories, + 180 for the layer-0 ranking, + 100 duplicate-score sequences) | about 720 passes |
| Family (5 × 120 × 24) | 14,400 |
| Gate on E (3 × 96 × 42, + 288 for the NO-MENTION clamp) | 12,384 |
| Link: natural and capture passes (5 × 96 × 9) | 4,320 |
| Link: B_x (96 × (54 + 57 + 54), + 864 LIST-BEFORE) | 16,704 |
| Link: A and N batches (96 × 21 × (12 + 13 + 12)), NO-MENTION ∅ (2,016), LIST-BEFORE ∅ and N (4,032) | 80,640 |
| Remap ranking (eager, exploratory) | about 7,800 passes |

That is about 128,000 sdpa row-forwards. Stage 3b ran these frames at about 0.026–0.03 s per row; with about 10 % for the second attention passes they take about 1.0–1.2 h, plus about 0.15 h of eager passes and about 0.8 h of setup (pip and pytest about 20–25 min: on a 4-core CPU the stage-7 test set took 14 min and the shared set 6.5 min; the 47 GB download about 6–10 min, five model loads about 8 min, preflight, scoring and archive about 5 min). That is about 2–2.5 GPU-hours on one A100-80GB, with a 4 h cap; at about $2 per A100-80GB hour (the runbook's rate) about $4–6, at most $10. The box needs ≥ 64 GB host RAM and ≥ 100 GB disk. `TEST_MODE` (Qwen2.5-0.5B, FP32, CPU, random bases from `--bases-override 896`, n = 2 cores, R = 2 stories, k* fraction 0.012, so k* = 5 of 336 heads, grid 1, 2, 4, 5, 8; three random sets; L4 = the 14 heads of layer 4) takes about 20 min of pytest plus about 15 min of runs on a 4-core CPU.

**Deadline.** The pipeline sets a deadline DEADLINE_H = 3.5 h after its start, so that the run stays within the 4 h cap. The link step checks it at the start of each format: once it has passed, that format's exploratory parts (the KO_x(k) curve batch, with the OPTIONS-AFTER set under LETTERS-AFTER, and the conditions A(L4), N(rand0), N(allG) and A(H*_OPTIONS-AFTER)) are skipped for every core, so each exploratory batch is complete or absent. The remap ranking checks it once, at its start, and is then skipped as a whole. No confirmatory batch is ever skipped. Every skip is recorded in the provenance of the results file and listed in the scorer's exploratory report; a later session may fill the skipped parts (the remap ranking is redone automatically, link with `FORCE_STEPS=link`), and the report then says so through the files' provenance.

**Exploratory** (subject to the deadline above):
- **SENTENCE-AFTER:** all lines.
- **Removal through the readers:** M with H* at G seeing P's key, ρ_K^x(H*), against ρ_K = 0.762 / 0.986.
- **Sufficiency:** R_x(H*) = [m(H*@G only) − m(P)] / [m(all@G only) − m(P)].
- **Curves:** KO_x(k) over k ∈ {8, 16, 32, 128}.
- **Remap ranking:** a3 for the remap on E, eager, per format: a3^rem(l, h) = ½[(A^{P+K_M} − A^P)[rowT → p] + (A^P − A^{P+K_M})[rowS → p]], averaged over seeds and cores, eligible heads only. Its top-64 overlap with H* (Jaccard and the hypergeometric P, as in H4) and KO_x of that set (in-sample on E, with its own in-batch P, x_all and x_notG rows).
- **Knockout of every head at G:** the stage-5 M1 for the remap (N(allG)).
- **Layer 4 alone:** A(L4).
- **The knockout of a random set:** N(rand0).
- **Head sets:**
  - the OPTIONS-AFTER set applied under LETTERS-AFTER (A);
  - the overlap of the sets across formats;
  - a3 from layer 0 (H's ranking) and its overlap with H*;
  - the layer profile of H*.
- **Duplicate and induction scores of H* at 24B:** H4 at 24B, on 100 random sequences.
- **Letter rows under LETTERS-AFTER.**
- **Breakdowns:** per seed; per core for the 36 E cores whose distractor location is B or S, against the rest.
- **Not run:** the f_star bases under the blocking, and Qwen2.5-72B.

## Outcome of P-2026-10-08-I (GPU stage 7; scored by analysis/stage7_score.py at the finalising commit 35df52f, which the run used)

**Run.**
- **Hardware and software:** one A100 80GB PCIe, Python 3.12.14, torch 2.11.0+cu128, transformers 5.18.0, numpy 2.5.3. Mistral-Small-24B-Instruct-2501 at the pinned revision 9527884b…, BF16, eager attention for the ranking passes and sdpa for every evaluation pass, as specified. The run used a clean checkout of 35df52f (`COMMIT.txt`: only `results/gpu_stage7/` and the archive untracked) with `TEST_MODE=0`.
- **Two sessions on one box.**
  - The first session (2026-10-09T11:24:59Z) stopped about a minute in, after the environment setup and before the unit tests, the release check and any model: the download of our predecessor's released reviewer repository from anonymous.4open.science returned HTTP 403 on that host (`results/gpu_stage7_attempt1/`). The same URL served the release from another network the same day.
  - The second session (11:39:09Z) used a copy of the same release, downloaded in a browser and uploaded to the box at the script's default path (`~/paper1/v5.5-reviewer-repository`). The run's release check found it equal to the pins before any test or model: `RELEASE.json` sha256 2dea297d… (`refit_remap.PINNED`), status PASS_EXCEPT_README as in stage 4, the nine `original_1000` bases equal to the stage-4 provenance and `native_story_120.json` equal to the manifest (22af9f5b…) (`RELEASE.txt`). The second session moved the first session's `FAILED.txt` aside (`FAILED.20261009T113909Z.txt`), as the script does on every start.
- **Order (second session):** the stage-7 unit tests (`log_pytest_stage7.txt`, 11:39:21Z: 73 passed, 1 skipped, the test that needs a cached 24B tokenizer, which the preflight then checks; Gate I-G0, `tests/test_stage7_link.py`, 17 passed), the shared-module tests (`log_pytest.txt`, 15 passed), preflight (11:50:55Z; 60 ranking stories, 96 cores, 1,800 prompts checked), the weights (11:51:08Z), rank (11:51:23Z, eager, 75 s), family (11:52:51Z), gate (12:01:39Z, 431 s), link (12:09:02Z, 5,701 s), the exploratory remap ranking (13:44:17Z) and the score (14:02:51Z). About 2.4 h from the start of the second session to the score.
- **What failed:** no step of the second session failed (no `FAILED.txt`, no `FAILED_EXPLORATORY.txt`). No exploratory part was skipped at the deadline.
- **Provenance and population:** every results file carries commit 35df52f, the pinned revision and one device; the head-set hash (ec052231…) and the means hash (MU 82a1e28a…) recomputed by the scorer equal those recorded by every later file. E is 96 cores in every format and condition, in the same order; R is make_cores(60, Random(0)) in each ranking format and shares no story with E. The scorer prints "provenance OK; population OK".
- **Re-score:** re-scoring the archive off the box with the committed scorer reproduces `STAGE7_SCORE.txt` byte for byte after its first line, which names local paths.
- **What changed between 35df52f and the results commit (51875ea):** only files under `results/` (the attempt-1 logs, 5dcfe79, and the results, 51875ea). No scorer, experiment, test, script or `ckeys/` file changed, and `docs/PREREGISTRATION.md` at 51875ea is byte-identical to 35df52f (sha256 3928cfa7…). This outcome is appended after the entry; the entry's text is unchanged.
- **Independent checks:** four recomputations from the raw files (`heads/rank.json`, `heads/gate.json`, `link/mistral.json`, `frames/mistral.json`), none importing the scorer, reproduce every gate value, point estimate and verdict to the printed precision; their bootstrap intervals agree with the scorer's to within 0.006. They also rebuild H* (the top 64 eligible heads by mean a3, in order, with no tie at rank 64), the random, active and layer-4 sets, the layer histograms and the overlaps between formats. A fifth check read the scorer against the entry: it applies each threshold, lower bound, κ rule, evaluability rule and combining rule as written. The one problem found is in the experiment code (the k = 128 curve point; see Deviations).
- **Summary line:** 6 MET (I1–I5, I7), 1 NOT MET (I6), 0 NOT EVALUABLE of 7.

Numbers below are from `STAGE7_SCORE.txt` unless marked. *(recomputed)* means recomputed from the raw files and not printed in the score file; "our arithmetic" marks a difference or ratio of printed values. Pairs are OPTIONS-AFTER / LETTERS-AFTER, the confirmatory formats; triples add SENTENCE-AFTER, which is reported without a verdict. Code arms: OPTIONS-AFTER (P1), LETTERS-AFTER (LETTER), SENTENCE-AFTER (POST), NO-MENTION (NONE), LIST-BEFORE (BEFORE). H* is the top 64 of the 1120 eligible heads (layers 5–39) by a3, 5 % of the 1280 heads.

### Gates

- **I-G0 (FP32 exactness): met.** `tests/test_stage7_link.py`: 17 passed, 0 failed, 0 skipped in the only run in `log_pytest_stage7.txt` (11:39:21Z, commit 35df52f). The same run: 73 passed, 1 skipped (`tests/test_refit_remap.py::test_fixed_tokenizer_detects_skipped_fix`, which needs a cached 24B tokenizer; the preflight then checks the tokenizer on every prompt). Shared modules: 15 passed.
- **I-G1 (a) (reproduction of stage 3b): met in all five formats, exactly.** Every φ, ψ_K and ψ_V equals the stage-3b value (difference +0.000). *(recomputed)* All 14,400 candidate rows of `frames/mistral.json` equal those of the stage-3b file bit for bit, although stage 3b ran on transformers 5.9.0 over two GPUs and stage 7 on transformers 5.18.0 on one; the provenance differs, the logits do not.
- **I-G1 (b) (in-run rows): met in all five formats.** The unblocked rows of the link batches give ψ_K and ψ_V within 0.002 of the family run and D within 0.2 %; the B_x batch's own a(∅)/(M − P) reproduces ψ_K (0.528 / 0.776 / 0.197).
- **I-G1 (c) (knockout floor): met.** The null knockout equals the unblocked batch (|κ^null − κ| 0.0000, |ψ̃_V^null − ψ_V| 0.0000, |t − 1| 0.0000 in each format).
- **I-G1 (d) (structural check, LIST-BEFORE): met.** Blinding and knocking out H*_OPTIONS-AFTER at the list rows, which precede p, changes no row: maximum difference 0 nats over 288 B_x rows and 2,016 N rows *(recomputed)*. This holds by construction; under LIST-BEFORE the key-only effect is +0.008 nats, so the B_x half has little power, and the N half (ψ_V 0.97) carries the check.
- **I-G2 (the readers at 24B): met in all three formats ("H1/H2 pattern at 24B: replicated").**

  | Quantity | OPTIONS-AFTER | LETTERS-AFTER | SENTENCE-AFTER | Threshold |
  |---|---|---|---|---|
  | (a) d_full (nats) | 17.020 [16.475, 17.568] | 18.186 [17.818, 18.547] | 8.686 [7.973, 9.353] | ≥ 3 (> 0) |
  | (b) d_G/d_full | 0.853 [0.840, 0.867] | 0.987 [0.981, 0.992] | 0.682 [0.635, 0.728] | ≥ 0.7 (≥ 0.5) |
  | mention-specific d_G/(d_full − d_full(NONE)) | 0.984 | 1.127 | 0.921 | printed |
  | (c) R(64) | 0.832 [0.788, 0.871] | 0.822 [0.775, 0.865] | 0.912 [0.878, 0.942] | ≥ 0.7 |
  | (c) KO(64) | 0.971 [0.964, 0.978] | 0.991 [0.989, 0.994] | 1.012 [1.003, 1.022] | ≥ 0.8 |
  | (c) R_rand / KO_rand (mean of 3) | 0.001 / 0.011 | 0.001 / 0.013 | 0.002 / 0.037 | ≤ 0.25 |
  | (d) \|none − clean\|, \|all_T − full\| | 0.061, 0.075 | 0.055, 0.061 | 0.045, 0.089 | ≤ 0.5 |
  | k80 | 64 | 64 | 32 | printed |

  d_full(NONE), the natural clamp from layer 5 under NO-MENTION, is 2.26 nats *(recomputed)*; under LETTERS-AFTER only 0.24 nats of the read lies outside G, so subtracting the NO-MENTION clamp over-corrects and the mention-specific ratio exceeds 1. R(64) is lower than at 7B (0.967 / 0.942 with k* = 40 / 52), and KO(64) is as high.
- **I-G3 (the remap acts): met.** D^∅ is 13.67 / 12.87 / 12.16 nats and 14.35 under NO-MENTION; the direct read at G, a(∅) − a(all@G), is 7.04 / 10.02 / 2.16 nats.
- **I-G4 (reading I5 and I6 as a value takeover; not a verdict gate): not passed.** Under A+(H*) (H* and the 32 heads of layer 4 at G), t = 0.556 [0.534, 0.578] < 0.6; Δ_V = +2.496 [+2.250, +2.737]. I6 is not met either, so the gate's rule (I5 and I6 met but I-G4 failed) does not arise; adding layer 4 to the ablation changes t by −0.009 (our arithmetic), so layer-4 reads at G carry almost none of the effect that is kept.
- **Evaluability:** every gate a verdict needs passed in both confirmatory formats, and NO-MENTION passed I-G1 (a, b) and I-G3 (1). κ^∅(NONE) and κ^∅(OPTIONS-AFTER) are defined at the point estimates and in every resample (0.0 % dropped). All seven predictions are evaluable.

### Predictions

| Prediction | Observed (OPTIONS-AFTER / LETTERS-AFTER) | Verdict |
|---|---|---|
| I1 the remap's key is read at the option words: g_K ≥ 0.7, lower bound ≥ 0.6 | 0.976 [0.963, 0.990] / 0.999 [0.997, 1.000]; g̃_K 1.133 / 1.109 | **met** |
| I2 through the natural readers: KO_x(H*) ≥ 0.8, lower bound ≥ 0.7 | 0.994 [0.989, 0.999] / 0.993 [0.990, 0.995]; r_K(H*) 0.030 / 0.009; ψ_K^x(H*) 0.016 / 0.007 against ψ_K(NO-MENTION) 0.070 | **met** |
| I3 specificity: KO_x ≤ 0.25 for each random set and the active set | 0.073, 0.059, 0.028, 0.040 / 0.082, 0.075, 0.029, −0.024 | **met** |
| I4 the key share falls toward NO-MENTION under A(H*): c_κ (OPTIONS-AFTER), c̃_K (LETTERS-AFTER) ≥ 0.5, lower bound ≥ 0.3 | c_κ 0.946 [0.905, 0.991] (κ 0.698 → 0.113 [0.080, 0.143]; NO-MENTION 0.080; 0.0 % of resamples dropped) / c̃_K 0.845 [0.818, 0.871] (two-sided: κ^A 0.661, c_κ 0.352) | **met** |
| I5 the value channel carries more (OPTIONS-AFTER): Δ_V > 0 with CI excluding 0 and ≥ F_V | +2.388 [+2.141, +2.631] nats, F_V 2.10; ψ̃_V 0.404, ψ_V^A 0.714; Δ_V/(K^∅ − K^A) 0.366 | **met** |
| I6 the behaviour is kept (OPTIONS-AFTER): t ≥ 0.75, lower bound ≥ 0.6 | t 0.565 [0.542, 0.589]; argmax rates unblocked / A(H*): M→T 0.82 / 0.80, P→S 1.00 / 1.00, B→B 1.00 / 1.00, initial location ≤ 0.01 | **not met** |
| I7 random sets change nothing: \|Δκ\|, \|Δψ̃_V\|, \|t − 1\| ≤ 0.10 for each random set and the active set | largest 0.019 / 0.019 | **met** |

**Reported lines (no verdict).**
- **SENTENCE-AFTER**, against the same thresholds, with its gates passed: g_K 0.901 [0.849, 0.946] (≥ 0.5); KO_x(H*) 1.023 [1.002, 1.048]; random and active KO_x 0.040, 0.078, 0.020, 0.003; c̃_K 1.465; Δ_V +4.220 [+3.816, +4.622] (F_V 1.29); t 0.944 [0.928, 0.960]; I7's largest deviation 0.031. Every threshold is met.
- **LETTERS-AFTER I5 and I6 (two-sided):** Δ_V +0.945 [+0.791, +1.102] nats, below the floor F_V 2.82 that applies under OPTIONS-AFTER; t 0.385 [0.358, 0.410]. Here the behaviour is lost as well as the margin: under A(H*) M's target is the argmax in 0.46 of cores (0.76 unblocked), P's source in 0.85 (1.00) and the natural answer in 0.83 (1.00). This is the case the entry left open ("a lettered answer may need the readers to map the copied location to its letter").
- **The knockout contrast N(H*)** (I-G1 (c) met): t 0.722 [0.697, 0.748] / 0.296 [0.252, 0.342] / 0.968 [0.956, 0.982], against 0.565 / 0.385 / 0.944 under A; Δ_V +5.140 / +1.746 / +4.722 nats; c_κ 1.016 / 0.569 / 1.314. Under OPTIONS-AFTER cutting H*'s edges from G to p keeps more of the remap's effect than mean-ablating H*, and the value channel takes up 0.779 of the lost key effect (0.366 under A). Under LETTERS-AFTER the knockout loses more (t 0.296; argmax rates M→T 0.39, P→S 0.64, B→B 0.69, with the answer moving to the initial location in 0.19–0.24 of rows). The entry's reading of a head-level recurrence of the stage-5 / H3 difference (t^N falls while t^A holds) applies in neither format: under OPTIONS-AFTER the knockout keeps more than the ablation (0.722 against 0.565); under LETTERS-AFTER both lose, the knockout more (0.296 against 0.385), and only the knockout moves answers to the initial location (0.19–0.24 of rows, as the all-heads knockout M1 did at Qwen2.5-7B).

**Status of the preregistered alternatives.** The applicable row of the entry's table is "I1–I4 and I7 met; I5 met, I6 not met | Partial takeover (G7's term) | The value channel carries more of M's effect without its readers, but not enough to keep the behaviour."
- **H_link with redundant writing (I1–I7 met):** not established. The head-level link is established at 24B (I1–I4, I7), and the values carry more of M's effect without the readers (I5), but the effect in nats is not kept (I6).
- **H_other (I1 met, I2 not met):** not supported. KO_x(H*) is 0.994 / 0.993; *(exploratory)* the heads ranked on the remap itself overlap H* in 57 / 55 / 59 of 64.
- **H_not_rows (I1 not met):** not supported. g_K is 0.976 / 0.999: M's key-only effect is read at the option rows. The NO-MENTION-sized route outside G that the entry expected (about 0.14 / 0.10 of M's key effect) is not seen: with every head at G blinded, a(all@G) is 0.17 / 0.01 nats *(recomputed)*, against an in-run NO-MENTION key effect of 1.00 nat, so g̃_K exceeds 1 (see Deviations).
- **Backup readers (I2 met, I4 not met):** not supported. I4 is met.
- **Non-specific (I3 or I7 not met):** not supported. Random and active sets give KO_x ≤ 0.082 and change κ, ψ̃_V and t by ≤ 0.019.

**Exploratory results (not scored).**
- **The head sets.** H* spans layers 5–37 in every format (OPTIONS-AFTER: 27 layers, at most 8 heads in one layer, layer 14). The sets overlap across formats in 59 of 64 (OPTIONS-AFTER and LETTERS-AFTER) and 43 of 64 (each with SENTENCE-AFTER; hypergeometric P 4.7e-86 and 1.2e-45). Their median duplicate-token score D is 0.104 / 0.104 / 0.108 and median induction score 0.039 / 0.041 / 0.041, within the range of the 7B full 5 % sets (median D 0.179 / 0.065). The top 64 of a ranking that includes layers 0–4 shares 56 / 57 / 59 heads with H*.
- **The heads that read the remap.** Ranking heads by a3 on the remap itself (P against P + K_M at the T and S rows, in-sample on E) gives a set H_rem that overlaps H* in 57 / 55 / 59 of 64 (Jaccard 0.80 / 0.75 / 0.86); KO_x(H_rem) is 1.003 / 0.993 / 1.034 (in-sample).
- **Further conditions under OPTIONS-AFTER.** Ablating the layer-4 heads at G alone changes nothing (t 1.006, κ 0.709); knocking out a random set changes nothing (t 0.985); knocking out every head's edge from G to p removes the key share (κ −0.009; t 0.739, Δ_V +6.22 nats). Removal through the readers ρ_K^x(H*) is 0.635 (frames ρ_K 0.762) and sufficiency R_x(H*) 0.724.
- **Splits.** g_K and KO_x(H*) are 0.961–0.989 and 0.992–0.996 across the three seeds under OPTIONS-AFTER, and 0.984 / 0.985 on the 36 cores with the distractor at B or S against 0.972 / 0.998 on the other 60.

### What I6 means

- **What t measures.** t = D^A / D^∅ is the ratio of M's effect on the margin m = log p(T) − log p(S), in nats, paired against P under the same blocking. The argmax rates printed beside it ask only whether T stays the top candidate. I6 was preregistered on t; the argmax rates do not change the verdict.
- **Where the nats go** *(recomputed, OPTIONS-AFTER)*. Writing D = K + V + I, with I the non-additive part: unblocked 13.674 = 7.222 + 3.130 + 3.322; under A(H*) 7.730 = 0.705 + 5.518 + 1.507. The key loses 6.52 nats (0.90 of K); 2.39 of them reappear in the values (0.37 of the lost key effect, I5); the non-additive part loses 1.81 nats (0.55 of I). Two-thirds of the fall in D is in the P row (m(P) −9.66 → −5.68, as log p(T | P) rises from −9.69 to −5.79) and one third in the M row (m(M) +4.02 → +2.05, with log p(T | M) unchanged at −0.70 and the source S less suppressed). Ablating the readers compresses the margins of every row, including the natural one (log p(B) − log p(S) in the B row: 11.51 → 7.12 nats, with B still the argmax in every core), without moving answers: M's target stays the argmax in 0.80 of cores (0.82 unblocked), P's source and the natural answer in 1.00, and at most 0.01 of answers go to the initial location.
- **Against the power check.** The Power table's "OPTIONS-AFTER, c = 0" scenario gives t = 0.57, the observed value. That scenario removed 0.81 of K and held V and I fixed in nats (Δ_V = 0). The data removed 0.90 of K, returned 0.37 of it in V (Δ_V +2.39) and lost 1.81 nats of I; these nearly cancel against the scenario (+0.69, −2.39 and +1.81 nats), so t lands at the no-takeover value although the value channel carries 2.4 nats more. With I held fixed, the observed f and c would give t = 0.698; the difference to 0.565 is ΔI / D^∅ = 0.133 exactly *(recomputed)*. The quantity the simulation held fixed, the non-additive part (0.24 of D), is the one that moved.
- **Reading.** The preregistered row applies: a partial takeover. Without its readers at the option words the remap's key effect is gone, and the values carry more of it, but its effect on the margin falls to 0.57 of its size, below the preregistered 0.75; its answers are kept.

### Deviations and disclosures

- **The k = 128 point of the head curves was not measured (experiment code; no gate or verdict affected).** The entry asks for R(k), KO(k) and the exploratory KO_x(k) at k ∈ {8, 16, 32, 64, 128}. `experiments/stage7_link.py` builds each curve from the stored H* list (`Hs[:k]`, lines 605–606 for the gate and 710 for the link's curve batch), which holds only the 64 heads of H*; the stored ranks 65–128 (`sets.arms[f].next`) are never used. The k = 128 rows of `gate.json` are therefore bit-identical to the k = 64 rows in every core, and the link's `x_k128` row is H* again in another batch (it differs from `x_H` by batch numerics, at most 0.37 nats). The score file prints these rows as R(128), KO(128) and KO_x(128) without saying so. k* = 64 and k80 (64 / 64 / 32) lie at or below 64, so no gate, verdict or k80 changes. The points labelled 128 are not reported anywhere, and nothing is said about whether R levels off beyond 64. Stage 6 does not have this problem (its curves slice the full ranking).
- **I-G1 (a) is exact, not within tolerance.** The family run equals the stage-3b file row for row (see Gates), as stage 4 also did. The run is fresh: its timings (98–511 s per format), tokenizer checks and provenance are its own, and the code has no cache path; the likely cause is deterministic BF16 kernels at the same torch build on the same chip. The paper says "reproduced exactly".
- **I-G1 (c) is met trivially.** The null knockout's second pass reuses the unblocked mask and reproduces the unblocked rows bit for bit, so the floor does not measure any numeric cost of the changed-mask path; it only gates the reported N lines, as specified.
- **The expected route outside the option rows is absent.** The entry estimated that about 0.14 / 0.10 of M's key effect would be read outside G (from the NO-MENTION key-only effect of about 1 nat). With every head at G blinded, 0.17 / 0.01 / 0.24 nats of M's key-only effect remain (a(all@G); a(∅) is 7.21 / 10.03 / 2.39) *(recomputed)*, against an in-run NO-MENTION key effect of 1.00 nat. This is why g̃_K exceeds 1 (1.133 / 1.109 / 1.547), g_K exceeds the expected 0.85 / 0.9, and c̃_K, c_κ exceed 1 in some blocked conditions (κ^A 0.012 under SENTENCE-AFTER and κ^N 0.070 under OPTIONS-AFTER fall below the NO-MENTION 0.080). These ratios are not bounded by 1; I4's lower bounds are unaffected.
- **R(64) is lower at 24B than at 7B on a story type the ranking set lacks** *(recomputed, exploratory)*. On the 60 cores of E whose distractor is not B or S (the story type in R, and in stage 6's E), R(64) is 0.944 / 0.942 / 0.974, as at 7B (0.967 / 0.942 under OPTIONS-AFTER); on the 36 cores with the distractor at B or S it is 0.612 / 0.602 / 0.749, while KO(64) stays at 0.956 / 0.982 / 1.006. On those stories H* is necessary but not sufficient alone. The entry anticipated a transfer check (I-G2 (c) passes on all 96 cores).
- **Ratios above 1 that are not errors.** KO(64) and KO_x(H*) under SENTENCE-AFTER (1.012, 1.023): blinding H* alone leaves slightly less key effect than blinding every head at G (0.19 against 0.24 nats *(recomputed)*), so the other heads at G read slightly against T. The mention-specific d_G ratio under LETTERS-AFTER (1.127): only 0.24 nats of the natural read lie outside G there, so subtracting the 2.26-nat NO-MENTION clamp over-corrects.
- **The random sets may contain heads of H*.** They are permutations of all 1120 eligible heads, as specified, and share 5 / 4 / 0 heads with H*_OPTIONS-AFTER *(recomputed)*, which orders their KO_x (rand0 > rand1 > rand2) and makes I3 slightly conservative.
- **Exploratory LETTERS-AFTER N:allG κ.** D under that knockout is 2.44 nats, below the κ rule's 3, so κ is undefined at the point estimate and in 99 % of resamples; the printed interval comes from the remaining 1 % and is not quoted.
- **LETTERS-AFTER I5 line.** It prints the criterion template ("> 0 (CI excl. 0) and >= F_V 2.82") without a verdict suffix, as the entry specifies for a two-sided line; Δ_V = 0.945 is below that floor.
- **Weights download.** The 47 GB snapshot took about 15 s (11:51:08 to 11:51:23), although the hub cache did not hold the model before the run (it was removed at the end). The revision is asserted by the download command and the loaded snapshot is recorded (`ENV.txt`), so provenance is unaffected.
- **The release copy.** The second session used a copy of the release that the author downloaded in a browser and uploaded to the box, because the box could not reach anonymous.4open.science. The script's checks (the pinned `RELEASE.json` hash, the nine bases and the stories file) passed before any test or model, as they do for a fetched copy.

## P-2026-10-10-J: GPU stage 8, measurement, fresh samples, natural text, interventions and mechanism (paper v5)

**Final.** Fixed in the commit titled "Finalise preregistration J", together with the four scorers (`analysis/stage8a_score.py` ... `analysis/stage8d_score.py`) and their shared Holm helper (`analysis/stage8_holm.py`), the stage-8 code and data (`ckeys`, `experiments`, `data`), the tests, `scripts/stage8_common.sh`, `scripts/stage8_models.json` (sha256 `b840b71ea644d3f01e634c67ef2c2b9bdb26cdf53d3fe6b45d34c5e78c0cb707`), `scripts/fetch_verified.py` and the four GPU scripts, before any stage-8 GPU run. Each GPU script refuses a draft entry, a modified tree, code that differs from that commit, or a J section that differs from its text there, and runs its part's FP32 unit tests before it loads any model. The build and its reviews were committed at 562e666 (build), c04b3eb (review fixes), 84e4f05 (audit fixes; the code of the final TEST_MODE runs, unchanged since) and the entry at 32d12f1 (draft). **Seen before finalisation:** see the section of that name below and in each part.

This entry has a common part (this section and the sections up to "Part A") and four parts, A to D. The common part fixes what holds for every part: the scripts, the model files, the statistics, the risk classes, the codes, the populations and the disclosure of pilots. Where a part's section and the common part disagree, the part's section states the exception and the reason. The designs, the critiques and the synthesis that amended them are kept in `docs/stage8_design/` (development record, not part of the anonymous release); the codes G1–G8 name the rules of this common part, and codes such as A-7 or C-3 in a part name the amendments of that synthesis.

### Parts and scripts (G1)

Stage 8 has four parts. They answer different objections and share no outputs:
- **Part A, natural text** (`scripts/gpu_stage8a.sh`, `analysis/stage8a_score.py`): counterfactual SQuAD v1.1 passages whose answer is a multi-token entity, in four families (two never run before): the format law, cue-conflict behaviour against a closed-book prior, a KIVI quantization difference-in-differences, and the reader heads on natural text (J-A1 to J-A8d, J-A-HA1 to J-A-HA3b).
- **Part B, fresh samples and new families, scored on the emitted forms** (`scripts/gpu_stage8b.sh`, `analysis/stage8b_score.py`): the templated factorial on fresh story cores with a second lexicon, new sentences and a null-sentence arm, in the four original and four new families, scored on the emitted forms and by generated answers; the published numbers re-scored on the original cores (J-B1 to J-B8, J-B-NULL, J-B-LB, J-B-SMALL).
- **Part C, interventions** (`scripts/gpu_stage8c.sh`, `analysis/stage8c_score.py`): identity edits obtained independently of the natural clamp (steering vectors from held-out stories and from unrelated sentences, a third-party sparse autoencoder, a rank-16 remap trained at the writing token) and a non-lexical boundary family, tested against a channel-ratio law at matched depth; a screen for a depth where the binding swap and the key route overlap (J-C1 to J-C6, J-C-BOUND, J-C-READa, J-C-READb, J-C-SCREEN, J-C-WIN).
- **Part D, mechanism** (`scripts/gpu_stage8d.sh`, `analysis/stage8d_score.py`): what the reader heads write: a per-layer flag tested for sufficiency, necessity and address-only content, a binding test, the signs of the key and value reads under a 'not mentioned' question and in IOI with their route, and the 1.5B/3B route (J-D1 to J-D8, J-D6a, J-D-ADDR, J-D-KN, J-D-SIGN-Q, J-D-SIGN-IOI, J-D-ROUTE-IOI, J-D-HOP2, J-D6-ROUTE).

The parts are independent. The user may run them in any order, one after another on one box, or on different boxes. Each part writes `results/gpu_stage8<part>/`, its archive `gpu_stage8<part>_results.tgz` and its score `STAGE8<PART>_SCORE.txt`. No part reads another part's results, and no verdict of one part depends on another part's run.

**The shared pipeline** (`scripts/stage8_common.sh`, sourced by the four scripts). Outside TEST_MODE it does the following, in this order:
1. It refuses to start (exit 1, logs archived) if any of these holds:
   - `docs/PREREGISTRATION.md` has no `## ` heading that contains P-2026-10-10-J;
   - this section still has its draft marker, a line that begins with the bold words of the draft notice (the check reads the section whose `## ` heading contains P-2026-10-10-J, with awk alone; the words quoted inside a line do not count);
   - tracked files have local changes;
   - the history of HEAD has no commit whose subject is exactly "Finalise preregistration J";
   - `ckeys`, `experiments`, `analysis`, `scripts`, `tests` or `data` differ between HEAD and the latest such commit;
   - this entry's own section of `docs/PREREGISTRATION.md` differs from its text at that commit. An outcome section added later under its own `## ` heading is allowed.
2. It pins transformers 5.18.0 and runs pip only when the installed version differs or a needed package is missing. It records the commit (`COMMIT.txt`), the environment (`ENV.txt`, `PIP_FREEZE.txt`) and the manifest's sha256 (in `ENV.txt` and at the head of `REVISIONS.txt`). It stops unless torch sees a CUDA device and the first one has at least 75 GiB (`MINGIB`, default 75).
3. It runs the part's FP32 unit tests on the CPU before it loads any model (`logs/pytest.log`). A failure stops the script. These tests are each part's exactness gate (J-<part>-G0, defined in the part). Every part's test files also include `tests/test_stage8_populations.py` (rule G6, across the parts) and `tests/test_stage8_holm.py` (the shared Holm helper). `TESTS=0` skips them only after a pass of the same test files at the same HEAD on the same host in the same results directory (`PYTEST_OK.txt`).
4. It fetches and verifies each model's files before that model's steps (next section). A part may fetch the next model in the background while the current one runs; such a prefetch only logs its result, and the fetch before the model's steps decides.
5. It runs the part's steps. Each step's log is `logs/<step>.log`. A step that succeeded in an earlier run with the same results directory is kept (`steps/<step>.done`) unless `FORCE=1` is set or `FORCE_STEPS` names it (by name or shell pattern). A failed step is listed in `FAILED.txt` and the other steps still run; a failed exploratory step is listed in `FAILED_EXPLORATORY.txt` instead and does not fail the run. A results file that the part declares for a step (`S8_OUTPUTS`) and that the failed step wrote is moved aside to `<file>.failed.<UTC>` and is not scored. `FAILED.txt`, `FAILED_EXPLORATORY.txt`, `FETCH_FAILED.txt` and `SKIPPED.txt` list the current run only: a later run moves the earlier files aside (`<name>.<UTC>.txt`), so a step skipped in one run and done in the next is not reported as skipped.
6. The deadline `STAGE8_DEADLINE` is `DEADLINE_H` hours after the start of the script: the clock starts before the checks of item 1, so setup counts. The default is set per part. Steps whose budget no longer fits are not started, in the order the part fixes, and are listed in `SKIPPED.txt`. A step that stops at the deadline partway through exits with status 3 and is also listed there. Both kinds are run again by a later session.
7. It runs the part's scorer, writes `MANIFEST.sha256` (the sha256 of every file in the results directory) and writes the archive. The archive holds no weights. A scorer exit status other than 0 (for example on a provenance or population MISMATCH; each part states its statuses) lists the score step in `FAILED.txt`; the score file is still written and archived. The script exits with status 1 when `FAILED.txt` exists, else 0.

In TEST_MODE (`TEST_MODE=1`), the script skips the guards, pip and the GPU check, keeps every model directory (`KEEP_CACHE=1`), and writes to `results/gpu_stage8<part>_test/` and `gpu_stage8<part>_test_results.tgz`. Every model the part names is replaced by Qwen2.5-0.5B-Instruct (from the Hugging Face cache or the Hub, not verified) in FP32 on the CPU, with n = 2–3. Every output is tagged `TEST_`, and its verdict lines are plumbing checks, not results.

Every results file carries a provenance block with these fields:
- the git commit, the library versions, the device and dtype;
- the model's repository, revision and attention implementation, and the sha256 of its `VERIFIED.json`;
- the population hashes;
- the chat wrapper used (`WRAPPER_USED`);
- every skipped item with its reason.

Results are written atomically (a temporary file, then `os.replace`). The run settings are BF16, `use_cache=False` for every scoring pass, and sdpa attention except for Gemma-2 (eager); a part names the steps that load a model with eager attention for another reason (Part A's reader heads and Part D's head sets read attention weights). Every scored quantity is a difference against a reference row in the same batch.

### Models and the sourcing of their files

`scripts/stage8_models.json` (sha256 `b840b71ea644d3f01e634c67ef2c2b9bdb26cdf53d3fe6b45d34c5e78c0cb707`) pins each model's official repository and its commit. The commits are those current on 2026-10-10. Each equals the commit that stages 1–7 pinned, where a stage pinned one. The manifest gives the size and hash of every file of the model directory, as the Hugging Face API reported them on 2026-10-10 (`api/models/<repo>/revision/<commit>?blobs=true`):
- sha256 of the content for files stored in LFS;
- the git blob id, sha1(`blob <size>\0` + content), for the other files.

The files are the top-level `*.json`, `*.safetensors`, `*.model`, `tokenizer.model.v3`, `*.txt` and `*.jinja`. Mistral's single-file `consolidated.safetensors` is excluded: the HF-format shards are used.

| Key | Repository | Commit | Weights (GB) | Files | Attention | Source of the bytes |
|---|---|---|---|---|---|---|
| `qwen7` | Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 15.23 | 11 | sdpa | official |
| `qwen14` | Qwen/Qwen2.5-14B-Instruct | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | 29.54 | 15 | sdpa | official |
| `qwen1.5` | Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 3.09 | 7 | sdpa | official |
| `qwen3b` | Qwen/Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 6.17 | 9 | sdpa | official |
| `qwen0.5` | Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775` | 0.99 | 7 | sdpa | official |
| `mistral7` | mistralai/Mistral-7B-Instruct-v0.3 | `c170c708c41dac9275d15a8fff4eca08d52bab71` | 14.50 | 12 | sdpa | official (HF-format shards) |
| `olmo7` | allenai/OLMo-2-1124-7B-Instruct | `470b1fba1ae01581f270116362ee4aa1b97f4c84` | 14.60 | 11 | sdpa | official |
| `llama8` | meta-llama/Llama-3.1-8B-Instruct (gated) | `0e9e39f249a16976918f6564b8830bc894c89659` | 16.06 | 10 | sdpa | with HF_TOKEN: official; else NousResearch/Meta-Llama-3.1-8B-Instruct@d10aef79, modularai/Llama-3.1-8B-Instruct-GGUF@96669450, RedHatAI/Llama-3.1-8B-Instruct@83c92747, unsloth/Meta-Llama-3.1-8B-Instruct@a2856192 |
| `gemma9` | google/gemma-2-9b-it (gated) | `11c9b309abf73637e4b6f9a3fa1e92e615547819` | 18.48 | 11 | eager | with HF_TOKEN: official; else unsloth/gemma-2-9b-it@fc7d4737, thr3a/gemma-2-9b-it@e99c393f, dnhkng/RYS-Gemma-2-9b-it@dd19021a, RedHatAI/gemma-2-9b-it@42ea3c67, jebish7/gemma-2-9b-it@e6111e3d |
| `gemma2b` | google/gemma-2-2b-it (gated) | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | 5.23 | 9 | eager | with HF_TOKEN: official; else SceneWorks/gemma-2-2b-it@684c553b, 1024m/gemma-2-2b-it-Base@7dc6b306, jebish7/gemma-2-2b-it@275bb885, unsloth/gemma-2-9b-it@fc7d4737, dnhkng/RYS-Gemma-2-9b-it@dd19021a (the last two for tokenizer files that all Gemma-2 sizes share) |
| `phi4` | microsoft/phi-4 | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` | 29.32 | 15 | sdpa | official |
| `falcon7` | tiiuae/Falcon3-7B-Instruct | `1e57a0ecd176c7c139f289c60a74e57f887c3dfb` | 14.91 | 10 | sdpa | official |
| `yi9` | 01-ai/Yi-1.5-9B-Chat (the fallback) | `1a0fc698cf883c4f5c325f026ca79f0ebd9955a5` | 17.66 | 11 | sdpa | official |
| `mistral24` | mistralai/Mistral-Small-24B-Instruct-2501 | `9527884be6e5616bdd54de542f9ae13384489724` | 47.14 | 19 | sdpa | official (HF-format shards) |

**Verification** (`scripts/fetch_verified.py`, rule G2). For each file, the fetcher tries these sources in order:
1. a copy already in the model's directory, re-hashed on every run;
2. a copy in the box's Hugging Face cache at the pinned commit of a repository that may serve the file (the official repository, for a gated one only when HF_TOKEN is set, then the ungated repositories listed for the file), hard-linked into the model's directory (so a later change or removal of the cache entry cannot change or remove the verified bytes; a symbolic link only where the cache is on another file system), and verified after it is linked;
3. a download from the official repository at the pinned commit (for a gated repository, only when HF_TOKEN is set);
4. for a gated repository, a download from each ungated repository the manifest lists for that file, in the listed order, each at its pinned commit.

A copy is accepted only if its size and hash equal the official ones. Otherwise the next source is tried. A source that fails with a download error is tried at most three times in all (with waits of 30 s and then 60 s), and only once when no retry can fix the error (HTTP 401, 403 or 404, or a gated or missing repository, revision or file).

`VERIFIED.json` is written into the model's directory, with the source of every file, only when every file verifies and the directory holds no other file. A model is **refused** (fetcher status 1, `FETCH REFUSED`, `FETCH_FAILED.txt`) before any of its steps runs only when a file of the manifest cannot be verified: every listed source delivered bytes that do not match, or answered with an error that a retry cannot fix. Only a refusal is a verification failure. The script stops instead, with its reason and without any fallback, when the fetch fails for a reason of the environment; a later run of the same script goes on from there:
- the disk cannot hold the files still to fetch plus 2 GiB (checked before any download; status 4);
- a source still fails with an error that a retry could fix (network, I/O) after its three tries, and no file was refused (status 5; the fetch stops at that file);
- the model's directory holds a file that the manifest does not list (status 6);
- a bad manifest (status 2) or an unexpected error of the fetcher (status 3).

The ungated sources were found by comparing the Hub's sha256 (LFS) or git blob id, and the size, of every file with the official ones. At least two of the listed sources hold each file of each gated model (the manifest lists up to four per file, preferring the copies the design named and organisations' complete copies). Because every byte is checked against the official hash, the assembled directories are byte-identical to the official releases, and the paper describes Llama-3.1-8B-Instruct, Gemma-2-9B-it and Gemma-2-2B-it as "official weights (sha256-verified)".

`tokenizer.model` of Mistral-7B-Instruct-v0.3 is committed in its repository as a 130-byte git blob that is an LFS pointer, and the Hub serves the 587,404-byte object it points to. The manifest therefore pins the object's sha256 (37f00374…), which the Hub reports as `x-linked-etag`, and not the pointer's blob id. This object is byte-identical to `tokenizer.model.v3`, whose blob id the manifest also pins.

**The fallback rule (G2).** The single technical fallback for a fresh family is 01-ai/Yi-1.5-9B-Chat (`yi9`). It can take the place of Llama-3.1-8B or Gemma-2-9B in Part A, of one of Llama-3.1-8B, Gemma-2-9B, Phi-4 and Falcon3-7B in Part B, and of Llama-3.1-8B in Part C. Part D runs no fresh family and has no fallback: a model of Part D whose files are refused is listed in `FAILED.txt` and is not evaluable in any line. The fallback replaces a model only when one of these happens:
- a file of that model fails verification (fetcher status 1);
- in Parts B and C, the model's tokenizer-only check, run before any output of the model, fails (Part B's J-B-G0b, Part C's preflight).

The fallback is decided before any output of the replaced model exists, at most once per part, and never on a fetch that failed for a reason of the environment (status 2–6). It is never used after any output of that model exists, and never after a competence, gate or verdict failure. The decision is written to the results directory (Part A: `FALLBACK.txt`; Parts B and C: the line "fallback: yi9 replaces <key>" in `COMMIT.txt`) and holds for every later run of the same part there: the replaced model is not run again, even when its files verify later, so the outputs of the fallback and of the replaced model are never mixed. In Part A, the factorial of every model, the fallback included, checks every item with that model's own tokenizer and skips the items that fail, with the count and the reasons reported. A refused fetch is also listed in `FETCH_FAILED.txt`, the fallback's revision and sources are in `REVISIONS.txt`, and the part's scorer names the replacement. If the fallback is itself refused, the replaced model's slot has no results and is not evaluable in any line; the line is then judged by the part's combination rule.

### Statistics, intervals and the combination of models (G3)

**Intervals.** Every interval is a percentile interval from 10,000 bootstrap resamples, with a fixed seed stated by the part. The index sets are shared by all arms, rows and scorings of a model, so contrasts are paired. Every ratio, share and difference of ratios is recomputed within each resample. The resampling unit is the cluster where clusters exist, with a two-stage cluster bootstrap that resamples the clusters, then the items within each drawn cluster:
- Part A: articles;
- Part B: the cells (lexicon, ordered location pair) (B-2);
- Part C: stories within (base, source) pairs, with the E4 seeds as a further level (C-13);
- Part D: stories (cores), resampled directly.

A part with clusters names the statistics that resample items directly instead (Part B's J-B6c core bootstrap and J-B8; Part C's pair statistics of J-C6 and the screen).

**Lines over several models.**
- A line that must hold "in every evaluable model" is an intersection-union test. It uses the 95% interval of each model, without a multiplicity correction. The part states how many evaluable models such a line needs. Every part combines such a line by one rule: the line is NOT MET as soon as one evaluable model does not meet it, whatever the number of evaluable models; it is NOT EVALUABLE only when no evaluable model fails it and fewer models than the part requires are evaluable; otherwise it is MET. (Part A also requires a fresh family among the evaluable models, and its head lines require both head models.)
- A line that must hold "in k of N models" follows the rule its part states. Only Part B has such lines (its N4 lines: 3 of 4): they use 98.75% intervals (Bonferroni over four models at a one-sided 2.5%), a model that is not evaluable for such a line counts as not meeting it, and with fewer than k evaluable models the line is NOT EVALUABLE.

**Interval criteria as tests.** Every interval criterion is written as a one-sided test of a named null. Example: "H0: s_ID ≤ 0.35, rejected when the lower bound of the 95% interval is > 0.35", a one-sided test at 2.5%; with the 98.75% interval it is a test at 0.625%. An equivalence criterion ("the interval lies inside (a, b)") is two such tests, one per bound, on the line's interval; Part C's TOST equivalences use the 90% interval (two tests at 5%), as Part C states. A point floor (for example "s_ID ≥ 0.5") is an effect-size condition, not a test, and is described as such. A line is met in a model only if all of its tests reject and all of its point floors hold.

**Holm sensitivity analysis.** This is reported, and no verdict uses it. It is computed identically in every part by the shared helper `analysis/stage8_holm.py` (tested in `tests/test_stage8_holm.py`, which every part's pytest step runs before any model). The family of a part is the interval components of its R-class account lines: one bound of one statistic in one model, taken from the models where the line is evaluable (Parts C and D also leave out a line that is NOT EVALUABLE as a whole; each part lists its family). For each component the helper computes the one-sided p-value from the normal approximation with the bootstrap standard error se (the standard deviation of the resampled statistic): p = Φ(−(est − bound)/se) when the alternative is "value > bound", and p = Φ((est − bound)/se) when it is "value < bound". A component with se = 0 gets the limit of that formula (0, 1 or 0.5 by the side of the bound the estimate lies on); a component whose estimate or se is undefined is never rejected and is left out of the family. The helper then applies Holm's step-down procedure at a familywise one-sided α of 0.025 over the family: the component of rank i (by increasing p) among m has the threshold 0.025/(m − i + 1), and the components are rejected in rank order up to the first that is not. For each R-class account line, the scorer prints the components whose decision changes when Holm's decision replaces the line's own interval decision, and the verdict the line would then get. Parts A and B recompute that verdict under the line's combination rule, with the point floors and the rule on undefined quantities unchanged. Parts C and D turn a MET line with a component that Holm does not reject into NOT MET and leave every other verdict unchanged (a component that Holm rejects and the interval rule does not is listed and changes no verdict).

**Evaluability.** Evaluability is decided only by the gates and by rows that the manipulation under test does not change. A quantity made undefined by the manipulation or the arm itself (for example a share whose denominator collapses under the tested condition) counts against the line (NOT MET, with the reason printed), never as NOT EVALUABLE. A confirmatory line whose rows were not run before the deadline is NOT EVALUABLE, and the reason printed with it says that the rows were not run.

### Risk classes, priors and the calibration summary (G4)

Before any stage-8 output, this entry records for every confirmatory line a class and a prior, P(MET | evaluable): the probability that the line is MET, given that it is evaluable. Each prior has a one-line justification from data in hand. The class follows the recorded prior:
- **L, replication.** The line is implied by data in hand on the same models and material, and its prior is at least 0.9.
- **M, extrapolation.** The prior is at least 0.8 (and the line is not L): an extrapolation of a regularity seen in every model in hand, or a consequence of the causal mask together with another line.
- **R, risky.** The prior is below 0.8, including every line on which no data in hand bears.
Where a line's description and its prior would point to different classes, the prior decides.

Each line also has a kind:
- **account lines** test the paper's account;
- **measurement-validity lines** test a measurement: scoring invariance, coverage, or the agreement of a score with generated answers.

The two kinds are tallied separately. Gates are not predictions and are not tallied.

**What each scorer prints.** Each scorer's SUMMARY gives, for each class (L, M, R) and separately for account and measurement-validity lines:
- the number of lines;
- the counts of MET, NOT MET, MET IN PART and NOT EVALUABLE lines;
- the observed met count against the expected met count, which is the sum of the priors of the lines with a verdict of MET, NOT MET or MET IN PART (the priors are conditional on evaluability, so the NOT EVALUABLE lines do not enter);
- the Brier score, the mean of (prior − 1[MET])² over the same lines.

It then gives the met count among the R-class account lines with a verdict, against their expected count. MET IN PART counts as not met in these tallies; no part of this entry defines it, so every SUMMARY prints its count as 0. NOT EVALUABLE lines are left out of both counts and are listed by code; the reason of each is printed with its line in PREDICTIONS. A line that a part marks as derived from its other lines (Part A's J-A3) is printed with its verdict in PREDICTIONS and is not tallied.

**In the paper.** The abstract reports only the met rate of the R-class account lines of this entry, over all four parts, next to the expected count: "k of n predictions we judged risky before the run held (expected k*)". The L and M tallies, the measurement-validity tallies and the Brier scores go in Table 3 and the appendix. The appendix also labels the lines of entries A–I by the same rule. Those labels are computed by a script from each entry's "seen before" record, are marked post hoc, and are never pooled with this entry's tallies.

**Pre-written consequences (E-2).** Each part names the single primary line behind each planned abstract clause and each Table 1 row. For every primary line, it gives four consequences, for MET and for NOT MET:
- the abstract text;
- the consequence for the title;
- the Table 1 row;
- the figure.

A claim enters the abstract only if its primary line is MET under its combination rule. MET IN PART appears in the body only, with the word "partly". NOT MET is replaced by the pre-written boundary statement.

The title is "Looked Up or Copied? Later Mentions Decide Whether a Model Reads an In-Context Value Through Its Key or Its Value". Only Part A's natural-text primary lines can change it, as named in Part A (A1 and A3 in the design). If these are NOT MET or NOT EVALUABLE, the title becomes the templates-only fallback given in Part A's section. No outcome of Part C changes the title.

### Codes and verdict words (G5)

Confirmatory lines are numbered J-A1, J-A2, … in Part A, and J-B…, J-C…, J-D… in the other parts. Named lines keep their part's prefix (for example J-B-NULL, J-D-SIGN-Q). Gates are J-A-G0, J-A-G1, …, and the G0 gate of every part is its FP32 unit tests. Each verdict is one of MET, NOT MET, NOT EVALUABLE, or MET IN PART (only where a part defines it; no part of this entry does). Each scorer writes `STAGE8<PART>_SCORE.txt` with the sections PROVENANCE, POPULATION, GATES, PREDICTIONS, REPORTED, SUMMARY and EXPLORATORY; Part C adds HEADLINE and Part D its D6 DECISION TABLE. PREDICTIONS has one line per confirmatory line, giving:
- the code;
- the class (L/M/R) and the kind (Part A, whose lines are all account lines, prints the class only);
- the prior;
- the verdict;
- the numbers and bounds it was decided on.

The GPU script prints the GATES and SUMMARY blocks at the end of the run.

### Populations (G6)

U is the union of make_cores(1000, Random(s)) for s = 0, 1, 2, 3. Stages 1–7 drew templated cores from seeds 0 and 1 only, so U contains every core used before. A core is compared by its full tuple: (agent, other, object, distractor, initial, distractor_location, base, source). U holds 3,981 distinct tuples; 19 of the 4,000 draws repeat. Its hash is sha256(json.dumps(sorted(U))), with each tuple as a list in that field order and the default separators of `json.dumps`: abd1f0530a3d08a2058743f59102f8dd6dd360af06fc65dd3277c5b5eb176c3d.

The rule: every templated population of stage 8 is disjoint from U and from every other stage-8 population, by full tuple. Each part's preflight and unit tests assert this for its own populations, and `tests/test_stage8_populations.py` (run at the build and by every part's pytest step before any model) asserts it across the parts, from the functions the GPU steps call (`ckeys.fresh.population` for Part B, `ckeys.edits.populations` for Part C, `experiments.stage8_flag.populations` for Part D). It also asserts that the three parts compute U identically (`ckeys.fresh.used_cores`, `ckeys.edits.universe`, `experiments.stage8_flag.u_set`) and that no confirmatory population is drawn from a pilot seed listed below. Each part lists its populations with their seeds, sizes and hashes.

There are two exceptions, both re-uses of the stage-1 cores drawn from Random(0) and labelled in their parts:
- S0 = make_cores(150, Random(0)), the cores of stages 1 and 3b, is re-measured on purpose by Part B's JB6 as the discovery sample. The sha256 of its 150 tuples in drawn order is fdd1bf1ba4d8d657f663f786c3ff92d0145e41cf02e123e602d594b183110121.
- Part D's fit set R = make_cores(60, Random(0)), the first 60 cores of S0 (the stage-6 ranking set), and its subset R′ (the 45 cores with distractor_location ≠ initial), re-used on purpose as the in-sample fit set: the flags are fit where stage 6 ranked the reader heads (Part D states which quantities are computed there).

The full space has 364,800 distinct cores (480 location configurations × 2 agent orders × 380 ordered object pairs). U enumerates every location configuration, so a fresh population is fresh in its tuples, not in its location configurations; Part B's lexicon and wording changes address this.

Seeds that the pilots below already drew from, and that no confirmatory population may use:
- make_cores seeds 0, 1, 7, 8, 9, 99, 101, 202 and 20261011;
- `ckeys.ioi.make_cores` seeds 5 and 6.

| Part | Population | Drawn from | Size | sha256 |
|---|---|---|---|---|
| A | R, E, YEAR, LEAK | SQuAD v1.1 dev questions, `data/stage8a_items.json` (`experiments.natural_factorial.populations`) | 71, 185, 80, 65 (build) | items file 40874aa1… (fixed by the finalising commit, not pinned in the code) |
| B | F | the first 150 cores of the Random(20261013) stream not in U | 150 | e87047c9… (with its rendering) |
| B | C (calibration) | the first 30 cores of the Random(20261014) stream not in U or F | 30 | a625fd13… (with its rendering) |
| B | S0 (re-use) | make_cores(150, Random(0)) | 150 | 48bb0a3a… (with its rendering) |
| C | E | Random(8101): π(S) ≠ B and π(X) ≠ B, not in U | 80 | a23465a2… |
| C | H (H_fit, H_cal) | Random(8102), not in U or E | 200 (150, 50) | 8cc62cef… |
| C | TSET, THOLD | Random(8103): π(S) ≠ B, not in U, E or H | 1,000, 100 | 06c56f93…, 136b7d94… |
| D | R, R′ (fit; re-use) | make_cores(60, Random(0)) | 60, 45 | 9036af1a…, fe348ba4… |
| D | E8 | Random(81), not in U or the pilots' and the other parts' seed streams | 100 | 2c198705… |
| D | BIND | Random(84), distractor_location ≠ initial, not in U, those streams or E8 | 100 | 495c1462… |
| D | F_IOI, E_IOI candidates | `ckeys.ioi.make_cores(90, Random(82))`, `(150, Random(83))`, disjoint from each other and from IOI seeds 0, 1, 5, 6 | 90, 150 | d41e791a…, 2b165365… |
| D | XFIT, XEVAL (exploratory) | `ckeys.tasks` paint and schedule cores, Random(85), Random(86), not in task seeds 0–3 | 60, 40 per task | 9b9a320f…, be234aaf…, 4a56ae02…, 8b4c7c05… |

The hashes are those of each part's own canonical form (each part states it; Part B's include each core's lexicon, sentence and candidate order). For Parts B–D the full values are pinned in `ckeys/fresh.py`, `ckeys/edits.py` and `experiments/stage8_flag.py` and asserted before any model. Part A's items file is fixed by the finalising commit (`data` is among the guarded directories); its sha256 is recorded in every Part A results file, and Part A's preflight stops the run unless the items rebuilt from the SQuAD file are byte-identical to it.

### Seen before finalisation: the principle and the common part (G7)

**The principle.** Every CPU pilot run for the design or the critique of this stage is listed in its part's "seen before finalisation" section with its numbers. This includes the critics' closed-book, cue-conflict, non-lexical-steering and core-overlap checks. The pilots' scripts and logs are in `pilots/stage8/` (the rewrite's design and critique read only literature and model nothing). They are listed by part:

| Part | Design pilots | Critique pilots |
|---|---|---|
| A | `pilots/stage8/partA/` (item builds, prompt checks, the 0.5B pilots `pilot05*`, continuation, split, power) | `pilots/stage8/critic_natural/` (closed-book `cb15.json`, cue-conflict `cc15.json`, 1.5B) |
| B | `pilots/stage8/partB/` (tokenizer check, Hub metadata, exactness, mini factorial at 0.5B, coverage at Llama-3.2-1B, in-hand statistics, power) | `pilots/stage8/critic_replication/` (JB9 chance, lexicon check; core overlap with seed 99) |
| C | `pilots/stage8/partC/` (pilots A and B, memory test, power) | `pilots/stage8/critic_intervention/` (non-lexical steering at 1.5B, κ compression) |
| D | `pilots/stage8/partD/` (0.5B flag, ablation, dissociation, IOI, injection; sign at 1.5B) | `pilots/stage8/critic_mechanism/` (`pilot_critic*`, `pilot_kv_inout`, 1.5B) |
| E (rewrite) | the rewrite design (literature only; no model was run) | its critique (literature only) |

No model output of any stage-8 confirmatory population has been seen. According to the pilots' logs and the design files, every pilot that ran a model ran it in FP32 on the CPU at 1.5B parameters or fewer (Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct). The larger models appear only in tokenizer-only and metadata checks. Each part's list is the authoritative record of its pilots.

**Seen for the common part.**
- **The Hub metadata of 2026-10-10.** This is the revision and the per-file sizes and hashes of the fourteen repositories in the manifest, and of about 2,100 public repositories whose names match Llama-3.1-8B-Instruct, Gemma-2-9B-it or Gemma-2-2B-it. They were searched for byte-identical copies of the gated files; every file of the three gated models has at least five (Gemma-2-2B-it's `generation_config.json`; at least ten for Gemma-2-9B-it and 23 for Llama-3.1-8B-Instruct). Every revision equals the one earlier stages pinned, where one was pinned. The 108 files that Part B's capture of the same morning also recorded have the same hashes, except one: Mistral-7B's `tokenizer.model`, for which that capture recorded the LFS pointer's blob id; the pointer is explained above. No model was run.
- **The fetcher's checks.**
  - The seven files of Qwen2.5-0.5B-Instruct were verified from the local Hugging Face cache.
  - These small files were downloaded and verified: Llama-3.1-8B-Instruct's `config.json` from NousResearch through huggingface_hub, its `tokenizer_config.json` from modularai over plain HTTPS, and Mistral-7B's `tokenizer.model` over HTTPS.
  - Two files of Gemma-2-2B-it were downloaded from SceneWorks/gemma-2-2b-it through huggingface_hub and matched the official sha256: `tokenizer.json` (17.5 MB) and `model-00002-of-00002.safetensors` (240.7 MB).
  - Downloads from the gated official repositories without a token were refused with HTTP 401, as expected.
- **The unit tests and plumbing runs.** `tests/test_fetch_verified.py` passed (29 tests): hashes, mirrors, retries, refusals against environment errors (statuses 1 and 3–6), the disk check, hard links from the cache, the manifest, the library's guards on throw-away repositories (including a final entry that quotes the words of the draft marker, which must run), a TEST_MODE dry run with a fake scorer, and `s8_fetch` and `s8_prefetch` with a stand-in fetcher. `tests/test_stage8_holm.py` (7 tests) and `tests/test_stage8_populations.py` (5 tests) passed. The TEST_MODE runs of the four scripts are listed in each part.

### Shared code (G8)

The parts build on code committed before this entry:
- **43700ae:**
  - `ckeys/surface.py`, the emitted-form trie scorer;
  - span tables in `ckeys/headsplice.py`;
  - the fused-qkv clamp sites of Phi-3/Phi-4 in `ckeys/clamp.py`;
  - `ckeys/squad_items.py`, `ckeys/natural_formats.py` and `scripts/build_stage8a_items.py`;
  - `tests/test_surface.py` and `tests/test_clamp_families.py`.
- **107332e:** `ckeys/generate.py` (greedy decoding under cache clamps, answer parsing, frame discovery) and `tests/test_generate.py`.
- **c6fd4a8:** the Part A item rules and `data/stage8a_items.json`.
- **The stage-8 build (committed with this entry):**
  - `scripts/stage8_common.sh`, `scripts/stage8_models.json`, `scripts/fetch_verified.py` and `tests/test_fetch_verified.py`;
  - `analysis/stage8_holm.py` (the Holm sensitivity analysis of every part) and `tests/test_stage8_holm.py`;
  - `tests/test_stage8_populations.py` (rule G6 across the parts);
  - each part's experiments, scorer, tests and GPU script.

The parts reuse these modules. Two were changed after their first commit, each with its tests: `ckeys/natural_formats.py` was extended for Part A in the stage-8 build (tested by Part A's gate files), and the stage-8 review changed `ckeys/generate.py` so that a chat template's end-of-turn token also ends a generated answer (`tests/test_generate.py`, which the GPU scripts of Parts A, B and C, the parts that use the module, run before any model; Part D does not use it). A change to a shared module needs a test that the GPU scripts run before any model.

### Compute and commands

| Part | GPU-hours (A100-80GB): core / with the exploratory and deadline-guarded steps | Default DEADLINE_H | Disk for weights |
|---|---|---|---|
| A | about 3.3 / about 4.1 | 4.5 | ≥ 60 GB (one model at a time; no prefetch) |
| B | about 3.2 / about 4.3 | 4.5 | ≥ 140 GB (two models at a time besides Qwen2.5-7B, kept until x1; 47 GB for J-B8) |
| C | about 3.5 / about 4.8 | 5.0 | ≥ 80 GB (two models and the four dictionaries) |
| D | about 2.3 / about 2.5 | 3.5 | ≥ 50 GB (both 7B models and a prefetch) |

The weights of all fourteen models total about 233 GB. A part needs only its own models. Each model's directory is deleted after its steps unless `KEEP_CACHE=1` is set or the model was already verified there before the run. The box needs at least 250 GB of disk; network access to huggingface.co (the models and Part C's dictionaries), rajpurkar.github.io (Part A's SQuAD file), github.com (the clone, and the published release that Part C's screen and J-C6 use) and anonymous.4open.science (Part B's J-B8 fetches the release of Anonymous (2026) there unless `P1R` names a local copy), and the Python package index when pip must install; and an optional HF_TOKEN, set only in the box's own terminal. The commands, on one A100-80GB:
```bash
# in a fresh clone of this repository (docs/GPU_RUNBOOK.md gives the clone line)
J=$(git log --format=%H -1 --grep='^Finalise preregistration J$') && [ -n "$J" ] && git checkout "$J"
bash scripts/gpu_stage8a.sh; bash scripts/gpu_stage8b.sh; bash scripts/gpu_stage8c.sh; bash scripts/gpu_stage8d.sh   # any order, any subset
TEST_MODE=1 bash scripts/gpu_stage8<part>.sh   # the CPU plumbing test of one part
```
`docs/GPU_RUNBOOK.md` (Stage 8) gives the options (`DEADLINE_H`, `FORCE`, `FORCE_STEPS`, `KEEP_CACHE`, `TESTS`, `MINGIB`, `OUT`, `S8_MODELS`, `PY`) and explains reruns and how to send the results back; each part names its own further options (for example `SQUAD` in Part A, `P1R` in Part B, a local copy of the release in Part C, and `TEST_ALL` for the TEST_MODE runs of Parts B–D).

### Part A. The format law on natural reading comprehension (counterfactual SQuAD)

**Code.** `ckeys/natural_formats.py` (prompt formats, closed-book prompts, answer frames, frame-aware decision tokens, answer parsing), `ckeys/natural_rows.py` (rows, tables and per-item passes), `ckeys/kvquant.py` (KIVI-style fake quantization), `experiments/natural_factorial.py` (stages preflight, frames, factorial, explore), `experiments/natural_heads.py` (heads on natural text), `analysis/stage8a_score.py` with `analysis/stage8a_parts/` (scorer), `scripts/gpu_stage8a.sh` (pipeline; it sources `scripts/stage8_common.sh`). Items: `ckeys/squad_items.py`, `scripts/build_stage8a_items.py`, `data/stage8a_items.json`, `data/stage8a_type_audit.tsv`. Tests: `tests/test_natural_clamp.py` and `tests/test_kvquant.py` (Gate J-A-G0), `tests/test_natural_heads.py` (Gate J-A-HA-G0), `tests/test_stage8a_score.py` (the scorer on synthetic inputs).

**Purpose.** Stages 1–7 showed the format law on templated stories with single-token values. Part A tests it on natural Wikipedia passages with multi-token, open-vocabulary answers, in four model families, two of them never run in stages 1–7. It answers four objections.
- *Generality (objection 2).* SQuAD v1.1 dev passages; answers are PERSON, PLACE or NUMBER entities named once in the passage; spans are mostly 2–4 tokens; Llama-3.1-8B and Gemma-2-9B are fresh families; intervals come from a cluster bootstrap over articles.
- *Emitted form (objection 4).* Each model's answer frame (the text it writes between "Answer:" and the entity) is fixed before the factorial from its own greedy generations on R items, and every decision token is scored after prompt + frame. Gate J-A-G3 requires the scored token to be the first generated token for B, S and X.
- *A behavioural consequence.* Cue-conflict rows give the same cache two sources, neither of them the original answer: the key from one entity and the value from another. The account predicts that the generated answer names the key's entity under a later options list and the value's entity without one (J-A6).
- *A practical consequence.* KIVI-style 2-bit quantization of the passage's cached values, against its keys, in multiple-choice and free-form formats, as a difference in differences (J-A7).
- *Prior controls (the critique's F1).* A question with its options and no passage already favours the original answer B (65 % at 1.5B, see Seen before finalisation). Every verdict on an accuracy (J-A7, J-A7b, J-A-HA3a, J-A-HA3b) is therefore made on prior-free items (closed-book argmax not B, and the Z passage does not give B). The identity measures ID_K, ID_V and s_ID are S-versus-X double differences on a common B background; a prior for B cancels from them to first order.
- *Heads on natural text.* The stage-6 reader-head analysis is repeated on the natural MCQ items at Qwen2.5-7B and Mistral-7B, with a transfer test of the template-found heads and a behavioural necessity test (J-A-HA1 to HA3).

Part A does not test the templated sentence effect on fresh cores (objection 5 belongs to Part B), the sign of negative key reads (Part D), or interventions (Part C). J-A8 is a channel-level replication of induction-head K-composition (Elhage et al. 2021; Olsson et al. 2022), class M (prior 0.80), and is not a novelty claim.

#### Material

**Source.** SQuAD v1.1 dev (Rajpurkar et al. 2016; CC BY-SA 4.0), `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json`, sha256 `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9` (asserted by `ckeys.squad_items.load_squad`): 48 articles, 10,570 questions. The counterfactual substitution follows Longpre et al. (2021).

**Item rules** (`ckeys/squad_items.py`; fixed before any model output):
1. e_B is the majority answer (given by at least 2 of the 3 annotators). It occurs exactly once in the passage (word-bounded, case-sensitive) and not in the question (case-insensitive).
2. Type: YEAR (`^(1[0-9]{3}|20[0-2][0-9])$`), NUMBER (`^(\d{1,3}(,\d{3})+|\d+)$`, not a year), or NAME (1–4 capitalised words with connectors). A NAME is PERSON when the wh-word is who/whom/whose and the answer is person-shaped (2–3 words of the person patterns, no word in the NONPERSON list); PLACE when the question has a place head noun (country, nation, city, town, village, capital, state, province, county, region, district, area, river, lake, island, continent), typed into the place classes country, city, region, water, island, continent (with the audited class corrections CLASS_FIX); ORG and the rest are dropped. "where" questions without a place head noun are dropped.
3. Substitutes S, X, Z (Z is only ever clamped, never shown): NAME from the majority answers of the same subtype, place class and word count in other articles; NUMBER by replacing the leading digit (seeded order); YEAR in the same century, 3 ≤ |Δ| ≤ 60. Each item has its own stream `random.Random(int(sha256(id)[:8], 16))`. Every substitute is absent from the passage and the question, shares no word with e_B, the other substitutes or D, and no content word of it occurs in the passage or question.
4. D, the fourth option: another majority answer of the same paragraph and subtype that occurs in the passage (D_in), else an absent pool entity. The option order is `random.Random(seed_of(id + "order")).shuffle([e_B, S, X, D])`, the same in every prompt of the item.
5. No partial mentions: no content word of e_B (≥ 3 characters, not a connector) occurs in the passage outside e_B's span or in the question. Items that fail only this rule form the LEAK stratum (exploratory; for example a surname re-mentioned after the full name).
6. Per tokenizer and format: the S, X and Z passages give prompts of B's token length that differ from B's only inside B's entity span P; the decision tokens of B, S, X, Z and D are pairwise distinct (stratum FT; YEAR items form stratum SP without this rule).
7. The type audit (`data/stage8a_type_audit.tsv`, sha256 `e4d48380a950a8d8608f7c2396da033225287259cfe24a54e12fae74284c5c56`) was done on entity lists alone, with no model output: 44 entities excluded by type (EXCLUDE), 9 place classes corrected (CLASS_FIX).

**Build** (`scripts/build_stage8a_items.py`, tokenizers of Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, and the tokenizer files of unsloth/Meta-Llama-3.1-8B-Instruct and unsloth/gemma-2-9b-it): candidates PERSON 425, PLACE 89, NUMBER 295, YEAR 529; with three substitutes and a distractor: PERSON 349, PLACE 42, NUMBER 191, YEAR 529; 486 FT items valid in every tokenizer and format (invalid: 7 in Qwen, 1 in Gemma, all NOM). Article split: `random.Random(20261010).sample(sorted(titles of the valid FT items), 12)` gives the 12 R articles (American_Broadcasting_Company, Economic_inequality, European_Union_law, Fresno,_California, Geology, Imperialism, Intergovernmental_Panel_on_Climate_Change, Islamism, Martin_Luther, University_of_Chicago, Victoria_(Australia), Yuan_dynasty); the rest are E. Caps (seeded shuffle of the id-sorted list, at most 2 items per paragraph): R at most 8 per article (Random(1)); E at most 10 (Random(2)); YEAR at most 10 per E article, first 80 (Random(3)); LEAK at most 10 per E article, first 80 (Random(4)).

| Split | Items | Articles | By type | Use |
|---|---|---|---|---|
| R | 71 | 12 | PERSON 38, NUMBER 28, PLACE 5 | frames (first 30), head ranking (first 60) |
| E | 185 | 34 | PERSON 80, NUMBER 77, PLACE 28 (country 20, region 4, city 4); D_in 47 | factorial (every valid item), head curves (first 80), head ablation (every valid item) |
| YEAR | 80 | 26 | YEAR | exploratory E4 |
| LEAK | 65 | 21 | PERSON 64, PLACE 1 | exploratory |

`data/stage8a_items.json` has sha256 `40874aa1033d1af829f5642916282a5440df2dad011c991b4ac2704a2c25b591`; `experiments.natural_factorial.populations()` returns the four splits' question ids. Part A's populations are SQuAD questions, not templated cores, so the G6 disjointness from U does not apply to them. The preflight downloads SQuAD, asserts its sha256, rebuilds the items with the same tokenizers at the Hub revisions the build used (Qwen2.5-7B-Instruct `a09a3545…`, Mistral-7B-Instruct-v0.3 `c170c708…`, unsloth/Meta-Llama-3.1-8B-Instruct `a2856192…`, unsloth/gemma-2-9b-it `fc7d4737…`; `BUILD_TOKENIZER_REVISIONS`) and stops the run unless the rebuild is byte-identical to the committed file. Each model's factorial then checks every E item in every format with its own tokenizer: item rule 6 as in the build (frame " "), and the decision tokens under the model's frame (below). It skips (and counts, with the reason) an item that fails either check in any format; the populations below are over the remaining items. The factorial file records the sha256 of the valid items' ids (sorted, one per line); the scorer checks it, and checks that the valid and the skipped items together are E.

#### Models and sourcing

BF16; sdpa attention except Gemma-2 (eager, soft-capping); use_cache=False in every scoring pass. Files come from `scripts/fetch_verified.py` via `s8_fetch` (rule G2): the official repository at the pinned revision with HF_TOKEN, otherwise byte-identical copies listed in `scripts/stage8_models.json`, each file verified against the official sha256 (LFS) or git blob id and size. Llama-3.1-8B-Instruct and Gemma-2-9B-it are "official weights (sha256-verified)".

| Key | Model | Revision | Attention | Role |
|---|---|---|---|---|
| llama8 | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 | sdpa | fresh family |
| gemma9 | google/gemma-2-9b-it | 11c9b309abf73637e4b6f9a3fa1e92e615547819 | eager | fresh family |
| qwen7 | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | sdpa | continuity; heads |
| mistral7 | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 | sdpa | continuity; heads |
| yi9 (fallback) | 01-ai/Yi-1.5-9B-Chat | 1a0fc698cf883c4f5c325f026ca79f0ebd9955a5 | sdpa | replaces llama8 or gemma9 |

The fallback replaces llama8 or gemma9 only when its files fail verification (`s8_fetch` status 1), before any output of it exists, and at most once. Any other fetch failure (not enough disk, a manifest error) stops the run with its reason, and no fallback runs. A replacement is written to `FALLBACK.txt` in the results directory and holds for every later session there: the replaced model is not run again, even when its files later verify. Yi counts as a fresh family. Gemma-2 rejects a system turn; the wrapper merges it into the user turn and records this (WRAPPER_USED).

#### Prompts, frames and rows

**Formats** (user turn; system turn "You are a helpful assistant."; generation prompt with thinking disabled; assistant prefill "Answer:"; `ckeys.natural_formats.parts`). PRE = "Read the passage and answer the question.\n\n"; FREE = "Answer with the exact words from the passage."; MC = "Answer with exactly one of the options."; o1..o4 the options in the item's order.
- NOM (no mention): PRE + "Passage: " + P + "\nQuestion: " + Q + "\n" + FREE.
- OPTA (options after): PRE + "Passage: " + P + "\nQuestion: " + Q + "\nOptions: o1; o2; o3; o4\n" + MC.
- OPTB (options before): PRE + "Options: o1; o2; o3; o4\n\nPassage: " + P + "\nQuestion: " + Q + "\n" + MC.
- MENA (sentence after): PRE + "Passage: " + P + " Related articles mention o1, o2, o3 and o4.\nQuestion: " + Q + "\n" + FREE.
- MENB (sentence before): PRE + "Passage: Related articles mention o1, o2, o3 and o4. " + P + "\nQuestion: " + Q + "\n" + FREE.
- LETA (letters after): PRE + "Passage: " + P + "\nQuestion: " + Q + "\nOptions:\nA. o1\nB. o2\nC. o3\nD. o4\nAnswer with the letter of the correct option."
- Closed book (no passage): CBOPT = "Answer the question.\n\nQuestion: " + Q + "\nOptions: o1; o2; o3; o4\n" + MC; CBLET = "Answer the question.\n\nQuestion: " + Q + "\nOptions:\nA. o1\n...\nD. o4\nAnswer with the letter of the correct option."

**Answer frame** (per model, stage frames, before the factorial). Greedy generations of the unclamped B prompt (at most 16 new tokens, stopped and read as in pass 3 below, without its early stop) of the first 30 R items in NOM and OPTA. For each, the text before the first occurrence of e_B (case-sensitive) is counted if it is one of FRAMES = {"", " ", " **", "**", " The ", " the "}. The frame is the most frequent of these under which at least 80 % of the E items are valid in every format (ties: the order of FRAMES); " " if none. It is written to `frames/<key>.json`; every later file records that file's sha256, and the scorer checks it. Frame " " gives exactly the continuation tokens the items were built with (a unit test on every item with the Qwen tokenizer; checked once in all five tokenizers, see Seen before finalisation).

**Decision tokens** (`decision_ids`). c_Y = the tokens of frame + e_Y after the prompt (prefix-stable); w = its leading tokens that lie inside the frame's characters (any frame characters left over must be whitespace merged into the next token); dec_Y = the next token, which must not be whitespace. w must be shared by B, S, X, Z and D, and the decision tokens pairwise distinct. LETA and CBLET: c_Y = frame + the letter of Y's option, which must be one token after w; Z has no letter. CBOPT: the four options only.

**Rows** (`ckeys.natural_rows`). A row on the B prompt writes, at every position of P and in every layer from 0, the key of one donor run and the value of another (the captured K/V of the B, S, X or Z prompt; ckeys.clamp sites, pre-RoPE projection outputs).
- Every format: ID = (B, B) (the in-batch reference; the identity), K_S = (S, B), V_S = (B, S), KV_S = (S, S), K_X = (X, B), V_X = (B, X), KV_X = (X, X).
- NOM, OPTA and LETA also: K_Z = (Z, B), V_Z = (B, Z), KV_Z = (Z, Z); cue conflict KS_VX = (S, X) and KX_VS = (X, S); flag only KS_VZ = (S, Z); copy fallback KZ_VS = (Z, S).
- Exploratory, NOM and OPTA: K_S@on and V_S@on (S's tables only from layer round(0.3 L), B's below); OPTA K_S^1 and K_S^r (S's key at the first span position only, or at the others only); NOM V_S^1 and V_S^r.

**Passes per item and format** (`experiments/natural_factorial.py`, one forward each).
1. Capture: the B, S, X and Z prompts + w in one batch: K and V at P in every layer, and each run's log-probabilities of the decision tokens.
2. Scoring: every row on B prompt + c_S, teacher-forced: the decision log-probabilities lp_Y(r) for Y in {B, S, X, Z, D} (LETA: B, S, X, D), the four-option mass, the argmax, and the log-probability of every token of c_S after the decision token.
3. Generation: greedy decoding with the KV cache (ckeys.generate.greedy), at most 16 new tokens; the clamps act in the prompt pass and decoding reads the clamped cache. A row stops at a token whose text contains a newline (not kept), at an EOS id of the generation config or the tokenizer, or at a chat end-of-turn token that the vocabulary has (`<end_of_turn>`, `<|im_end|>`, `<|eot_id|>`, `<|end|>`, `<|endoftext|>`, `</s>`, `<eos>`; `ckeys.generate.END_OF_TURN`). Rows: NOM, OPTA, LETA every row; OPTB, MENA, MENB ID, KV_S, KV_X; plus the unclamped S prompt. The answer is read up to the first special token of the tokenizer (its special tokens and the added tokens marked special). The end-of-turn stop and this reading rule exist because the generation configs of Gemma-2-9b-it and Yi-1.5-9B-Chat list only `<eos>` (id 1) and `<|endoftext|>` (id 2) as EOS, not the end of the chat turn (`<end_of_turn>`, `<|im_end|>`), whose text would otherwise be read as part of the answer. A row stops early once its answer class can no longer change: a special token; or the first line has at least W + 2 normalised words, W the most words of a candidate; or, in the letter formats, two characters after the skipped leading markdown, neither of them the replacement character of an incomplete multi-byte character. The answer class, g1 and the generated prefix are unchanged by this (tested on every item along many continuations).
4. KIVI (NOM, OPTA, LETA): the S prompt with no quantization, with the keys, or with the values of every passage token fake-quantized at 2 bits in every layer (below), greedy as in 3.
5. Closed book, once per item, before the formats: CBOPT and CBLET, the four options' decision log-probabilities after prompt + w and their argmax. When the decision tokens of a closed-book prompt are not valid (rule above), that prompt has no argmax for the item.

**Order and deadline.** Formats run in the order NOM, OPTA, OPTB, MENA, LETA, MENB. The pipeline passes `--reserve-min` = the core minutes of the models after this one in the order llama8, gemma9, qwen7, mistral7 (`CORE_MIN`: llama8 32, gemma9 48, qwen7 58, mistral7 58; the fallback gets the reserve of the place it takes) plus 10, plus 20 for the same model's heads step at Qwen2.5-7B and Mistral-7B. Before LETA (MENB), the factorial projects that format's time as OPTA's (MENA's) measured time; if it would end later than the deadline minus the reserve, LETA runs with the rows ID, KV_S, KV_X, KV_Z, KS_VX, KX_VS only, and MENB is skipped. Each such decision is recorded in the file's provenance; lines that need the dropped rows are NOT EVALUABLE in that model. The step then exits with status 3, is listed in `SKIPPED.txt` and runs again in a later session. Exploratory passes are dropped before either: a model's explore step starts only when at least the reserve (without the heads minutes) plus 14 minutes remain, each of its parts is skipped once the deadline has passed, and so is the exploratory part of the heads step (ID_K and ID_V under N*).

#### Measures

- |P|: the number of prompt tokens that overlap e_B's characters (the span P), in the model's tokenizer and in the format the measure uses (OPTA for J-A1b).
- Shifts: Δ_Y(r) = lp_Y(r) − lp_Y(ID).
- ID_K = ½[(Δ_S(K_S) − Δ_S(K_X)) + (Δ_X(K_X) − Δ_X(K_S))]; ID_V the same with V_S and V_X. s_ID = mean ID_K / (mean ID_K + mean ID_V), a ratio of means over the population, recomputed in every resample.
- who(r): the entity row r's generation names. Entity formats: the first of B, S, X, Z, D with match(g, e) (SQuAD normalisation of the generation's first line and of e: lowercase, punctuation and articles removed, whitespace collapsed; the first line equal to e, or starting with e followed by a space), else "other". Letter formats: the option whose letter the generation starts with (leading whitespace and `*`, `_`, backquote, brackets and quotes skipped; the letter not followed by another letter), else "other".
- g1(r): the generated token after w, when the generation starts with w.
- Competence: competent(i, f) holds when who(ID) = B, who(KV_S) = S and who(KV_X) = X in format f. C(F) = the items competent in every format of F.
- Prior-free: prior-free(i, F) holds when, for every format of F, the closed-book argmax over the four options exists and is not B (CBOPT; CBLET for LETA), and who(KV_Z) ≠ B in every format of F with a KV_Z row. PF(F) = C(F) ∩ prior-free.
- Cue conflict (format f): key-source rate = mean ½[1(who(KS_VX) = S) + 1(who(KX_VS) = X)]; value-source rate = mean ½[1(who(KS_VX) = X) + 1(who(KX_VS) = S)]. Flag only: S-rate of KS_VZ. Copy fallback: S-rate of KZ_VS.
- KIVI (`ckeys/kvquant.py`): asymmetric uniform quantization with round-to-nearest, scale (max − min)/(2^b − 1), zero point min, at the passage token positions in every layer, on the k_proj / v_proj outputs (pre-RoPE). Keys per channel: positions in consecutive groups of 32, one scale per channel and group. Values per token: channels in consecutive groups of 32, one scale per position and group. A last shorter group is quantized as it is. acc_f(r) = 1(who = S) on the S prompt under r ∈ {none, K2, V2}; drop_f(C) = acc_f(none) − acc_f(C). DiD = [drop_NOM(V) − drop_OPTA(V)] − [drop_NOM(K) − drop_OPTA(K)], paired by item. The relative reconstruction errors ||x′ − x|| / ||x|| of K and V are reported per format (no strength matching).
- Continuation (A8): L_S^cont(r) = the sum of the log-probabilities of the tokens c_S[t], t > j, with c_S[t] ≠ c_B[t] (index-aligned; t beyond c_B counts as differing). d_C^cont = mean[L_S^cont(C_S) − L_S^cont(ID)] and d_C^dec = mean[lp_S(C_S) − lp_S(ID)] for C ∈ {K, V, KV}. I_cont = (d_KV^cont − d_K^cont − d_V^cont)/d_KV^cont; I_dec likewise.
- Hybrid rate: h(K_Z, f) = P(who(K_Z) ≠ B | g1(K_Z) = dec_B), a ratio of means.

**Heads** (`experiments/natural_heads.py`; Qwen2.5-7B and Mistral-7B; eager attention, BF16, use_cache=False). Here an item is valid when it is valid in NOM and OPTA for the model's tokenizer and frame; the heads file records how many R and E items are valid (`n_valid`), and "the first 60 (80) valid items" means all of them when fewer are valid. HA3's prior-free populations come from the same model's factorial, whose items are valid in all six formats.
- G = the token rows of the four option strings in OPTA's options line; G_Y those of option Y. The decision position is the last row of B prompt + w; m = lp_S − lp_B there. K_S = S's keys at every position of P in every layer (HeadSplice with span tables).
- Q+ = every row whose first character follows the question's last character (in OPTA the options line, so Q+ contains G; then the instruction, the chat tokens, the prefill and the frame), plus every teacher-forced answer row. Q+ never contains P.
- a3(l, h) = ½[(A^{K_S} − A^{ID})[G_S → P] + (A^{ID} − A^{K_S})[G_B → P]], where A[G_Y → P] is the mean over the rows of G_Y of the attention summed over P, and K_S is clamped from layer 0. It is averaged over the first 60 valid R items. N* = the top k* heads, k* = ceil(0.05 × heads) = 40 (Qwen) and 52 (Mistral).
- T* = the first k* heads of `arms.P1.rankings.a3` in `results/gpu_stage6/heads/Qwen2.5-7B-Instruct.json` (sha256 ed828a9b…) and `Mistral-7B-Instruct-v0.3.json` (sha256 88ababd9…).
- Random sets: the first k* heads of each of three successive permutations of all heads (in layer-major order) drawn from one `numpy.default_rng(2)` (stage 6). C* (exploratory): the top k* heads by direct logit attribution at the decision row in NOM on the same R items. A head's o_proj contribution is passed through the final RMSNorm, linearised at the run's own scale, and projected on W_U[dec_B] minus the mean of W_U over the other three options' decision tokens.
- μ_f(l, h): the mean of head (l, h)'s o_proj input over the R items and the Q+ rows of format f (NOM, OPTA), in the clean B run on B prompt + c_B. It is saved with its sha256.
- On the first 80 valid E items (OPTA): d_full = mean[m(full K_S clamp) − m(clean)], from two single passes (the B prompt + w unclamped, and with K_S clamped at P in every layer). For each of N*, T* and the random sets, the stage-6 sufficiency and knockout batches, in which each row names the heads that see K_S in the rows G. Sufficiency rows: the top-k heads of the set's full ranking for k in KS = {1, 2, 5, 10, 20, k*, 2k*}; none (no head); all_G (every head); all_T (every head in every row, the full clamp). Knockout rows: every head but the top-k, for k in KS; none_KO (no head); all_G,KO (every head). The full rankings are the a3 ranking for N*, the stage-6 ranking for T* and the permutation for a random set. d_G = mean[m(all_G) − m(none)] of N*'s batch. R(k) = mean[m(top-k) − m(none)] / mean[m(all_G) − m(none)] and KO(k) = 1 − mean[m(all but top-k) − m(none_KO)] / mean[m(all_G,KO) − m(none_KO)], each within the set's own batches (for N*, the first denominator is d_G). Random-set values are means over the three draws of these ratios.
- Ablation, every valid E item, OPTA and NOM: one batch whose rows are the conditions none, N*, T*, rand0–2 and C*. Each row mean-ablates its condition's heads at Q+ with μ_f, under the KV_S clamp on B prompt + c_S. Recorded per row: the argmax chain over all of c_S, the decision argmax = dec_S, and the chain over c_S after the decision token. The same conditions on the clean B prompt + c_B give the option margin m_B = lp_B − max(lp_S, lp_X, lp_D) and the chain over c_B.

#### Statistics

- Per model and line, the population is fixed by the line (below), over the model's valid E items.
- Two-stage cluster bootstrap: resample the population's articles with replacement, then the items within each drawn article with replacement. 10,000 resamples; seed 20261010; one fixed index set per population; every format of a line resampled jointly.
- Every statistic is a function of item means (ratios and shares are ratios of means) and is recomputed in every resample. A resample in which the statistic is not a number (for example 0/0) is dropped, and the share dropped is printed with the interval. Intervals are 95 % percentile intervals of the remaining resamples.
- Each interval criterion is a one-sided test of a named null at 2.5 %: "H0: θ ≤ t rejected" means the lower bound is > t; "H0: θ ≥ t rejected" means the upper bound is < t; "H0: θ outside (a, b) rejected" means the interval lies inside (a, b) (two one-sided tests).
- A point condition (for example s_ID ≥ 0.5) is an effect-size condition on the point estimate. A point estimate that is not a number fails it.
- **Combination.** A factorial line (J-A1 to J-A8d) holds "in every evaluable model". This is an intersection-union test at the 95 % intervals without correction, combined by the common rule (G3): the line is NOT MET as soon as one evaluable model does not meet it, whatever the number of evaluable models. Otherwise it is MET if there are at least 3 evaluable models, at least one of them a fresh family (Llama, Gemma, or Yi as fallback), and NOT EVALUABLE if not (`comb_models`). A head line is NOT MET if it is not met in Qwen2.5-7B or in Mistral-7B, MET if it is met in both, and NOT EVALUABLE otherwise (`comb_both`; the part requires both models). A model without a results file (not run) is not an evaluable model.
- **Holm sensitivity** (reported; no verdict uses it). The family is every interval component (one bound of one statistic in one model; an interval-inside criterion gives two) of Part A's R-class lines, in every model where the line has a verdict (MET or NOT MET, including NOT MET by the rule on undefined quantities below). Each component's one-sided p is the normal approximation from its bootstrap standard error se (the standard deviation of the resamples): p = Φ(−(est − bound)/se) for H1 θ > bound and Φ((est − bound)/se) for H1 θ < bound (se 0 is taken as 1e-12, so p is 0, 1 or 0.5 by the side of the bound the estimate lies on). A component whose estimate or se is not a number is left out of the family and is not rejected. Holm's step-down at a familywise one-sided 0.025 is applied by `analysis/stage8_holm.py`, the helper shared by the four parts. The scorer prints, per R line, every component whose decision differs between the interval rule and Holm, and the line's combined verdict when the Holm decisions replace the interval decisions (the point conditions and the rule on undefined quantities unchanged).
- **Size floors.** J-A-G2: at least 80 competent items per format. J-A1b: at least 40 items with |P| ≥ 3. J-A8: at least 40 items in its population. J-A8d: at least 40 items with g1(K_Z) = dec_B in each format. Lines on prior-free populations (J-A7, J-A7b, J-A-HA3a, J-A-HA3b): at least 30 items. Below a floor the line is NOT EVALUABLE in that model. The J-A8d floor counts items selected by the K_Z row under test; it is kept as a size floor (an exception to the common evaluability rule) because too few such items leave the hybrid rate unestimated rather than contradicted.
- **Undefined by the arm itself** (the common rule). A J-A-G4 failure (s_ID's own denominator, mean ID_K + mean ID_V, below 2 nats in a format whose s_ID the line uses) and, for J-A8, d_KV^cont(NOM) below 2 nats (I_cont's denominator) count against the line: NOT MET in that model, with the reason printed, never NOT EVALUABLE. A line that is not evaluable in a model for another reason (a gate, rows not run, a size floor) is NOT EVALUABLE there even when its quantity is also undefined.
- **Scorer checks that do not change verdicts.** A line that cannot be computed in a model (a missing record) is NOT EVALUABLE there, with the error printed. A provenance or population MISMATCH (results files from more than one commit; a frames, items, SQuAD, stage-6 or μ hash that differs; outside TEST_MODE a dtype other than BF16, or an attention other than sdpa, eager for Gemma-2 and for the heads; a model's valid and skipped items that do not make up E, or a population hash that differs; head ranking items that are not the first min(60, n_R) valid R items in rank order, or evaluation items that are not min(80, n_E) in number, where n_R and n_E are the numbers of valid R and E items the heads file records (`n_valid`; without that record, 60 and 80); a missing preflight.json or a TEST_MODE file outside TEST_MODE) is printed and makes the scorer exit with status 2; the verdicts are printed as computed.

#### Gates

- **J-A-G0, FP32 exactness** (CPU, Qwen2.5-0.5B-Instruct, 1e-4 in log-probabilities; `tests/test_natural_clamp.py`, 14 tests; `tests/test_kvquant.py`, 4 tests; all must pass, none skipped, none failing). The scorer reads the last pytest session in the pipeline's logs that ran these files; with no such log the gate is NOT EVALUABLE and counts as failed. The checks:
  - The KV_S row equals the unclamped S run at the decision position and on every continuation token (decision log-probabilities, argmax, option mass, continuation log-probabilities), in NOM, OPTA and LETA, for a PERSON and a NUMBER item. The ID row equals the clean B run, and the capture batch's own log-probabilities equal the plain runs.
  - A batch equals its rows run singly.
  - The cue-conflict, Z, K_S, V_X, onset and piece rows equal clamps built by hand from separately captured runs.
  - In NOM and LETA, greedy generation under KV_S equals greedy generation of the S prompt and the cache-free stepwise reference, and under K_S the cache path equals the stepwise reference. The argmax chain over the greedy tokens holds. The early stop never changes the answer class or g1 and only shortens the generation, in the model's own generations; and (tokenizer only) on every committed item, for many continuations of each candidate and letter, the answer class at the first token prefix where the stop fires equals the class of the whole continuation.
  - With `<|im_end|>` removed from the generation config's EOS list (as Gemma-2's `<end_of_turn>` is missing from its own), the decoder still stops at it (the end-of-turn rule of pass 3), and the rows name B, S and X with the same tokens and g1 as with the full list. An answer followed by `<|im_end|>` and more text is read as B once cut at the first special token, and as "other" without the cut.
  - Frame " " reproduces the build's continuation tokens on every item (Qwen tokenizer), and frame " **" separates cleanly on a test item (w ends inside the frame; five distinct decision tokens).
  - KIVI: the quantizer equals an independent per-group reference; 16 bits is the identity; the hooked run equals a layer-by-layer reference built from captured tables, the reference quantizer and ckeys.clamp; the recorded relative errors are the reference's; a mixed batch equals its rows; decoding with the cache equals the stepwise reference.

  J-A-G0 failing makes J-A1 to J-A8d, J-A-HA3a and J-A-HA3b NOT EVALUABLE in every model (J-A-HA1 and J-A-HA2 depend on J-A-HA-G0). A failing test also stops the pipeline before any model is loaded.
- **J-A-G1, BF16 floor** (per model and format, all valid items): mean over items of max_Y |lp_Y(KV_S) − lp_Y(S run)| ≤ 0.3 nats, Y over the format's decision tokens and the S run the capture batch's S row; the same for ID against the B row; and the KV_S generation equals the S-prompt generation, token for token, in ≥ 97 % of items.
- **J-A-G2, competence:** n_comp(f) ≥ 80.
- **J-A-G3, emitted form** (competent items): median p(dec_B | ID) ≥ 0.5; and g1(ID) = dec_B, g1(KV_S) = dec_S and g1(KV_X) = dec_X each in ≥ 95 % of items.
- **J-A-G4, s_ID defined** (per line and cell): mean ID_K + mean ID_V ≥ 2 nats on the line's population, in each format whose s_ID the line uses. A failure counts against the line (NOT MET, see Statistics), not against its evaluability.
- **J-A-HA-G0:** `tests/test_natural_heads.py` (5 tests, all passing). The checks: every head in every row seeing K_S at the whole span equals the full K_S clamp; no head equals clean; every head in the rows G equals RowSplice(G) with the span table; a batch whose size equals |P| equals its rows (the span axis is never taken for the batch axis); ablation with each head's own o_proj input as its mean equals the clean run; ablation with a fixed mean equals an independent o_proj pre-hook; the argmax chain over greedy tokens holds and fails for a changed token; a3 of a self-clamp is 0; DLA equals o_proj applied to the head's slice alone; Q+ starts after the question, reaches the last answer row, contains G and does not contain P.
- **J-A-HA-G1, BF16 floor** (the first 80 valid E items): mean |m(none) − m(clean pass)| and mean |m(all_T) − m(full clamp pass)| ≤ max(0.5 nats, 0.02 × d_full), with none and all_T from N*'s sufficiency batch.
- **J-A-HA-G2, the option rows are the readers:** d_full ≥ 3 nats and d_G/d_full ≥ 0.6 (point estimates). If it fails, every head line is NOT EVALUABLE in that model.
- **Evaluability.** A model is evaluable for a line when J-A-G0 passes; J-A-G1, G2 and G3 pass in every format the line uses; the rows the line needs were run (not reduced or skipped at the deadline); and the line's size floor is met. J-A-G4 and J-A8's d_KV^cont floor decide NOT MET, not evaluability. A head line in a model needs J-A-HA-G0 to G2 (and k* among KS); HA3a and HA3b also need that model's factorial, J-A-G0, and J-A-G1 to G3 in OPTA (HA3a) or NOM (HA3b).

#### Confirmatory lines

Classes follow G4 and the recorded prior, P(MET | evaluable) under the combination rule, recorded before any run: L = implied by data in hand on the same models and material, with prior ≥ 0.9; M = prior ≥ 0.8 (an extrapolation of a regularity seen in every model in hand); R = prior < 0.8. A line's class never disagrees with its prior (`tests/test_stage8a_score.py` checks this). J-A3 is derived from J-A1 and J-A2: it is reported with a verdict word but not counted.

| Code | Class | Prior | Population | Criterion (all parts must hold in the model) | Prior's basis (data in hand) |
|---|---|---|---|---|---|
| J-A1 | M | 0.80 | C(OPTA) | s_ID ≥ 0.5 (point); H0: s_ID ≤ 0.35 rejected; H0: ID_K ≤ 0 rejected | templated OPTIONS-AFTER s_ID 0.79–0.87 at every 7–14B model in hand; natural 0.5B pilot 0.43, above templated 1.5B 0.36; simulated power 0.72–0.82 at a true 0.55 for four models |
| J-A1b | R | 0.55 | C(OPTA), \|P\| ≥ 3 (n ≥ 40) | s_ID ≥ 0.4 (point); H0: s_ID ≤ 0.25 rejected | no data on long spans; smaller n |
| J-A1c | M | 0.85 | C(LETA) | s_ID ≥ 0.5 (point); H0: s_ID ≤ 0.35 rejected | lettered formats are the most key-borne in hand (24B LETTERS-AFTER ψ_V 0.018; natural d_K 18.7 nats) |
| J-A2 | M | 0.85 | C(NOM) | s_ID ≤ 0.2 (point); H0: s_ID ≥ 0.3 rejected; H0: ID_V ≤ 0 rejected | natural 0.5B pilot s_ID −0.01; templated NO-MENTION key read ≤ ~1 nat in every model |
| J-A3 | derived | 0.70 | C(OPTA, NOM) | s_ID(OPTA) − s_ID(NOM) ≥ 0.4 (point); H0: difference ≤ 0 rejected | J-A1 and J-A2 together give ≥ 0.3; 0.4 needs more |
| J-A4 | L | 0.90 | C(OPTA, OPTB) | ID_K(OPTB)/ID_K(OPTA) ≤ 0.15 (point); H0: ratio ≥ 0.25 rejected; H0: ID_K(OPTA) − ID_K(OPTB) ≤ 0 rejected | LIST-BEFORE key read −0.001 at 24B; OPTB pilot item −0.52 nats against ID_V 6.23 (the causal mask; the second-order route is the only open one) |
| J-A5 | R | 0.50 | C(MENA, NOM) | s_ID(MENA) − s_ID(NOM) ≥ 0.10 (point); H0: difference ≤ 0 rejected; H0: ID_K(MENA) ≤ 0 rejected | templated sentence s_ID 0.19–0.45 at 7B+, natural MENA pilot item s_ID 0.09 |
| J-A5B | M | 0.80 | C(MENA, MENB) | H0: ID_K(MENA) − ID_K(MENB) ≤ 0 rejected | before-arms are near 0 by the mask; the after-arm read was positive in every templated model |
| J-A6a | R | 0.45 | C(OPTA) | key-source rate ≥ 0.6 (point); H0: rate ≤ 0.5 rejected; key − value ≥ 0.3 (point); H0: key − value ≤ 0 rejected | 0.5B OPT-A flips K_S 0.14 vs V_S 0.57 (value wins at 0.5B); 1.5B pilot: 1 competent item, both cue rows not key-source |
| J-A6b | R | 0.60 | C(LETA) | as J-A6a, on letters | letters are key-borne in every templated model |
| J-A6c | R | 0.45 | C(NOM) | value-source rate ≥ 0.6 (point); H0: rate ≤ 0.5 rejected; value − key ≥ 0.3 (point); H0: value − key ≤ 0 rejected | 0.5B NOM flips V_S 0.86 vs K_S 0.00 at the decision token, but full-match flips 0.43 (hybrids); the account predicts hybrid continuations when key and value disagree (J-A8) |
| J-A6d | R | 0.45 | C(OPTA) for the OPTA half, C(NOM) for the NOM half | OPTA: S-rate of KS_VZ ≥ 0.6 (point), H0: ≤ 0.5 rejected; NOM: S-rate ≤ 0.2 (point), H0: ≥ 0.3 rejected | NOM half near-certain; OPTA half untested, and the B prior pulls against it (KS_VZ answered B in the 1.5B pilot item) |
| J-A6e | R | 0.40 | C(OPTA) | S-rate of KZ_VS ≥ 0.5 (point); H0: S-rate ≤ 0.4 rejected | the account: a key that matches no option leaves the lookup silent and the value copy gives S; the prior says B; 1.5B pilot item: "other" |
| J-A7 | R | 0.30 | PF(NOM, OPTA) (n ≥ 30) | DiD ≥ 0.10 (point); H0: DiD ≤ 0 rejected | no data in hand on cache quantization |
| J-A7b | R | 0.35 | PF(NOM, OPTA) (n ≥ 30) | drop_NOM(V) − drop_OPTA(V) ≥ 0.10 (point); H0: ≤ 0 rejected | as J-A7 |
| J-A8 | M | 0.80 | C(NOM, OPTA) ∩ leak-free PERSON/PLACE (FT) with ≥ 1 continuation token differing from c_B (n ≥ 40); d_KV^cont(NOM) < 2 nats counts as NOT MET (I_cont undefined) | (a) I_cont(NOM) ≥ 0.5 (point), H0: I_cont ≤ 0.3 rejected; (b) I_dec(NOM) ≤ 0.25 (point), H0: I_dec ≥ 0.35 rejected; (c) \|d_KV^cont(OPTA)/d_KV^cont(NOM)\| ≤ 0.2 (point), H0: ratio outside (−0.3, 0.3) rejected | 0.5B pilot: I_cont 1.41 vs I_dec 0.04; OPT-A continuation d_KV 0.00; induction-head theory |
| J-A8d | R | 0.60 | C(NOM, OPTA) ∩ PERSON/PLACE with ≥ 2 tokens of c_B from the decision token (≥ 40 items with g1(K_Z) = dec_B per format) | h(K_Z, NOM) ≥ 0.2 (point), H0: ≤ 0.1 rejected; h(K_Z, OPTA) ≤ 0.05 (point), H0: ≥ 0.1 rejected | 0.5B hybrids seen under V_S ("Robert M. Newton"); memorised completions may lower h |
| J-A-HA1 | R | 0.75 | first 80 valid E items, OPTA | R_N(k*) ≥ 0.7 (point), H0: ≤ 0.6 rejected; KO_N(k*) ≥ 0.7 (point), H0: ≤ 0.6 rejected; mean random R(k*) ≤ 0.15 and KO(k*) ≤ 0.15 (points) | stage 6 (templated, 7B): R 0.967/0.942, KO 0.977/0.971, random ≤ 0.011; multi-token options untested |
| J-A-HA2 | R | 0.40 | as HA1 | R_T(k*) ≥ 0.5 (point), H0: ≤ 0.35 rejected; KO_T(k*) ≥ 0.5 (point), H0: ≤ 0.35 rejected | no data on transfer to natural text |
| J-A-HA3a | R | 0.30 | PF(OPTA) among the ablated items (n ≥ 30) | drop(N*) = acc(none) − acc(N*) of the faithful argmax chain over c_S ≥ 0.3 (point), H0: ≤ 0 rejected; each random set's drop ≤ 0.1 (point) | stage 6 H3 kept the base answer in 0.87–1.00 of stories after the same ablation; stage 7 I6 lost part of the behaviour at 24B (t 0.55) |
| J-A-HA3b | R | 0.65 | PF(NOM) among the ablated items (n ≥ 30) | Δ = acc(N*) − acc(none) of the decision argmax (= dec_S) and of the chain over c_S after the decision token: \|Δ\| ≤ 0.05 (points), H0: Δ outside (−0.1, 0.1) rejected, for each | N* is selected at the option rows, which NOM lacks; Q+ includes the answer rows, so the line is not true by construction |

Class totals: L 1 line (sum of priors 0.90: J-A4), M 5 (4.10: J-A1, J-A1c, J-A2, J-A5B, J-A8), R 14 (6.75); J-A3 derived. Every line is an account line; Part A has no measurement-validity line.

**Pre-written expectations, not thresholds.** s_ID(OPTA) 0.6–0.8; s_ID(NOM) ≤ 0.05; s_ID(MENA) 0.15–0.3; key-source rate in OPTA 0.6–0.8; value-source rate in NOM 0.5–0.7 (hybrids); KIVI relative error about 0.15 for keys and 0.4 for values (0.5B); prior-free n per model 30–60.

**Copy fallback (J-A6e), read two-sided.** S-rate ≥ 0.5: the value copy answers when the key matches no option. B-rate ≥ 0.5: the question-and-options prior answers when the lookup is silent. "other" (including D or X) ≥ 0.5: neither. The scorer prints all rates.

#### What each primary line means for the paper (pre-written)

The title is "Looked Up or Copied? Later Mentions Decide Whether a Model Reads an In-Context Value Through Its Key or Its Value", with a templates-only fallback ("... on Templated Stories"). The main natural-data figure is Fig. N: (a) s_ID by format, with the templated values as ghost markers; (b) cue-conflict rates; (c) KIVI drops by channel and format; (d) d_K, d_V, d_KV at the decision token and on the continuation.

| Line | MET | NOT MET |
|---|---|---|
| J-A1 (with J-A2; J-A3 derived) | Abstract: "On natural reading-comprehension passages with multi-token answers, in four model families including two not used before, a later options list makes the model read the passage entity through its cached keys (s_ID ...), and without it through its values (s_ID ...)." Title kept. Table 1 row "natural passages, options after: key read" marked supported. Fig. N(a). | Graded (A-12). If J-A3 is met and s_ID(OPTA) is in [0.35, 0.5) with lower bound > 0.2: "a substantial but not dominant key read on natural text"; the title is kept; Table 1 row "partly". Otherwise (J-A3 not met, or s_ID lower bound < 0.2): the key lookup is a property of templated prompts; the title takes the templates-only fallback; Table 1 row "not shown on natural text"; Fig. N(a) is kept as a boundary. J-A2 NOT MET: "the copy half of the dichotomy is withdrawn for natural text". |
| J-A5 | Abstract: "... and so does a natural sentence that mentions the candidates after the passage." Table 1 row "natural mention sentence" supported. Fig. N(a), MENA bar. | The abstract says "a later options list" where it said "later mentions" for natural text; the templated sentence effect rests on Part B; Table 1 row "not shown on natural text". |
| J-A6a and J-A6c (with J-A6b, d, e) | Abstract: "With the key from one entity and the value from another, the same cache answers the key's entity under a later options list and the value's entity in free form." Table 1 row "behaviour follows the channel" supported. Fig. N(b). | The channel attributions are not shown to decide behaviour on natural text. The behavioural sentence is dropped from the abstract; Fig. N(b) is reported as a boundary with the decision-token rates. If only J-A6c fails with high "other" rates, the text reports hybrids, as J-A8 predicts. |
| J-A7 | Practitioner sentence: "Quantizing the cached values of a passage to 2 bits breaks free-form answers more than multiple-choice answers, beyond what the same quantization of the keys does; KV-cache methods evaluated in multiple-choice format can miss it." Fig. N(c). | "KV-cache methods" is deleted from the practitioner sentence; Fig. N(c) is reported with the reconstruction errors as an exploratory panel. |
| J-A-HA3a (with HA1, HA2) | Table "natural readers": "the reader heads found at the option rows on natural passages are necessary for faithful multiple-choice answers; random sets are not". | "The natural readers can be removed without losing faithful MCQ answers: a copy route keeps them" (the H3 pattern). HA1 NOT MET: the natural readers are not a sparse 5 % set. HA2 NOT MET: the template readers do not transfer to natural text. |
| J-A8 | Text only (no abstract clause): "a third regime, a channel-level replication of induction-head K-composition (Elhage et al. 2021; Olsson et al. 2022): the key addresses the next piece and the value supplies it". Fig. N(d). | "Multi-token continuations on natural text are not read by K-composition at the channel level"; Fig. N(d) is kept as a boundary. |

Every verdict, including failures, enters the claim-status table. The abstract states the met rate among R lines.

#### Reported (no verdict)

- J-A3 (derived) and the A-12 graded reading of J-A1 (per model, from s_ID(OPTA) on C(OPTA) and its interval).
- Every accuracy outcome on all competent items and on prior-free items: the accuracy for B under V_Z, K_Z and KV_Z, and the cue-conflict rates on PF.
- The cue-conflict rates at the decision token (g1 = dec_S or dec_X).
- The earlier A6 flips (φ_dec(K_S), φ_dec(V_S), and full matches) and the earlier A7 accuracies.
- n_comp per format; P_CB(B) by closed-book format and type.
- s_ID and the option mass on all valid items per format.
- The KIVI relative errors and faithful accuracies on all competent items.
- d_K^cont(NOM) on the J-A8 population: its sign is reported; Part D owns the sign account.
- The heads: faithful and clean-B chain accuracies and the option margin m_B under every condition.

#### Exploratory (no verdicts)

- E1: onset K_S/V_S from round(0.3 L) as fractions of the full-depth effect (OPTA keys, NOM values), reported two-sided.
- E2: piece rows (first span position against the rest).
- E3: moderators of s_ID(OPTA) and s_ID(NOM): type, D_in (two-sided), and |P| in {1, 2, ≥ 3}.
- E4: the YEAR stratum in NOM and OPTA, rows ID, K_S, V_S and KV_S at the first token where c_S differs from c_B. Written expectation: the decision digit is value-only in every tokenizer (I_dec ≈ 0), because the span tokens before it have identical K/V in the B and S runs.
- LEAK stratum: ID_K, ID_V and s_ID in NOM and OPTA on its competent items (a natural in-passage re-mention).
- E5: KIVI at 3 bits (K3, V3; NOM, OPTA) on the S prompt, and none/K2/V2 on the unmodified B prompt (benchmark accuracy; NOM, OPTA, LETA), on every valid E item.
- E6: full-string s_ID from the decision token on, with the X-target batch.
- Heads: the R/KO curves, the layer profile, C* (copy heads by DLA) ablated in OPTA and NOM, T* ablation, and ID_K/ID_V with N* ablated at Q+ (OPTA, 80 items).
- The per-model s_ID(OPTA) with its unweighted mean across models; the Holm sensitivity analysis is printed with the summary.

#### Seen before finalisation

No GPU output of any part-A code. Every model run below was on this container's CPU (4 cores), with Qwen2.5-0.5B-Instruct or Qwen2.5-1.5B-Instruct.
1. **Design pilots** (`pilots/stage8/partA/`):
   - Data scan: 3,081 typed answers occur exactly once; an earlier build (v5, before the leak and place-class rules) had 706 FT items and 529 YEAR items. Its four-tokenizer check found 697 valid items, median prompt 207–255 tokens (maximum about 770), single-token spans 14 % (Qwen), 11 % (Mistral), 36 % (Llama), 16 % (Gemma), and E = 239 items (superseded by the current 185).
   - Exactness at 0.5B FP32: KV_S reproduced the S run at the decision token within 1e-5 to 3e-5 nats (4 item-format checks).
   - Effect pilot: 0.5B, passages windowed to ≤ 320 characters, 7 items in NOM and OPT-A and 1 item in OPT-B and MEN-A; 6 of the 7 items ran in BF16, a deviation from the FP32 pilot policy.
     - NOM: ID_K −0.13 (sd 0.58), ID_V +11.01 (sd 4.94), s_ID −0.01; competent 5/7; median p(dec_B) 0.96.
     - OPT-A: ID_K +2.64 (sd 1.78), ID_V +3.56 (sd 1.39), s_ID 0.43; competent 6/7; p(dec_B) 0.99; mass 0.996.
     - The single item: OPT-B ID_K −0.52 against ID_V 6.23; MEN-A ID_K +0.61 against ID_V 5.89.
     - Decision flips: NOM K_S 0.00 / V_S 0.86; OPT-A K_S 0.14 / V_S 0.57. Full-match flips: NOM V_S 0.43, with hybrids such as "Robert M. Newton".
   - Continuation pilot: 0.5B FP32, 10 multi-token items from the first item file, including since-dropped ORG items.
     - NOM decision token: d_K +1.07, d_V +13.90, d_KV +15.62, I 0.04.
     - NOM continuation: d_K −10.24, d_V +3.95, d_KV +15.34, I 1.41.
     - OPT-A decision token: d_K +9.21, d_V +8.49, d_KV +11.39. OPT-A continuation: |d| ≤ 0.31 in every row, d_KV 0.00.
   - Power simulation (35 clusters, ICC 0.15): A1 power 0.92–0.95 per model and 0.72–0.82 for four models at a true 0.55.
   - The stage-6 T* rankings were extracted, and the stage-1 GPU log shows about 10k prompt tokens/s at Qwen2.5-7B.
2. **Critique pilots** (`pilots/stage8/critic_natural/`):
   - Closed book: Qwen2.5-1.5B FP32 on the 239 E items of the v5 build.
     - Options wording: B wins 65.3 % (PERSON 80.9 %, PLACE 81.8 %, NUMBER 20.6 %); the renormalised p(B) exceeds 0.9 in 49.4 %; mean four-option mass 0.985.
     - Free-form wording: B wins 62.8 %; mass 0.260.
   - Cue conflict: Qwen2.5-1.5B FP32, windowed, 2 PERSON/PLACE items; stopped at the 14-minute cap.
     - Item 1, OPT-A (competent): KV_Z → B, K_S → B, V_S → B, KS_VX → B, KX_VS → "Myhill" (other), KZ_VS → other, KS_VZ → B.
     - Item 1, NOM: KV_X → other, KS_VX → "Mike Myhill" (other), KX_VS → "Pierre André Grillet" (other).
     - Item 2: not competent (KV_S → D in OPT-A; every NOM row "Germany").
   - Item checks on the v5 build:
     - 36 of 110 PERSON E items had a content word of e_B elsewhere in the passage (31 after the span; 73 after over all valid items). This led to the LEAK stratum.
     - 8 of 66 E PLACE items had S and X of the question's place class. This led to the place classes.
3. **This build:**
   - Unit tests: `tests/test_natural_clamp.py` 14 passed, `tests/test_kvquant.py` 4, `tests/test_natural_heads.py` 5, `tests/test_stage8a_score.py` 15; the shared `tests/test_stage8_populations.py` 5 and `tests/test_stage8_holm.py` 7.
   - Generation configs (official files, by their git blob ids in `scripts/stage8_models.json`): Gemma-2-9b-it lists EOS 1 only and Yi-1.5-9B-Chat EOS 2 only. At Qwen2.5-0.5B with `<|im_end|>` removed from the EOS list, the answers " Bing Crosby" and " 24" were followed by `<|im_end|>` and were parsed "other" before the special-token rule (the review fixes, commit c04b3eb, made the shared decoder also stop at the chat end-of-turn tokens; commit aa92bb6 updated the gate test to match).
   - The 1.5B closed-book pilot covers 123 of the 185 E items: the options-wording argmax is not B in 59 of them (NUMBER 44 of 56, PERSON 11 of 49, PLACE 4 of 18), so the prior-free populations are mostly NUMBER items.
   - The item rebuild check: byte-identical, about 6 CPU minutes.
   - A tokenizer-only frame check on the 256 R and E items (no model):
     - Frame " " equals the build in all five tokenizers.
     - "**" (no space) is invalid for every item in every tokenizer (":" merges with "**").
     - "" is invalid for 90 items (Qwen, Llama).
     - " **" is invalid for 2 (Qwen, Llama), 18 (Mistral) and 12 (Yi) items.
   - `TEST_MODE` plumbing runs at Qwen2.5-0.5B FP32 with windowed passages and 2 E items, both NUMBER items:
     - Frame " " (4 of 4 R generations).
     - s_ID: OPTA +0.337 (2 items), NOM −0.009, LETA +0.468, MENA +0.158, OPTB −0.023, MENB −0.018.
     - Closed-book argmax never B.
     - KIVI relative error: keys 0.15, values 0.43–0.44.
     - The verdict lines of these runs are plumbing checks (size floors waived), not results.
   - The final `TEST_MODE` run of `scripts/gpu_stage8a.sh` on commit 84e4f05, whose code is the finalised code (CPU, two threads, sharing the CPU with Part B's run; 73 minutes) completed with no failed or skipped step: pytest 67 passed in 25 minutes (J-A-G0 and J-A-HA-G0 MET); the preflight rebuild was byte-identical; the score read "2 MET, 10 NOT MET, 8 NOT EVALUABLE of 20 counted lines; provenance OK; population OK" (plumbing). An earlier run on the reviewed code (commit aa92bb6; CPU, two threads, the four keys all at Qwen2.5-0.5B; 75 minutes) completed with no failed or skipped step: pytest 50 passed in 25 minutes; the preflight rebuild was byte-identical; every frames, factorial, heads and explore step completed; the score read "2 MET, 10 NOT MET, 8 NOT EVALUABLE of 20 counted lines; provenance OK; population OK" (plumbing at n = 2, size floors waived). An earlier full run on the build before review (about 65 minutes) had also completed with no failed step:
     - pytest: 45 tests passed in 15 minutes;
     - the preflight rebuild was byte-identical;
     - every frames, factorial, heads and explore step completed, and so did the score, the manifest and the archive.
     - Its score file, a plumbing check: the four keys gave identical numbers. J-A2 and J-A4 MET; J-A1, J-A5 and the J-A6, J-A7 lines NOT MET on 1–2 items; the other lines NOT EVALUABLE.
     - Heads at 0.5B with k* = 5: R_N(5) 0.66, KO_N(5) 0.82, R_T(5) −0.01; J-A-HA-G2 not met (d_full 2.34 nats < 3).
     - With N* ablated at Q+: ID_K 1.26 → 0.19 nats, ID_V 2.48 → 1.34 nats.
     - Exploratory, 2 items: OPTA key read at the first span position 0.02 of the full and at the rest 0.93; YEAR I_dec −0.01 (NOM).

#### Compute (A100-80GB, BF16; estimated from token counts)

Mean prompt lengths over R and E (Qwen tokenizer) are NOM 229, OPTA 246, OPTB 246, MENA 248, MENB 248 and LETA 255 tokens (maximum about 780; other tokenizers within ±10 %).
- **Factorial, per item**, about 37.5k prompt-token passes:
  - NOM, OPTA and LETA, about 8.3k–9.2k each: capture 4 rows, scoring 14 rows, generation 15 rows, KIVI 3 rows.
  - OPTB, MENA and MENB, about 3.7k each: capture 4 rows, scoring 7 rows, generation 4 rows.
  - About 144 greedy decode steps, before the early stop.
- **Factorial, per model:** 185 items give about 6.9M prompt tokens (about 11.5 min at 10k tokens/s) and about 26k decode steps. At about 30 ms each and about 60 % remaining after the early stop, that is about 8 min. With closed book and hooks, about 22 min for Llama, Qwen and Mistral, and about 33 min for Gemma-2 (eager, 42 layers).
- **Frames:** about 2 min per model.
- **Heads, per model** (eager): rank about 1 min; curves on 80 items about 4.8M token passes (doubled layers in the splice), about 15 min; ablation about 0.7M tokens, about 3 min; explore about 1 min. About 20 min.
- **Exploratory, per model:** about 12 min.
- **Overhead:** pytest about 10 min; preflight about 6 min; fetches about 4 × 4 min; about 12 model loads at about 1 min.
- **Total:** core about 3.3 GPU-h; with every exploratory pass about 4.1 h. DEADLINE_H defaults to 4.5. At about $2 per A100-hour, about $7–9.

#### Commands

```
J=$(git log --format=%H -1 --grep='^Finalise preregistration J') && git checkout "$J"
bash scripts/gpu_stage8a.sh                 # HF_TOKEN optional; SQUAD=<local dev-v1.1.json> optional
TEST_MODE=1 bash scripts/gpu_stage8a.sh     # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2), outputs TEST_<key> in results/gpu_stage8a_test
python analysis/stage8a_score.py --results results/gpu_stage8a     # re-score an archive
```
The pipeline runs:
1. pytest (the J-A-G0 and J-A-HA-G0 files, the scorer's tests, `tests/test_generate.py`, `tests/test_clamp.py`, `tests/test_head_splice.py`, and the tests every part runs, `tests/test_stage8_populations.py` and `tests/test_stage8_holm.py`; 67 tests), run in every session (TESTS=0 skips them only after a pass at the same commit on the same host); a failure stops the script;
2. preflight (the SQuAD download into `results/gpu_stage8a/squad/` unless SQUAD is set, and the item rebuild); a failure stops the script;
3. for each key in llama8, gemma9, qwen7, mistral7: s8_fetch, frames, factorial, heads (qwen7 and mistral7), explore (when time allows), s8_drop; yi9 in the place of llama8 or gemma9 when that model's fetch returns status 1 (a verification failure) and no output of it exists, or when `FALLBACK.txt` records such a replacement from an earlier session;
4. the score: `results/gpu_stage8a/STAGE8A_SCORE.txt`, `results/gpu_stage8a/MANIFEST.sha256` and the archive `gpu_stage8a_results.tgz` in the repository root.

The step names (for `FORCE_STEPS`) are preflight, frames_<key>, factorial_<key>, heads_<key> and explore_<key>; pytest and the score run in every session.

### Part B. Fresh samples and new families, scored on the forms the models emit

**Code.** `ckeys/fresh.py` (populations F, C and S0; the second lexicon; the eight sentences and their null versions; the per-core candidate order; the prompt builders; the per-item answer pattern, stop rule, frame extraction and form sets), `experiments/fresh_factorial.py` (stages tokcheck, calib, g3, eval), `experiments/paper1_frames.py` (the flag `--score E`, with `--frames` and `--model-dir`, for J-B8; the default path is unchanged), `analysis/stage8b_score.py` with `analysis/stage8b_parts/` (`stats.py`, `data.py`, `lines.py`, `tables.py`), `scripts/gpu_stage8b.sh` (pipeline; it sources `scripts/stage8_common.sh`). Tests: `tests/test_fresh.py` and `tests/test_fresh_factorial.py` (Gate J-B-G0), `tests/test_stage8b_score.py` (the scorer on synthetic inputs), and the cross-part `tests/test_stage8_populations.py` (G6) and `tests/test_stage8_holm.py` (the Holm helper), which J-B-G0 also runs. Shared code used unchanged: `ckeys/surface.py`, `ckeys/generate.py`, `ckeys/clamp.py`, `ckeys/encoding.py`, `ckeys/story.py`, `experiments/format_factorial.py` (`row_specs`, `LABEL`, `run_item`, `provenance`), `experiments/stage7_link.py` (`release_check`, for J-B8), `analysis/stage8_holm.py` (the Holm helper of every part).

**Purpose.** Part B answers four objections.
- *Objection 5 (the sentence effect was confirmed on the cores and models where it was found).* The 2×2 and the no-mention arm are re-run on 150 story cores never used before, in the four models of stages 1 and 3b (P4, a replication) and in four model families never examined (N4, out of sample). The fresh population is new in wording and lexicon, not only in seed: eight new neutral sentences, a second six-word lexicon in half of the cores, and a per-core candidate order.
- *Objection 4 (the free-form formats were scored on lower-case " w" tokens that carry 0.00–0.03 of the probability).* Every candidate is scored as the exact chain-rule probability summed over the surface forms the models emit (E), in the same forward pass as the published lower-case score (L) and the reviewer-named 12-form score (Σ). Coverage is measured on every row and gates every E-based criterion. A behavioural measure on greedy generations (β) is added, and the published scale-free results are re-measured under E on the original cores (J-B6).
- *"7 of 10 models are Qwen".* Llama-3.1-8B-Instruct, Gemma-2-9B-it, Phi-4 and Falcon3-7B-Instruct are added, plus Gemma-2-2B-it for the scale side of the sentence claim (J-B-SMALL).
- *Measurement validity in the sentence arm.* A null sentence of the same position, syntax and length that names no candidate (POST-NULL) separates "a re-mention opens the key" from "any intervening text opens it" (J-B-NULL), and a behavioural line tests whether the sentence read changes generated answers (J-B5b).

Part B does not test natural text (Part A), interventions (Part C) or the mechanism of the negative reads (Part D). J-B9 of the design is dropped (B-1).

#### Populations

All populations are lists of story cores of `ckeys.story.make_cores` (fields agent, other, object, distractor, initial, distractor_location, base, source). A core is compared by its full tuple. U (common part, G6) contains every core of stages 1–7.

| Population | Rule | Size | Hash (pinned in `ckeys/fresh.py`) |
|---|---|---|---|
| F (evaluation) | the first 150 cores of the stream make_cores(·, Random(20261013)) whose tuple is in neither U nor earlier in the stream | 150 | `pop_hash` e87047c9c877a21db89bf5081d082de748ea5d77b620e33135d178c3bf24b14f |
| C (calibration only) | the first 30 cores of the stream make_cores(·, Random(20261014)) in neither U, F nor earlier in the stream | 30 | a625fd13d1dd67bc0c01c3a173807c7b71ee8347451c139d93ffc20f1c6486e9 |
| S0 (discovery sample, JB6 and JB-G2) | make_cores(150, Random(0)), the cores of stages 1 and 3b, re-measured on purpose | 150 | 48bb0a3ad22463ee1831cabaef714d87b2a7cb7f721e23d877111f19ba0d6900 |

- The hashes are `pop_hash` = sha256(json.dumps([[*tuple, lexicon, sentence, order], …])) in population order (sentence null for S0). The sha256 of S0's tuples alone is fdd1bf1b… (common part). U's hash abd1f053… is asserted by the tests.
- The stream make_cores(n, rng) draws the same cores as n calls of make_cores(1, rng). F uses the first 151 draws of its stream (one draw is in U; 4 of the first 400 draws are). F covers all 30 ordered (base, source) pairs and all 20 objects. No pilot touched seeds 20261013 or 20261014.
- **Lexicon.** In F and in C, the half of the cores with the smallest h = sha256(json.dumps(core tuple)) (ranks 0 … n/2 − 1 by h) use LEX2 = (bin, crate, tray, jar, bucket, chest); the rest use LEX1 = (box, basket, shelf, drawer, cabinet, closet). A core is drawn in LEX1 and rendered by index (LEX1[i] → LEX2[i]), so every rule of make_cores and of pick_x holds in either lexicon. F: 75 and 75; C: 15 and 15. Disjointness from U is checked on the LEX1 tuple, which is stricter than on the rendered one.
- **Sentence.** SENTENCES[rank(h) mod 8]: F 19, 19, 19, 19, 19, 19, 18, 18 cores; C 4, 4, 4, 4, 4, 4, 3, 3. The same sentence is used in the core's POST and PRE arms, and its null version in POST-NULL.
- **Order.** random.Random(int(h[:16], 16)).sample(range(6), 6): the order of the six candidates in the list and in the sentence, the same in every arm of the core. S0 keeps LEX1, the canonical order and the ROOM sentence.
- **Clusters.** (lexicon, base, source): 55 of the 60 possible cells occur in F, with 1–8 cores each.
- **X.** The third location of a core is `story.pick_x` of its LEX1 core, rendered in its lexicon.

**The eight sentences** (slots filled with the core's six candidates in its order; each also used with NULL_NOUNS = (rug, clock, mirror, poster, radio, globe) in the same order for POST-NULL):
1. "There is a {0}, a {1}, a {2}, a {3}, a {4} and a {5} in the house."
2. "The hallway also has a {0}, a {1}, a {2}, a {3}, a {4} and a {5}."
3. "In the kitchen there are a {0}, a {1}, a {2}, a {3}, a {4} and a {5}."
4. "A {0}, a {1}, a {2}, a {3}, a {4} and a {5} stand along the wall."
5. "The house contains a {0}, a {1}, a {2}, a {3}, a {4} and a {5}."
6. "Nearby there are a {0}, a {1}, a {2}, a {3}, a {4} and a {5}."
7. "The attic holds a {0}, a {1}, a {2}, a {3}, a {4} and a {5}."
8. "Along one wall sit a {0}, a {1}, a {2}, a {3}, a {4} and a {5}."

ROOM ("The room has a box, a basket, a shelf, a drawer, a cabinet and a closet.") is 21 tokens (with its leading space) in every Part-B tokenizer. The eight sentences are +2, +1, +2, +1, 0, 0, 0, +1 tokens longer in the Qwen, OLMo-2, Llama-3.1, Gemma-2, Phi-4 and Falcon3 tokenizers; +2, +1, +2, +1, 0, +1, +1, +1 in Mistral-7B; +2, +1, +2, +1, 0, +1, 0, +1 in Yi-1.5. The length is the same with either lexicon, with the null nouns and in any order, because each of the 18 words is one token after a space and before ",", "." and " and" in every tokenizer (checked by the tests on the cached tokenizers, and on each model's verified files by J-B-G0b). No null noun is a candidate of either lexicon or a story object.

#### Models and sourcing

As in the common part (`scripts/stage8_models.json`; files verified by `scripts/fetch_verified.py`; BF16, sdpa, eager for Gemma-2, transformers 5.18.0, `use_cache=False` in every pass of the trie scorer; the J-B-G3 fallback `score_cached` and the generation use the KV cache).
- P4: `qwen7`, `qwen14`, `mistral7`, `olmo7` (F and S0).
- N4: `llama8`, `gemma9`, `phi4`, `falcon7` (F). The fallback `yi9` takes the slot of a model whose files fail verification or whose tokenizer check (J-B-G0b) fails, before any output of that model exists; at most one slot is replaced (G2). The pipeline records the decision in `COMMIT.txt` ("fallback: yi9 replaces <key>"); a later session of the same run keeps it (the replaced model is not run again). The scorer takes the replaced slot from that record (without one: the first N4 model, in the order above, that `FETCH_FAILED.txt` lists as refused and that has no F results); `yi9` fills the slot only when it has F results, otherwise the slot keeps the replaced model, which has no results, and is not evaluable. A fetch that fails for another reason (disk, manifest) stops the run and triggers no fallback.
- `gemma2b` (F; J-B-SMALL). `mistral24` (J-B8). `qwen1.5`, `qwen3b` (exploratory X2).
- Gemma-2's chat template rejects a system turn; the system text is merged into the user turn and recorded (`WRAPPER_USED`).
- Mistral-Small-24B's tokenizer is loaded with `fix_mistral_regex`, as in stages 2–7.

#### Prompts, rows and scorings

**Arms** (user turn; system turn "You are a helpful assistant."; generation prompt with thinking disabled; assistant prefill "Answer:"; `ckeys.fresh.raw_prompt`). With PREFIX = "Read the story and answer the question.\n\nStory: ", the story and question as `ckeys.story.record` renders them in the core's lexicon, L = "Choices: " + the six candidates in the core's order joined by ", ", and T = "\nAnswer with one word.\nAnswer:":
- AFTER: PREFIX + story + "\nQuestion: " + question + "\n" + L + T;
- BEFORE: "Read the story and answer the question.\n" + L + "\n\nStory: " + story + "\nQuestion: " + question + T;
- NONE: PREFIX + story + "\nQuestion: " + question + T;
- POST: PREFIX + story + " " + sentence + "\nQuestion: " + question + T;
- PRE: PREFIX + sentence + " " + story + "\nQuestion: " + question + T;
- POST-NULL: as POST with the null sentence;
- P1 (S0 only): as AFTER with "\nAnswer with exactly one choice.\nAnswer:".
With LEX1, the canonical order and ROOM, every builder equals `ckeys.encoding.raw_prompt` byte for byte (tested on S0, also LETTER, which no step runs: B-9). Arms run: F all six (AFTER, BEFORE, NONE, POST, PRE, POST-NULL); S0 P1, AFTER, BEFORE, POST, PRE, NONE; C the six F arms and P1.

**Items.** The B, S and X prompts (moved-to location base, source, X) must have one length and differ at exactly one token p, the writing token; an item failing this in an arm is skipped and counted (0 of F, C and S0 in every arm and every Part-B tokenizer at the build). A population's statistics use the items valid in every arm run.

**Rows** (`format_factorial.row_specs`, unchanged): the self-clamp ID; K_S, V_S, KV_S from l0 ∈ {0, round(0.0625 L), round(0.3 L)}; K_X, V_X, KV_X from 0, all on the B prompt; plus the clean B, S and X runs. A row clamps the key and/or value at p in every layer ≥ l0 to the captured value of the donor run (pre-RoPE projection outputs, `ckeys.clamp`; the fused qkv_proj slices at Phi-4).

**Scorings** of candidate w in run Z, from one forward pass per batch row over the prompt and the token trie of every proper prefix of every form (`ckeys.surface.score`; 4D tree mask, explicit position ids):
- L(w | Z) = log p(" w"), the published score;
- Σ(w | Z) = logsumexp over the 12 forms φ + w and φ + W, φ ∈ {" ", "", " The ", " the ", "The ", "the "}, W = w capitalised;
- E(w | Z) = logsumexp over the 32 fixed forms (Φ_Σ plus " In the ", " in the ", " On the ", " on the ", " At the ", " at the ", " Inside the ", " inside the ", " **", "**") and the model's discovered frames, instantiated with the item's names.
A form's log-probability is the exact chain rule over its continuation tokens; identical token sequences are counted once and no sequence may be a proper prefix of another (`FormSet`). A discovered frame that breaks this rule for an item is dropped for that item (the last admitted frame first, until the set builds) and recorded (`frames_dropped`). An item has one form set, the same in every arm.

**Frame discovery** (stage calib, on C, before any evaluation item of the model). Greedy generations (at most 16 new tokens; stop at a newline, at EOS or an end-of-turn token of the chat template, or once the decoded text names a candidate of the item's lexicon) of the clean B, S and X runs of the 30 C cores in the seven calibration arms (630 generations). The frame of a generation is the text before the first candidate of the item's lexicon, with the item's agent, other agent, object and distractor replaced by {a}, {b}, {o}, {d}; a generation without a candidate, or whose frame has a newline or is longer than 60 characters, gives no frame (it still counts in its arm's total). A frame not among the 16 fixed E frames is a candidate if it occurs in at least 2 % (and at least 2) of some arm's generations; at most 16 candidates, the most frequent over all arms first (ties by the string) (`ckeys.generate.discover_frames`). In that order a candidate is admitted only if the form set of every C item builds with the fixed frames, the frames admitted before it and it. The frames are written to `frames/<key>.json`; the evaluation prints its sha256 before the first item and records it; the scorer checks it.

**Generation** (stage eval, every arm of F and S0): one batch of 8 rows per item and arm, the clean B, S and X runs and the rows ID, K_S, K_X, V_S, V_X at l0 = 0 (`ckeys.generate.greedy`: the prompt pass with the clamps active, then cached one-token steps; equal to cache-free stepwise argmax decoding under the clamps, J-B-G0), at most 16 new tokens with the stop rule above. a(Z) is the first candidate of the item's lexicon in the generated text ((?i)\b(w1|…|w6), plurals allowed), lower-cased, else "other".

**Stages** (`experiments/fresh_factorial.py --stage`, one process each): tokcheck (J-B-G0b), calib, g3 (J-B-G3), eval (F; and S0 for P4, which also runs `format_factorial.run_item` itself on every S0 item and arm: the published path, for J-B-G2). The pipeline's steps per model, in the order of `KEYS` (qwen7, qwen14, mistral7, olmo7, llama8, gemma9, phi4, falcon7, gemma2b): `tok_<key>`, `calib_<key>`, `g3_<key>`, `evalF_<key>` and, for P4, `evalS0_<key>`. A failed `tok_<key>` stops that model (and, for an N4 model, triggers the fallback rule); the eval steps run only when the frames file and the G3 file exist. J-B8 (after the core, deadline-guarded; steps `jb8_release`, `tok_mistral24`, `calib_mistral24`, `jb8`): the release of Anonymous (2026) is checked with stage 7's `release_check` (the manifest and the RELEASE.json pin, the nine Mistral `original_1000` bases, the stories file) before the model; then tokcheck and calib at Mistral-Small-24B (`calib_mistral24` runs only after `tok_mistral24` succeeded, and `jb8` only after `calib_mistral24` succeeded and the frames file exists; a failed tokenizer check is listed in `SKIPPED.txt` as the reason jb8 did not run); then `experiments/paper1_frames.py --model mistral --p1-root <release> --score E --frames frames/mistral24.json --model-dir <verified directory>` on the release's native cores with the released bases, formats P1, NONE, BEFORE, POST, LETTER (a discovered frame that makes one form a proper prefix of another with a native core's names is dropped for that core, the last admitted first, and listed with the item). Then the exploratory steps `tok_`, `calib_`, `g3_` and `x2_<key>` for qwen1.5 and qwen3b, and `x1_fp32`, `x1_eager`; then the scorer, the manifest and the archive.

#### Measures

For an item in arm f under scoring σ ∈ {L, Σ, E}, with d_w(row) = σ(w | row) − σ(w | ID):
- ID_K = ½[(d_S(K_S) − d_S(K_X)) + (d_X(K_X) − d_X(K_S))] at l0 = 0; ID_V likewise with V_S, V_X; ID_KV with KV_S, KV_X. Bars are means over items.
- D(f) = mean ID_K(f) + mean ID_V(f); D_A = D(AFTER).
- s_ID(f) = mean ID_K(f) / D(f), defined when D(f) > 0, D(f) ≥ 0.2 · max(D_A, 0) and mean ID_V(f) ≥ −0.05 · max(D_A, 0). (In hand, stage 3b: D(f)/D_A ≥ 0.49 in every arm and P4 model.)
- r(f) = mean ID_K(f) / mean ID_K(AFTER) and δ(a, b) = [mean ID_K(a) − mean ID_K(b)] / mean ID_K(AFTER), defined when mean ID_K(AFTER) > 0 and ≥ 0.1 · D_A (the anchor; in hand 0.59–0.86 of D_A).
- β_K = ½[(1[a(K_S) = S] − 1[a(K_X) = S]) + (1[a(K_X) = X] − 1[a(K_S) = X])] per item; β_V likewise; b_ID = β_K / (β_K + β_V). The behavioural counterpart of a statistic is the same formula with β_K and β_V in place of ID_K and ID_V, with the same definedness rules and thresholds.
- Flip rate under f: ½[P̂(a(K_S) = S) + P̂(a(K_X) = X)].
- Mass of a row under σ: Σ over the six candidates of exp σ(w | row). **Coverage** of a cell (model × arm): the minimum over its 16 rows (13 clamp rows, clean B, S, X) of the mean E mass.
- **Floor** of a cell: mean |m^E(ID) − m^E(clean B)|, m = E(S) − E(B).
- Generated accuracy acc_B(f) = P̂(a(clean B) = B); acc_S likewise; the "other" rate of clean B.
- Agreement A of a cell: among the 8 generated rows of every item whose answer names a candidate, the share whose answer is that row's E-argmax over the six candidates.
- Competent item in arm f: the E-argmax names B in the clean B run and S in the clean S run.

#### Statistics

- **Bootstrap.** Two-stage cluster bootstrap: resample the (lexicon, base, source) clusters with replacement, then, within each drawn cluster, as many items as it holds with replacement; 10,000 resamples, seed 20261013. The index set is fixed by the population's valid items and their clusters (so it is the same in every model with the same valid items) and is shared by every arm, scoring and β, so every contrast is paired. Every statistic is a function of item means and is recomputed from the resample's means in every resample. S0 has one lexicon, so its clusters are the ordered pairs. JB6 (c)'s original rules use the item bootstrap of their entries (same seed). J-B8 resamples the native cores (item bootstrap, same seed), separately in each format. A subset of items (J-B3's competent items) gets its own bootstrap of the same kind and seed.
- **Levels.** P4f lines (every evaluable P4 model, an intersection-union test), the S0 lines J-B6a and J-B6b, and the single-model lines (J-B-SMALL, J-B8) use 95 % percentile intervals. N4 lines ("3 of 4") use 98.75 % intervals. The bounds are numpy percentiles (linear interpolation) of the defined resamples.
- **Interval criteria** are one-sided tests of named nulls: "H0: θ ≤ t, rejected when the lower bound > t" (and the mirror for upper bounds); "inside (a, b)" is two one-sided tests, each at the interval's level (both bounds strictly inside). Point floors are effect-size conditions on the point estimate. In a model, a line is NOT EVALUABLE when any of its components is (a model-level gate, an arm not run, competence, the anchor, the floor, a cell without a counterpart, an anchor-undefined statistic); otherwise it is MET when every test rejects and every point condition holds, and NOT MET otherwise.
- **Definedness.** A statistic undefined at the point, or in more than 5 % of the resamples: by the anchor rule (r, δ and their differences) → NOT EVALUABLE; otherwise by the arm's own rule (s_ID and its differences), or a point estimate that is not a number → the criterion is NOT MET. The anchor rule is checked first. Resamples where a statistic is undefined by either rule are dropped.
- **Coverage switch (B-8).** On F, a statistic is computed under E when every cell it uses (its arms and AFTER, whose D_A or ID_K scales it) has coverage ≥ 0.8; otherwise its behavioural counterpart is used. J-B3's three competent-only points share one scoring, chosen over the cells POST, NONE, PRE and AFTER by the same rule (NOT EVALUABLE when none applies). J-B-LB has no counterpart and is NOT EVALUABLE in such a model. Under E, a cell whose floor exceeds 0.05 × mean ID_K^E(AFTER) makes the statistic NOT EVALUABLE (the floor is not checked under the counterpart). No switch applies to J-B5 and J-B5b (behavioural by definition) or to the S0 lines J-B6a–c (which use E and L as their rows state).
- **Combination.** P4f, J-B6a and J-B6b: NOT MET as soon as an evaluable P4 model does not meet the line, whatever the number of evaluable models; otherwise NOT EVALUABLE if fewer than 3 P4 models are evaluable, else MET. N4: NOT EVALUABLE if fewer than 3 slots are evaluable; else MET iff at least 3 of the 4 slots meet it (a slot not evaluable counts as not meeting). J-B6c needs all four P4 models (see its row). J-B7: NOT MET if its P4 part (the P4f rule) or its N4 part (the N4 rule) is NOT MET; else NOT EVALUABLE if either part is; else MET. Single-model lines (J-B-SMALL, J-B8): the model's verdict. Part B defines no MET IN PART.
- **Holm** (common part; `analysis/stage8_holm.py`, the same helper in every part): reported, no verdict uses it. The family is the one-sided interval components of this part's R-class account lines (J-B3-N4, J-B5b-P4f, J-B5b-N4, J-B-NULL-N4, J-B-LB-N4, J-B-SMALL) in the models where the line is evaluable; each component is given as its point estimate, its bootstrap SE (the SD of the defined resamples, n − 1 denominator), its bound and its direction, the helper takes one-sided p = Φ(−(est − bound)/SE) for "> bound" and Φ((est − bound)/SE) for "< bound" and steps down at familywise α = 0.025. The two-sided "inside" criteria, the point conditions and the components made NOT MET by their own undefinedness are not in the family and keep their decisions. Per line the scorer prints every component whose decision differs under Holm and the verdict the line would get, under the line's combination rule, with Holm's decisions in place of the interval decisions.
- **TEST_MODE.** Outputs tagged TEST_ (or `--test`) are scored with the population sizes, J-B-G1, J-B-G2, the competence gate and the BF16 and attention checks of the provenance waived, with J-B3's competent-item floor at 2 instead of 30, and with a missing pytest log not failing J-B-G0; their verdicts are plumbing checks, not results.
- **Per-family reporting (B-12).** For every N4 line the scorer names each slot's verdict and counts the slots that do not meet it (NOT MET or NOT EVALUABLE): none, exactly one, exactly two, or three or more, the count that selects the pre-written sentence.

#### Gates

- **J-B-G0, exactness** (FP32, CPU, before any model; 1e-4 nats). `tests/test_fresh.py`, `tests/test_fresh_factorial.py`, `tests/test_stage8b_score.py`, and the shared `tests/test_surface.py`, `tests/test_generate.py`, `tests/test_clamp_families.py`, `tests/test_stage8_populations.py`, `tests/test_stage8_holm.py` (24, 12, 20, 13, 5, 5, 5 and 7 tests). The scorer reads the last pytest session in `logs/` that ran these files (outside TEST, a results directory without a pytest log fails J-B-G0): in each file at least that many tests must pass or be allowed skips, none may fail or error, and a skip is allowed only for the tests that need a cached tokenizer (names containing "cached_tokenizer" or "study_tokenizers"), which J-B-G0b repeats on each model's verified files. The checks, each against an independently computed reference:
  1. the trie score and score_cached equal one plain forward per form (`score_reference`) at Qwen2.5-0.5B under per-row K and V clamps, on F prompts of both lexicons; the identity row's answer position equals a plain pass;
  2. run_item's clamp rows (K_S from 0, KV_S from 0.3 L) and clean run equal score_reference under a separately constructed single-row clamp;
  3. run_item's 8-row generation batch equals cache-free stepwise argmax decoding of each row alone under its own clamp;
  4. run_item's lower-case fields equal `format_factorial.run_item` on 2 S0 items (P1, POST); plain_item equals it; the J-B-G3 statistics pass in FP32 and fail on a perturbed record;
  5. `paper1_frames.py --score E`: the lower-case vectors equal `lp_rows` in every run of `run_core` (locations and letters); the default path writes byte-identical output to the file before the flag existed (run as a subprocess); a discovered frame that breaks the prefix rule with a core's names is dropped for that core, not fatal;
  6. score_cached equals score_reference on tiny random Gemma-2 (eager, sliding layers, soft-capping) and Phi-3 (fused qkv) models;
  7. frame admission rejects a frame that breaks the disjointness rule; the tokenizer check (J-B-G0b) fails when a null sentence would not have the after-sentence's length;
  8. F and C disjoint from U and from each other, deterministic, sizes and hashes pinned; the S0 and U hashes; the attributes; the prompts (byte-identical on S0); parsing with both lexicons; the token constraints above on every cached Part-B tokenizer;
  9. the scorer on synthetic results with known answers (every line MET; NOT MET and NOT EVALUABLE paths; the coverage switch and the floor; own versus anchor undefinedness; the fallback slot, after a refused fetch and after a failed tokenizer check; the pytest gate, and a missing pytest log outside TEST; J-B8's model-level gates; J-B3's competent-only scoring over POST, NONE, PRE and AFTER and its definedness rules; J-B-G2 with an S0 file cut before AFTER; bootstrap determinism and two-stage structure; the combination rules; every line's class against its prior; the Holm family and the call of the shared helper; the pipeline's J-B-G0 file list, its J-B8 step chain and its disk figure);
  10. the stage-8 populations disjoint across the parts and the shared Holm helper (common part, G6 and Holm).
  J-B-G0 failing (outside TEST also: no pytest log) makes every line NOT EVALUABLE, J-B8 included.
- **J-B-G0b, tokenizer check** (per model, before it loads; tokenizer only): the 18 words single-token and stable before ",", ".", " and"; every sentence within ±3 tokens of ROOM and of one length with either lexicon, the null nouns and three orders (so POST-NULL has POST's length); the form set of every F, C and S0 item builds and decodes back; 0 skipped items in every arm of F, C and S0. A failure stops that model; for an N4 model it is a pre-output technical failure and the fallback rule applies.
- **J-B-G1, files**: `VERIFIED.json` of every model run (common part), copied by the pipeline to `verified/<key>.json` in the results. A model without that copy is not evaluable.
- **J-B-G2, reproduction** (flagged, no verdict depends on it): per P4 model and S0 arm, the published-path pass against the committed files (stage 1 for P1, stage 3b's 2×2 otherwise): |Δs_ID^L| ≤ 0.03 and |Δr^L| ≤ 0.03 (E3's tolerance), with s_ID^L = mean ID_K^L / (mean ID_K^L + mean ID_V^L) and r^L = mean ID_K^L / mean ID_K^L(AFTER) computed in each pass (the committed AFTER is stage 3b's), without the definedness rules. An S0 file without the AFTER arm (an eval step stopped by the deadline) is reported as not evaluable for that model, and the other models are still compared. J-B6 is a same-pass comparison and is scored either way.
- **J-B-G3, trie floor** (per model, on the first 30 F cores in NONE, POST and AFTER; BF16): the trie pass against the plain published path on the same prompts; per arm |s_ID^L(trie) − s_ID^L(plain)| ≤ 0.02 (ratios of item means, without the definedness rules) and mean |L(trie) − L(plain)| (all 16 rows, six candidates) ≤ 0.05 × mean ID_KV^L(plain) (B-10); all three arms must pass. Pass: the evaluation scores with the trie. Fail: it scores with score_cached, the exact per-node path (the prompt with the plain causal kernel and the KV cache, then every trie node as a cached continuation; its answer position is the plain path's). The scorer checks that each evaluation used the scorer its G3 chose; otherwise the model is not evaluable.
- **J-B-G4, per cell**: coverage ≥ 0.8 for E (else the behavioural counterpart); floor ≤ 0.05 × mean ID_K^E(AFTER) (else the E statistic is NOT EVALUABLE).
- **J-B-G5, per model**: competence, generated acc_B (over all F items) ≥ 0.8 under NONE and AFTER for every F line except J-B7 (J-B1 to J-B5, J-B5b, J-B-NULL, J-B-LB, J-B-SMALL), and also under POST (J-B3, J-B-SMALL), POST and POST-NULL (J-B-NULL), POST and PRE (J-B5b), BEFORE (J-B-LB); otherwise the line is NOT EVALUABLE in that model, and the model is not replaced. An arm the line uses that was not run makes it NOT EVALUABLE in that model ("arms not run (deadline)"). Anchor: mean ID_K(AFTER) ≥ 0.1 · D_A with lower bound (at the line's level) > 0 under the scoring used; otherwise the r- and δ-based components, and so the line, are NOT EVALUABLE in that model.
- **Model-level**: J-B-G0, J-B-G0b, J-B-G1, a G3 file with J-B-G3's scorer choice used by every evaluation, and the frames file's sha256 must hold, else every line is NOT EVALUABLE in that model. The S0 lines also need every S0 arm run. J-B8 is gated the same way at Mistral-Small-24B (it has no J-B-G3): J-B-G0, its tokenizer check (`tokcheck/mistral24.json`), its verified file set (outside TEST) and the sha256 of `frames/mistral24.json` equal to the one its jb8 file recorded.
- **Provenance and population** (reported, no verdict depends on them): the scorer checks one commit across the results files, BF16 and the fixed attention of every eval, g3 and frames file (X1's files excepted), no TEST_ file outside TEST, the pinned population hashes, and valid plus skipped items = 150 per population; a mismatch is printed (MISMATCH), the score file is still written and the scorer exits with status 2.

#### Confirmatory lines

Kind A = account line, V = measurement-validity line (tallied separately, G4). The class follows the recorded prior P(MET | evaluable) (G4): L = implied by data in hand on the same models and material, prior ≥ 0.9; M = prior ≥ 0.8; R = prior < 0.8 (the scorer's tests check every line). All lines use F and E (or the counterpart) unless stated. Priors were recorded before any stage-8 output; the in-hand numbers are lower-case L on S0, stage 3b, 95 % / 98.75 % (base, source) cluster intervals recomputed at the build (see Seen before finalisation).

| Code | Class, kind, prior | Criterion (per model) | Justification of the prior |
|---|---|---|---|
| J-B1-P4f | L, A, 0.90 | s_ID(AFTER) ≥ 0.50 (point); H0: s_ID(AFTER) ≤ 0.40 rejected; H0: ID_K(AFTER) ≤ 0 rejected | s_ID(AFTER) 0.777 / 0.863 / 0.819 / 0.593, lower bounds ≥ 0.574; new lexicon and E untested |
| J-B1-N4 | M, A, 0.85 | the same, 98.75 % | OPTIONS-AFTER s_ID 0.67–0.88 in all 9 models from 3B |
| J-B2-P4f | L, A, 0.90 | H0: r(BEFORE) − r(NONE) ≥ 0.05 rejected; H0: r(PRE) − r(NONE) ≥ 0.05 rejected; H0: s_ID(BEFORE) ≥ 0.10 rejected; H0: s_ID(PRE) ≥ 0.10 rejected | r(BEFORE) − r(NONE) −0.057 / −0.033 / −0.035 / −0.079, r(PRE) − r(NONE) −0.052 / −0.031 / −0.036 / −0.070, upper bounds ≤ −0.023 |
| J-B2-N4 | M, A, 0.85 | the same, 98.75 % | implied by J-B4 and the causal mask; at or below 0 in 10 of 10 models |
| J-B3-P4f | M, A, 0.80 | r(POST) ≥ 0.15 and ≤ 0.75 (points); H0: r(POST) ≤ 0.10 rejected; H0: r(POST) ≥ 1.0 rejected; H0: δ(POST, PRE) ≤ 0 and H0: δ(POST, NONE) ≤ 0 rejected; H0: s_ID(POST) ≤ 0.10 rejected; H0: s_ID(POST) − s_ID(NONE) ≤ 0 rejected; on the items competent in POST, NONE, PRE and AFTER the points δ(POST, PRE), δ(POST, NONE) and s_ID(POST) − s_ID(NONE) > 0 (fewer than 30 such items: NOT EVALUABLE; one scoring for the three, chosen over POST, NONE, PRE and AFTER; each point under the definedness rules on those items, so an undefined anchor makes it NOT EVALUABLE and an own-undefined s_ID difference NOT MET) | r(POST) 0.268 / 0.337 / 0.506 / 0.233 and s_ID(POST) 0.317 / 0.396 / 0.407 / 0.189 with ROOM; eight new sentences and a second lexicon untested |
| J-B3-N4 | R, A, 0.50 | the same, 98.75 % | never tested outside Qwen, Mistral and OLMo; the design's 0.6, lowered for the added s_ID and competence criteria |
| J-B4-P4f | L, A, 0.90 | s_ID(NONE) ≤ 0.10 (point); H0: s_ID(NONE) ≥ 0.15 rejected; H0: ID_V(NONE) ≤ 0 rejected | s_ID(NONE) 0.059 / 0.052 / 0.039 / 0.033, upper bounds ≤ 0.071 |
| J-B4-N4 | M, A, 0.85 | the same, 98.75 % | 0.03–0.06 in all 10 models in hand |
| J-B5-P4f | M, A, 0.85 | β_K(AFTER) ≥ 0.50 and β_V(NONE) ≥ 0.50 (points); H0: β_K(AFTER) ≤ 0.40 and H0: β_V(NONE) ≤ 0.40 rejected; H0: β_K(f) ≥ 0.10 rejected for f = NONE, BEFORE, PRE; H0: β_K(AFTER) − β_V(AFTER) ≤ 0 rejected | 4-candidate argmax proxy: β_K(AFTER) 0.82–1.00, β_V(NONE) 0.92–1.00, β_K(NONE, BEFORE, PRE) 0.00; generation untested |
| J-B5-N4 | M, A, 0.80 | the same, 98.75 % | as J-B1/J-B4 in N4 |
| J-B5b-P4f | R, A, 0.45 | H0: β_K(POST) − β_K(PRE) ≤ 0 rejected | proxy β_K(POST) 0.05 [0.02, 0.08], 0.27, 0.30, 0.02 [0.00, 0.04] against 0.00 under PRE: OLMo is at the edge |
| J-B5b-N4 | R, A, 0.40 | the same, 98.75 % | as J-B3-N4, and behaviour is a stronger requirement |
| J-B-NULL-P4f | M, A, 0.80 | H0: r(POST-NULL) ≥ 0.10 rejected; H0: s_ID(POST-NULL) ≥ 0.10 rejected; H0: δ(POST, POST-NULL) ≤ 0 rejected | stage 5 (G9): a sentence naming B and two other candidates gave ID_K below NONE (−0.32 / −0.30 / −0.21 nats) at Qwen2.5-7B / 14B / Mistral-7B; r(NONE) 0.051 / 0.029 / 0.036 / 0.044 |
| J-B-NULL-N4 | R, A, 0.55 | the same, 98.75 % | its δ half needs a sentence read in new families (J-B3-N4's risk, weaker) |
| J-B-LB-N4 | R, A, 0.65 | H0: ID_K(BEFORE) − ID_K(NONE) ≥ 0 rejected (98.75 %); no behavioural counterpart | −1.18 / −1.18 / −0.55 / −0.68 nats at P4 under L (98.75 % upper bounds ≤ −0.34); 9 of 10 models in hand; under E the small NONE read is untested (0.5B pilot: E-scored ID_K(NONE) ≈ 0) |
| J-B-SMALL | R, A, 0.40 | at Gemma-2-2B (95 %): H0: r(POST) ≥ 0.10 rejected | Qwen2.5-1.5B / 3B POST ID_K −0.47 / −0.04 nats (stage 1); competence at 2B uncertain (gated) |
| J-B6a | R, V, 0.50 | S0, every evaluable P4 model (≥ 3; combination above): coverage ≥ 0.8 (point) in each of the 6 arms | 0.5B pilot E mass of clean B 0.87 (NONE), 0.91 (POST); Llama-3.2-1B 0.43–0.54 with fixed frames; the minimum over 16 rows is stricter |
| J-B6b | R, V, 0.45 | S0, every evaluable P4 model (≥ 3; combination above), every arm with M^L(clean B) < 0.5 (the mean over items of the clean B run's L mass): H0: \|s^E − s^L\| ≥ 0.05 rejected and (except AFTER) H0: \|r^E − r^L\| ≥ 0.05 rejected (paired, 95 %, two one-sided tests); a model without such a cell is not evaluable | M^L(clean B) in hand: POST/PRE/NONE ≤ 0.03 at Qwen2.5-7B/14B and Mistral-7B, BEFORE 0.34 / 0.15 at 14B / Mistral; OLMo ≥ 0.84 everywhere; 0.5B pilot s_ID L vs E: +0.020 / −0.001 (NONE), −0.146 / −0.181 (POST) |
| J-B6c | R, V, 0.75 | S0, P4, all four models (one without S0 results, with a failed model-level gate or with an S0 arm not run: NOT EVALUABLE): the verdicts of E1a, E1b, E1c (stage 3b), B2 (stage 1 prediction 2) and C4's BEFORE bound (stage 2), recomputed under E with their original rules, equal those under L in the same pass (primary JB6) | the lines are far from their bounds in hand except E1c's POST − PRE, which needs the sentence read to survive E |
| J-B7 | R, V, 0.35 | F: in each of the six arms, A ≥ 0.95 (at least one answer naming a candidate), clean-B "other" rate ≤ 0.10 and coverage ≥ 0.80 (points; no competence gate; an arm not run: NOT EVALUABLE); MET iff every evaluable P4 model (≥ 3 evaluable) and ≥ 3 of the 4 N4 slots meet it (combination above) | 0.5B pilot A 0.97; verbose families (Llama-3.2-1B "On the shelf") threaten coverage |
| J-B8 | R, V, 0.55 | Mistral-Small-24B, the native cores of Anonymous (2026)'s release (B, S, T distinct), in each of the formats P1, NONE, BEFORE, POST: H0: \|φ^E − φ^L\| ≥ 0.05, \|ψ_K^E − ψ_K^L\| ≥ 0.05 and \|ψ_V^E − ψ_V^L\| ≥ 0.05 rejected (paired core bootstrap, 95 %, two one-sided tests); the mean over cores of M^E of the natural B run ≥ 0.80 (point). LETTER reported. No jb8 results (not run: the deadline, a release that is absent or fails its check, or refused files; the reason from `SKIPPED.txt`), a file not scored with `--score E`, a failed model-level gate (J-B-G0; the tokenizer check of mistral24; its verified file set outside TEST; the sha256 of `frames/mistral24.json` differing from the one the jb8 file recorded), or one of the four formats not run → NOT EVALUABLE | the list formats change little under E; POST and NONE carry the risk |

The original rules in J-B6c (per model; "CI excluding 0" is the lower bound of the 95 % interval > 0): E1a ID_K(AFTER) > 0 with CI excluding 0 in 4/4; E1b ID_K(BEFORE) ≤ 0.5 and ID_K(PRE) ≤ 0.5 nats (point means) each in ≥ 3/4; E1c paired ID_K(AFTER) − ID_K(BEFORE) > 0 (CI excluding 0) in 4/4 and paired ID_K(POST) − ID_K(PRE) > 0 (CI excluding 0) in ≥ 3/4; B2 paired ID_K(P1) − ID_K(NONE) > 0 (CI excluding 0) in ≥ 3 of the 4 P4 models (the original "≥ 4 of 5" at the P4 size); C4 ID_K(BEFORE) ≤ 0.5 nats (point mean) in all four. Intervals: 95 % item bootstrap, seed 20261013. J-B6c is MET iff each of the five line verdicts is the same under E as under L; the per-model components that change are printed.

φ, ψ_K, ψ_V in J-B8 are stage 3b's with m = σ(T) − σ(S) and σ ∈ {L, E} from the same trie pass: φ = [m(M) − m(P)] / [m(T) − m(S)], ψ_K = [m(P + K_M) − m(P)] / [m(M) − m(P)], ψ_V = [m(P + V_M) − m(P)] / [m(M) − m(P)], with M, P, P + K_M and P + V_M averaged over the seeds 101–103 within a core, each ratio a ratio of means over the cores, and the E-minus-L difference recomputed in every resample. Its E uses Mistral-Small-24B's own frames (calibration on C) and, under LETTER, the forms " X", "X", " **X", "**X" of each letter.

**Expected values** (the author's, not thresholds). N4: s_ID(AFTER) 0.6–0.9; r(POST) 0.2–0.5; r(BEFORE) − r(NONE) and r(PRE) − r(NONE) about −0.05; s_ID(NONE) 0.03–0.07; β_K(AFTER) ≥ 0.8; β_V(NONE) ≥ 0.8; r(POST-NULL) ≈ r(NONE). P4f: the S0 values within about 0.05 with ROOM replaced by the eight sentences.

#### What each primary line means for the paper (pre-written)

| Primary line | Abstract clause (MET) | NOT MET: replacement | Title | Table 1 row | Figure |
|---|---|---|---|---|---|
| J-B1-N4 | "In four model families never examined before, a list of the candidates after the story makes the writing token's key carry most of the value's identity (s_ID ≥ 0.5 in k of 4)." | "The key read after a list does not generalise beyond the families where it was found (k of 4 new families)." The list result stays as a property of Qwen, Mistral and OLMo. | No change (Part A decides the title). | "Later list, new families": s_ID(AFTER) per family, MET. NOT MET: the row reads "not general (k/4)". | Fig. "fresh": s_ID per arm and family. |
| J-B3-N4 | "A neutral sentence naming the candidates after the story opens an intermediate key read (r = x–y of the list read) in k of 4 new families." | Exactly one family failing: "…in three of the four new families; not in F." Exactly two: "The sentence read is family-dependent: it appears in Qwen, Mistral, OLMo and in families A and B, not in C and D." Three or more: the sentence claim is restated as specific to Qwen, Mistral and OLMo; only the list result is called general. | No change. | "Later sentence, new families": r(POST) per family. | Fig. "fresh" (POST bars). |
| J-B4-N4 | "Without a later mention, the identity is copied through the value in every family (s_ID(NONE) ≤ 0.1)." | "Keys carry part of the identity without later mentions in k families"; the word "copied" is qualified. | No change. | "No mention": s_ID(NONE) per family. | Fig. "fresh". |
| J-B5-N4 (with J-B5-P4f) | "Clamping only the writing token's key changes the generated answer to the clamped value in x–y % of stories when the candidates are listed after it and in at most z % when they are not; clamping only its value does the reverse." | "The log-probability account does not carry over to generated answers in k families" (the β numbers are given). | No change. | "Generated answers": β_K(AFTER), β_V(NONE). | Fig. "crossover": β_K against β_V per arm and family (the main-text behavioural figure). |
| J-B5b-P4f / J-B5b-N4 | "The sentence read changes generated answers too: the key alone moves the answer in x % of stories after a sentence and in y % before it." | "The sentence read is visible in log-probabilities but not in generated answers (flip rate x % after, y % before)." | No change. | "Sentence, generated": flip rate POST vs PRE. | Fig. "crossover" (POST, PRE points). |
| J-B-NULL-N4 (with -P4f) | "A sentence of the same length naming no candidate opens no key read, so the read needs the re-mention." | "Intervening text alone opens part of the key read in k families"; J-B3's reading "a re-mention opens the key" is withdrawn in those families. | No change. | Note under "Later sentence": the null control. | Fig. "fresh" (POST-NULL bars). |
| J-B6c (with J-B6a, J-B6b) | "Scored on the forms the models actually emit (coverage ≥ 0.8 in every cell), every published verdict on the original stories is unchanged, and s_ID moves by at most 0.05." | "Emitted-form scoring changes the published verdict of {line}; E-scored numbers replace the lower-case ones as primary throughout the paper, and the change is stated." J-B6b alone not met: "the key share moves by more than 0.05 under emitted forms in {cells}"; the numbers are reported. | No change. | Appendix table "invariance" (tab_invariance). | Appendix: mass per cell (tab_mass). |
| J-B7 | (no abstract clause) | "The emitted-form score does not cover or agree with generation in {cells}; there the behavioural measures carry the claims." | No change. | Appendix. | Appendix. |

The abstract's risk summary counts the R account lines of this part with the other parts' (common part).

#### Reported (no verdict)

r, s_ID, β_K, β_V, ID_K^E, ID_V^E and the flip rate per model and arm (the numbers behind the crossover figure, also written to `fig_crossover.csv`; the tables `tab_fresh.tex`, `tab_mass.tex` and `tab_invariance.tex` are written beside the score); the relative changes of ID_K and ID_V under E against L on S0 (tolerance 15 %, printed per cell, list cells marked); the J-B8 detail (φ, ψ_K, ψ_V under L and E, the mass); J-B-G2 per arm; the per-family verdicts with the count of slots not meeting each N4 line, which selects the pre-written sentence; LETTER in J-B8.

#### Exploratory (no verdict)

Σ and L versions of every statistic; the competent-only population; the key share per onset (0, 0.0625 L, 0.3 L); per-lexicon and per-sentence estimates of r(POST), s_ID(POST) and s_ID(AFTER) (wording and lexicon as fixed effects, B-2); the LIST-BEFORE sign in every model; b_ID under POST; the frame census and the uncovered first-token mass; the trie against the published path on S0 (all items); X1 (Qwen2.5-7B, first 60 S0 cores, AFTER, POST, NONE, FP32 sdpa and BF16 eager, against the BF16 sdpa run); X2 (Qwen2.5-1.5B and 3B on S0, P1, AFTER, POST, NONE, under L, Σ and E, with the competent-only breakdown). X3 (Gemma-2-27B) is dropped (B-13).

#### Seen before finalisation

No output of any study model on F, C or S0 under the stage-8 code has been seen; no model above 0.5B parameters was run by the build. Seeds 20261013 and 20261014 were touched only by the population construction (no model). The design and critique pilots (all CPU, FP32; scripts and logs kept with the stage-8 build notes):
- **Part-B pilot 1, tokenizers** (`partB/tok_check.py`, `tok_check.json`; no model): Llama-3.1-8B (unsloth files), Gemma-2-9B, Phi-4 and Falcon3-7B have all six lower-case " w" and capitalised " W" single tokens and 0 of 150 cores of Random(8) skipped in any of the 7 arms; Mistral-7B's " Shelf", " Drawer", " Cabinet", " Closet" are multi-token (44 trie nodes for Σ); granite-3.1-8b (" shelf" two tokens, 145/150 skipped) and deepseek-llm-7b (" cabinet" two tokens) are infeasible; Yi-1.5-9B passes.
- **Pilot 2, Hub metadata** (`partB/hf_manifest.py`, `hf_manifest.json`): the gated repositories' files have byte-identical ungated copies (common part).
- **Pilot 3, populations**: seed 20261013 has 4 of its first 400 draws in U; its first 150 cover all 30 pairs and 20 objects; C (seed 20261014) is disjoint. (The design's hash e1f28077 was of the LEX1 cores alone with sorted keys; the pinned hash above includes the attributes.)
- **Pilot 4, exactness of the trie** (`partB/test_surface_pilot.py`, `exact_pilot.log`, 1280 s): tiny random Qwen2, Mistral, Llama, OLMo2 (sdpa and eager), Gemma2 (eager) and Phi3: trie = separate passes to ≤ 3.8e-6 nats, answer position = plain to ≤ 1.9e-6; Qwen2.5-0.5B, clean and under K_S / V_S clamps: ≤ 4.7e-5.
- **Pilot 5, mini factorial at Qwen2.5-0.5B** (`partB/mini_pilot.py`, `mini_qwen05.log`; 6 cores of seed 20261011): candidate mass L / Σ / E 0.135 / 0.694 / 0.873 (NONE) and 0.238 / 0.800 / 0.912 (POST); s_ID under L / Σ / E +0.020 / −0.002 / −0.001 (NONE) and −0.146 / −0.171 / −0.181 (POST); β_K / β_V 0.00 / 0.75 (NONE), 0.00 / 0.58 (POST); the Σ-argmax agreed with generation in 0.97 of 30 rows in each arm; first tokens " In" 0.11 and " On" 0.09 under NONE.
- **Pilot 6, coverage at Llama-3.2-1B-Instruct** (`partB/coverage_pilot.py`, `cov_llama1b.log`; clean runs, n = 8 and 12): L / Σ / E mass NONE 0.017–0.025 / 0.25–0.29 / 0.43–0.47; POST 0.017–0.025 / 0.23–0.28 / 0.49–0.52; PRE 0.023 / 0.29 / 0.54; AFTER 0.85 / 0.93 / 0.93; BEFORE 0.32 / 0.85 / 0.86; generated accuracy 0.25–0.75.
- **Pilot 7, in-hand statistics** (`partB/power_inhand.py`; stage-3b rows): r(POST) 0.268 / 0.337 / 0.506 / 0.233, s_ID(AFTER) 0.777 / 0.863 / 0.819 / 0.593, s_ID(NONE) 0.059 / 0.052 / 0.039 / 0.033 (Qwen2.5-7B / 14B / Mistral-7B / OLMo-2-7B); r(BEFORE), r(PRE) −0.035 to 0.000; cluster SDs 1.5–2× the core SDs.
- **Pilot 8, behavioural proxy** (stage-3b 4-candidate argmax): β_K(AFTER) 0.82–1.00, β_V(AFTER) 0.00–0.19, β_V(NONE) 0.92–1.00, β_K(NONE, BEFORE, PRE) 0.00, β_K(POST) 0.02–0.30.
- **Pilot 9, power** (`partB/power_sim.py`, `power_jb2.py`; 99.375 % cluster bootstrap): JB3 r(POST) met with probability 0.00 / 0.50 / 1.00 / 1.00 at true 0.12 / 0.15 / 0.18 / 0.22 (×1 noise); JB2 1.00 at true 0 or 0.05, 0.08–0.55 at 0.08; JB4 ≥ 0.91 at s_ID(NONE) ≤ 0.09, ≤ 0.06 at 0.11; JB1 0.18–0.54 at s_ID(AFTER) 0.50, 0.87–1.00 at ≥ 0.55.
- **Critic pilots** (`critic_replication/`): `jb9_chance.py` (stage-3b rows): P(JB9 met | F and S0 exchangeable) ≈ 0.62, Mistral r(POST) cluster SD 0.033; `lexicon_check.py` (tokenizers): bin, crate, tray, jar, bucket, chest, trunk, bowl, barrel, bag, tin, pot, cart single-token and stable in the 8 Part-B tokenizers and Yi; an inline core-overlap count with seed 99: 150 of 150 location configurations and 148 of 150 with X occur in U, 109 of 150 (B, S, X) triples in S0; an inline β proxy (above).
- **Build checks** (this entry's code; tokenizers, committed rows, and Qwen2.5-0.5B only):
  - token counts of ROOM and the eight sentences in every Part-B tokenizer (above; Mistral-Small-24B as Mistral-7B); 28 candidate null nouns tested for single tokens (curtain, stool, vase, kettle and heater fail in Mistral-7B, Yi or Mistral-Small-24B and were not used);
  - the tokenizer check (J-B-G0b, stage tokcheck) on the cached tokenizers of all 13 keys (Gemma-2-2B through the identical Gemma-2-9B files): every check passes; trie nodes per item 31–37 (Qwen, OLMo-2, Llama-3.1, Phi-4), 29–32 (Falcon3), 25 (Gemma-2), 83–116 (Mistral-7B), 72–97 (Yi-1.5), 79–88 (Mistral-Small-24B), with the fixed frames;
  - the in-hand numbers of the prior table, recomputed from the committed stage-3b rows with this scorer's bootstrap (95 % / 98.75 %): r(NONE) 0.051 / 0.029 / 0.036 / 0.044; r(BEFORE) − r(NONE) −0.057 / −0.033 / −0.035 / −0.079; r(PRE) − r(NONE) −0.052 / −0.031 / −0.036 / −0.070; ID_K(BEFORE) − ID_K(NONE) −1.18 / −1.18 / −0.55 / −0.68 nats; s_ID(POST) − s_ID(NONE) +0.258 / +0.343 / +0.368 / +0.156; D(f)/D(AFTER) ≥ 0.49; ID_K(AFTER)/D(AFTER) 0.78 / 0.86 / 0.82 / 0.59; M^L(clean B) under POST/PRE/NONE 0.000–0.027 (Qwen, Mistral) and 0.84–0.88 (OLMo), BEFORE 0.999 / 0.338 / 0.147 / 0.982;
  - the TEST_MODE run of `scripts/gpu_stage8b.sh` (Qwen2.5-0.5B, FP32, shared 4-core CPU; keys qwen7, llama8, gemma2b, then jb8, x2 at qwen1.5 and x1; n = 2 items of F and S0, 2 of C, 3 native cores for jb8 with random bases): pytest 66 passed; every step succeeded (a first session was stopped from outside during evalS0_qwen7; the second kept the four finished steps and completed the rest); the release check passed (RELEASE.json 2dea297d…, status PASS_EXCEPT_README, bases and stories equal to the pins); J-B-G3 met in every model (|Δs| < 1e-4, mean |ΔL| ≤ 1e-4); the score file's summary read "1 MET, 1 NOT MET, 19 NOT EVALUABLE of 21 lines; provenance OK; population OK" (plumbing only: one model per set, n = 2, random bases); about 60 min of runs after an 18-minute pytest; in a smoke run (`qwen7` tag), J-B-G3 gave |Δs| ≤ 7e-7 and mean |ΔL| ≤ 1e-5 (FP32), the frames census admitted no frame (" " 35 of 42 generations, " In the " 6, " On the " 1), and the trie's L equalled the published path on S0 to 1e-5;
  - the unit tests (J-B-G0): 66 passed, 0 failed, 0 skipped (every Part-B tokenizer was cached) in the TEST_MODE run's pytest (18 min on a shared 4-core CPU); separately `tests/test_fresh_factorial.py` 10 passed in 8 min, `tests/test_stage8b_score.py` 10 passed, `tests/test_fresh.py` 24 passed.
  - the final TEST_MODE run of `scripts/gpu_stage8b.sh` on commit 84e4f05, whose code is the finalised code (one session of 70 minutes, sharing the CPU with Part A's run): pytest 91 passed in 24 minutes (J-B-G0 MET, with the cross-part population and Holm tests); every step succeeded and none was skipped; the score read "1 MET, 8 NOT MET, 12 NOT EVALUABLE of 21 lines; provenance OK; population OK" (plumbing; the change from the earlier reading is the unified combination rule);
  - an earlier TEST_MODE run on the reviewed code (commit aa92bb6; the same keys and sizes; one session of 73 minutes): pytest 73 passed in 25 minutes; every step succeeded and none was skipped; the score read "1 MET, 1 NOT MET, 19 NOT EVALUABLE of 21 lines; provenance OK; population OK" (plumbing only);
  - a review of the build before finalisation (tokenizers and Qwen2.5-0.5B only; no study model): the eight sentences have one token length with LEX1, LEX2, the null nouns and seven orders in every cached Part-B tokenizer, in Mistral-Small-24B's and in Qwen2.5-1.5B/3B's; J-B8's letter form sets build in all of them; Mistral-Small-24B's tokenizer loaded from a local directory gets `fix_mistral_regex` applied; the unit tests of the reviewed code: `tests/test_fresh.py` 24, `tests/test_fresh_factorial.py` 12 and `tests/test_stage8b_score.py` 14 (the counts J-B-G0 then required);
  - the fixes after the audit of this entry against the code (a missing pytest log fails J-B-G0 outside TEST; J-B8's model-level gates, and calib and jb8 at Mistral-Small-24B only after its tokenizer check passed; J-B3's competent-only scoring and definedness; J-B-G2 with an S0 file cut before AFTER; the P4f combination; the two cross-part test files in J-B-G0): `tests/test_stage8b_score.py` 20 passed, `tests/test_stage8_populations.py` 5 and `tests/test_stage8_holm.py` 7 passed; the TEST_MODE archive re-scored with `--test` exits 0 with provenance OK and population OK, and reads 0 MET, 0 NOT MET, 21 NOT EVALUABLE because its pytest log predates the new J-B-G0 file list and counts (the next TEST_MODE run writes a matching log); scored with J-B-G0 taken as passed (a check outside the archive), it reads 1 MET, 8 NOT MET, 12 NOT EVALUABLE: the seven lines that changed from NOT EVALUABLE to NOT MET have one evaluable model, which fails them, and J-B8 stays evaluable under its gates.

#### Compute (A100-80GB, BF16; estimated from stage 3b's 24 s per 150-item arm at 7B)

Per item and arm: the 3-row and 13-row trie passes (about 1.25× stage 3b's tokens) and one 8-row generation batch (on average 3–10 cached decode steps). About 60 s per F arm at 7–8B, 110 s at 14B, 100 s at Gemma-2-9B (eager); S0 adds the published-path pass (about 24 s per arm at 7B). Each step reloads the model (four loads per P4 model, three per N4 model).

| Step | Minutes |
|---|---|
| pytest (J-B-G0, CPU) | 10 |
| qwen7 / mistral7 / olmo7 (tokcheck, calib, G3, F, S0) | 22 each |
| qwen14 | 40 |
| llama8 / falcon7 | 12 each |
| gemma9 / phi4 | 19 each |
| gemma2b | 7 |
| yi9 (only in a replaced slot) | 14 |
| score, manifest, archive | 5 |
| core total | about 3.2 h (the sum of the rows above without yi9; range 2.7–3.7) |
| jb8 (47 GB, release check, calibration, frames in five formats) | about 40 |
| x2 (each of qwen1.5, qwen3b), x1 | about 15 and 12 |

The default deadline is DEADLINE_H = 4.5 h, so that the core (about 3.2 h) and then jb8 (it starts only with 45 minutes left), x2 and x1 (about 4.3 h in all) fit. Downloads (about 158 GB for the nine core models, 47 GB for jb8, 9 GB for x2) are prefetched one model ahead; weights are deleted after each model unless KEEP_CACHE=1; at most two models are on disk besides Qwen2.5-7B, which stays until x1 (≥ 140 GB free). Every eval step stops before an arm that would start inside the reserve the later models need (`--reserve-min`: the minutes of the later models' rows above plus 10; exit 3, redone by the next session; the scorer marks lines that need a missing arm NOT EVALUABLE with "arms not run (deadline)"); jb8 starts only if 45 minutes remain, each x2 model only if 15, x1 only if 12, and a step skipped for the deadline is listed in `SKIPPED.txt`. At about $2 per A100-hour, $7–9.

#### Commands

```bash
J=$(git log --format=%H -1 --grep='^Finalise preregistration J$') && [ -n "$J" ] && git checkout "$J"
bash scripts/gpu_stage8b.sh                    # HF_TOKEN optional; P1R=<the unpacked release of Anonymous (2026)> optional
TEST_MODE=1 bash scripts/gpu_stage8b.sh        # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2): qwen7, llama8, gemma2b, jb8,
                                               # qwen1.5, x1 (TEST_ALL=1: every key)
python analysis/stage8b_score.py --results results/gpu_stage8b    # re-score an archive
```

### Part C. Independently obtained identity edits and the channel-ratio law

**Code.** `ckeys/edits.py` (populations E, H, TSET, THOLD; prompts; prefix captures with a residual write; the edit
vectors; the PAR / PERP and LEX / NONLEX components; clamp rows and their scoring), `ckeys/neutral.py` (the 24 frozen
neutral sentences and their forms), `ckeys/sae.py` (the BatchTopK dictionaries, E3), `ckeys/das_at.py` (the rank-16 remap at
the writing token, E4; imports `ckeys/das.py` unchanged), `experiments/stage8_edits.py` (stages preflight, calib, dasfit,
eval, readers, explore), `experiments/stage8_overlap.py` (the overlap screen), `experiments/prakash_caa.py` (J-C6; extends
`experiments/prakash_swap.py` by import), `analysis/stage8c_score.py` with `analysis/stage8c_parts/` (`stats.py`,
`law.py`, `readers.py`, `prakash.py`), `scripts/gpu_stage8c.sh` (it sources `scripts/stage8_common.sh`). Tests:
`tests/test_stage8_edits.py` and `tests/test_sae.py` (Gate J-C-G0), `tests/test_stage8c_score.py` (the scorer on
synthetic inputs). Shared code used unchanged: `ckeys/surface.py`, `ckeys/clamp.py`, `ckeys/headsplice.py`,
`ckeys/readerblind.py`, `ckeys/interventions.py`, `ckeys/das.py`, `ckeys/story.py`, `ckeys/encoding.py`,
`ckeys/variants.py`, `ckeys/causaltom.py`, `experiments/prakash_swap.py`, and the helpers imported from
`experiments/format_factorial.py`, `experiments/ioi_factorial.py`, `experiments/row_restricted_keys.py` and
`experiments/stage6_heads.py`. `ckeys/generate.py` (greedy decoding, frame discovery) is used with the stage-8 review's
change: the chat templates' end-of-turn tokens also end a generated answer (`tests/test_generate.py`, run in J-C-G0).

**Purpose.** Part C answers objection 3 and parts of objections 1, 4, 5 and 8.
- *Objection 3 (the intervention lesson rested on one remap family, and its out-of-sample test failed).* The lesson is
  restated as a falsifiable law about any identity edit: an edit that writes a value at the writing token is read through
  the key and the value channel in the same ratio as the model's own written value at the same depth (the
  **channel-ratio law**, C-1). It is tested on five families obtained independently of the natural clamp: in-task
  steering vectors (E1), steering vectors from unrelated neutral sentences (E2), the features of a third-party sparse
  autoencoder (E3), a rank-16 DAS remap trained at the writing token with the released recipe of Anonymous (2026) (E4),
  and steering vectors that write a non-English or synonym form of the value (E5).
- *The H11 tautology.* A full-residual identity patch at p is exactly the natural clamp (the lemma below), so it cannot
  test the law; it is kept as the equivalence control T. Out-of-sample status is defined functionally (C-3): an edit's
  displacement is split into its projection on the story's natural displacement (PAR) and the rest (PERP), and into its
  projection on the 5-dimensional English lexical span L_l and the rest (LEX / NONLEX). Only the risky families (E3, E4,
  E5) and identity-carrying PERP / NONLEX components count as an out-of-sample test.
- *The share statistic is compressed (the critique's F1).* κ = K / (K + V) cannot tell an edit read like the natural edit
  from one whose channels are read 2× differently (`kappa_compress.py`: MAD 0.08 and maximum gap 0.16 pass the old rule at
  c_K / c_V = 2). The law is therefore stated on log channel ratios, with an equivalence test (TOST) and a within-design
  sensitivity gate (J-C-G8) that must classify synthetic 0.5× rows as departures.
- *A predicted failure (C-4).* The paper's own token-form result predicts that a non-lexical identity edit is under-read by
  the key channel. J-C-BOUND tests this as a confirmatory line, so the law has a stated boundary.
- *Kind versus depth (the critique's F3).* J-C6 is relabelled: it only tests whether an identity edit at block 28 of
  Qwen2.5-14B is key-flat. Whether a binding swap can be tested at a depth where the identity key route is still open is
  decided by an overlap screen (J-C-SCREEN, J-C-WIN) on Prakash et al.'s material at Qwen2.5-7B and Llama-3.1-8B.
- *Objection 8 (reliance on an anonymous submission).* Every family can be rebuilt from public material: mean differences,
  a public sparse autoencoder at a pinned revision, and a public training recipe applied to public models.
- *Objections 4 and 5.* All scores use the emitted-form score E (`ckeys/surface.py`, frames discovered on this part's own
  calibration set; Part C reads no Part-B file) with a coverage gate; L is the robustness score. The populations use
  seeds never used before (8101–8103), and Llama-3.1-8B is a family never examined in stages A–I.

Part C does not test natural text (Part A), the fresh-sample replication (Part B) or the mechanism of the flag (Part D).
It covers single-position identity edits on templated belief stories. Binding and ordering edits are outside the law;
H7–H10 stand as failures and are not reinterpreted.

#### The law and the lemma

**Lemma** (unit-tested in J-C-G0). B and S differ only at the writing token p.
1. An edit confined to (p, l), the output of 0-based block l at p, reaches every other position only through p's key and
   value in blocks ≥ l + 1: under the causal mask p's residual reaches other tokens only through attention, and p's K/V in
   blocks ≤ l are computed from block inputs that the edit does not change. So B with the edit's K/V written at p from
   block l + 1 is exactly the edited run, and the key-only and value-only rows are exact.
2. Writing h_S,l(p) gives S's K/V at p from l + 1: this edit is the natural clamp KV(S) with onset l + 1.

Every edit is therefore applied as a clamp row whose tables are captured in a prefix pass (positions 0..p of the B prompt
with the edit's vector written at (p, l)). The prefix through p is the same in all four formats (checked for every core
in every Part-C tokenizer), so an edit's tables are the same tensors in every format; only the readers after p differ.
The edit prefix passes run in forwards of the natural prefix pass's shape (five rows, B, S, X, π(S), π(X); the last
forward padded), so an edit's tables are computed by the natural pass's kernels: a row's activations depend on the batch
only through the kernels its shape selects, so T reproduces the natural tables bitwise in BF16 as in FP32, and an edit
differs from the natural tables only through what it writes (one 66-row forward would differ from the five-row natural
pass by the BF16 batch floor, about the size of J-C-G1's bound ν_T ≤ 0.02: review check below).

**The channel-ratio law (C-1).** For an edit Z, channel C ∈ {K, V, KV}, format f and depth l:
ψ_C^Z(f, l) = mean ID_C^Z / mean ID_C^nat. The law states that one scaling per edit holds for both channels and in every
format: ψ_K^Z = ψ_V^Z. Its statistic is λ = log(ψ_K / ψ_V) (definitions below). The rivals: R1 copy-only (λ ≤ log 0.5,
the key under-reads the edit), R3 key-flat (λ ≥ log 2), and graded departure (anything else that is not equivalent).

#### Populations

All populations are lists of story cores of `ckeys.story.make_cores`, compared by full tuple. A stream is make_cores(1,
rng) drawn repeatedly from one Random(seed), which gives the same cores as one make_cores(n, rng) call. A population takes
the first cores of its stream that are in none of the excluded sets, are not repeated, and satisfy its condition.

| Population | Rule | Size | sha256 (pinned in `ckeys/edits.py`) |
|---|---|---|---|
| E (evaluation) | Random(8101); excluded U; π(S) ≠ B and π(X) ≠ B | 80 (136 draws; 24 ordered (B, S) clusters) | a23465a211577f4c6e9efe78d4c7a588b598c05437406001145a420fec9d8c1b |
| H (calibration) | Random(8102); excluded U ∪ E; no condition | 200 (201 draws); H_fit = the first 150, H_cal = the last 50 | 8cc62cefc9865a475f51fd554ece829981b4b46dcdaa7ae66e9261cae3b6b584 |
| TSET (E4 training) | Random(8103); excluded U ∪ E ∪ H; π(S) ≠ B | 1,000 | 06c56f93dce02d93a957a4e78556f4694ea3d994385a80212a2da1edd3bff16f |
| THOLD (E4 held out) | the next 100 cores of the TSET stream that pass TSET's rule (TSET and THOLD: the first 1,100; 1,410 draws) | 100 | 136b7d940050222b1737d20df4aba78e1e327eac16ab86b50730b471d5e261f4 |

- X = `story.pick_x(core)`, π = `story.PAIR_SWAP` (box ↔ basket, shelf ↔ drawer, cabinet ↔ closet).
- The hashes are sha256(json.dumps([[*tuple] for each core in order])), with the tuple fields agent, other, object,
  distractor, initial, distractor_location, base, source. U's hash abd1f053… (common part) is asserted by the tests.
- The four populations are pairwise disjoint and disjoint from U (asserted by the preflight and the tests). No pilot drew
  from seeds 8101–8104 (8104 seeds only the random directions of control R).
- **Prakash et al.'s material** (J-C6, the screen): their template-2 stories, raw wrapper and seed-10 pool of 320 pairs
  (`ckeys/causaltom.py`, release at 0579347 with every file hash and the pool hash 4451da1a… asserted). Per model the
  population is the first 150 pairs that pass their LM filter under NO-MENTION (every passing pair if fewer, reported). At
  Qwen2.5-14B the filter is re-run and compared with stage 6's population (recorded). J-C6's means use every pool pair
  outside its population (170 when 150 pairs pass).

#### Models, dictionaries and sourcing

- Qwen2.5-7B-Instruct (`qwen7`): E1–E5. Mistral-7B-Instruct-v0.3 (`mistral7`): E1, E2, E4, E5. Llama-3.1-8B-Instruct
  (`llama8`): E1, E2, E5. Depths l ∈ {3, 7, 11, 15} in every model (C-9 adds l = 15 at Mistral-7B and Llama-3.1-8B).
  Qwen2.5-14B-Instruct (`qwen14`): J-C6 only.
- Files and fallback as in the common part (`scripts/stage8_models.json`, `scripts/fetch_verified.py`). Llama-3.1-8B is the
  new family; if its files fail verification or its preflight fails (a pre-output check), `yi9` takes its place (families
  E1, E2, E5) before any output of it exists; the replacement is decided once and every later session keeps it
  (COMMIT.txt). The scorer puts Yi-1.5-9B in Llama-3.1-8B's slot of every line on the E population when there is no eval
  file of Llama-3.1-8B and there is one of Yi-1.5-9B. The overlap screen is then run at Qwen2.5-7B only (the Prakash
  material is located only for the Qwen and Llama-3 tokenizers).
- BF16, sdpa attention, transformers 5.18.0, `use_cache=False` in every scoring pass.
- **The dictionary (E3).** andyrdt/saes-qwen2.5-7b-instruct at revision c37e53c4bb07127ad17ab88f28b93d4e87142e59 (the
  repository head on 2026-10-10, last modified 2025-05-26), `resid_post_layer_{3,7,11,15}/trainer_1`: BatchTopK, k = 64,
  131,072 latents, trained on the output of 0-based block l (config `io: "out"`). The sha256 of each file is asserted at
  download:

  | Layer | ae.pt (3,758,637,401 bytes) | config.json | eval_results.json | Published FVE |
  |---|---|---|---|---|
  | 3 | 93f70d8aac4976fa… | 5667fd2bd9c18892… | 1986b2b392dff5a0… | 0.931 |
  | 7 | 94be36b5ba215103… | 19b6c408283063c1… | 4892e73fb50f0939… | 0.862 |
  | 11 | 36bddbd229d59c11… | bd21d02ae9d119d2… | 9eb49624a999891a… | 0.827 |
  | 15 | 2efefadd8d85ad1a… | 9fd2d432683dfb84… | 176ad5c35b2e098e… | 0.807 |

  The full hashes are in `ckeys/sae.PINS`. The state dict (checked on the pinned file's pickle header) holds
  encoder.weight [131072, 3584], encoder.bias, decoder.weight [3584, 131072], b_dec, threshold (one global value) and k.
- **The reader heads (J-C-READ).** H*_{>7} = the stage-6 top-k* heads by a3 under OPTIONS-AFTER
  (`results/gpu_stage6/heads/<model>.json`, committed, sha256 ed828a9b701d60f5… and 88ababd91bdb3155…; k* = 40 at
  Qwen2.5-7B and 52 at Mistral-7B) that lie in blocks > 7: 32 and 43 heads. The three random sets have the same sizes and
  are drawn from all heads of blocks > 7 (numpy default_rng(2), the first heads of three permutations). The file's sha256 is
  recorded by every readers file.

#### Prompts, edits, rows and scorings

**Formats.** LETTERS-AFTER (LETTER), OPTIONS-AFTER (P1), SENTENCE-AFTER (POST) and NO-MENTION (NONE), as
`ckeys.encoding.raw_prompt`, wrapped by `chat_text` (system turn "You are a helpful assistant.", generation prompt,
assistant prefill "Answer:"). Every E core has prompts of one length per format for B, S, X, π(S) and π(X), which differ
only at p; p and the prefix through p are the same in the four formats (asserted for every core in the preflight).

**Edit vectors** (written at (p, l); t ∈ {S, X} the target, B the base value; means over H_fit; FP32):
- **T** (equivalence control): h_t,l(p).
- **R** (specificity control): h_B + ‖μ1(t) − μ1(B)‖ r_t, r_t a unit Gaussian direction per depth and value
  (torch Generator seed 8104, drawn depth by depth).
- **E1** (CAA-in): h_B + μ1(t) − μ1(B), μ1(x) = mean of h_x,l(p) over the 150 H_fit cores with x written at p.
- **E2** (CAA-out): h_B + μ2(t) − μ2(B), μ2(x) = mean block-l output at the token " x" over the 24 neutral sentences
  (`ckeys/neutral.py`, sha256 593ce65ce4eb5b468527c6447601cc6670f9576e6434398dc105fb6c21bcce2a; each sentence is the user
  turn of the chat template with no prefill; every sentence places every form exactly once in all four Part-C
  tokenizers).
- **E3** (SAE, Qwen2.5-7B): with a(h) = relu(W_enc (h − b_dec) + b_enc) masked by the threshold and d_j = decoder.weight[:,
  j]: h_B + Σ_{j∈F_t} (β ā_j(t) − a_j(h_B)) d_j − Σ_{j∈F_B∖F_t} a_j(h_B) d_j. Every other latent and the SAE error stay at
  B's. ā_j(x) = mean a_j(h_x,l(p)) over H_fit; s_j(x) = ā_j(x) − max_{y≠x} ā_j(y); F_x = the top-k_F latents by s_j(x) with
  s_j(x) > 0. **The k_F / β rule** (on H_cal under NONE, targets S and X): k_F is the smallest of {4, 8, 16, 32, 64} whose
  flip-to-target rate (the share of (H_cal core, t) whose E-argmax in the edit's KV row is t) is ≥ 0.7 with β = 1; if
  none, k_F = 64 and β is the smallest of {2, 4} that reaches 0.7; if none, k_F = 64, β = 4 and E3 is recorded as
  "ineffective at calibration" at that layer (its rows still run; J-C-G5 decides).
- **E4** (DAS at p, Qwen2.5-7B and Mistral-7B): h_B + Uᵀ U (h_π(t),l − h_B), U the rank-16 basis fitted at (p, l) with the
  released recipe (`ckeys/das_at.py`): the pair-swap objective (base B′, source S′, target π(S′)) on TSET, six-way
  cross-entropy over the candidate scores (each the logsumexp of the logits of its single-token forms " x" and " X"),
  AdamW (lr 1e-3, weight decay 0, gradient clip 1.0), batch 1, one shuffled epoch (random.Random(seed)); NO-MENTION with the
  "Answer:" prefill (the released recipe read at the reply start: a disclosed deviation). Seed 101 starts from the PCA basis
  of h_S′ − h_B′ over TSET; seed 102 from a random orthonormal basis (C-7). π is an involution, so the source π(t) makes the
  remap write t, and no natural run is in this state. The NONE flip rate on THOLD is reported per fit.
- **E5** (the lexical boundary family, C-4): FR, DE and SYN: h_B + α (μ5(form(t)) − μ5(form(B))), μ5 as μ2 with the
  French, German or synonym form of each value (`ckeys/variants.py _F`), read at the last token of the form, over the
  neutral sentences in which all six forms of the sub-family occur exactly once as their own token sequence (all 24 in
  every Part-C tokenizer at build; a sub-family clean in fewer than 20 sentences of a model's tokenizer is not run in that
  model); NL: h_B + α NONLEX(μ1(t) − μ1(B)). α ∈ {1, 2} per sub-family and depth on H_cal under NONE only: α = 1 if
  φ_Hcal(NONE) ≥ 0.3 at α = 1, else α = 2, where φ_Hcal is the sum over the 50 H_cal cores of the edit's ID_KV divided by
  the sum of the natural ID_KV (E scores).
- **Components** (C-3) of an edit's displacement d = h_Z − h_B against the story's natural displacement dn = h_t − h_B and
  the lexical span L_l = span{μ2(x) − μ2(box) : x ≠ box} (5-dimensional, orthonormal basis Q_l by QR):
  PAR = h_B + c dn with c = ⟨d, dn⟩ / ⟨dn, dn⟩; PERP = h_B + d − c dn; LEX = h_B + Q Qᵀ d; NONLEX = h_B + d − Q Qᵀ d; for
  Z ∈ {E1, E2, E3, E4 (each seed)}, and LEX / NONLEX of the natural displacement itself.

**Rows of a cell** (format f, depth l; every row on the B prompt; K/V at p replaced in blocks ≥ l + 1 by the tables of a
donor run; "K row" = (K_Z, V_B), "V row" = (K_B, V_Z), "KV row" = (K_Z, V_Z)):
- every format: the self row (B's own tables, the in-batch reference); the natural rows K, V, KV of S and X; T, E1, E2,
  E3, E4 (two seeds), E5 (FR, DE, SYN, NL) K, V, KV toward S and X; R KV toward S and X;
- the synthetic rows of J-C-G8 (C-2): SKλ = the natural K table interpolated toward B, K_B + λ (K_t − K_B) for λ ∈ {0.5,
  0.8}, as its K row (with V_B) and its KV row (with V_t), its V row being the natural V row; SVλ likewise with V
  interpolated and the natural K;
- P1 and NONE: PERP and NONLEX of each Z, and NONLEX of the natural displacement, K, V, KV;
- NONE: PAR and LEX of each Z and LEX of the natural displacement, KV.
A cell holds 494 rows at Qwen2.5-7B (85 / 151 / 85 / 173 in LETTER / P1 / POST / NONE), 442 at Mistral-7B and 338 at
Llama-3.1-8B, per depth and story. Rows are scored in forwards of 64 rows; each forward is led by its own self row, and
every row is stored as its score minus that self row's.

**Scorings.** E: `ckeys.surface.score`, the exact chain-rule log-probability summed over the 32 fixed forms
(FRAMES_E_FIXED: " ", "", " The ", " the ", "The ", "the ", " In the ", " in the ", " On the ", " on the ", " At the ",
" at the ", " Inside the ", " inside the ", " **", "**", each with w and W) plus the frames discovered on H; under LETTER
the letters' forms " X", "X", " **X", "**X". L: log p(" w") (log p(" X") under LETTER). **Frame discovery** (calib, before
any E item): greedy generations (≤ 16 new tokens, stop at a newline, EOS or a completed candidate) of the clean B, S and X
runs of the 50 H_cal cores in P1, POST and NONE (450 generations); `ckeys.generate.discover_frames` (a frame in ≥ 2 % and
≥ 2 of some format's generations, at most 16); a frame is admitted, in order, only if every H_cal and E form set still
builds (no form a proper prefix of another). The frames are recorded in `calib/<key>.json`.

#### Measures

For an instance Z (an edit, a component, T, R or a synthetic row set), channel C ∈ {K, V, KV}, format f and depth l, with
d_w(row) = score_w(row) − score_w(self row) (E unless stated; letters of S and X under LETTER):
- ID_C^Z = ½ [(d_S(C(Z_S)) − d_S(C(Z_X))) + (d_X(C(Z_X)) − d_X(C(Z_S)))] per story; ID^nat with the natural rows.
  For E4 the per-story value is the mean over the two seeds (the seed level of the bootstrap).
- ψ_C^Z(f, l) = mean ID_C^Z / mean ID_C^nat; φ_Z = ψ_KV^Z; ι^Z = 1 − (mean ID_K^Z + mean ID_V^Z) / mean ID_KV^Z;
  κ^Z = mean ID_K^Z / (mean ID_K^Z + mean ID_V^Z) and σ = κ^nat (descriptive only, C-1).
- **Channel used** in (f, l): mean ID_C^nat ≥ 2 nats and ≥ 0.1 mean ID_KV^nat (point estimates).
- **λ statistics** of Z at depth l (ψ floored at 0.01 before the log, so λ stays finite and a dead channel is a departure):
  W(f, l) = log ψ_K(f, l) − log ψ_V(f, l) for f ∈ {P1, POST} with both channels used in (f, l);
  A_L(l) = log ψ_K(LETTER, l) − log ψ_V(NONE, l) with K used under LETTER and V under NONE;
  A_P(l) = log ψ_K(P1, l) − log ψ_V(NONE, l) with K used under P1 and V under NONE.
  A statistic needs its cells evaluable (J-C-G4) and Z effective at l (J-C-G5; for components: carrying identity). If Z's
  ι (point) exceeds 0.5 in one of its cells (the natural cell passing by J-C-G4), the statistic counts as a departure
  ("interaction", C-9's ι rule: Z's KV effect against the sum of its K and V effects); the synthetic rows SK and SV have no
  ι rule.
- **Equivalence** of a statistic: H0: λ ≤ −log 1.25 and H0: λ ≥ log 1.25 both rejected, i.e. the 90 % interval lies inside
  (−0.223, 0.223) (TOST at 5 % per side), with at most 5 % of resamples undefined.
- **The law in a combo** (a model, an instance and a set of depths): MET iff it has ≥ 3 λ statistics and every one is
  equivalent (an intersection-union test); NOT MET otherwise; NOT EVALUABLE with fewer than 3 statistics. Its **pattern**:
  the pooled λ̄ (the mean of its statistics, recomputed per resample): equivalent if every statistic is; else R1 copy-only
  if λ̄ ≤ log 0.5 (point); R3 key-flat if λ̄ ≥ log 2 (point); otherwise graded departure (λ̄ printed).
- **Flip rate** of Z under f: the share of (story, t) whose E-argmax over the six candidates in the KV(Z_t) row is t (the
  natural flip rate with the natural KV rows).
- **Coverage** of a cell: the minimum over the natural rows (self, K / V / KV of S and X) of the mean E mass, a row's E
  mass being Σ_w exp E_w over the six candidates.
- **KO** (J-C-READ; OPTIONS-AFTER, l = 7, L scores; `experiments/stage8_edits.py` stage readers,
  `analysis/stage8c_parts/readers.py`): every row is on the P1 B prompt with B's key and value at p pinned from block 8. In
  a donor's row the donor's key at p from block 8 (natural: S's or X's; an edit: the key its prefix pass captured; E4: seed
  101) is seen by every head of blocks ≥ 8 at every query position (spec "all", equal to the K-only clamp row), except,
  for a set (H*_{>7}, or one of the three random sets), that the set's heads see B's key at the six option-word positions
  of the "Choices:" list. ID_K^D[spec] is ID on these rows (each minus the in-batch self row, B's own tables), and
  KO_D(set) = 1 − mean ID_K^D[set] / mean ID_K^D[all], recomputed per resample.
- **On the binding-swap material** (J-C6, J-C-SCREEN, J-C-WIN; H's definitions, `analysis/stage8c_parts/prakash.py`):
  per pair, arm and format the exchange rows r0 (B with B's K/V), r1 (the patched run M), r2 (B with M's keys), r3 (B
  with M's values) and r4 (B with M's K/V) at the patched positions from block d + 1, with m = log p(target) − log p(s_q);
  ψ_K = mean[m(r2) − m(r0)] / mean[m(r1) − m(r0)], ψ_V likewise with r3, κ = ψ_K / (ψ_K + ψ_V); Φ = mean[m(r1) − m(r0)];
  IIA (for CAA: its flip rate) = the share of pairs whose r1 top token is the target word. The κ rule: ψ_K + ψ_V ≥ 0.5 and
  ψ_K, ψ_V ≥ −0.1 (a resample that fails it is undefined). Gate b0: mean |m(r4) − m(r1)| and mean |m(r0) − m(B)| ≤ 0.3
  nats; Gate b2: Φ ≥ 3 nats. An arm is usable in a format when Gates b0 and b2 and the κ rule (point) hold.
  s_ID(f, l0) = mean ID_K / (mean ID_K + mean ID_V) of the natural clamp of S's or X's K/V at p from block l0, against the
  self-clamp row. In the screen, Φ(l) and IIA(l) are those of BIND patched at block l under NO-MENTION (the BIND sweep).
- **Table statistics** per story, depth, Z and t (pooled over blocks ≥ l + 1 and all KV heads): ν = ‖[K;V]_Z − [K;V]_t‖ /
  ‖[K;V]_t − [K;V]_B‖ (one pooled ratio, C-3); the K and V cosines of the displacements and their norm ratios; the residual
  cosine cos(d, dn) at (p, l); the projections ⟨Δ_Z, Δ_t⟩ / ⟨Δ_t, Δ_t⟩ of the K and V displacements (O1; the K projection
  also over the KV groups of the reader heads H*).
- **Story-level distance** D_Z = mean_s (|ID_K^Z − ID_K^nat| + |ID_V^Z − ID_V^nat|) / mean_s ID_KV^nat (reported).

#### Statistics

- **Bootstrap.** Hierarchical (C-13): resample the 24 ordered (base, source) clusters of E with replacement, then the
  stories within each drawn cluster with replacement; 10,000 resamples, seed 20261012, one index set per model, shared by
  every format, depth, row and scoring, so every contrast is paired. The E4 seeds are a further level: every resample also
  draws two seeds with replacement from {101, 102}. Every ratio, log ratio and share is recomputed in every resample.
  Prakash et al.'s pairs (J-C6, the screen) use the pair bootstrap with the same seed and size.
- **Levels.** TOST at 90 % (5 % per side) for equivalence; every other interval criterion is a one-sided test at 2.5 % on
  the 95 % interval ("H0: θ ≤ t rejected: lower bound > t"). Lines over several models are intersection-union tests on
  these intervals (no correction; NOT MET as soon as one evaluable model or combo fails, see the combination rule).
  Point floors are effect-size conditions.
- **Holm** (common part, decision D2): reported, no verdict uses it. The family is the interval components of this part's
  R-class account lines that have a verdict (J-C3, J-C4, J-C5, J-C-BOUND, J-C-READb, J-C-WIN; J-C-SCREEN has none): the
  two one-sided components (H1: λ > −log 1.25 and H1: λ < log 1.25; own decision: the 90 % interval) of every λ statistic
  of a J-C3 or J-C4 combo that has a verdict and of every identity-carrying component of J-C5, the upper bound of each
  pooled λ̄_E5 that has a verdict (H1: λ̄_E5 < log 0.8), the lower bound of each counted KO_Z(H*) (H1: KO > 0.5) and the
  window's H9 bound (H1: κ(OPTIONS-AFTER) − κ(NO-MENTION) > 0). A component whose estimate or SE is undefined is left out.
  The shared helper `analysis/stage8_holm.py` computes one-sided p values from the bootstrap SE (Φ(−(est − bound)/se) for
  '>', Φ((est − bound)/se) for '<') and Holm's step-down at familywise 0.025; the scorer prints, per R line, the components
  whose decision changes and the verdict with Holm's decisions in place of the interval decisions (a MET line with a
  component no longer rejected becomes NOT MET; every other verdict is unchanged).
- **SUMMARY** (common part): the 11 lines are tallied by class (M: J-C1, J-C2, J-C-READa, J-C6; R: the other seven; no
  class-L line; Part C defines no MET IN PART), with the observed against the expected met count and the Brier score over
  the lines with a verdict, and the met rate among the R lines with a verdict.
- **Evaluability.** Only the gates and rows the tested edit does not change decide evaluability. A model with no eval
  results, or with fewer than 60 evaluated stories (a run cut by the deadline), is not evaluable. If a part of the scorer
  raises, its lines are NOT EVALUABLE (traceback printed, exit status 1). TEST_MODE outputs (tag TEST_) are plumbing checks.
- **Scorer checks that do not change verdicts.** A provenance or population MISMATCH (results files from more than one
  commit; outside TEST_MODE a model revision other than the manifest's, a dtype other than BF16 or an attention other
  than sdpa; an eval, readers or explore file that records a calibration sha256 other than the calibration file's; a
  results file that is unreadable or has no provenance block; a preflight E hash other than the pin, or a skipped item;
  eval stories not indexed 0..n−1 in order, or an eval file whose recorded E hash is not the pin) is printed and makes the
  scorer exit with status 2, as do results without a passing J-C-G0; the verdicts are printed as computed.

#### Gates

- **J-C-G0, exactness** (FP32, CPU, Qwen2.5-0.5B, before any model; 1e-4 in the logits): `tests/test_stage8_edits.py`
  (15 tests), `tests/test_sae.py` (7), `tests/test_stage8c_score.py` (19), the shared `tests/test_generate.py` (5; frame
  discovery uses `ckeys/generate.py`, which the stage-8 review changed), `tests/test_stage8_populations.py` (5; rule G6
  across the parts) and `tests/test_stage8_holm.py` (7; the Holm helper, D2), and the shared `tests/test_surface.py`,
  `tests/test_clamp.py`, `tests/test_head_splice.py`, `tests/test_prakash.py` in the same run (95 tests). The scorer reads
  the last pytest run in `logs/pytest.log`: J-C-G0 is MET when each of the six files named with a count has at least that
  many passed tests (15, 7, 19, 5, 5 and 7) and none failed, errored or skipped, and no test of the run failed or errored (a
  shared test of the other four files that skips because a study tokenizer is not available does not fail the gate);
  without the log it is NOT EVALUABLE. The checks, each against an
  independently computed reference:
  1. (i) writing h_S,l(p) equals the natural KV(S) clamp from l + 1 (tables from the S run), l ∈ {0, 3, 10}, NONE and P1;
  2. (ii) for random vectors, B with the edit's captured K/V from l + 1 equals the edited run (two rows, per-row tables);
  3. (iii) prefix tables equal full-run tables (1e-5); (iv) the prefix and its tables are identical in the four formats
     (exact);
  4. the scored rows (E and L, minus the in-chunk self row) equal `score_reference` under a separately built clamp with
     per-row tables, in P1 and LETTER; the self row's L equals a plain forward;
  5. (viii) T through the pipeline: its tables equal the natural ones (1e-5), its rows the natural rows, so κ_T = σ; through
     eval's own passes (66 edit rows in forwards of the natural pass's shape) T's tables equal the natural ones bitwise;
     the row layout of every cell (synthetic rows interpolate exactly; components where stated);
  6. the component algebra and the table statistics on synthetic tables with known answers;
  7. (vii) the J-C-READ rows: the "all" row equals the plain K-only clamp row, the empty row equals clean B;
  8. the dictionaries' io = "out" site: a forward hook on block l equals the output of `model.layers[l]` and
     hidden_states[l + 1] (tiny random Qwen2, C-8);
  9. the BatchTopK code against direct references (encoder, threshold mask, decoder orientation, the state-dict layout and
     its hash, FVE, selectivity, the edit; an edit of every differing latent with zero error reproduces h_t);
  10. the DAS-at-p code (forms, PCA and random inits, orthonormal fits, the identity when source = base, the eval map);
  11. (ix) on Prakash et al.'s release, the CAA arm with μ(S) − μ(s_q) = h_S − h_B at [p, p+1] gives the ID arm's rows;
  12. the populations (sizes, hashes, disjointness), the neutral sentences (hash, cleanliness), the no_grad guard, the
      reader sets, the edit vectors from a calibration record;
  13. the scorer on synthetic results with known answers (MET, NOT MET and NOT EVALUABLE paths of every line and gate;
      the combination: NOT MET on any evaluable failure; E3 in J-C-READb only when layer 7 passes J-C-G3; J-C6's
      evaluability with BIND usable; outcome (b) read only from evaluable models; the Holm family of the R lines, its
      components and the verdicts under Holm through `analysis/stage8_holm.py`; J-C-G0's file list against the script's
      pytest step, and no line computed and no outcome named without a passing J-C-G0);
  14. greedy decoding under cache clamps equals cache-free stepwise argmax decoding with the clamps active (Qwen2.5-0.5B
      and tiny random models), answer parsing and frame discovery, and a chat template's end-of-turn token ends a
      generated answer (`tests/test_generate.py`);
  15. the stage-8 populations of Parts B, C and D: U computed identically, the sizes, disjointness from U and from each
      other, no pilot seed (`tests/test_stage8_populations.py`, rule G6);
  16. the Holm helper: p values, degenerate and undefined components, hand-worked and brute-force step-down results
      (`tests/test_stage8_holm.py`).
  J-C-G0 failing or not run makes every line NOT EVALUABLE: no model counts in any line on the E population, and the
  HEADLINE prints "J-C-G0 not passed: no outcome" (no Section-5 outcome is selected).
- **J-C-G1, the equivalence control** (per model): at every depth and for each target t, the mean over stories of ν_T
  ≤ 0.02; in every evaluable cell (J-C-G4), |ψ_C^T − 1| ≤ 0.05 (point) in each used channel C ∈ {K, V}, and
  mean_s |ID_C^T − ID_C^nat| ≤ 0.25 nats for C ∈ {K, V}; at least one evaluable cell. Failing makes the model not
  evaluable. Since the edit passes have the natural pass's shape, T's tables equal the natural tables bitwise unless the
  pipeline is wrong, and T's rows are scored in the same forward as the natural rows: J-C-G1 checks the pipeline on the
  GPU (as J-C-G0 does in FP32), not a BF16 floor.
- **J-C-G2, the natural format effect** (per model): σ(LETTER, 3) − σ(NONE, 3) ≥ 0.5 (point) and H0: ≤ 0 rejected
  (computed whether or not these two cells pass J-C-G4). Failing makes the model not evaluable.
- **J-C-G3, the dictionary's alignment** (Qwen2.5-7B, per layer; C-8): FVE of the layer-l dictionary on the block-l outputs
  ≥ published − 0.05 (0.881 / 0.812 / 0.777 / 0.757), and greater than its FVE on the outputs of blocks l − 1 and l + 1. FVE
  is dictionary_learning's (1 − Σ var(h − ĥ) / Σ var(h)) over every prompt position ≥ 1 of the clean B NONE prompts of the
  200 H cores (position 0, the attention sink, is left out). FVE at p (H_fit, six values) is reported. Failing makes E3 not
  evaluable at that layer (in J-C3, in J-C5 and, at l = 7, in J-C-READb; a layer without a J-C-G3 record counts as
  failing). (Layer 15 is expected to be marginal: its published FVE is 0.807.)
- **J-C-G4, the cell** (per model, format, depth; point estimates): mean ID_KV^nat ≥ 2 nats; ι^nat ≤ 0.5; mean ID_K^nat
  and mean ID_V^nat ≥ −0.1 mean ID_KV^nat; coverage ≥ 0.8. Otherwise the cell enters no statistic.
- **J-C-G5, efficacy** (per model, family, depth; under NONE; C-11): flip rate ≥ 0.8 × the natural flip rate (point), and
  φ ≥ 0.5 (point) with H0: φ ≤ 0.3 rejected (lower 95 % bound > 0.3). E5 sub-families (gated in) and components (carrying
  identity): φ(NONE) ≥ 0.3 (point) with H0: φ ≤ 0 rejected (C-3, C-4). Otherwise the family is "ineffective at l"
  (reported, not counted). The gate needs the NONE cell at l to pass J-C-G4; otherwise the family is not evaluable at l.
  E4 uses the seed level (flip rate and φ of the two seeds together).
- **J-C-G6, specificity** (per model, depth): |mean ID_KV^R| ≤ 0.1 mean ID_KV^nat under NONE. A failure flags the depth
  in the report; no line depends on it.
- **J-C-G7, the discrepancy anchor** (J-C6): BIND at block 28 reproduces stage 6 (point estimates): IIA under NO-MENTION
  ≥ 0.95 and |κ_BIND(f) − κ_H(f)| ≤ 0.05 in every format f run, for κ_H = 0.618 / 0.906 / 0.864 (NO-MENTION / QNAMES /
  OPTIONS-AFTER). Failing makes J-C6 NOT EVALUABLE.
- **J-C-G8, sensitivity** (per model; C-2): over all depths, with the same statistics and no efficacy gate, T is
  classified equivalent (every statistic equivalent), SK50 is a departure (some statistic not equivalent) with pooled
  λ̄ < −log 1.25 (point), and SV50 is a departure with λ̄ > log 1.25 (point); each of the three needs at least one
  statistic, so a model with none fails the gate. SK80 and SV80 are reported, and so is the design's old κ rule (MAD
  ≤ 0.12 with upper bound ≤ 0.17, maximum gap ≤ 0.25) on the same rows. Failing makes the law lines (J-C1 to J-C5,
  J-C-BOUND) NOT EVALUABLE in that model.

#### Confirmatory lines

All lines are account lines (kind A); Part C has no measurement-validity line. The prior is P(MET), given that the line
is evaluable, recorded before any stage-8 output. The class follows the prior (decision D1: L = implied by data in hand on
the same models and material with prior ≥ 0.9; M = prior ≥ 0.8; R = prior < 0.8), so no Part-C line is class L. Tiers per
C-6: Tier 0 = E1 at every depth and E2 at l ∈ {3, 7} at Qwen2.5-7B and Mistral-7B (J-C1; C-6 labels it L, its prior
0.80 makes it class M); Tier 1 = E2 at l ∈ {11, 15} and Llama-3.1-8B (J-C2, M); Tier 2 = E3, E4, PERP / NONLEX and E5
(R).

| Code | Class, prior | Criterion | Justification of the prior (data in hand) |
|---|---|---|---|
| J-C1 | M, 0.80 | The law MET in each evaluable combo of (Qwen2.5-7B, E1, l ∈ {3,7,11,15}), (Qwen2.5-7B, E2, {3,7}), (Mistral-7B, E1, all), (Mistral-7B, E2, {3,7}); NOT MET if any evaluable combo is NOT MET; MET needs ≥ 3 evaluable combos, else NOT EVALUABLE | 0.5B pilot: κ_E within 0.03 of σ in LETTER, P1, NONE; 1.5B pilot: φ 0.91–1.04 at blocks 1–9; the critic's 1.5B pilot: E2 transfer 0.94 at block 9. Power (below): the TOST passes at c_K/c_V = 0.9 and fails at 0.8 |
| J-C2 | M, 0.80 | The law MET in each evaluable combo of (Qwen2.5-7B, E2, {11,15}), (Mistral-7B, E2, {11,15}), (Llama-3.1-8B, E1, all), (Llama-3.1-8B, E2, all); NOT MET if any evaluable combo is NOT MET; MET needs ≥ 2 evaluable combos, else NOT EVALUABLE | E2 at 1.5B block 13 under LETTER reached 0.50× (natural effect 0.35 nats, below the channel floor); E2's transfer at block 9 was 0.94; Llama never measured. The class is fixed by C-6; the prior is its floor |
| J-C3 | R, 0.55 | The law MET in (Qwen2.5-7B, E3, the layers passing J-C-G3); NOT EVALUABLE with fewer than 3 statistics (also when no layer passes J-C-G3) | No data on E3; selective latents at p are likely the value's lexical features, which the readers read like the natural edit |
| J-C4 | R, 0.35 | The law MET in each evaluable combo of (Qwen2.5-7B, E4, all), (Mistral-7B, E4, all) (seed level); ≥ 1 evaluable combo, else NOT EVALUABLE | The remap writes the projection of π(t)'s state to produce t; under the token-form account its key part points at π(t), a departure; no data at p |
| J-C5 | R, 0.35 | In every model passing J-C-G1, J-C-G2 and J-C-G8, for every identity-carrying (φ(NONE) ≥ 0.3, H0: φ ≤ 0 rejected) PERP or NONLEX component of an E1–E4 effective at that depth (E3 at the layers passing J-C-G3; E4 at the seed level): each of its statistics W(P1, l) and A_P(l) equivalent (one pooled test over every statistic of every such component, with no minimum count); NOT EVALUABLE if no identity-carrying component has a λ statistic | The critic's 1.5B pilot: NONLEX(E1) φ 0.11–0.13, 0 flips; a carrying non-lexical component is predicted to be copy-read |
| J-C-BOUND | R, 0.50 | For every E5 sub-family with λ statistics at the depths where it is gated in, in every model passing J-C-G1, J-C-G2 and J-C-G8: pooled λ̄_E5 over those statistics ≤ log 0.5 (point) with H0: λ̄_E5 ≥ log 0.8 rejected (upper 95 % bound < −0.223), and E2's pooled λ̄ over those of the same statistics (code and depth) that E2 has, where E2 is effective, ≥ log 0.8 (point). A sub-family whose statistics E2 has none of is not evaluable; ≥ 1 sub-family with a verdict, else NOT EVALUABLE (also when no sub-family is gated in) | The critic's 1.5B pilot (2 stories): French vectors φ(NONE) 0.34–0.39, transfer φ(LETTER)/φ(NONE) 0.05 at block 9 against 0.94 for E2; synonyms φ 0.06–0.08 |
| J-C-READa | M, 0.85 | Qwen2.5-7B and Mistral-7B (each passing J-C-G1 and J-C-G2), P1, l = 7, the readers rows (L scores): the gate KO_nat(H*_{>7}) ≥ 0.7 and mean ID_K^nat[all] ≥ 2 nats (point; failing makes the model's families not evaluable); for E1 and E2 where effective at l = 7 (J-C-G5) and mean ID_K^Z[all] ≥ 2 nats: KO_Z(H*_{>7}) ≥ 0.7 (point) with H0: KO_Z ≤ 0.5 rejected, and the mean KO over the three random sets ≤ 0.25 (point); ≥ 1 counted (model, family), else NOT EVALUABLE | Stage 6: KO(k*) 0.977 / 0.971, random sets ≤ 0.011; E1 and E2 are aligned with the natural displacement (residual cosine 0.80–0.997 at 1.5B) |
| J-C-READb | R, 0.60 | The same for E3 and E4 (the readers rows of seed 101; efficacy at the seed level); E3 is not evaluable, and not counted, when layer 7 fails J-C-G3 | E3 and E4 need not be aligned with the natural edit in the key-read subspace |
| J-C6 | M, 0.85 | Qwen2.5-14B, block 28: in every evaluable format (NO-MENTION and OPTIONS-AFTER evaluable): κ_CAA ≤ 0.30 (point), κ_BIND − κ_CAA ≥ 0.30 (point) with H0: κ_BIND − κ_CAA ≤ 0 rejected (paired; a resample where either arm fails the κ rule is undefined, at most 5 % undefined); a format is evaluable when CAA and BIND are both usable there (Gates b0 and b2 with Φ ≥ 3 nats, the κ rule; BIND is the comparison arm, so its failure makes the format not evaluable, never NOT MET) and CAA's flip rate under NO-MENTION is ≥ 0.5; J-C-G7 passed | Stage 6: ID@28 κ 0.129 / 0.007 / 0.006 and s_ID(f, 29) 0.017 / −0.028 / −0.015 |
| J-C-SCREEN | R, 0.50 | No overlap window in any evaluable screened model (Qwen2.5-7B, Llama-3.1-8B; point estimates): no block l with BIND Φ(l) ≥ 3 nats and IIA(l) ≥ 0.5 under NO-MENTION and s_ID(OPTIONS-AFTER, l + 1) ≥ 0.4 (onset l + 1 measured for every such block, besides the grid round(x n_L / 28), x ∈ {0, 3, …, 27}, for a model of n_L blocks). A screen is evaluable when BIND's IIA reaches 0.7 at some block and ≥ 50 pairs pass the filter; ≥ 1 evaluable screen, else NOT EVALUABLE | 14B: BIND IIA 0.00 at blocks 0–23 and 0.72 at 24 (0.5 L), s_ID(OPTIONS) 0.587 at 14, 0.218 at 18, 0.041 at 24; Llama-3-70B's released BIND band 25–44 of 80 (0.31 L) leaves room for a window at 8B |
| J-C-WIN | R, 0.30 | In every evaluable screened model with a window, at l_w = the earliest window block: H's H7 (\|κ(f) − s_ID(f, l_w + 1)\| ≤ 0.25 in every usable f, Pearson r ≥ 0.9 over the three formats, which needs all three usable: otherwise H7 is not met) and H9 (κ(OPTIONS-AFTER) − κ(NO-MENTION) ≥ 0.4 (point), H0: ≤ 0 rejected (paired, at most 5 % undefined); κ(QNAMES) between the two when QNAMES is usable), with H's Gates b0, b2 and κ rule (BIND reproduces at l_w by the window's definition, IIA ≥ 0.5, in place of H's Gate b1 at 0.7) and s_ID(f, l_w + 1) defined for a usable f; NO-MENTION and OPTIONS-AFTER usable, else not evaluable in that model; ≥ 1 model with a verdict, else NOT EVALUABLE (also without a window) | 14B (block 28): BIND κ 0.62–0.91 against s_ID ≈ 0, gaps 0.60–0.93 |

**Expected values** (the author's, not thresholds): λ within ±0.15 for E1 and E2 at l ≤ 7; E3 effective at l ≤ 7;
E4 effective, with λ < 0 under LETTER; E5-FR gated in at some depth with λ̄ ≈ −1; E5-SYN and E5-NL not gated in; no window
at Qwen2.5-7B.

#### Feasibility (C-9; from data in hand)

The number of λ statistics a combo can have is set by which channels the natural edit uses at each depth. Stage 1's
committed per-story rows give the one-sided natural effects at onsets 0, 0.0625 L and 0.3 L; identity contrasts are about
half of them where the effect is identity-specific (Qwen2.5-7B at onset 0: ID_K 20.4 against one-sided 40.2 under LETTER).

| Model, depth (onset) | LETTER K | P1 K / V | POST K / V | NONE V | λ statistics expected |
|---|---|---|---|---|---|
| Qwen2.5-7B, l = 3 (4; stage 1 at 2: one-sided 40.2; 39.6 / 8.6; 11.0 / 23.1; 36.8) | used | used / used | used / used | used | 4 |
| Qwen2.5-7B, l = 7 (8; stage 1 at 8: 23.2; 17.1 / 8.6, ι 0.38; 5.8 / 23.1; 36.8) | used | used / used | marginal / used | used | 3–4 |
| Qwen2.5-7B, l = 11 (12; no data; at 14B s_ID(OPTIONS) is 0.22 at 0.38 L) | likely | marginal / used | no / used | used | 1–3 |
| Qwen2.5-7B, l = 15 (16) | unlikely | no / used | no / used | used | 0–1 |
| Mistral-7B, l = 3 (4; stage 1 at 2: 30.0; 26.3 / 2.5; 12.1 / 19.9; 28.8) | used | used / not (ID_V ≈ 1.2) | used / used | used | 3 |
| Mistral-7B, l = 7 (8; stage 1 at 10: 4.2; 2.6 / 2.3; 1.5 / 21.2) | marginal | marginal / not | no / used | used | 0–2 |
| Mistral-7B, l = 11, 15 | no | no | no | used | 0 |
| Llama-3.1-8B | no data; Mistral-like expected | | | | 3–5 over l ≤ 7 |

Consequences declared now: the Tier-1 combos of E2 at l ∈ {11, 15} are expected to be NOT EVALUABLE at both Qwen2.5-7B
and Mistral-7B, so J-C2 is expected to rest on the two Llama-3.1-8B combos; the Mistral-7B combos of J-C1 are expected
to have 3–5 statistics, so J-C1 may be NOT EVALUABLE (fewer than 3 evaluable combos, none of them NOT MET) if two of
them fall short. Refined in review from the same data: stage 1's identity-to-margin ratios at onset 0 applied to its
onset-2 margins give, at Mistral-7B, identity contrasts of about 14.6 (LETTER K), 14.0 / 2.2 (P1 K / V), 7.4 / 11.7
(POST K / V) and 13.7 (NONE V) nats, and at onset 10 about 2.0, 1.4 / 2.0, 0.9 / 12.5 and 13.7; stage 6's three
top-ranked reader heads at Mistral-7B lie in blocks 6–7 (18 of its 52 in blocks 4–9), so l = 3 (onset 4) keeps them and
should resemble onset 2 (A_L, A_P and W(POST) used, W(P1) marginal: 3–4 statistics), while l = 7 (onset 8) loses them
(1–3 statistics). Each Mistral-7B combo of J-C1 is then expected to have 4–7 statistics. Every A statistic and the
efficacy gate need the NONE cell (coverage ≥ 0.8), so a model whose NONE coverage falls below 0.8 has no λ statistic at
any depth; this risk is shared by all depths. The law at depth is therefore tested mainly through V-only cells (φ) and
the reported κ against σ; this is a limit of the design.

#### Combination and the headline rule (C-6)

- J-C1, J-C2, J-C4, J-C-BOUND, J-C-READa, J-C-READb and J-C-WIN are intersection-union tests over their evaluable
  sub-combos, with the minimum numbers stated in the table: such a line is NOT MET as soon as one evaluable sub-combo is
  NOT MET, whatever the number evaluable; MET when every evaluable sub-combo is MET and at least the minimum number is
  evaluable; NOT EVALUABLE only when none is NOT MET and fewer than the minimum are evaluable. J-C3, J-C5 and J-C6 are
  single tests.
- J-C-G0 failing or not run makes every line NOT EVALUABLE and the HEADLINE names no outcome; J-C-G1 or J-C-G2 failing
  removes the model from every line on the E population (J-C1 to J-C5, J-C-BOUND, J-C-READa, J-C-READb); J-C-G8 failing
  removes it from the law lines (J-C1 to J-C5, J-C-BOUND). The lines on the binding-swap material (J-C6, J-C-SCREEN,
  J-C-WIN) have their own gates (J-C-G7, the screen's evaluability, Gates b0 and b2, the κ rule) and do not depend on
  J-C-G1, J-C-G2 or J-C-G8.
- **Headline** (printed by the scorer):
  - "Supported out of sample" requires the law MET in E3 and E4 (J-C3 and J-C4 MET) or in the identity-carrying
    components (J-C5 MET), and J-C-BOUND MET.
  - Tier 0 alone (J-C1 MET without the above) supports at most "consistent for near-natural steering vectors".
  - The design's JC4 (depth tracking) is a reported consistency check and never enters the headline (C-10).

#### What each primary line means for the paper (pre-written; E-2)

No outcome of Part C changes the title (common part). Table 1 row: "Attributing an intervention to a channel"
(Section 5). Figure 4: (a) λ per family, depth and format with the ±log 1.25 band; (b) κ against σ (descriptive);
(c) the decomposition (φ of PAR, PERP, LEX, NONLEX against the full edit); (d) KO per family beside the natural readers.

| Primary line | Abstract clause (MET) | NOT MET: replacement | Table 1 row | Figure |
|---|---|---|---|---|
| J-C3 with J-C4 (Tier 2) | "Identity edits obtained without the natural clamp — features of a public sparse autoencoder and a remap trained at the writing token — are read through the key and the value in the same ratio as the model's own written value (\|λ\| < log 1.25 in every cell)." | Per pattern: R1 "Edits that change the value through dictionary features or a learned remap are read mainly through the value (λ̄ = x): the key channel does not read them as it reads the written word." R3 / graded: "… are read through the key x times more (less) than the natural value; channel attribution depends on the edit." | "Independent edits (E3, E4)": MET / the pattern | 4a |
| J-C-BOUND | "Steering vectors that write the value as a French or German word, or without its English lexical component, are read by the value alone (λ ≤ log 0.5): the key channel reads only the token form of what an edit writes." | "Non-lexical identity edits are read like lexical ones (λ̄ = x); the token-form account of the key readers is contradicted for edits." Not evaluable: "No non-lexical identity edit was strong enough to test the boundary (φ < 0.3)." | "Boundary (E5)" | 4a, 4c |
| J-C1 (Tier 0) | (Body only) "Steering vectors from held-out stories and from unrelated sentences follow the law in every format and depth where both channels are used." | "Even steering vectors close to the natural edit depart from the natural channel ratio (λ̄ = x): the format law does not transfer to edits as a channel-ratio law." | "Near-natural edits (E1, E2)" | 4a |
| J-C-SCREEN | (Body only) "Binding forms only after the identity key route closes in every model screened (no block with a working binding swap and s_ID ≥ 0.4)." | "A binding swap works at block l_w, where the identity key route is open (s_ID = x); there the binding swap is read as the law predicts / is not (J-C-WIN)." | Note under row "Prakash et al.'s binding swap" | Appendix |

The abstract's risk summary counts the R lines of this part with the other parts' (common part).

#### Pre-committed Section-5 text per outcome (C-12)

Section 5 is the intervention section of paper v5 (Section 4 of v4). Exactly one of these paragraphs replaces its general
claim. The scorer's HEADLINE names the outcome, checked in the order (a), (a′), (c), (b), (e), (d), (f); the numbers x, a, b
are filled from the score file. Without a passing J-C-G0 the HEADLINE prints "J-C-G0 not passed: no outcome" and no
paragraph is selected.
The phrases "30–70 % away" and "the law's boundary is the kind of edit, not its depth" are deleted from the paper.
- **(a) Law met on Tier 2 and the boundary met** (J-C3 and J-C4, or J-C5, MET; J-C-BOUND MET): "Which cache channel an
  identity edit appears to act through is predictable before the edit is run: an edit is read through the key in proportion
  to the token-form part of what it writes. Edits from a public sparse autoencoder and a remap trained at the writing token
  keep the natural channel ratio within 25 % in every cell where both channels are used, while steering vectors that write
  the value in another language, or without its English lexical component, are read by the value alone."
- **(a′) Law met on Tier 2, boundary not met** (J-C3 and J-C4, or J-C5, MET; J-C-BOUND NOT MET or not evaluable): "Identity
  edits obtained without the natural clamp keep the natural channel ratio within 25 % in every cell where both channels are
  used; the predicted boundary does not appear (non-lexical edits are read like lexical ones, λ̄ = x, or none was strong
  enough to test it). Channel attribution follows the format, whatever the identity edit writes."
- **(b) Acts only through the natural lexical code** (none of (a), (a′) and (c) applies; at least one effective family
  is counted, and every effective family E1–E4 (E4 at the seed level), at every depth where it is effective, in every model
  that passes J-C-G1, J-C-G2 and J-C-G8 (E3 at the layers passing J-C-G3), has φ(PAR) or φ(LEX) ≥ 0.8 φ of the edit under
  NONE (point estimates); and no PERP or NONLEX component of those families carries identity, J-C5 not MET):
  "The edits we could build act
  through the same low-dimensional lexical code as the natural write; for them the channel ratio is a linear consequence
  of that code and is not evidence that channel attribution is independent of the edit."
- **(c) Boundary met without Tier-2 support** (J-C-BOUND MET; J-C3 and J-C4 not both MET, and J-C5 not MET): "The key
  channel reads only the
  token-form part of an edit: non-lexical identity edits are copy-read (λ ≤ log 0.5). Independent lexical edits do not keep
  the natural ratio (λ̄ = x), so the ratio is not a property of the format alone."
- **(d) Departure on Tier 2** (none of (a), (a′), (c), (b) and (e) applies, and a combo of J-C3 or J-C4 is NOT MET; the
  pattern of each such combo, R1, R3 or graded departure with λ̄, is printed): for R1 and R3 the replacement texts of J-C3 / J-C4 in the table above; for a graded departure: "Independent
  identity edits are read through the key x times as strongly as the natural value (95 % interval [a, b]); channel
  attribution of an intervention is approximate, and we report the departure per family."
- **(f) Tier 0 at most** (none of the above; printed as "consistent for near-natural steering vectors only" when J-C1 is
  MET, else as "not supported or not evaluable"): "For steering vectors close to the
  natural edit the channel ratio is kept; edits that differ from it could not be tested (reason), so we make no claim
  beyond near-natural steering vectors." If J-C1 is also not evaluable, Section 5 keeps only the motivating case and states
  that the law was not testable.
- **(e) Failed** (none of (a), (a′), (c) and (b) applies; J-C1 NOT MET, which one evaluable Tier-0 combo NOT MET suffices
  for): "Even steering vectors close to the natural edit depart from the natural channel ratio;
  the format law describes the model's own written value and does not predict how an edit is attributed."
- **The screen.** No window: "Binding forms only after the identity key route closes in every model screened, so a binding
  swap cannot be compared with an identity edit at a depth where the key route is open; H7–H10 stand." Window, J-C-WIN MET:
  "Inside the overlap window at block l_w, the binding swap is read as the law predicts." Window, J-C-WIN NOT MET: "Inside
  the overlap window the binding swap departs from the natural read (gaps x); the law does not extend to binding edits."
- **J-C6.** MET: "At block 28 of Qwen2.5-14B an identity edit is copy-read like the natural value (κ ≤ 0.30), while the
  binding swap there is key-read; key-flat reading is not a property of depth for identity edits." It does not separate
  kind from depth, because the natural read there is itself copy-only. NOT MET: "At block 28 even an identity edit is
  key-read (κ = x)."

#### Reported (no verdict)

- The π(t) diagnostic (C-7), per model with E4 (Qwen2.5-7B, Mistral-7B) and depth: under LETTER in the K rows of E4,
  d_π(t) − d_t averaged per story over the two seeds and both targets, its mean with its 95 % interval (story bootstrap). A
  lower bound > 0 is read as "the key raises π(t)'s letter: a lexical-key departure"; an upper bound < 0 as "the key
  raises t's letter".
- The lexical-code reading (C-3c), per model, family E1–E4 and depth where effective: φ of PAR and LEX against φ of the
  edit under NONE (point); "acts through the natural lexical code" if either is ≥ 0.8 of it.
- JC4 as a consistency check (C-10): for every (model, family E1–E4, format) with Δσ = σ(f, 3) − σ(f, 15) ≥ 0.25 (point),
  both cells evaluable and the family effective at both depths: Δκ ≥ 0.5 Δσ and |Δκ − Δσ| ≤ 0.2 (point); consistent if it
  holds in ≥ 80 % of ≥ 3 qualifying pairs.
- Printed with the lines and gates: the per-seed E4 laws (under J-C4) and J-C-G6 per depth (GATES). Printed in the
  EXPLORATORY section: the κ / σ table; ν, the K / V and residual cosines per cell and family.

#### Exploratory (no verdict)

L-scored λ; O1 (the projections of each edit's K / V displacement on the natural one, all heads and the reader heads' KV
groups, against the observed ψ); greedy-generation flip rates of the KV rows at l = 7 under NONE and P1 on the first 40
stories (O4); D per cell; the frame census; the SAE's FVE at p; the E5 calibration table. Dropped (C-13): Gemma-2 with
Gemma Scope, the α dose, BIND-p, LIST-BEFORE, the SAE-lexical / SAE-task split. In-hand context, stated as exploratory: at
14B the binding swap has IIA 0.00 at blocks 0–23 and 0.72 at 24, while s_ID(OPTIONS-AFTER) is 0.218 at 18 and 0.041 at 24.

#### Power (seen before finalisation)

`power_lambda.py` (build notes; the committed stage-1 per-story identity contrasts at onset 0 of Qwen2.5-7B and
Mistral-7B; 200 replicates of n = 80 stories with the two-stage (B, S) cluster bootstrap, 1,000 resamples; an edit's
per-story contrasts are c_C × the natural ones × (1 + ε), ε ~ N(0, τ) per story and channel; two depths, 8 statistics):
P(every statistic equivalent) at c_K / c_V = 1, 0.9, 0.85, 0.8, 0.7, 0.5 and 2:
- τ = 0.1: 1.00, 1.00, 0.65 / 0.58, 0.00, 0.00, 0.00, 0.00 (Qwen / Mistral where they differ);
- τ = 0.3: 0.90 / 0.85, 0.05 / 0.03, 0.00, 0.00, 0.00, 0.00, 0.00.
The criterion therefore accepts imbalances up to about 0.9 and rejects 0.8 and beyond; per-story edit noise of 30 % costs
10–15 % power at c = 1. The design's power check of the old rule (`power_c.py`) gave P(met) 1.00 at a 0.15 share
under-read, which the new rule rejects.

#### Seen before finalisation

No output of any study model above 0.5B parameters on E, H, TSET or THOLD under the stage-8 code has been seen. The
design and critique pilots (CPU, FP32; scripts and logs kept with the stage-8 build notes, `partC/` and
`critic_intervention/`):
- **Pilot A, Qwen2.5-1.5B-Instruct, 4 stories (seed 101), NONE and LETTER, blocks 1 / 5 / 9 / 13** (`pilot_c.py`,
  `pilotA.json`, `pilotA_summary.txt`; planned for 8 stories, stopped at 4 after about 14 min). E1 (20 held-out cores,
  seed 202) and E2 (12 neutral sentences) at α = 1: φ under NONE 1.00–1.02 and 0.97–1.02 at every block; under LETTER
  0.91–1.04 while the natural effect is ≥ 2 nats (blocks ≤ 9). ν_KV of E1 0.04 / 0.16 / 0.26 / 0.33, of E2
  0.12 / 0.32 / 0.62 / 0.70; residual cosine of E1 0.997 / 0.987 / 0.969 / 0.961 and of E2 0.968 / 0.916 / 0.839 / 0.803;
  ν_T 1.7e-6. Natural LETTER effect 7.2 / 2.2 / 0.35 nats at blocks 5 / 9 / 13. R: no flip, non-specific shift of m_SB up to
  1.5 nats. BIND-p: Φ ≤ 0.6 nats.
- **Pilot B at 1.5B**: no data (killed twice by the memory limit; the first kill came from a clamp forward outside
  no_grad, the reason for the no_grad guard).
- **Pilot B, Qwen2.5-0.5B-Instruct, 6 stories, LETTER / P1 / NONE, blocks 1 and 9** (`pilotB05.json`,
  `pilotB05_summary.txt`; about 9 min): key-only / value-only rows of each edit's own K/V from l + 1; exactness 0.0. κ
  (T / E1 / E2): LETTER onset 2 0.77 / 0.77 / 0.78; P1 onset 2 0.24 / 0.23 / 0.23; P1 onset 10 0.09 / 0.07 / 0.06; NONE
  0.01 at both onsets. LETTER at onset 10 not evaluable (0.26 nats). Story-level D ≤ 0.022 wherever the effect is ≥ 1.6
  nats (0.19 / 0.29 under LETTER at onset 10). BIND-p Φ ≤ 0.07 nats.
- **Smoke and memory runs**: `smoke.json` (Qwen2.5-1.5B, one story, NONE, block 1, the stage-A rows) and `memtest.json`
  (Qwen2.5-0.5B, one story, NONE, blocks 1 and 9, the stage-B rows; its log shows the clamp forward outside no_grad that
  the guard now prevents).
- **Tokenizer and Hub checks** (design): 138 of 400 cores of make_cores(400, Random(8101)) met the π constraints with
  single-position differences in Qwen2.5-7B, Mistral-7B, Llama-3.1-8B and Gemma-2-9B; the 12 pilot sentences were clean;
  the dictionary repository's file listing (`andyrdt_tree.json`).
- **Power of the old rule** (`power_c.py`, stage-1 rows): P(JC3 met) 1.00 under the law and under a 0.15 under-read, 0.01–
  0.04 under a 0.25 under-read, 0.00 under copy-only and key-flat.
- **The critic's κ compression** (`kappa_compress.py`, stage-1 summaries): at c_K / c_V = 2 or 0.5 the old rule passes
  (MAD 0.077–0.081, maximum 0.139–0.172), Pearson r ≈ 0.99; JC2's crossover ≥ 0.82 for c between 1/3 and 3.
- **The critic's non-lexical pilot** (`pilot_nonlex.py`, `pilot_nonlex_summary.txt`; Qwen2.5-1.5B, 2 of 4 planned stories,
  blocks 9 / 13 / 17, NONE / P1 / LETTER; about 14 min): natural ID 10.45 / 10.24 / 10.17 (NONE), 6.13 / 4.87 / 3.91 (P1),
  2.10 / 0.39 / 0.06 (LETTER). φ under NONE: E2 0.99 / 1.01 / 1.01; SYN 0.08 / 0.06 / 0.06; FR 0.39 / 0.36 / 0.34; E1
  1.01 / 1.02 / 1.02; NONLEX(E1) 0.13 / 0.12 / 0.11; flips 0 for SYN, FR and NONLEX(E1). Transfer φ(LETTER)/φ(NONE) at block
  9: E2 0.94, FR 0.05, E1 1.11, NONLEX(E1) 1.04. Residual cosines with the natural displacement: E2 0.83–0.84, SYN 0.16–0.18,
  FR 0.38–0.42, E1 0.95–0.97, NONLEX(E1) 0.46–0.48 (norm 0.46–0.48×).
- **In-hand results used for the priors and the feasibility table**: stage 1's per-story rows (above); stage 6's head
  rankings (fixed sets of J-C-READ) and its Prakash outputs (the BIND sweep, ID@28, the s_ID onset sweep, the κ anchors);
  stage 2's 24B onsets.
- **Build checks** (this part's code; tokenizers, Hub metadata, and Qwen2.5-0.5B only):
  - every E core (all formats, also π(S), π(X)) and every H core passes the single-position and shared-prefix rules in the
    Qwen2.5-7B, Mistral-7B, Llama-3.1-8B (verified mirror files in the local cache) and Yi-1.5-9B tokenizers; p = 55, 54, 77,
    57; P1 prompts 103 / 106 / 125 / 116 tokens; trie nodes of the E form set 37 / 116 / 37 / 97;
  - the 24 sentences are clean for all four form families (EN, FR, DE, SYN) in Qwen2.5-0.5B / 7B / 14B, Mistral-7B and
    Llama-3.1-8B (24 / 24 each); French and German forms are 2–4 tokens;
  - the Hub metadata of the dictionaries at c37e53c4… (the sha256 of 3 files × 4 layers, the configs, the published FVE)
    and the pickle header of layer 3's ae.pt (a 128 KB range request: the key names and shapes above);
  - Prakash et al.'s material locates in the Qwen2.5-7B and Llama-3.1-8B tokenizers (lengths 180 / 186 / 196 / 203 and
    181 / 187 / 197 / 204; state words at 154, 166 and 155, 167);
  - the in-hand numbers of the feasibility table and the power table above;
  - the T control's table floor (`bf16_nuT.py`, Qwen2.5-0.5B on the CPU, the first three E stories, depths 3 / 7 / 11 /
    15): ν_T < 1e-5 at every depth in FP32 and in BF16 (the natural tables from a 5-row prefix batch, T's from a 66-row
    batch); FP32 table differences ≤ 1.5e-5 at |K| up to 217;
  - plumbing runs of every stage at Qwen2.5-0.5B (FP32, CPU, l = 7 only), on the first two E stories, the first two H_cal
    cores, 20 TSET / 4 THOLD pairs and the first two Prakash pool pairs (all seen): no frame admitted; E5 φ on H_cal with
    α = 1 / 2: FR 0.15 / 0.09, DE −0.06 / −0.09, SYN 0.14 / 0.10, NL 0.11 / 0.20; a random dictionary (d × 1024) flips
    ≤ 0.25 at every k_F; the two E4 fits (20 pairs) flip 0.00 / 0.25 of THOLD. At l = 7 only the P1 and POST cells pass
    J-C-G4 (LETTER ID_KV 0.52 nats; NONE coverage 0.77); σ(P1, 7) +0.05, σ(POST, 7) −0.07; κ under P1 / POST: E1 +0.03 /
    −0.08, E2 +0.02 / −0.07, E3 −0.01 / +0.01, E4 (seed 101) −0.17 / +0.02; ν: E1 0.19, E2 0.32, E3 0.92, E4 0.91 / 0.78;
    residual cosines E1 0.96, E2 0.86, E3 0.37, E4 0.56 / 0.55; D under P1: E1 0.15, E2 0.06, E3 0.46, E4 0.98; O1
    predicted against observed ψ_K(P1): E1 0.97 / 0.63, E2 0.89 / 0.45, E3 0.62 / −0.08, E4 0.56 / −0.27; the π(t)
    diagnostic +0.28 [+0.07, +0.50]; J-C-G2 0.45 (not met at 0.5B); readers at a pseudo-random head set: ID_K^nat(P1, all)
    0.20 nats; greedy flips under NONE: natural 0.75, E1 0.50, E2 0.75, the others 0.00; the screen at 0.5B: BIND IIA 0.00
    at every swept block (Φ ≤ 4.6 nats), s_ID(OPTIONS-AFTER) 0.37 / 0.36 / 0.27 / −0.02 at onsets 0 / 3 / 5 / 8; J-C6's
    code at block 12 of 0.5B: CAA κ +0.01 / −0.08 (NO-MENTION / OPTIONS-AFTER), BIND κ +1.02 / +0.76, ν(CAA, ID) 0.18,
    Gate b0 ≤ 1.2e-5;
  - TEST_MODE plumbing runs at Qwen2.5-0.5B (FP32, CPU, two E stories, one depth l = 7, H_fit 3, H_cal 2, 4 sentences, a
    random dictionary). The final run, on commit 84e4f05, whose code is the finalised code: pytest 95 passed (26 min;
    J-C-G0 MET, with the decoder, cross-part population and Holm tests); release, preflight, calib, dasfit and eval in a
    first session that this environment's three-hour limit stopped during the readers step; a second session (TESTS=0)
    kept those steps and ran readers, overlap, explore, J-C6 and the score, exit 0, no step failed or skipped; the score
    read "provenance OK; population OK; J-C-G0 MET" with HEADLINE outcome (f) (plumbing). An earlier whole-script run (`TEST_MODE=1 bash scripts/gpu_stage8c.sh`, 2026-10-10, one
    session in a fresh OUT, 58 min on 2 CPU threads) ended with exit 0 and no step FAILED: release and pool hashes OK;
    pytest 70 passed, 0 skipped (24 min); preflight, calib (the calibration file's sha256 792256ad… identical to the
    earlier plumbing runs), dasfit, eval (2 stories, 8 min each), readers, overlap, explore, J-C6 and score all ran; the
    scorer: provenance OK, population OK, J-C-G0 MET, J-C-G1 MET (2 cells), J-C-G2 NOT MET (σ(LETTER, 7) − σ(NONE, 7)
    +0.45 [+0.29, +0.50]; TEST_MODE uses l = 7 for l = 3), J-C-G8 NOT MET (no λ statistic at 0.5B), J-C-G3 NOT MET (random
    dictionary), every line NOT EVALUABLE, HEADLINE outcome (f); its PREDICTIONS and later sections are identical to the
    previous whole-script run's. Calibration on H at 0.5B (seen):
    no frame admitted; E5 φ on H_cal at l = 7 with α = 1 / 2: FR 0.15 / 0.09, DE −0.06 / −0.09, SYN 0.14 / 0.10, NL
    0.11 / 0.20; the random dictionary's flip rate ≤ 0.25 at every k_F;
  - the unit tests (J-C-G0), each file alone on 2 CPU threads: `tests/test_stage8_edits.py` 14 passed in 314 s (the
    longest, the scored rows against the reference, 188 s), `tests/test_sae.py` 7 passed in 0.1 s,
    `tests/test_stage8c_score.py` 12 passed in 41 s; with the shared files of the pytest step (surface 13, clamp 5,
    head_splice 7, prakash 12), 70 passed and none skipped. The review then added one test to `tests/test_stage8_edits.py`
    (`test_T_tables_bitwise_through_the_eval_passes`) and two to `tests/test_stage8c_score.py` (the Holm family and
    verdicts; outcome (b) read only from evaluable models): 15, 7 and 14 tests, 73 in the pytest step;
    `tests/test_stage8c_score.py` alone: 14 passed in 50 s on one CPU thread. The audit of the entry against the code
    then changed the scorer and the pytest step (E3 counts in J-C-READb only when layer 7 passes J-C-G3; BIND's usability
    decides J-C6's evaluability; a line over combos or models is NOT MET on any evaluable failure; no line is computed and
    no outcome named without a passing J-C-G0; `tests/test_generate.py`, `tests/test_stage8_populations.py` and
    `tests/test_stage8_holm.py` added to the step and to J-C-G0) and added five tests to `tests/test_stage8c_score.py`
    (each fails on the code before the change): 19 passed in 50 s on 2 CPU threads; the three shared files 17 passed in
    25 s; 95 tests in the pytest step. Re-scoring the TEST_MODE archive above with the changed scorer (`--test`) exits 0
    with provenance OK and population OK; its pytest log is the 73-test step from before the audit, so J-C-G0 is NOT MET
    there (the three shared files have no test in it), every line is NOT EVALUABLE as before and the HEADLINE reads
    "J-C-G0 not passed: no outcome".
- **Review checks** (CPU; Qwen2.5-0.5B FP32 / BF16 and the study tokenizers; no study model above 0.5B run):
  - the BF16 floor of the tables: the natural tables of the first three E stories in one BF16 prefix pass against the
    same pass in FP32 differ by a pooled ν of 0.020 / 0.019 / 0.017 / 0.015 at l = 3 / 7 / 11 / 15 (the natural
    displacement is 0.20 / 0.19 / 0.60 / 0.69 of |[K;V]_B| there), so a T computed in a forward of another shape would sit
    at J-C-G1's bound ν_T ≤ 0.02; in FP32 one 66-row edit forward differs from the 5-row natural pass by up to 7.7e-6 and
    five-row forwards reproduce it bitwise, hence the edit passes in the natural pass's shape (test
    `test_T_tables_bitwise_through_the_eval_passes`);
  - the synthetic rows of J-C-G8 in the TEST run's eval file (0.5B, l = 7, two stories): ψ_K(SK50) 0.08 / 0.41 and
    ψ_V(SV50) 0.40 / 0.47 in P1 / POST (a half-interpolated key is read sub-linearly);
  - the feasibility refinement above (stage 1's Mistral-7B margins, stage 6's reader ranking);
  - the 24 sentences clean for EN, FR, DE and SYN (24 / 24 each) and the DAS forms single-token in the Mistral-7B,
    Llama-3.1-8B, Qwen2.5-7B and Yi-1.5-9B tokenizers; the E population's 136 draws and 24 clusters.

#### Compute (A100-80GB, BF16)

From stage 3b's 24 s per 150-item arm at 16 rows per item (about 0.010 s per 120-token row at 7B) scaled by the prompt
plus trie length (about 140 tokens at Qwen2.5-7B and Llama-3.1-8B, 220 at Mistral-7B):

| Step | Minutes |
|---|---|
| release, pytest (J-C-G0, CPU; 95 tests: 24–34 min on 2 shared CPU threads in the build environment for the 73 of the build, about 1 min more for the 22 added in the audit) | 20 |
| qwen7: fetch, preflight, calib (incl. 4 × 3.76 GB dictionaries), dasfit (8 fits × 1,000 steps), eval (80 × 1,976 rows), readers | 80 |
| mistral7: the same without E3 (eval 80 × 1,768 rows at 220 tokens) | 62 |
| llama8: no dasfit, no readers (eval 80 × 1,352 rows) | 42 |
| overlap screens (qwen7, llama8) | 2 × 15 |
| jc6 (qwen14: filter, 3,910 mean passes, 150 × 3 formats × 3 arms) | 25 |
| explore (three models) | 3 × 8 |
| score, manifest, archive | 5 |
| core total (release, pytest, the three models, score) | 209 (about 3.5 h) |
| with the screens, J-C6 and the exploratory steps | 288 (about 4.8 h) |

The default deadline is DEADLINE_H = 5.0 h, which holds every step in one session with about 12 minutes to spare. A
model's reserve is the core minutes of the models after it (80 / 62 / 42 for Qwen2.5-7B / Mistral-7B / Llama-3.1-8B, 48
for Yi-1.5-9B) plus 10. Every eval step reserves that plus 8 for its own readers step at Qwen2.5-7B and Mistral-7B, and
stops between stories when the time left minus its reserve falls below 1.2 times its running time per story (exit 3;
the next session resumes from the stories done). An overlap screen runs only if 15 minutes plus the model's reserve
remain, an exploratory step only if 8 + 25 minutes (its own and J-C6's) plus the model's reserve remain, J-C6 only if 25
minutes remain; so a slower host loses the exploratory steps first, then J-C6, then the overlap screens, and a second
session (same OUT) runs what was dropped. Disk: ≥ 80 GB (two models and the four
dictionaries). At about $2 per A100-hour, $7–10.

#### Commands

```bash
J=$(git log --format=%H -1 --grep='^Finalise preregistration J$') && [ -n "$J" ] && git checkout "$J"
bash scripts/gpu_stage8c.sh                    # HF_TOKEN optional; PRAKASH_REPO=<checkout of the release at 0579347> optional
TEST_MODE=1 bash scripts/gpu_stage8c.sh        # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2, l = 7): every step of qwen7,
                                               # the screen and J-C6 (TEST_ALL=1: mistral7 and llama8 too)
python analysis/stage8c_score.py --results results/gpu_stage8c    # re-score an archive
```

### Part D. What the reader heads write, the sign of the key read, and the 1.5B / 3B route

**Code.** `ckeys/flag.py` (the o_proj hooks OCap and Inject, the head decomposition, the flag algebra and its controls),
`ckeys/questions.py` (the question arms Q_IN and Q_OUT, registered with `ckeys.encoding.register_arm`; the chat-wrapped
in-sentence IOI arm INLINE_CHAT; the IOI row layout), `experiments/stage8_flag.py` (stages preflight, sets, fit, inject,
ablate, bind, sign, diss, before, xtask), `analysis/stage8d_score.py` with `analysis/stage8d_parts/` (`stats.py`,
`lines.py`), `scripts/gpu_stage8d.sh` (it sources `scripts/stage8_common.sh`). Tests: `tests/test_flag.py` and
`tests/test_questions.py` (Gate J-D-G0), `tests/test_stage8d_score.py` (the scorer on synthetic inputs). Shared code used
unchanged: `ckeys/headsplice.py` (HeadSplice, HopSplice), `ckeys/clamp.py`, `ckeys/interventions.py`, `ckeys/encoding.py`,
`ckeys/story.py`, `ckeys/ioi.py`, `ckeys/knockout.py`, `ckeys/surface.py`, `ckeys/tasks.py`,
`experiments/format_factorial.py` (`run_item`), `experiments/ioi_factorial.py` (`run_item`), `experiments/stage6_heads.py`
(`Stage6.base_runs`, `configure_hop`), `experiments/row_restricted_keys.py` (`run`, RowSplice). No shared module was
changed.

**Purpose.** Part D answers the novelty objection ("QK/OV plus duplicate-token heads already imply it") and the sign
questions of the reviews.
- *What the readers write.* Hop 1 is a duplicate-token match: a later copy of a candidate word attends to the writing token
  p. The account under test is that the matching heads H* write a flag into the matched word's own row, one vector per
  layer, the same in every story, and identity-free (an address, not content). The answer then finds the flagged row by
  key (hop 2) and outputs that row's own token. Part D tests sufficiency (J-D1), specificity against structured controls
  (J-D2), necessity (J-D3), the hop-2 route (J-D4), identity-freeness (J-D-ADDR, J-D-KN) and whether the flag of an
  initial-state sentence carries its binding (J-D5).
- *The circularity of the a3 ranking.* H* was ranked by the attention change under the same key clamp. The injection and
  ablation tests use no key clamp at evaluation, and the flag is fit on the ranking set R and evaluated on fresh stories.
- *The sign of the key read.* The critique showed that under the "not mentioned" question Q_OUT the value read flips sign
  too (1.5B pilot: ID_K −5.13, ID_V −8.24), so Q_OUT shows task semantics, not a polarity of the readers. The sign block
  is therefore built around channel dissociation: in Q_OUT both channels are negative (J-D-SIGN-Q); in IOI's in-sentence
  re-mention (INLINE) the key read is negative while the value read is positive (J-D-SIGN-IOI). The same frozen H* must
  carry the negative IOI read (J-D8-IOI); the negative read must be applied at the answer row's key read of the listed
  names (J-D-ROUTE-IOI); and the hop-2 heads that carry the flag's effect in belief lists must also carry it in Q_OUT and
  IOI (J-D-HOP2).
- *The 1.5B / 3B dissociation* ("attention present without the read"). J-D6 tests whether the flag written at the
  sentence rows mediates the natural ratio of the sentence read to the list read at 1.5B, 3B and 7B; J-D6-ROUTE tests the
  consequence of stage 5's G4 (at 1.5B hop-2 attention does not follow the key, d(ans → r_S) +0.015): a negative effect of
  the injected flag must then be read by value.

**What Part D does not claim (D-1, replacing the design's headline).** "Re-mentions of the clamped candidates after p
open a large key read whose sign is set downstream. Without them, smaller reads of either sign occur (IOI BEFORE −2.0 /
−3.8 nats at Qwen2.5-7B / 14B; LIST-BEFORE ≤ 0 in 9 of 10 models, about −2.0 at 72B). Part D does not explain these." The
design's sentence "existence is decided by later mentions; sign by the reader" and its claim that LIST-BEFORE is about 0
at 7B are withdrawn. IOI BEFORE at Qwen2.5-7B is tested exploratorily with a stated two-sided rule (Exploratory, E-BEFORE).

#### Populations

Belief cores are compared by their full tuple (agent, other, object, distractor, initial, distractor_location, base,
source); U is the common part's union of make_cores(1000, Random(s)), s = 0–3 (sha256 abd1f053…). EXCL is the union of
make_cores(1000, Random(s)) for the pilot seeds 7, 8, 9, 99, 101, 202, 20261011 and Part B's seeds 20261013 and 20261014,
and of make_cores(3000, Random(s)) for Part C's seeds 8101–8104. Each is a superset of every population drawn from that
seed, so a core outside EXCL is outside every other stage-8 population. A stream is make_cores(1, rng) drawn repeatedly
from one Random(seed) (the same cores as one make_cores(n, rng) call); a stream population takes the first cores that are
not excluded, not repeated, and satisfy its condition. Hashes are sha256(json.dumps([[*tuple] for each core in order])),
with the belief tuple above, the IOI tuple (template, pattern, place, object, io_b, io_s, io_x, subj, order) and the task
tuple (agent, object, distractor, initial, distractor_location, base, source), pinned in
`experiments/stage8_flag.POP_SHA` and asserted by the preflight and the tests.

| Population | Rule | Size | sha256 |
|---|---|---|---|
| R (fit) | make_cores(60, Random(0)): stage 6's ranking set. In U by design: the flags are fit in sample, where H* was ranked | 60 | 9036af1a838a12d58a7a7eb40f70dd800dd659bfd6e95ed1ab5a2ce10862f936 |
| R' | the cores of R with distractor_location ≠ initial (the initial-state flags) | 45 | fe348ba40a19182a67cee382a3533c1af009273acb740c01d418662a570122a2 |
| E8 (evaluation) | Random(81) stream, not in U ∪ EXCL | 100 (110 draws) | 2c198705c595d8068467a188d46e5c22793e079f837ba39bc1e0b43fb14b0f13 |
| A (ablation) | the first 60 cores of E8; Qwen2.5-3B's dissociation uses the same 60 | 60 | (prefix of E8) |
| BIND (J-D5) | Random(84) stream with distractor_location ≠ initial, not in U ∪ EXCL ∪ E8 | 100 (140 draws; 43 object-first, 57 distractor-first) | 495c14620cd22e38af0ac67c376441442c40264996db98872d2bae4009a7ee92 |
| F_ioi candidates | `ckeys.ioi.make_cores(90, Random(82))`; F_ioi = the first 60 valid in INLINE for the model's tokenizer | 90 → 60 | d41e791a42cc3a833b32220c54f6aa5b129bfda79e904fb8bf5c382ebf0f8b9a |
| E_ioi candidates | `ckeys.ioi.make_cores(150, Random(83))`; E_ioi = the first 100 valid in every IOI arm the model runs (INLINE, INLINE_CHAT; Qwen2.5-7B also AFTER, BEFORE) | 150 → 100 | 2b165365fc60e69af9bbe0d943bb9ea7fc30ec65a2747cd6f04630e58ddd0334 |
| XFIT paint / schedule (exploratory) | Task.make_cores(1, rng) stream of Random(85) not in Task.make_cores(1000, Random(s)), s = 0–3 | 60 each | 9b9a320f… / 4a56ae02… |
| XEVAL paint / schedule (exploratory) | the same with Random(86), also not in XFIT | 40 each | be234aaf… / 8b4c7c05… |

- X = `story.pick_x(core)` (absent from the story and ≠ B, S). The K_N event word of a core is
  N_WORDS[int(h[:8], 16) mod 6], h the hex sha256 of json.dumps(core, sort_keys=True), N_WORDS = (garage, kitchen,
  pocket, desk, hallway, porch): one token after a
  space in the Qwen2.5 and Mistral-7B-v0.3 tokenizers, not a candidate (checked). I' = the location absent from
  {init, dloc, B, S, X} (exactly one exists when dloc ≠ init). The order stratum of a core is "object-first" when the
  queried object's name sorts before the distractor's (`ckeys/story.py` sorts the two initial-state sentences by object
  name), else "distractor-first".
- Validity (IOI): `encode_runs` not None (B, S, X differ only at p), `check_occurrences`, the four listed names found in the
  row groups of `ckeys.ioi.RowTask`. With the Qwen2.5 and the Mistral-7B-v0.3 tokenizers no candidate is skipped (F_ioi =
  the first 60, E_ioi = the first 100); the per-model indices are recorded by the preflight and every IOI file.
- Disjointness (asserted): E8 and BIND are disjoint from U, EXCL and each other, and from the critics' pilot cores
  (Random(0) indices 16–46, inside U; Random(7); Random(9)); the IOI candidate lists are disjoint from each other and from
  `ckeys.ioi.make_cores(1000, Random(s))` for s = 0, 1, 5, 6 (stage 5 and the pilots); the task lists from each other and
  from stage 3's seeds. The test also checks E8 and BIND against Part B's F, C and Part C's E, H, TSET, THOLD themselves
  when their modules are importable. No pilot drew from seeds 81–86.
- In TEST_MODE every population is cut to its first 2 cores.

#### Models and sourcing

- **Qwen2.5-7B-Instruct (`qwen7`) and Mistral-7B-Instruct-v0.3 (`mistral7`)**: every line except the dissociation lines
  J-D6a, J-D6 and J-D6-ROUTE. **Qwen2.5-1.5B-Instruct (`qwen1.5`), Qwen2.5-3B-Instruct (`qwen3b`) and Qwen2.5-7B**: J-D6a
  and J-D6 (Mistral-7B runs no dissociation step). J-D6-ROUTE is scored at Qwen2.5-1.5B only. Files and revisions as in
  the common part (`scripts/stage8_models.json`, `scripts/fetch_verified.py`; no token needed). Part D uses no fresh
  family, so the fallback rule G2 does not apply: a model whose files are refused leaves its lines NOT EVALUABLE.
- BF16, transformers 5.18.0, `use_cache=False` in every pass, sdpa attention except the ranking passes of `sets` (eager).
- **The reader set H\*.** At 7B, frozen from `results/gpu_stage6/heads/<model>.json` (committed): `arms.P1.rankings.a3[:k*]`
  with k* = 40 (Qwen2.5-7B, 18 layers) and 52 (Mistral-7B, 19 layers); the random sets are the first k* heads of the three
  stored permutations; the active-at-G set is `sets.active_kstar`. The file sha256 (ed828a9b… / 88ababd9…), the revision
  and k* recorded there, and the canonical hash of {H, rand, active, k*} (dea841d0… / 55a1b17b…) are asserted. At 1.5B and
  3B, H* is ranked here with stage 6's code: `Stage6.base_runs` phase 1 (eager) on R under OPTIONS-AFTER (P1), a3 averaged
  over R, k* = ceil(0.05 × the number of heads, layers × heads per layer) = 17 (1.5B, 28 × 12) and 29 (3B, 36 × 16), the
  random sets from numpy default_rng(2) as stage 6, the
  active-at-G set as stage 6 (top k* by mean o_proj-input norm at G outside the top 2k* by a3). L* = the layers holding an
  H* head.
- **The hop-2 ranking** (J-D-HOP2, both 7B models; eager, on R under P1): hop(l, h) = ½[(A^{K_S} − A^B)[END, r_S] +
  (A^B − A^{K_S})[END, r_B]], the attention change from the answer position to the option rows of S and B under the key
  clamp K_S, averaged over R; the top 10 heads; a control set of 10 random heads (numpy default_rng(15)). The same passes
  give an in-run a3 whose top-k* overlap with the stored H* is reported.

#### Prompts, rows and the flag

- **Formats.** OPTIONS-AFTER (P1) and SENTENCE-AFTER (POST) as in the paper (`ckeys.encoding.raw_prompt`); Q_IN and Q_OUT
  are P1 with the question line replaced by "Which of the choices is mentioned in the story?" / "… is not mentioned in the
  story?"; every belief prompt is chat-wrapped (`chat_text`, the "Answer:" prefill). IOI (`ckeys/ioi.py`): INLINE (raw;
  the four names as a spaced parenthetical after the IO mention), INLINE_CHAT (the same sentence inside AFTER's chat
  wrapper and instruction: HEAD + "\n\nSentence: " + inline sentence + "\n" + INSTR + "\nAnswer:"), AFTER and BEFORE
  (chat). BOS iff the tokenizer has one.
- **Rows.** p = the writing token (the one position where the B and S runs differ). G = the six option rows (the location
  words of the "Choices:" list, or of the room sentence under POST), in canonical order, all after p; r_w = the row of word
  w. IOI: the four listed names (`RowTask.groups(...)["options"]`), r_{IO_B}, r_{IO_S}, r_{IO_X} named by token.
- **Clamps.** The natural key clamp K_W (the stage-6 clamp): the pre-RoPE k_proj output at p, every layer, set to that of
  the run in which W is written. Used for every flag fit. In the scored clamp rows (K_S, K_X, K_N, V_S, V_X), a key row
  holds p's value at the base run's and a value row holds p's key at the base run's, every layer (as
  `format_factorial.run_item`); Gate J-D-G0 checks these rows against `run_item`. The HeadSplice rows (transfer and
  gate batches) swap only the key, as in stage 6, so their all-heads all-rows row equals the key-only natural clamp.
- **The flag.** With z^B and z^K the o_proj inputs (concatenated head outputs) of the clean base run and of the run under
  K_S, and W_O^{l,h} the o_proj columns of head h (no bias in either model):
  δ_l(s) = ½ Σ_{h∈H*_l} W_O^{l,h} ([z^K_h(r_S) − z^B_h(r_S)] + [z^B_h(r_B) − z^K_h(r_B)]), Δ_l = mean_{s∈R} δ_l(s), l ∈ L*.
  The fit batch per story is [B, B + K_S, S] (one forward). The same formula gives: Δ^KV (the natural S run instead of
  K_S); the leave-one-word-out flags Δ^{−w} (the stories of R with B ≠ w and S ≠ w; 36–46 stories each); Δ^act (the
  active-at-G heads instead of H*, at their own layers, rescaled by one factor to the total squared norm of Δ); Δ^POST
  (POST rows on R); at 7B only, on R', Δ^init (the key at p_init from the run whose initial location is I'; rows r_I' and
  r_init) and Δ^dloc (the key at p_dloc from the run whose distractor location is I'; rows r_I' and r_dloc), and on F_ioi
  (INLINE) Δ^IOI (rows r_{IO_S}, r_{IO_B}). Δ, every other flag and every control (with the ablation directions and their
  means μ) are saved in `fit/<tag>.pt`, whose sha256 every later file that uses them records; the per-story flags are not
  saved (their summaries are in `fit/<tag>.json`).
- **Injection** I(spec): add the listed vectors to the o_proj output y_l at the listed rows, every l of the vector's
  layers, in the clean base run, with no key clamp; computed in FP32 inside the hook and cast back. "move(v, w)" = +v at
  r_w and −v at r_B.
- **Directional ablation** A(u, μ) at rows Rset: y_l[r] ← y_l[r] − (⟨y_l[r], u_l⟩ − μ_l) u_l for l ∈ L*, r ∈ Rset, in FP32;
  μ_l = the mean over R and the absent-word option rows (words not in {B, init, dloc}) of ⟨y_l[r], u_l⟩ in the clean run.
  Applied in every pass of `format_factorial.run_item` (its three clean passes and its 13-row batch), as stage 6's H3.
- **Controls.** Isotropic directions (3 draws, torch seed 11) and head-span vectors Σ_{h∈H*_l} W_O^{l,h} z, z ~ N(0, I)
  (3 draws, seed 12), each norm-matched to |Δ_l|; the layer-permuted flag (a fixed derangement of L*, random.Random(13),
  rescaled); the story's own write δ(s) with its component along Δ̂_l removed, norm-matched (orth); the mean clean-run H*
  output at the six option rows over R, as a unit direction, norm-matched (meanH); the active-set flag Δ^act; the top
  principal direction of the clean attention output y_l at the option rows over R (pc1); three isotropic unit directions
  for the ablation (seed 14).

**Batches** (the rows each stage stores; candidate log-probabilities over the full vocabulary, at the last position):
- *inject* (E8, P1; one forward of 29 rows): none, none2 (Gate J-D-G2), K_S, K_X, K_N (the key of the run whose event word
  is the core's N word), add 0.5Δ / 1Δ / 2Δ at r_X, move(Δ, X), move(Δ, S), −Δ at r_B, move(Δ^{−S}, S), move(Δ^{−X}, X),
  move(Δ^KV, S), move(Δ^KV, X), move with iso0–2, head-span 0–2, the layer-permuted flag, orth, meanH, Δ^act and the
  story's own write, +Δ at the first row of the "Choices:" line and at the "Question" row, +Δ^POST at r_X. Then a capture
  pass [none, move(Δ, X)] of the K and V at G and the hop-2 route batch (HopSplice over G, two passes in every layer)
  [none, move, move + ans_K, move + ans_V, move + ans_KV, move + other_KV]: in ans_C only the answer row reads the clean
  run's C at G; in other_KV every row but the answer reads the clean K and V at G.
- *ablate* (A, P1): `run_item` under no ablation, A(Δ̂), A(pc1), A(meanH), A(iso0–2) at the six option rows; exploratory
  (skipped when the deadline has passed before the step starts): A(Δ̂) at r_init and r_dloc only, and at r_B only.
- *bind* (BIND, P1, queries direct (answer B), other_agent (answer init), irrelevant_object (answer dloc)): [none,
  +ev, +init, +dloc, +iso, K_X], with ev = Δ^P1, init = Δ^init, dloc = Δ^dloc and iso = iso0 each rescaled per layer to the
  mean of the three flags' norms, added at r_X.
- *sign*, E8 in Q_IN and Q_OUT: `format_factorial.run_item` (ID_K, ID_V, the clean argmax and mass); the transfer batch
  (HeadSplice, 24 rows: for K_S and for K_X, the sets ∅, H*, rand0–2 at G; every head at G (allG); every head but H* at G
  (koH) and but rand0–2 (korand0–2); every head at the rows of B, S and X only (allGc); every head at every row (allT));
  [none, none2, K_S, K_X, V_S, V_X, +Δ at r_X, iso0–2 at r_X]; and the hop-2 batch [none, inj, inj + top-10, inj + rand-10,
  inj + every head], inj = +Δ at r_X, in which the listed heads at the answer row read the clean run's keys at G
  (HeadSplice with the span G). P1 on E8: the hop-2 batch. E_ioi in INLINE: `ioi_factorial.run_item` with
  `ckeys.ioi.identity_measures` (ID_K, ID_V; Gate e); the transfer batch at the listed names (Gc = the rows of IO_B, IO_S,
  IO_X); [none, none2, K_S, K_X, +Δ^IOI at r_{IO_X}, move(Δ^IOI, IO_X), +Δ^P1 at r_{IO_X}, iso]; the IOI route batch
  (HopSplice over the listed names; capture pass [none, K_S, inj]) [none, K_S, K_S + ans_K, K_S + ans_V, K_S + ans_KV,
  K_S + other_KV, inj, inj + ans_K, inj + ans_V, inj + ans_KV, inj + other_KV], inj = +Δ^IOI at r_{IO_S} and −Δ^IOI at
  r_{IO_B} (as K_S writes it); the hop-2 batch with inj = +Δ^IOI at r_{IO_X}. INLINE_CHAT (both models) and AFTER (Qwen2.5-7B
  only; Mistral-7B failed stage 5's Gate e there): `run_item` and the transfer batch.
- *diss* (qwen1.5, qwen3b, qwen7; E8, n = 100, 60 at 3B; P1 and POST): per format a pass [B, S, X] (case-marginalised
  clean scores, keys and values at p), the write pass [B, B + K_S] (δ^f(s) at r_S, r_B), the batch [none, none2,
  +Δ^P1 at r_X, +Δ^POST at r_X, iso0 at r_X, K_X, K_S]; under POST a capture pass [none, inj] of the K and V at G and the
  route batch [none, inj, inj + ans_K, inj + ans_V, inj + ans_KV], inj = +Δ^P1 at r_X (HopSplice; the answer row and its
  trie nodes read pass B); at 1.5B and 3B under P1 the gate batch (HeadSplice [∅, H*, allG] × {K_S, K_X}). Scores: the
  case-marginalised log-probability of each location w, log Σ_f p(f) over the forms " w", " W", "w", "W"
  (`ckeys.surface.FormSet` with frames " " and "", exact for multi-token forms through the trie; 5 nodes in the Qwen
  tokenizer).

#### Measures

Six-way renormalised log-probabilities ℓ_w = log p(w) − log Σ_{c∈C} p(c) over the six candidates (lower-case list forms in
the list formats; the case-marginalised scores in *diss*); four-way over IO_B, IO_S, IO_X and the subject in IOI.
Δℓ_w(row) = ℓ_w(row) − ℓ_w(none) in the same batch; c = ℓ_S − ℓ_X (the normalisation cancels).
- N_X = mean Δℓ_X(K_X); ι(row) = mean Δℓ_X(row) / N_X; π_X(row) = the fraction of stories whose argmax over C is X.
- ID_K (in-batch) = ½[c(K_S) − c(K_X)]; ID_inj(v) = ½[c(move(v, S)) − c(move(v, X))]; ID_inj^LOO uses Δ^{−S} in the
  S row and Δ^{−X} in the X row.
- r_ans^inj(C) = 1 − mean Δℓ_X(move + ans_C) / mean Δℓ_X(move), C ∈ {K, V, KV}; r_other likewise.
- ρ_K(u) = mean ID_K(A(u)) / mean ID_K(none), ID_K as in the paper (`run_item`: full-vocabulary log-probs of the lower-case
  candidates, every layer from 0, the other channel held at the base run's).
- J-D-KN: per story the Pearson correlation over the six candidates of Δℓ(K_N) with Δℓ(−Δ at r_B), averaged over stories
  (a story where either vector is constant has no correlation and counts as 0; the scorer prints how many); the B-loss
  ratio β = mean Δℓ_B(K_N) / mean Δℓ_B(−Δ at r_B); the specificity mean[Δℓ_S(K_N) − Δℓ_X(K_N)].
- J-D-ADDR: the norm-weighted mean over L* of cos(Δ^K_l, Δ^KV_l), weights |Δ^K_l| (Δ^K = Δ^P1); ID_inj(Δ^KV) / ID_inj(Δ^K).
- J-D5: Ψ_bind(s) = [Δℓ_X(init | other_agent) − Δℓ_X(init | irrelevant_object)] − [Δℓ_X(dloc | other_agent) −
  Δℓ_X(dloc | irrelevant_object)], each Δ within its query's batch; pooled mean, stratum means, stratum difference
  (object-first minus distractor-first). Secondary (role confounded with order): Ψ_ev(s) = [Δℓ_X(ev | direct) −
  Δℓ_X(ev | other_agent)] − [Δℓ_X(init | direct) − Δℓ_X(init | other_agent)].
- Transfer (Q and IOI cells): id(Set) = ½[(c(Set; K_S) − c(∅; K_S)) − (c(Set; K_X) − c(∅; K_X))], c = lp(S) − lp(X);
  d_G = mean id(allG); R(H*) = mean id(H*) / d_G; KO(H*) = 1 − mean id(koH) / d_G; R_rand and KO_rand = the means of the
  three random sets' values; ID_K^T = mean id(allT) (the full key clamp, key only, in batch); d_Gc / ID_K^T reported (a
  cell counts as a re-mention read only if ≥ 0.5). Signed ratios of means, valid for negative reads.
- IOI route: m = ℓ_{IO_S} − ℓ_{IO_B}; r(C) = 1 − mean[m(K_S + ans_C) − m(none)] / mean[m(K_S) − m(none)]; the injected-
  flag rows likewise.
- Hop-2 carry: carry(Set) = 1 − mean Δℓ_X(inj + Set) / mean Δℓ_X(inj), for Set = top-10, rand-10, every head.
- D6, on the competent cores (case-marginalised argmax = B in the clean B run and = S in the clean S run, in both P1 and
  POST): ρ = mean Δℓ_X(POST; +Δ^P1 at the sentence row of X) / mean Δℓ_X(P1; +Δ^P1 at the list row of X); ρ_nat = mean
  N_X(POST) / mean N_X(P1), both in-run on the same cores and scorer. ω = mean_s Σ_l ⟨δ^POST_l(s), Δ̂^P1_l⟩ / mean_s Σ_l
  ⟨δ^P1_l(s), Δ̂^P1_l⟩ on every E8 core of the run; κ = the norm-weighted mean over L* of cos(Δ^POST_l, Δ^P1_l), weights
  |Δ^P1_l| (from the fit on R). D6b (reported, competent cores): mean Δℓ_X(P1; +Δ^POST at r_X) / mean Δℓ_X(P1; +Δ^P1 at
  r_X). Route at POST (competent cores): r^inj(C) = 1 − mean Δℓ_X(inj + ans_C) / mean Δℓ_X(inj), each Δ against the route
  batch's none row.

#### Statistics

- **Bootstrap.** Stories (cores) resampled with replacement, 10,000 resamples, numpy default_rng(20261010); one index set per
  population size, shared by every row, arm and statistic computed on those stories, so contrasts between rows (and between
  Q_IN and Q_OUT, which use the same E8 cores in the same order) are paired. Every ratio of means is recomputed in every
  resample; strata means use the resampled stories of each stratum (a resample with none of a stratum is dropped).
- **Interval criteria** are one-sided tests of named nulls at 2.5 %: "H0: θ ≤ t, rejected when the lower bound of the 95 %
  interval is > t" and the mirror for upper bounds; "CI inside (a, b)" is two such tests (equivalence). Point floors are
  effect-size conditions. A line is met in a model when every test rejects and every point condition holds.
- **Combination.** Lines over Qwen2.5-7B and Mistral-7B ("2/2"): MET if met in both; NOT MET if not met in at least one
  evaluable model, also when that model is the only evaluable one; NOT EVALUABLE otherwise (in particular with one
  evaluable model that meets it). Lines over 1.5B, 3B and 7B: MET if met in every evaluable model and at least two are
  evaluable; NOT MET if not met in one evaluable model, however many are evaluable; NOT EVALUABLE otherwise. J-D6-ROUTE is
  a single-model line (1.5B). No interval is corrected for these combinations (intersection-union). A line whose
  computation raises an error in the scorer is NOT EVALUABLE in that model, with the traceback printed under the line;
  the error is recorded as "<code>/<model>", the SUMMARY prints "SCORER ERROR in [...]" and the scorer exits with status
  1 (a failed score step, in TEST_MODE too). Part D defines no MET IN PART.
- **Summary** (common part, G4): the 17 account lines are tallied by class (L 2, M 5, R 10), each class with its MET,
  NOT MET and NOT EVALUABLE counts, the observed met count against the sum of the priors of the lines with a verdict, and
  the Brier score; then the met rate among the R lines with a verdict.
- **Holm** (sensitivity, no verdict uses it; common part, G3, identical in every part): the shared helper
  `analysis/stage8_holm.py` runs Holm's step-down at familywise one-sided α = 0.025 over the interval components of the
  R-class account lines of Part D that have a verdict (the family named by D-11), taking each line's components from the
  models where it is evaluable. A component is one one-sided test {est, se, bound, direction}: the point estimate, the
  bootstrap standard error (standard deviation of the defined resamples), the null's bound, and p = Φ(−(est − bound)/se)
  for H1: θ > bound or Φ((est − bound)/se) for H1: θ < bound (an equivalence is two components). Every test of a line is
  a component whatever its point conditions, so the family does not depend on them. A component whose estimate or se is
  not finite is left out of the family (the scorer prints how many). The scorer prints, per R line, the components whose
  decision changes under Holm and the verdict with Holm's decisions in place of the interval decisions: a MET line with a
  component no longer rejected becomes NOT MET; a component that Holm rejects and the interval rule does not is listed
  and changes no verdict (a NOT MET line stays NOT MET).
- **TEST_MODE** outputs (tag TEST_, scorer option `--test`) are scored with the size checks, the revision, dtype,
  attention and stage-6 sets checks and J-D-G6's count of 40 competent cores waived, and never give exit status 2 (a
  scorer error still gives status 1); their verdicts are plumbing checks.

#### Gates

- **J-D-G0, exactness** (FP32, CPU, Qwen2.5-0.5B, before any model; 1e-4 in the log-probabilities unless stated).
  `tests/test_flag.py` (13 tests), `tests/test_questions.py` (11), `tests/test_stage8d_score.py` (18), run in one pytest
  step with the shared `tests/test_head_splice.py` (7; the HeadSplice / HopSplice the part's hooks compose with),
  `tests/test_stage8_populations.py` (5; rule G6 across the parts) and `tests/test_stage8_holm.py` (7; the shared Holm
  helper). The scorer reads the last pytest run in `logs/pytest.log`: the gate is met when the three part-D files and the
  two shared population and Holm files have at least 13, 11, 18, 5 and 7 PASSED tests, and no test of the run (every
  tests/ file, `tests/test_head_splice.py` included) FAILED, ERROR or SKIPPED. A skip is counted from the line pytest
  prints for each skipped test and from its summary, which also lists a whole file skipped at import; a log whose last
  run lacks the population or Holm file fails the gate. With no log the gate is NOT EVALUABLE. Each check compares a
  hooked row with an independent reference:
  1. a zero injection equals the clean run (1e-5);
  2. an injection at layer l changes that layer's attention output by exactly the added vector at the row and nothing
     else (1e-5), and leaves the final hidden states before the row unchanged;
  3. a mixed batch (none, add, move, a key row, a value row with an add) equals each row run alone;
  4. directional ablation with μ = the row's own projection equals clean, and after ablation with another μ the
     projection equals μ while unmasked rows are untouched;
  5. Σ_h head_out(z_h) from OCap's capture equals the o_proj output captured by a separate hook;
  6. nonzero-add composition (D-11): Inject(v) with a no-op HopSplice (both passes read the injected run's own K/V at G),
     with an empty-set HeadSplice, with HeadSplice "ablate" whose μ is its own o_proj input (three o_proj calls per masked
     layer), and with a no-op all-heads HeadSplice splice, each equals Inject(v) alone (an add applied twice would differ);
  7. the transfer batch's allT rows equal a separately constructed full key clamp, and its ∅ rows the clean run;
  8. the scored batches' K_S, K_X and V_S rows equal `format_factorial.run_item`'s rows;
  9. the INLINE_CHAT item code equals `ioi_factorial.run_item` on INLINE (1e-5), and an INLINE_CHAT item has floors < 1e-3;
  10. the case-marginalised trie score with a no-op HopSplice equals the plain trie score, and equals the plain sum over
      forms where every form is one token;
  11. the flag from the run's write pass equals a flag computed from an independent o_proj pre-hook capture and a
      separately constructed clamp;
  12. the per-head hop-2 batch (J-D-HOP2): with every head listed, the answer row reading the clean run's keys at G
      through HeadSplice equals HopSplice's ans_K row under the same injection (an independent two-pass path), its inj
      row equals the injection alone and its none row the clean run;
  13. the route batch of J-D6-ROUTE (an injected POST run whose answer row reads the clean run's K and V at G, HopSplice
      mask extended over the trie nodes) scored through the trie equals plain causal forwards of prompt + each form in
      which the answer row and every form position read the clean K and V (an independent reference);
  and, tokenizer only (`tests/test_questions.py`, the tokenizers at the revisions pinned in `scripts/stage8_models.json`:
  Qwen2.5-0.5B's, byte-identical to the 1.5B, 3B and 7B tokenizers at their pins, and Mistral-7B-v0.3's), on the first 50
  cores of E8 and BIND, on every core of R' (all 45), and on the first 50 IOI candidates: B, S, X and N differ only at p
  in P1, POST, Q_IN and Q_OUT; the six option rows lie after p; the BIND prompts are valid under the three queries; the
  initial-state runs differ only at p_init / p_dloc; the IOI runs of INLINE, INLINE_CHAT, AFTER and BEFORE differ only at
  p with the listed names after p (before p under BEFORE), valid in at least 80 % of the 200 candidate-arm pairs; the
  question arms' text; the populations, their pins and disjointness; the K_N words; the case-marginalised forms; the task
  prompts. The scorer's tests: bootstrap determinism and ratio recomputation, the one-sided tests and their Holm
  components, the class of every line against its prior (G4), the Holm family handed to `analysis/stage8_holm.py` (R
  lines with a verdict, evaluable models only, every component whatever the point conditions) and the shared helper
  itself on this part's family, the combination rule, every line's statistic on rows built to a known value, the D6
  decision table, an end-to-end scoring with MET, NOT MET and NOT EVALUABLE lines, a failed G0 making every line NOT
  EVALUABLE, an exception in one line's per-model scoring giving that model NOT EVALUABLE, "SCORER ERROR" and exit
  status 1, the J-D-G0 log rule (a skip in any file of the last run, a whole file skipped at import, the two shared files
  required and listed in the script's pytest step), and the scorer's pinned hashes equal to those of
  `experiments/stage8_flag.py`. J-D-G0 not met (failed or not run) makes every line NOT EVALUABLE.
- **Per-model checks** (the scorer's PROVENANCE and POPULATION sections): one commit over every file; the manifest's
  revision; BF16; sdpa (eager for sets); the sets hash and the flags hash recorded by every later file equal the sets
  and fit files'; at 7B the stage-6 sets hash and source; the preflight's population hashes equal the pins, with no
  overlap, and every later file records the same hashes; sizes: inject 100, ablate 60, bind 100, sign 100 in each of
  Q_IN, Q_OUT, P1, INLINE and INLINE_CHAT (both 7B models), diss 100 / 60 / 100 (1.5B / 3B / 7B). Outside TEST_MODE a
  MISMATCH, or results without a passing J-D-G0, makes the scorer exit with status 2 (a failed score step); the verdicts
  are still computed and printed as defined here.
- **J-D-G1, the natural effect** (per 7B model): N_X under P1 on E8 ≥ 3 nats with lower bound > 0. Otherwise J-D1–J-D4,
  J-D-ADDR and J-D-KN are NOT EVALUABLE in that model (J-D3 also needs mean ID_K(none) ≥ 1 nat with lower bound > 0).
- **J-D-G2, batch floor** (per batch type with a duplicated none row: inject, the Q and IOI six- and four-candidate
  batches, diss): mean over stories of the mean |lp(none2) − lp(none)| over the candidates ≤ 0.05 nats (full-vocabulary
  log-probabilities; in diss the case-marginalised scores, over P1 and POST together). A failing floor makes these lines
  NOT EVALUABLE in that model (a check of the model's batched numerics in that cell): the inject floor
  J-D1–J-D4, J-D-ADDR and J-D-KN (with J-D-G1); the Q_IN and Q_OUT floors J-D7, the Q_OUT floor also J-D-SIGN-Q and
  J-D-HOP2; the INLINE floor J-D-SIGN-IOI, J-D-ROUTE-IOI and J-D-HOP2; the diss floor J-D6a and J-D6. It does not compare
  against single passes (stage 6's Gate a2 failed on batch-shape offsets that cancel in in-batch differences).
- **J-D-G3, competence** (per model × arm, clean B run): Q_IN argmax over C a mentioned location (B, init or dloc) in
  ≥ 0.90 of stories; Q_OUT argmax an unmentioned location in ≥ 0.90; mean candidate mass ≥ 0.50; IOI arms: stage 5's
  Gate e (two-way accuracy ≥ 0.75, four-way ≥ 0.50, mean LD > 0 with lower bound > 0). A failing cell is NOT EVALUABLE
  for every line that reads it (J-D7, J-D-SIGN-Q, J-D-SIGN-IOI, J-D8-Q, J-D8-IOI, J-D-ROUTE-IOI, J-D-HOP2), and such a
  line is NOT EVALUABLE in that model, except J-D8-Q, whose two cells combine as stated in its row.
- **J-D-G4, transfer evaluable** (per Q or IOI cell): |d_G| ≥ 1 nat with the CI excluding 0, the same sign as ID_K^T, and
  |d_G| / |ID_K^T| ≥ 0.5. A failing cell is NOT EVALUABLE for J-D8-Q (Q_IN, Q_OUT) or J-D8-IOI (INLINE).
- **J-D-G5, binding queries** (per model): argmax over C = init under other_agent and = dloc under irrelevant_object in
  ≥ 0.80 of BIND each; otherwise J-D5 is NOT EVALUABLE in that model.
- **J-D-G6, dissociation population** (per model of D6): ≥ 40 competent cores; the P1 injection effect Δℓ_X(+Δ^P1) > 0 with
  lower bound > 0 on them; at 1.5B and 3B, R(k*) of the in-run H* under P1 on the run's E8 cores ≥ 0.6 (point). Otherwise
  J-D6 and J-D6-ROUTE are NOT EVALUABLE in that model (J-D6a does not use this gate).
- **J-D-G7, ratio denominators** (per line): the denominator of a ratio line must have |mean| ≥ the stated size with its CI
  excluding 0: Δℓ_X(move) in the route batch ≥ 1 nat (J-D4); Δm(K_S) ≥ 1 nat (J-D-ROUTE-IOI); Δℓ_X(inj) ≥ 0.5 nat in
  Q_OUT and INLINE (J-D-HOP2); Δℓ_X(inj) under POST ≥ 0.2 nat (J-D6-ROUTE); for J-D3, mean ID_K(none) on A ≥ 1 nat
  (signed) with lower bound > 0; otherwise the line is NOT EVALUABLE there.

#### Confirmatory lines

Kind A = account line; Part D has no measurement-validity lines (its floors and competence checks are gates). Priors were
recorded before any stage-8 output; "in hand" means the stage 5 and 6 results and the disclosed pilots (Seen before
finalisation). The class follows the recorded prior P(MET | evaluable) by the common rule: L = implied by data in hand on
the same models and material, prior ≥ 0.9; M = prior ≥ 0.8; R = prior < 0.8 (so J-D4 and J-D8-Q, implied by stage 6 but
at prior 0.85, are M). Models: 2/2 = Qwen2.5-7B and Mistral-7B; 3 = Qwen2.5-1.5B, 3B, 7B (≥ 2 evaluable).

| Code | Class, kind, prior | Criterion (per model) | Models | Justification of the prior |
|---|---|---|---|---|
| J-D1 | R, A, 0.55 | ID_inj^LOO / ID_K ≥ 0.5 (point); H0: ID_inj^LOO / ID_K ≤ 0.35 rejected; π_X(move(Δ^{−X}, X)) ≥ 0.5 (point) | 2/2 | 0.5B critic pilot: ratio 1.05 with and without LOO; π_X untested above 0.5B (0.12 at 0.5B); no 7B data on a mean flag |
| J-D2 | R, A, 0.55 | for each of orth, meanH and Δ^act: ι(move Δ) − ι(move c) ≥ 0.4 (point) and H0: ι(move Δ) − ι(move c) ≤ 0 rejected; ι(move c) / ι(move Δ) ≤ 0.3 (point) | 2/2 | 0.5B pilot: orth 0.17 against 0.96; meanH and the active-set flag untested |
| J-D3 | R, A, 0.50 | ρ_K(Δ̂) ≤ 0.5 (point) and H0: ρ_K(Δ̂) ≥ 0.6 rejected; ρ_K(pc1) ≥ 0.7 and ρ_K(meanH) ≥ 0.7 (points) | 2/2 | 0.5B pilot ρ_K 0.08 (random 0.99); at 7B whole-head ablation of H* removed 0.76 / 0.94 (stage 6 H3); one direction per layer is a much smaller intervention |
| J-D4 | M, A, 0.85 | r_ans^inj(KV) ≥ 0.6 (point) and H0: r_ans^inj(KV) ≤ 0.45 rejected; H0: r_ans^inj(K) − r_ans^inj(V) ≤ 0 rejected | 2/2 | implied by stage 6 H5 if the injection mimics the natural write: r_ans(KV) 0.899 / 0.859, r_ans(K) 0.797 / 0.683, r_ans(V) 0.080 / 0.023 |
| J-D-ADDR | M, A, 0.80 | weighted cos(Δ^K, Δ^KV) ≥ 0.9 (point); ID_inj(Δ^KV) / ID_inj(Δ^K) in [0.8, 1.25] (point) with H0: ≤ 0.7 and H0: ≥ 1.4 both rejected | 2/2 | 0.5B critic pilot: cos 0.98–1.00 at the main layers, per-story 0.92–0.99, ι(KV flag) 0.99 against 0.96 |
| J-D-KN | R, A, 0.40 | mean per-story Pearson ≥ 0.8 (point); β in [0.7, 1.3] (point); H0: mean[Δℓ_S − Δℓ_X](K_N) ≤ −0.1 \|ID_K\| and H0: ≥ +0.1 \|ID_K\| both rejected (ID_K the in-batch point estimate) | 2/2 | 1.5B critic pilot (P1, 'garage'): Δℓ_B −0.90 under K_N against −1.50 under K_S; Δℓ_S +1.76, Δℓ_X +1.87 (difference within 0.1 ID_K = 0.35 at the point, per-story \|Δℓ_S − Δℓ_X\| 1.16); the −Δ at r_B comparison was never run |
| J-D5 | R, A, 0.25 | H0: mean Ψ_bind ≤ 0 rejected (pooled); the 95 % CI of the stratum difference includes 0 | 2/2 | no data in hand; the other_agent question may not use the option lookup |
| J-D7 | M, A, 0.85 | mean ID_K(Q_OUT) ≤ −1 nat (point) and H0: ID_K(Q_OUT) ≥ 0 rejected; mean ID_K(Q_IN) ≥ 1 (point) and H0: ID_K(Q_IN) ≤ 0 rejected; H0: ID_K(Q_IN) − ID_K(Q_OUT) ≤ 0 rejected (paired) | 2/2 | 1.5B pilots: Q_OUT −4.17 (12/12 negative), −5.13; Q_IN +2.12, +3.20; extrapolated to 7B |
| J-D-SIGN-Q | M, A, 0.85 | H0: ID_K(Q_OUT) ≥ 0 rejected and H0: ID_V(Q_OUT) ≥ 0 rejected | 2/2 | 1.5B critic pilot ID_K −5.13 (se 0.57), ID_V −8.24 (se 0.74) |
| J-D-SIGN-IOI | L, A, 0.90 | H0: ID_K(INLINE) ≥ 0 rejected and H0: ID_V(INLINE) ≤ 0 rejected (fresh IOI cores) | 2/2 | stage 5: ID_K −3.26 [−3.47, −3.05] / −2.33 [−2.46, −2.19], ID_V +2.63 [+2.39, +2.88] / +1.27 [+1.14, +1.40] (n = 200 each, Gate e passed) |
| J-D8-Q | M, A, 0.85 | in Q_IN and in Q_OUT (both cells): R(H*) ≥ 0.6 (point) and H0: R(H*) ≤ 0.45 rejected; KO(H*) ≥ 0.6 (point) and H0: KO(H*) ≤ 0.45 rejected; R_rand ≤ 0.15 and KO_rand ≤ 0.15 (points). Per model: MET if met in both cells, NOT MET if not met in an evaluable cell, NOT EVALUABLE otherwise | 2/2 | stage 6 H1/H2 under P1: R 0.967 / 0.942, KO 0.977 / 0.971, random ≤ 0.011; hop 1 precedes the question semantics |
| J-D8-IOI | R, A, 0.55 | the same criteria in IOI INLINE | 2/2 | 0.5B pilot R 0.95, KO 0.97 (n = 24); untested at 7B and on a different task |
| J-D-ROUTE-IOI | R, A, 0.40 | K_S rows: r(KV) ≥ 0.6 (point); H0: r(K) − r(V) ≤ 0 rejected | 2/2 | no data in hand; stage 5's AFTER row splice put the read on the listed names (1.05) but not on the clamped names alone (0.27) |
| J-D-HOP2 | R, A, 0.25 | in Q_OUT and in INLINE: carry(top-10) ≥ 0.5 (point) and H0: carry(top-10) ≤ 0.2 rejected ("same reader, opposite sign"); the scorer prints "different reader" when both carries are ≤ 0.2 (points) with upper bounds < 0.5, else "intermediate" (both NOT MET) | 2/2 | no data in hand; ten heads may not carry a distributed hop 2 even in P1 |
| J-D6a | L, A, 0.90 | ω ≥ 0.6 (point) and H0: ω ≤ 0.4 rejected; κ ≥ 0.7 (point) | 3 | implied by stage 5's G2 (hop-1 attention follows the clamped key one for one at 1.5B / 3B) and the shared prefix; 0.5B pilot ω 0.99, cos 0.72–1.00 |
| J-D6 | R, A, 0.35 | sign(ρ) = sign(ρ_nat) (points) and the 95 % CI of ρ − ρ_nat inside ±max(0.1, 0.5\|ρ_nat\|) (two one-sided tests; the tolerance at the point estimate of ρ_nat) | 3 | ρ_nat in hand −0.20 (1.5B), ≈ 0 to +0.07 (3B), 0.23–0.26 (7B); 0.5B pilot: POST injection −0.46 against K_X −1.06; at 3B ρ_nat ≈ 0 makes the sign condition a coin flip |
| J-D6-ROUTE | R, A, 0.35 | at 1.5B under POST: H0: r^inj(V) − r^inj(K) ≤ 0 rejected | 1.5B | no data in hand; follows from stage 5's G4 only if the injected sentence flag is read at all |

**Expected values** (the author's, not thresholds): ID_inj^LOO / ID_K 0.6–1.0; ρ_K(Δ̂) 0.2–0.5; r_ans^inj(KV) ≈ 0.85;
ID_K(Q_OUT) −3 to −8 nats; R(H*) in IOI INLINE 0.5–0.9; ρ ≈ ρ_nat within ±0.1 at 1.5B and 7B.

#### Combination and the D6 decision table

Every line above is combined over its models as stated under Statistics. The D6 outcome (printed by the scorer, fixed now;
the first row whose condition holds, in this order):

| Outcome | Condition |
|---|---|
| (a) Mediated, read by value at 1.5B | J-D6 MET and J-D6-ROUTE MET |
| (b) Mediated; the 1.5B route not shown to be by value | J-D6 MET, J-D6-ROUTE not MET or not evaluable |
| (c) Not mediated | in some model where J-D6 is evaluable, ρ_nat has upper bound < 0 while \|ρ\| ≤ 0.1 (point) with the 95 % CI of ρ inside (−0.2, 0.2): the natural negative read is not carried by the sentence-row flag |
| (d) Mixed | anything else, reported as such |

The design's branch "never written" is dropped (stage 5's G2 already excludes it; D-8).

#### What each primary line means for the paper (pre-written; E-2)

No Part-D outcome changes the title (the common part). The main text reports three numbers for the flag (D-13):
ID_inj^LOO / ID_K, ρ_K and r_ans^inj(K); ι, ω, κ, ρ, Ψ, π, the gates and the controls go to the appendix.

| Primary line | Abstract clause (MET) | NOT MET: replacement | Title | Table 1 row | Figure |
|---|---|---|---|---|---|
| J-D1 | "The reader heads write one vector per layer into the re-mentioned word's row. Injected with no key clamp into the row of an option absent from the story, a version fit without that word moves the answer there (x of the key read's identity effect; the answer changes in y % of stories)." | "A mean flag does not reproduce the key read as an injection at 7B (ID_inj^LOO / ID_K = x): the reader heads' write is story-specific or non-linear." The routing result stands without the flag. | No change. | "What the readers write": ID_inj^LOO / ID_K per model. NOT MET: "not sufficient as a rank-1 write". | Fig. "flag", panel (a): ι by dose with the structured controls. |
| J-D3 | "Removing that one direction per layer at the option words removes the key read (ρ_K = x), while the top principal direction and the mean reader output do not." | "One direction per layer does not carry the key read (ρ_K = x): other carriers exist; the flag is sufficient but not necessary" (if J-D1 met), or the flag paragraph is withdrawn (if not). | No change. | Same row, ρ_K column. | Fig. "flag", panel (b). |
| J-D4 | "The answer reads the injected flag by key, as it reads the natural one (r_ans(K) = x against r_ans(V) = y)." | "The injected flag acts by a route other than the answer's key read." | No change. | Same row, r_ans column. | Fig. "flag", panel (a) inset. |
| J-D-SIGN-IOI with J-D8-IOI | "The same frozen reader heads that serve the belief lookup carry an opposite-sign key read in IOI's in-sentence re-mentions, while the value read keeps its positive sign (R(H*) = x, ID_K = y < 0 < ID_V = z)." | J-D-SIGN-IOI not met: "the in-sentence IOI read does not dissociate the channels at 7B". J-D8-IOI not met: "the negative IOI key read is carried by other heads; the reader heads are task-specific." | No change. | "Sign": ID_K, ID_V, R(H*) in P1, Q_OUT, IOI INLINE. | Fig. "flag", panel (c), the one main-text sign panel. |
| J-D-ROUTE-IOI and J-D-HOP2 | (body only) "The negative IOI read is applied at the answer's key read of the listed names (r(KV) = x), by the same hop-2 heads that read the flag in lists (carry = y)." | ROUTE not met: "the negative IOI read is applied elsewhere (r_other = x)". HOP2 not met: "a different (or a distributed) reader applies the negative read"; the scorer's reading (different reader / intermediate) is quoted. Polarity is never presented as a discovery. | No change. | Appendix. | Appendix. |
| J-D6 with J-D6-ROUTE | (body only) the D6 outcome sentence: (a) "At 1.5B the sentence re-mentions carry the same flag, the answer reads it by value, and its effect has the sign and size of the natural read." (b) as (a) without the route clause. (c) "At 1.5B the flag is written in sentences but does not mediate the small negative natural read." (d) "Mixed." | as the outcome table | No change. | "Scale": ω, ρ, ρ_nat at 1.5B, 3B, 7B. | Fig. "flag", panel (d). |
| J-D5 | (body only) "The flag of an initial-state sentence carries which object it binds, in both sentence orders (Ψ_bind = x)." | "The initial-state flag does not carry binding in a way separable from order" (Ψ_bind ≤ 0, or the strata differ). | No change. | Appendix. | Appendix. |

The novelty statement (D-13): (i) a key-only edit moves an identity-free address (J-D1 with the LOO flag, J-D-ADDR), which
explains why identity edits look key-carried in list formats; (ii) the same frozen heads in one model serve belief MCQ and
IOI with opposite key signs while the value keeps its sign (J-D-SIGN-IOI, J-D8-IOI); (iii) the scale-dependent route at
1.5B (J-D6, J-D6-ROUTE). Wang et al. (2022) is cited for the duplicate-token signal and S-inhibition (a negative use of
duplicate detection is known); Feng & Steinhardt (2023) for injectable additive binding vectors. The abstract's risk
summary counts the R lines of this part with the other parts'.

#### Reported (no verdict)

ι for add 0.5Δ / 1Δ / 2Δ (dose response, monotone on the points), add-only ι(1Δ) ≥ 0.25 with lower bound > 0, ι(move),
π_X(move), ID_inj with the full flag over ID_K, N_X; the sanity rows (isotropic, head-span, layer-permuted, +Δ at the
"Choices" and "Question" rows: ι, and the π_X shift) and the rank-1 capture ι(move Δ) / ι(move δ(s)); r_other of J-D4; the
B-identity share of the per-story flag's variance (against the null (groups − 1) / (n − 1)) and the logit-lens cosine
max |cos(Δ_l, W_U[location])| against random token rows; ρ_K of the isotropic directions (sanity ≥ 0.85); Ψ_ev (role
confounded with order); per Q and IOI cell ID_K, ID_V, R(H*), KO(H*), d_G / ID_K^T and d_Gc / ID_K^T; the Q-cell D9 rows
(Δℓ_X(+Δ^P1 at r_X) against N_X(cfg), isotropic) and Δℓ_B under K_S (Q_OUT: > 0 expected); the injected IOI flag's route
(r(C), r_other); the hop-2 carries under P1 (the ranking format), every head and random-10; D6b; the INLINE_CHAT and
AFTER cells (Gate e, ID_K, ID_V, R, KO, d_Gc / ID_K^T).

#### Exploratory (no verdict)

- **E-BEFORE** (Qwen2.5-7B, the first 60 cores of E_ioi, BEFORE; D-1): ID_K with p's attention to the listed names' rows
  knocked out (`ckeys.knockout`, the pairs {p} × rows, in the S, X and B runs) against no knockout; fraction of ID_K
  removed f = 1 − mean ID_K^KO / mean ID_K. Two-sided rule: f ≥ 0.5 "p-as-re-mention" (under BEFORE p is itself a later
  mention of the listed name); f ≤ 0.2 "a second-order read elsewhere"; else intermediate. The same with every list row
  knocked out; and the RowSplice localisation (`row_restricted_keys.run` with `ckeys.ioi.RowTask`, groups self, sentence
  tail, options, choices, tail): the fraction of the full key effect each group carries.
- **E-XTASK** (D-10; both 7B models): the belief flag Δ^P1 against each task's own flag (fit on XFIT with H*) at the X
  option row of the paint and schedule P1 tasks (`ckeys.tasks`), on XEVAL; ratio = mean Δℓ_X(Δ^P1) / mean Δℓ_X(own flag).
  Two-sided: ≥ 0.6 "a shared pointer"; ≤ 0.3 "a task-specific flag"; no direction is predicted.
- **E-QOUT** (D-12): the fraction of Q_OUT stories whose answer is the story's own word B under K_S and under V_S.
- **E-RECON** (D-12): the H3 / G7b reconciliation, the base-answer and the initial-location answer rates under A(Δ̂) at
  r_init and r_dloc only, at r_B only, and at all six rows.
- **E-GEOM**: c_l, φ_l, the cross-layer cosines of Δ, the cosines between Δ^P1, Δ^KV, Δ^POST, Δ^init, Δ^dloc and Δ^IOI; the
  hop-2 top-10 and the in-run a3 overlap with H*.
- **E-IOIINJ**: Δ^IOI, move(Δ^IOI) and Δ^P1 at the listed IO_X row, isotropic, K_X (four-way).
- **E6**: ι on the raw margin m_X = lp(X) − lp(B) for add, move and the LOO move.

Not run (D-12): the story-duplicate projection pass and the eager hop-2 attention probe of the design; the design's IOI
AFTER injection.

#### Seen before finalisation

No stage-8 output of any model above 0.5B parameters has been seen; the build ran no trained model larger than 0.5B. Seeds
81–86 were touched only by population construction, the 0.5B TEST_MODE run (n = 2), the 0.5B gate tests (the first core
of E8 and of the E_ioi candidates) and the review's 0.5B exactness probes (the first three cores of E8, random injected
vectors; listed under Build checks). The design and critique pilots
(all CPU, FP32, transformers 5.18.0; scripts and logs kept with the stage-8 build notes):
- **Design pilot (1), flag fit and injection** (`partD/pilot_flag.py`, `pilot05_P1.log`, `pilot05b.log`; Qwen2.5-0.5B):
  a3 ranking on 16 stories of Random(0) (k* = 17, layers 2–16); flag fit on the same 16; per-story consistency 0.84–0.97 at
  the main layers (3, 7, 11, 13), 0.50–0.65 elsewhere; mean cross-layer cosine 0.07. On 16 Random(1) stories (m_X, against
  K_X +3.25 nats): ι add 0.45, move 0.80, α 0.5 / 1 / 2 0.19 / 0.45 / 0.69, isotropic +0.01 / −0.05 / −0.16, "Choices"
  −0.04, "Question" +0.04; clean accuracy 0.62, K_X made X the argmax in 6 %. Q_IN (n = 10): ι add 0.42, move 0.73;
  Q_OUT (n = 10): clean argmax B 0.00, K_X effect on m_X −0.55, add −0.55.
- **Design pilot (2), directional ablation** (`pilot_ablate.py`, `pilot_abl05.log`; 16 fit, 16 Random(1)): ID_K 1.29 → 0.10
  (ρ_K 0.08); two random unit directions 0.99, 0.99; ID_V 3.26 → 3.00; base argmax 0.62 → 0.44.
- **Design pilot (3), Q_IN / Q_OUT at 0.5B** (n = 10): Q_IN competence 1.00, ID_K +2.21; the P1 flag at r_X +2.26 (move
  +2.80; K_X +3.31); random −0.41 to +0.08. Q_OUT weakly competent (30 % of argmaxes mentioned), ID_K −0.28 (se 0.22),
  injection −0.15 / −0.38 against K_X −0.60.
- **Design pilot (4), sign at Qwen2.5-1.5B** (`pilot_sign.py`, `pilot_sign15.log`; n = 12, Random(1)): Q_OUT competence
  12/12, ID_K −4.17 (se 0.41), 12/12 negative, Δlp(B) under K_S +6.45, argmax B under K_S 3/12; Q_IN competence 12/12, ID_K
  +2.12 (se 0.55), 12/12 positive. The only design pilot above 0.5B.
- **Design pilot (5), dissociation at 0.5B** (`pilot_diss.py`, `pilot_diss05.log`; 16 fit, 16 eval, case-marginalised over
  the single-token forms): ω 0.99, per-layer cos(Δ^POST, Δ^P1) 0.72–1.00; POST: case-marginalised mass 0.84, accuracy 0.56,
  ID_K −0.74; the P1 flag at the sentence row of X −0.46 (move −0.75) against K_X −1.06, random +0.10; the POST flag at the P1
  list row +1.63 against the P1 flag's +1.47.
- **Design pilot (6), IOI INLINE reader transfer** (`pilot_ioi.py`, `pilot_ioi05.log`; n = 24, IOI Random(5)): ID_K
  −2.75; the listed names carry 0.81; R(H*) 0.95, KO(H*) 0.97, random 0.00.
- **Design pilot (7), IOI flag injection** (`pilot_ioi_inject.py`, `pilot_ioi_inj05.log`; IOI Random(6) fit, Random(5)
  eval, n = 20): the IOI flag at the listed IO_X row −2.20 (se 0.25) against K_X −4.63; the belief flag −0.58 (se 0.13);
  random −0.02; cos(Δ^IOI, Δ^belief) −0.03 to 0.86 per layer.
- **Critic pilot `pilot_critic.py` / `pilot_critic2.py`** (`critic_mechanism/`, Qwen2.5-0.5B; flags fit on Random(0)
  indices 16–39 / 16–45, evaluated on 24 Random(7) stories): cos(Δ^K, Δ^KV) 0.98–1.00 at the main layers (per story
  0.92–0.99); B-identity share of the per-story flag 0.45–0.88 against a null of 0.17–0.22; max |cos(Δ, W_U[loc])| 0.06–0.11
  against 0.025 for random tokens; ID_K (six-way) +1.28; ID_inj / ID_K 1.04–1.05, with the LOO flag 1.048 (fit sizes
  15–23); ι(move) 0.96, ι(add) 0.75; the own write 1.18; its orthogonal part 0.13 / 0.15 (0.16 / 0.17 norm-matched); the KV
  flag 0.99; K_S raises ℓ_X by +0.83 against N_X +2.25.
- **Critic pilot `pilot_kv_inout.py`** (`pilot_kv_inout15.log`; Qwen2.5-1.5B, 16 Random(9) stories): Q_OUT mass 1.00,
  ID_K −5.13 (se 0.57), ID_V −8.24 (se 0.74), all 16 negative; Q_IN ID_K +3.20, ID_V +9.37; P1 ID_K +3.53, ID_V +9.00. K_N
  ('garage'): P1 Δℓ_B −0.90 (K_S −1.50), Δℓ_S +1.76, Δℓ_X +1.87; Q_OUT Δℓ_B +5.82 (K_S +5.98), Δℓ_S +0.17, Δℓ_X +0.60;
  Q_IN Δℓ_B −3.55 (K_S −2.17).
- **Critic's inline counts**: in make_cores(100, Random(81)) 74 cores have dloc ≠ init (36 object-first, 38
  distractor-first). (E8 is now the Random(81) stream outside U and EXCL; see Deviations.)
- **In-hand numbers quoted above** are from the committed stage-5 and stage-6 score files and the paper's tables.
- **Build checks** (this part's code; tokenizers, committed files and Qwen2.5-0.5B only):
  - make_cores(100, Random(81)) holds 3 cores of U (draws 7, 74, 90); the E8 stream therefore takes 110 draws to give 100
    cores outside U and EXCL; the BIND stream 140 draws.
  - the preflight on the Qwen2.5 and Mistral-7B-v0.3 tokenizers: every prompt of R, R', E8 (P1, POST, Q_IN, Q_OUT, with the
    K_N runs) and BIND (three queries) valid; no IOI candidate skipped (F_ioi 60, E_ioi 100 in both); 5 trie nodes (Qwen) and
    20 (Mistral) for the case-marginalised forms.
  - the unit tests (J-D-G0) after the review: `tests/test_flag.py` 13, `tests/test_questions.py` 11,
    `tests/test_stage8d_score.py` 15 (against the shared `analysis/stage8_holm.py`) and the shared
    `tests/test_head_splice.py` 7, all passed (46 in 4.5 min on a shared CPU, 2 threads). One earlier run of the same set
    failed test 1 alone (the session's first plain forward differed from the zero injection by 1.4e-3 in the log-probs,
    while every later exactness test of that run passed); the cause was not found, and 18 fresh processes repeating that
    comparison, some under CPU oversubscription, gave exactly 0. A J-D-G0 failure stops the pipeline before any model.
    After the audit of this entry against the code (a per-model scorer exception now gives exit status 1, skips are
    counted in every file of the run, and the step runs the shared population and Holm tests): `tests/test_stage8d_score.py`
    18, with `tests/test_stage8_populations.py` 5 and `tests/test_stage8_holm.py` 7 added to the step; the six files
    passed together (61 in 4.0 min, 2 threads).
  - review probes (Qwen2.5-0.5B, FP32; code checks, no scientific quantity): the hop-2 batch with every head listed
    against HopSplice ans_K on the first E8 core, max difference 3.8e-5 in the log-probs; the HopSplice mask over the trie
    nodes in the POST route batch on the first three E8 cores with random injected vectors (answer row reading the clean
    K/V moves the case-marginalised scores by 0.19–0.77 nats; giving the trie nodes pass A instead moves them by at most
    2.4e-6, so the trie nodes' own reading is immaterial at this scale); `format_factorial.run_item`'s encoding equals
    `prep`'s on every E8 and BIND core in P1, POST, Q_IN and Q_OUT for both tokenizers (0 mismatches).
  - the TEST_MODE runs of `scripts/gpu_stage8d.sh` (Qwen2.5-0.5B, FP32, n = 2 per population, k* = 5 heads ranked in-run,
    keys qwen7 and qwen1.5; a shared 4-core CPU; outputs in a scratch directory). First run: the pytest step with the
    shared `tests/test_clamp.py` added, 46 passed in 22 min (its row_restricted regression alone took over 10 min under
    load, which is why the gate leaves `tests/test_clamp.py` out); every step succeeded. Final
    run before review: pytest 41 passed in 3.4 min; preflight, sets, fit, inject, ablate (3.2 min), bind, sign
    (4.4 min), diss at qwen7, then the qwen1.5 steps, then the exploratory before and xtask steps, then the score; no step
    failed or was skipped; a rerun kept every step. The score file read "provenance OK; population OK; J-D-G0 MET"; the
    plumbing verdicts of 2 stories at 0.5B: J-D6a MET, J-D8-Q and J-D8-IOI NOT MET, every other line NOT EVALUABLE (J-D-G1
    N_X +2.42 < 3 nats; Q_OUT and INLINE_CHAT fail J-D-G3 at 0.5B; no competent POST core). Numbers seen (2 stories, 0.5B,
    k* = 5): ι(add 0.5 / 1 / 2Δ) 0.22 / 0.45 / 0.77, ι(move) 0.59, ID_inj(full flag)/ID_K 0.46, rank-1 capture 0.92;
    isotropic ρ_K 1.00–1.01; Q_IN ID_K +3.72, Q_OUT +0.42; INLINE ID_K −2.58, ID_V +3.63, R(H*) 0.27; INLINE_CHAT ID_K
    −3.51; AFTER −2.87; hop-2 carry under P1 0.29 (top 10), 1.02 (every head); the injected IOI flag's r(KV) 0.98; IOI
    BEFORE knockout fraction −0.45; cross-task ratio 1.70 (paint), 0.29 (schedule). The final run, on commit 84e4f05, whose code is the
    finalised code (19 min), passed pytest (61 tests in 4.2 min, J-D-G0 MET with the cross-part
    population and Holm tests) and every step, none failed or skipped, with "provenance OK; population OK". An earlier run on
    the reviewed code (commit c04b3eb, 22 min)
    passed pytest (46 tests in 4.5 min) and every step, with no step failed or skipped; its score read "provenance OK;
    population OK; J-D-G0 MET" and the same plumbing pattern (J-D-G1 N_X +2.42, below 3 nats, at 0.5B). These are plumbing values of the
    0.5B stand-in; no confirmatory population of any study model was run. Re-scored after the audit fixes, that archive
    gives exit status 0 with "provenance OK; population OK; J-D-G0 NOT MET", because its pytest run predates the two
    shared files and the three added scorer tests; with the 61-test run appended to a copy of its pytest log, J-D-G0 is
    MET and every other line of the score file is unchanged.

#### Compute (A100-80GB, BF16)

Calibrated on stage 6 (about 0.3 s per batched forward with hooks at 7B). Forward calls per story: fit 4 (P1 2, POST 2) +
2 (R') + 2 (F_ioi); inject 5 (one with two attention passes); ablate 36 (9 conditions × `run_item`'s 4); bind 6; sign 9 per
Q arm, 2 for P1, 11 for INLINE, 6 for INLINE_CHAT and AFTER; diss 8 (9 at 1.5B and 3B, with the gate batch). Each step
but the preflight (tokenizer only) loads the model (about 1 min at 7B).

| Step | Minutes (Qwen2.5-7B / Mistral-7B) |
|---|---|
| pytest (J-D-G0 and the shared tests, CPU) | 10 |
| fetch (15 GB, prefetched one model ahead) | 5 |
| preflight, sets (eager), fit | 2 + 3 + 4 |
| inject | 4 |
| ablate | 12 |
| bind | 4 |
| sign | 22 / 18 |
| diss (qwen7 only) | 5 |
| model loads (7 / 6 steps) | 7 / 6 |
| per 7B model (without the fetch) | about 63 / 53 |
| Qwen2.5-1.5B, 3B (preflight, sets, fit, diss) | about 10 and 12 |
| exploratory (before and xtask at Qwen2.5-7B; xtask at Mistral-7B), loads included | about 6 / 6 |
| score, manifest, archive | 3 |
| **core total** | **about 2.3 GPU-h (the model steps and the score; about 2.6 h of wall-clock time with the pytest step and the first fetch); about 2.5 GPU-h with the exploratory steps** |

The default deadline is DEADLINE_H = 3.5 h. The core steps of every model always run, in the order Qwen2.5-7B,
Mistral-7B, Qwen2.5-1.5B, Qwen2.5-3B; the exploratory steps (before and xtask at Qwen2.5-7B, xtask at Mistral-7B) run
after every core step, each only if at least 8 minutes remain before the deadline and the model's flags file exists
(dropped first); the exploratory ablation conditions inside ablate are skipped when the deadline has passed before the
ablate step starts. The two 7B models' weights stay on disk until their exploratory steps are done or skipped (no
model is downloaded twice); the 1.5B and 3B weights are deleted after their steps, every model's at the end, unless
KEEP_CACHE=1. Disk: ≥ 50 GB. Batch sizes: at most 29 rows per forward, 24 with a second attention pass, 11 with two
passes; under 5 GB of activations beyond the weights. At about $2 per A100-hour, $5–6.

#### Commands

```bash
J=$(git log --format=%H -1 --grep='^Finalise preregistration J$') && [ -n "$J" ] && git checkout "$J"
bash scripts/gpu_stage8d.sh                    # no HF token needed
TEST_MODE=1 bash scripts/gpu_stage8d.sh        # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2): qwen7 and qwen1.5
                                               # (TEST_ALL=1: also mistral7 and qwen3b)
python analysis/stage8d_score.py --results results/gpu_stage8d    # re-score an archive
```

#### Deviations from the design (docs/stage8_design/design_mechanism.md) as amended (docs/stage8_design/SYNTHESIS.md, section D), and the choices the amendments left open

- E8 is the Random(81) stream outside U and EXCL (110 draws), not make_cores(100, Random(81)): the latter holds 3 cores of U,
  which G6 forbids. BIND is the Random(84) stream with dloc ≠ init (100 eligible cores from 140 draws) rather than "~140
  cores so that ~100 are eligible".
- Gate tolerances: 1e-4 in the log-probabilities for the ablation check (the design's 1e-5 is below the FP32 rounding of
  y − (c − μ)u, measured 1.1e-5); 1e-5 is kept for the zero injection and the exact added vector.
- J-D-HOP2's per-head splice is a key splice (HeadSplice: the listed heads at the answer row read the clean run's keys at
  G); per-head value splicing is not available in the shared hooks, and the account says the flag is read by key.
- J-D-ROUTE-IOI's verdict uses the natural K_S rows; the injected-flag rows are reported with the same statistics.
- J-D-KN's specificity condition is an equivalence test of the mean difference (95 % CI inside ±0.1 |ID_K|); the Pearson and
  β conditions are point conditions.
- J-D6's tolerance uses the point estimate of ρ_nat; J-D6 and J-D6a need two of the three models evaluable.
- Ratio denominators are gated (J-D-G7) with the sizes stated there, which the amendments did not fix.
- The scored K and V rows hold the other channel at the base run's value (run_item's convention); the flag fits use stage
  6's key-only natural clamp, as the design defines K_W.
- The two 7B models' weights are kept on disk until the exploratory steps at the end, so that no model is fetched twice.
- Priors follow the critique's "implied by data in hand" column, and every class follows its prior by the common rule
  (L ≥ 0.9, M ≥ 0.8, R < 0.8): J-D4 and J-D8-Q, which the critique called consistency checks implied by stage 6, have
  priors 0.85 and are therefore M; J-D2 (structured controls) is class R, prior 0.55, which the amendments did not state.
