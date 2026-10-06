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

## P-2026-10-05-H: GPU stage 6, the reader heads and the second hop; the exchange on Prakash et al.'s intervention and the out-of-sample law (paper v3)

**DRAFT, not yet final.** To be fixed in the commit titled "Finalise preregistration H", with `analysis/stage6_score.py`, the stage-6 code and `scripts/gpu_stage6.sh`, before any stage-6 GPU run; the GPU script refuses a draft entry or a modified tree and runs the FP32 unit tests before loading any 7B model. Finalised before the stage-5 outputs are inspected; nothing here depends on them.

**Context.** The claims table lists "second hop" and "head identity" as hypothesis, not tested (critique item 3), and the exchange has only been applied to our own predecessor's intervention (item 4) with an in-sample r = 0.98 between the remap's key share and s_ID (item 5). Part (a) names the heads that read the writing token's key at Qwen2.5-7B and Mistral-7B, tests whether they are canonical duplicate-token heads, ablates them, and gives only the answer position the base K/V of the option words. Part (b) runs the key-only / value-only exchange on Prakash et al.'s (2026) reversed-sentence binding swap (BIND) and on an identity edit at the same positions (ID), on their stories, word lists, raw wrapper and seed-10 pool, at Qwen2.5-14B, and tests the law kappa ≈ s_ID depth-matched. Disclosed pilots: (a) Qwen2.5-0.5B FP32 (a3-ranked top heads are duplicate heads, k80 = 12 of 336, ablation of 16 heads ID_K 1.69 → −0.09, answer-only hop r_ans(KV) 0.37 and 0.04 on two stories); (b) 0.5B prototype of the exchange (exact, B+KV_M = M to 0.0; psi_K = +1.54, psi_V = −0.61 under NO-MENTION, which the evaluability rule must mark not evaluable); the release's own 14B per-layer IIA (1.00 at blocks 28-34), which predicts l* = 27-28.

**Runs:** `scripts/gpu_stage6.sh`: `experiments/stage6_heads.py` (Qwen2.5-7B, Mistral-7B, BF16, eager attention, use_cache=False; ranking set R = make_cores(60, Random(0)), evaluation set E = make_cores(60, Random(1)); OPTIONS-AFTER confirmatory, SENTENCE-AFTER reported) and `experiments/prakash_swap.py` (Qwen2.5-14B, BF16, sdpa; `MODEL=llama70` optional on 2 GPUs). **Scoring:** `analysis/stage6_score.py` (writes `results/gpu_stage6/STAGE6_SCORE.txt`), committed with this entry.

### (a) Which heads read the key, and does the answer read the re-mentions directly?

a3(l,h) = ½[(A^{K_S} − A^B)[rowS → p] + (A^B − A^{K_S})[rowB → p]] (attention change under the key clamp, ranked on R only); HeadSplice lets chosen heads see K_S in the option rows G; R(k) = mean[m(top-k) − m(none)] / mean d_G with d_G the option-row ceiling; KO(k) = 1 − the same with the top-k heads blinded; k* = ceil(0.05 × n_heads) = 40 (Qwen2.5-7B) / 52 (Mistral-7B); mean-ablation replaces a head's output at the six option rows by its position-matched mean over R; duplicate score D, induction score I on 100 random repeated-token sequences, task-side T_dup from the other repeated location words; HopSplice gives only the answer position (row T−1) the base run's K and/or V of rows G while K_S is clamped at p: r_ans(C) = 1 − [m(row C) − m(ID)]/[m(K_S) − m(ID)], r_other for all other later rows, r_all for all rows. Every batched quantity is differenced against an in-batch reference row.

**Gates:** **Gate a1** (FP32, CPU, Qwen2.5-0.5B, `tests/test_head_splice.py`, seven tests, 1e-4 in the logits): HeadSplice all-heads == full key clamp, empty == clean, all heads in rows G == RowSplice(G), group slices == RowSplice(group), batched grid == single runs, self-ablation == clean; HopSplice all-rows == plain K/V clamp of rows G, no-overwrite == clean/full, answer-only with the K_S-run cache == full, batched == single. **Gate a2** (BF16 floor, per model and format): mean |m(none row) − m(clean pass)| and mean |m(all row) − m(full clamp pass)| ≤ max(0.5 nats, 0.02 × mean d_full); hop exactness row within 0.1 nats of the K_S row. **Gate a3** (D3 replicates on E): mean d_full ≥ 10 nats under OPTIONS-AFTER and > 0 under SENTENCE-AFTER; mean d_G / mean d_full ≥ 0.8 under OPTIONS-AFTER, both models.

**Predictions (OPTIONS-AFTER, both models unless stated):**
- **H1, sparsity.** With the top-k* heads by a3 seeing K_S in rows G and every other head the base key: R(k*) ≥ 0.8 with lower bound ≥ 0.7. (k80 for the a3, single-head f+ and leave-one-out d− rankings reported.)
- **H2, necessity and specificity.** KO(k*) ≥ 0.8; three random sets of k* heads give R_rand(k*) ≤ 0.25 and KO_rand(k*) ≤ 0.25 (means over draws).
- **H3, ablation.** Mean-ablation of the top-k* heads at the six option rows in every row of the 13-row ID_K/ID_V batch and its clean passes: (a) rho_K = mean ID_K(abl)/mean ID_K(none) ≤ 0.5 with upper bound ≤ 0.6; (b) clean-B candidate mass under ablation ≥ 0.9 in ≥ 80 % of stories; (c) the random sets and the size-matched active-at-G control set each leave rho_K ≥ 0.75; (d) paired ID_V(abl) − ID_V(none) > 0 with CI excluding 0 and mean ≥ 0.25 × [ID_V(NO-MENTION) − ID_V(OPTIONS-AFTER)] from stage 1 (2.7 nats at Qwen2.5-7B, 2.8 at Mistral-7B). The base-argmax rate ≥ 0.8 is predicted only if (d) is met.
- **H4, canonical duplicate-token heads.** Over the causal set C = top-k_C by a3 (k_C = min(k80, k*)): median D ≥ 0.2, median I ≤ 0.1, median T_dup ≥ 0.2. Secondary: |top-10(a3) ∩ top-10(D)| ≥ 3 (hypergeometric P ≤ 1.7e-4), Spearman rho(a3, D) and rho(a3, T_dup).
- **H5, the second hop at the answer.** r_ans(KV) ≥ 0.5 with lower bound ≥ 0.4 and r_ans(KV) > r_other (paired, CI excluding 0); consistency r_all(KV) ≥ 0.8; secondary r_ans(K) > r_ans(V); strong version r_ans(KV) ≥ 0.7 reported. SENTENCE-AFTER reported against r_ans(KV) ≥ 0.35 and r_all ≥ 0.5.
- **H6, the same readers for list and sentence.** |top-20(a3, OPTIONS-AFTER) ∩ top-20(a3, SENTENCE-AFTER)| ≥ 10 per model (null P ≤ 1.3e-12).

Alternative: the read is distributed beyond 5 % of heads (H1/H2 fail, curves reported); the readers are task-tuned lookup heads rather than general duplicate heads (H4 (i) fails); the flag reaches the answer through the instruction/template rows, a three-hop route (H5 fails with r_all ≥ 0.8); the two routes do not trade off at the head level (H3d fails, as at 0.5B).

### (b) The exchange on Prakash et al.'s intervention; the law out of sample

Qwen2.5-14B-Instruct; their template-2 stories, raw prompt (no chat template), pool of 320 pairs under random.seed(10) (sha256 asserted); primary population = the first 150 pairs whose clean and counterfactual prompts the model answers correctly under NO-MENTION. Formats (readout only): NO-MENTION (their question), QNAMES (the question names the four candidates s1, s2, S, X), OPTIONS-AFTER ("Choices: ... Answer with exactly one choice."), LETTERS-AFTER exploratory. BIND: at the output of block l* set the clean residual at [166,167,154,155] to the counterfactual's at [154,155,166,167]; ID: at block l*_ID set the queried state span to a donor story's with s_q → S. l* and l*_ID = the earliest layer maximising IIA on the NO-MENTION sweep (ties within 0.01 → earliest). Exchange rows r0 = B+KV_B (self), r1 = M, r2 = B+K_M, r3 = B+V_M, r4 = B+KV_M, r5 = M+K_B, r6 = M+V_B, K/V of the patched positions from block l_patch+1 on; psi_K = [m(r2) − m(r0)]/[m(r1) − m(r0)], psi_V likewise with r3, kappa = psi_K/(psi_K + psi_V); kappa_w (state words only) secondary. Natural clamp on the same clean stories at the queried state word, onsets l0 ∈ {l*+1, l*_ID+1, 0, 3, 14} plus an exploratory sweep, giving s_ID(f, l0).

**Gates (per format and arm):** **Gate b0** exactness: mean |m(r4) − m(r1)| ≤ 0.3 nats and r0 within 0.3 nats of the unbatched B (≤ 1e-3 in the FP32 TEST_MODE). **Gate b1** reproduction: IIA(l*) ≥ 0.7 under NO-MENTION for BIND; IIA_ID(l*_ID) ≥ 0.7 for ID. **Gate b2** effect size: Phi = mean[m(r1) − m(r0)] ≥ 3 nats. **kappa evaluability:** psi_K + psi_V ≥ 0.5 and psi_K, psi_V ≥ −0.1 (point estimates; resamples failing the rule are dropped, a dropped fraction > 5 % fails the CI); otherwise psi_K, psi_V and the interaction are reported and the cell is "interaction-carried" if the interaction ≥ 0.5. **Gate b3** (interpretation of H10 as a dissociation): ID_K(OPTIONS-AFTER, l0 = l*+1) > 0 with CI excluding 0 and s_ID(OPTIONS-AFTER, l*+1) ≥ 0.5, and kappa_ID at l* under OPTIONS-AFTER ≥ 0.5.

**Predictions:**
- **H7, the law on the binding swap, depth-matched.** For every evaluable f ∈ {NO-MENTION, QNAMES, OPTIONS-AFTER}, with NO-MENTION and OPTIONS-AFTER evaluable: |kappa(f) − s_ID(f, l*+1)| ≤ 0.25 (the paper's in-sample maximum deviation is 0.18), and Pearson r(kappa, s_ID(·, l*+1)) ≥ 0.9 over the three. The same against s_ID(f, 0) is secondary.
- **H8, shared prediction under NO-MENTION.** psi_V ≥ 0.5 and psi_K ≤ 0.25, with the CI of psi_V − psi_K excluding 0.
- **H9, crossover on their intervention (H_read).** kappa(OPTIONS-AFTER) − kappa(NO-MENTION) ≥ 0.4 with the paired CI excluding 0; kappa(QNAMES) between the two if evaluable.
- **H10, the rival (H_binding).** kappa(f) ≤ 0.25 for every evaluable f (incompatible with H7/H9 whenever s_ID(OPTIONS-AFTER, l*+1) ≥ 0.5). Read as "keys carry identity, values carry binding" only if Gate b3 passes; otherwise "not evaluable as a dissociation at this depth".
- **H11, positive control: the law on an identity edit on their material.** Part 1 (own depth): for every evaluable f, |kappa_ID(f) − s_ID(f, l*_ID+1)| ≤ 0.25, and kappa_ID(OPTIONS-AFTER) − kappa_ID(NO-MENTION) ≥ 0.4 with CI excluding 0. Part 2 (the binding depth, evaluable if IIA_ID(l*) ≥ 0.7): |kappa_ID at l*(f) − s_ID(f, l*+1)| ≤ 0.25 for every evaluable f. Met if Part 1 holds and Part 2 holds or is not evaluable.
- **H12, optional Llama-3-70B.** If run: Gate b1 with l* ∈ 30..40, and the 14B verdicts of H8, H11 Part 1 and whichever of {H7, H9} or H10 was met are reproduced at the same thresholds.

Alternative: H10 is the explicit rival of H7/H9; a flat-high kappa (address read by the answer position) and interaction-carried cells are reported as such; failure of H11 means the in-sample law does not transfer to their stories and wrapper even for an identity edit.

**Exploratory:** f+/d− rankings and grids, layer profile, split-half reliability of a3, zero-ablation and top-10/20, K-only/V-only hop rows and row restrictions, previous-token scores, optional Qwen2.5-14B (a); IIA/Phi sweeps under every format, FP32 re-check at l* ± 2, words-only and periods-only exchanges, rho_K/rho_V and the interaction, QNAMES2, LETTERS-AFTER as a fourth point, the full onset sweep of s_ID on their stories, a multi-position row splice if kappa(OPTIONS-AFTER) ≥ 0.3, per-pair scatter, the overlap with their 80 validation pairs (b).
