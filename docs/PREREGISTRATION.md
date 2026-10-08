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

**DRAFT, not yet final.** To be fixed in the commit titled "Finalise preregistration I", with `analysis/stage7_score.py` (and `analysis/stage7_parts/`), the stage-7 code (`ckeys/readerblind.py`, `experiments/stage7_link.py`) and `scripts/gpu_stage7.sh`, before any stage-7 GPU run; the GPU script refuses a draft entry, a modified tree or code that differs from that commit, and runs the FP32 unit tests (Gate I-G0) before it loads any model. Results go into the revision of paper v3; `paper/versions/paper2_v3.pdf` is not changed. **Seen before this draft:** every committed 24B number quoted below (stages 2, 3b and 4), the 7B head results of stage 6, and a Monte-Carlo power check run on the committed stage-3b per-core rows (see Power). No head-level output of any model at 24B has been seen. The only runs of the stage-7 code before finalisation will be the unit tests, `TEST_MODE` plumbing runs (Qwen2.5-0.5B, FP32, random bases, two stories) and code-review checks on the same model and on a tiny random-weight Mistral; no trained model larger than 0.5B. In the `TEST_MODE` runs every prediction is NOT EVALUABLE (I-G1 (a) has no stage-3b reference for that model, and I-G3 (1) fails because random bases give D ≈ 0); the exactness and floor gates (I-G0, I-G1 (b)–(d)) and single parts of other gates (e.g. I-G2 (b), I-G4) can pass there, and did. Any other pilot is listed here before the entry is finalised.

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
7. Every position ≤ p is identical under every B_x, A and N.
8. The pinned stories file hash equals the manifest entry; the base hashes equal the stage-4 provenance.
9. The run's mean-ablation path (the ranking pass's capture of the o_proj inputs at G, the means MU, the mean table, A) leaves the natural B row unchanged when the means are that story's own values, and zero means change it. The hooks also hold at the 24B head geometry (heads × head_dim ≠ hidden size) on a tiny random-weight Mistral: this path, B_x(all) = P + K_M, A with each head's own input, N(all heads) = the 4D knockout, and positions ≤ p unchanged.

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
