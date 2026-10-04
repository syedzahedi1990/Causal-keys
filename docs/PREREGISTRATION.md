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
