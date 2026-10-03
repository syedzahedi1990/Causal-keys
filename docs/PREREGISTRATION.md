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
