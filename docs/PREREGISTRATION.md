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
