# GPU runbook

No tunnels or remote shells. The GPU box runs a fixed script at a known commit, and results come back as one archive.

## Stage 1: format x key/value factorial at 1.5–14B (preregistered P-2026-10-03-B)

**Hardware:** 1× A100 80GB (≈ $0.7–1.0/h on Vast) or 1× H100. Disk ≥ 100 GB, PyTorch template. Runtime is about 1 hour.

### Option A: you launch it (works now, no keys shared)
1. Rent the instance on Vast and open its **Jupyter → Terminal** (or SSH).
2. Run:
   ```bash
   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
   export HF_TOKEN=...      # optional, type it in the terminal; adds Llama-3.1-8B and Gemma-2-9B
   bash scripts/gpu_stage1.sh
   ```
3. When it prints `DONE`, download `Causal-keys/gpu_stage1_results.tgz` from Jupyter's file browser. Put it in a Google Drive folder named `causal-keys-results`, then tell Claude. Claude reads it through the Drive connector.
4. Destroy the instance.

### Option B: Claude launches it (needs a new session)
1. Add `VAST_API_KEY` (and optionally `HF_TOKEN`) as environment variables in the cloud environment settings. Then start a new session.
2. In Vast, connect Google Drive once under **Account → Cloud Connections**.
3. The new session then:
   - creates the instance through the Vast API, with an on-start command that clones this repo at a pinned commit and runs `scripts/gpu_stage1.sh`;
   - follows progress through the Vast logs API;
   - copies `gpu_stage1_results.tgz` to Drive with Vast's cloud copy and reads it through the Drive connector;
   - destroys the instance.

## Stage 2: Paper 1 bases at 24B/72B plus the natural factorial at 24–72B (preregistered P-2026-10-03-C)

**Hardware:** 2× 80GB GPUs (2× A100 80GB ≈ $1.3–2/h, or 2× H100 ≈ $3–4/h). Disk ≥ 300 GB, PyTorch template. Runtime is about 2–3 hours, mostly model downloads.

With 1 GPU, the 24B and 32B parts still run and the 72B parts are skipped.

Run it in the instance's **Jupyter → Terminal**:
```bash
git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
bash scripts/gpu_stage2.sh
```

The script:
- downloads Paper 1's released data from the anonymous repository;
- pins transformers 5.9.0 and the Paper 1 model revisions;
- runs the Paper 1 frames for Mistral-24B and Qwen-72B;
- runs the natural factorial for Mistral-24B, Qwen2.5-32B and Qwen2.5-72B;
- writes `gpu_stage2_results.tgz`.

Put that file in the Google Drive folder `causal-keys-results` and tell Claude. Then destroy the instance.

## Stage 4: refit Paper 1's remap under NO-MENTION (preregistered P-2026-10-05-F; paper v2 only)

**Hardware:** 2× 80GB GPUs (2× H100 ≈ $3–4/h, or 2× A100 80GB ≈ $1.3–2/h). Each fit needs one card with ≥ 75 GiB. With 1 GPU the two fits run one after the other. Disk ≥ 150 GB, PyTorch template. Runtime is about 2–3 h on 2× H100 or 3–5 h on 2× A100.

Run it in the instance's **Jupyter → Terminal**:
```bash
git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
F=$(git log --format=%H -1 --grep='^Finalise preregistration F') && [ -n "$F" ] && git checkout "$F"
bash scripts/gpu_stage4.sh
```
The second line checks out the commit that finalises preregistration P-2026-10-05-F and its scorer. The script refuses to start while the entry is still marked DRAFT or while tracked files have local changes.

The script:
- needs Python ≥ 3.11 (otherwise it makes a Python 3.12 venv `.venv-s4` with uv, torch 2.11.0+cu128 as in stages 2/3b), pins transformers 5.9.0 (stops if that fails) and accelerate 1.13.0, stops if no GPU is visible or a card used for a fit has less than 75 GiB, and records the environment and the commit;
- downloads Paper 1's released repository (checked against the reviewed copy's sha256 pins), runs a preflight (release pins, prompt checks, `fix_mistral_regex` installed and tokenizer ids unchanged by it) that stops the script if it fails, then downloads the Mistral-Small-24B weights (once);
- runs Paper 1's unmodified `gpu/train.py` twice through `experiments/refit_remap.py`: `fit_none` (no later mention) and `fit_p1` (Paper 1's own format), in parallel on GPUs 0 and 1. The wrapper replaces at import time the prompt function (fit_none only), the release-integrity check (README.md tolerated) and, by default (`LOADER=ours`), the model loader (same load arguments; Paper 1's exact Python/package pins waived); every replacement is listed in `fit_*/FRAME.json`;
- validates the bases, evaluates both refits next to the released bases in Paper 1's frames (with and without the "Answer:" prefill), and scores with `analysis/stage4_score.py`;
- writes `gpu_stage4_results.tgz` (fit run directories included, no model weights).

Put `gpu_stage4_results.tgz` in the Google Drive folder `causal-keys-results` and tell Claude. Then destroy the instance. A step failed if the script prints a line starting with `FAILED <step>` (the same lines are in `FAILED.txt`). It then exits with status 1. `FAILED.txt` gives each step's exit code; an exit code above 128 means the step was killed by a signal (137 = SIGKILL, often the host's out-of-memory killer). Upload the archive anyway: the logs are inside it, including `log_env.txt` for the environment setup. `GATES FAILED` or `NOT MET` in the score summary are results, not step failures: the script still ends with "no failures" and exit status 0, and the archive is uploaded as it is, without a rerun.

To redo a failed step, run `bash scripts/gpu_stage4.sh` again with the same `OUT`: finished fits (`fit_*/FRAME.json` status COMPLETE) are reused; an unfinished fit (failed, killed, or failed validation) is moved aside to `fit_*.failed.<UTC time>`, which stays in the archive and is noted in `COMMIT.txt`, and is fitted again; validation, evaluation and scoring run only when both fits are COMPLETE. Logs and `COMMIT.txt` are appended to, and an old `FAILED.txt` is moved aside. If the box's Python is older than 3.11, the `.venv-s4` made by the first run is reused (it is made again only if torch cannot be imported from it). To re-score an archive: `PYTHONPATH=. python analysis/stage4_score.py --root results/gpu_stage4 --p1-root <local copy of Paper 1's reviewer repository>`. Without `--p1-root` the scorer uses the GPU box's path from the frames file, so off the box G1 is SKIPPED and the gates are reported as not evaluated.

## Stage 5: what a later mention must be, and do, to read the writing token's key (preregistered P-2026-10-05-G; paper v3)

**Hardware:** 1× 80GB GPU (1× A100 80GB ≈ $0.7–1.0/h on Vast, or 1× H100). Disk ≥ 200 GB (the HF cache is cleaned between models), PyTorch template. Runtime is about 5 h (about 1 h of that is downloads; `EXTRA=1` adds roughly 1 h).

Run it in the instance's **Jupyter → Terminal**:
```bash
git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
G=$(git log --format=%H -1 --grep='^Finalise preregistration G') && [ -n "$G" ] && git checkout "$G"
bash scripts/gpu_stage5.sh
```
The second line checks out the commit that finalises preregistration P-2026-10-05-G, its scorer `analysis/stage5_score.py` and the stage-5 code. The script refuses to start while the entry is still marked DRAFT, while tracked files have local changes, or when the code at HEAD (`ckeys experiments analysis scripts tests` and the entry) differs from that commit.

The script:
- pins transformers 5.18.0 (the stage-1/3b environment; stops if that fails), records the environment (`ENV.txt`, `PIP_FREEZE.txt`) and the commit (`COMMIT.txt`), and stops if no GPU with ≥ 75 GiB is visible (`MINGIB=<GiB>` lowers the floor for a partial rerun);
- runs the FP32 unit tests (`pytest tests/`, Qwen2.5-0.5B and GPT-2 small on the CPU) and two preflights before loading any 7B model: `ckeys.subsets.check_rules` (the 13 subset arms on 2000 cores; S6 = SENTENCE-AFTER and L6 = LIST-AFTER byte for byte) and `analysis/ioi_entry_facts.py` (the facts quoted by part (e) must equal `results/gpu_stage5/ioi/ENTRY_FACTS.txt`, which the commit "Finalise preregistration G" must include: the script stops if the file is missing or differs); any failure stops the script;
- loops over the models (GPT-2 small and XL, Qwen2.5-1.5B/3B/7B/14B, Mistral-7B, OLMo-2-7B, Qwen2.5-7B base; `EXTRA=1` adds Qwen3-8B and the exploratory models of parts (a) and (b)), downloading each once and running every part that lists it: (a) `remention_attention.py`, (b) `attention_knockout.py`, (c)+(d) one `format_factorial.py` run with the standard, subset and variant arms (seed 0; seed 1 on the subset arms at the three primary models) and the row-restricted splice, (d) `form_competence.py` and `form_attention.py`, (e) `ioi_factorial.py`, the GPT-2 small attention probe and the IOI row splice; `PARTS=a,b,c,d,e` selects parts (a part not listed is skipped everywhere; parts (c) and (d) share one seed-0 factorial file per model, which always carries both the subset and the variant arms, so selecting either part runs both arm sets), `ONLY=<comma-separated model names>` restricts the loop to those models (a name not in the model list is an error, as is a `PARTS`/`ONLY` choice that selects no step), a step whose results file already exists in `OUT` and is readable JSON is kept unless `FORCE=1` (a file written by a failed step, or one interrupted while being written, is moved aside to `<file>.failed.<UTC time>`, noted in `COMMIT.txt`, and the step is run again; under `FORCE=1` an earlier file is replaced only when the step writes a new one, so a forced step that fails before writing leaves it in place, noted in `COMMIT.txt`; the switches `TEST_MODE`, `EXTRA`, `KEEP_CACHE`, `FORCE`, `TESTS` take 0 or 1 only; part (a) writes its `.npz` sidecar before its `.json`, so a readable `.json` certifies both), and `MINGIB=<GiB>` lowers the 75 GiB GPU-memory floor (the 7B parts fit a 40 GB card; 14B needs about 30 GB in BF16 plus the eager attention of part (a)); it resolves every model's Hub revision before its steps, passes it to each step as `--revision` (so each results file's provenance carries it) and appends it, with the snapshot that was loaded, to `ENV.txt` (`ENV.txt` and `COMMIT.txt` are appended to on a rerun, under a dated header);
- scores with `analysis/stage5_score.py` (gates first, one verdict line per prediction G1-G22, provenance, population checks, then every part's full report) into `results/gpu_stage5/STAGE5_SCORE.txt`;
- writes `gpu_stage5_results.tgz` (results, logs, score; no model weights). The archive is a few hundred MB, far larger than those of stages 1-4: part (a) stores per-head float16 attention arrays for every core, arm and condition, including the full answer rows (about 300 MB at the four Qwen2.5 models, 600 MB with `EXTRA=1`); the Jupyter download and the Drive upload take minutes.

Put `gpu_stage5_results.tgz` in the Google Drive folder `causal-keys-results` and tell Claude. Then destroy the instance. A step failed if the script prints a line starting with `FAILED <step>` (the same lines are in `FAILED.txt`); the other steps still run, the archive is still written, and the script then exits with status 1. Upload the archive anyway: the logs are inside it. `GATED OUT`, `NOT EVALUABLE` or `NOT MET` in the score summary are results, not step failures: the script still ends with "no failures" and exit status 0. A `provenance MISMATCH` or `population MISMATCH` in the score summary (not one commit, a results file without provenance, a cell without the preregistered n or the same cores) is different: the score step then FAILS with exit status 2 (the score file is still written) and the script exits 1; upload the archive, the cause is read off the PROVENANCE / POPULATION sections. To redo a failed part on the same box, run the script again with the same `OUT`, `TESTS=0` (pytest already passed on that commit) and `PARTS=<that part>` (and `ONLY=<model>` for one model): every step whose results file exists and is readable is kept, a file written by the failed step was already moved aside to `<file>.failed.<UTC time>` so that step is run again, each model's Hub revision is read back from `OUT/REVISIONS.txt` (so the rerun loads the same snapshot as the earlier parts; the scorer flags a model whose files carry different revisions), `ENV.txt`, `COMMIT.txt` and `PIP_FREEZE.txt` are appended to, and everything in `OUT` is re-scored. On a new box, clone and check out the same commit, unpack the earlier archive first so that `results/gpu_stage5` holds the other parts (`tar xzf gpu_stage5_results.tgz`), then run with `PARTS=<that part>` (`TESTS=0` if pytest already passed on that commit; `MINGIB=40` on a 40 GB card for the 7B parts); without the earlier files the other parts score as "no results" / NOT EVALUABLE and the new archive lacks them. A rerun of (c) or (d) redoes the shared seed-0 factorial for both parts. If a GPT-2 step of part (e) FAILED with `clamp not exact` (the FP32 batch-noise floor above 1e-3 on that GPU), `IOI_EXACT=no` on the rerun keeps the file (the floors and `exact_violations` stay in its provenance and in the score); the default asserts exactness as the entry says.   To re-score an archive off the box: `PYTHONPATH=. python analysis/stage5_score.py --root results/gpu_stage5`.

The CPU plumbing test (`TEST_MODE=1 bash scripts/gpu_stage5.sh`, about 40 minutes on a 4-core box: 22 minutes of pytest, then 18 minutes of runs) runs every part at Qwen2.5-0.5B-Instruct in FP32 on the CPU, also on a box with a GPU (part (e) also at GPT-2 small), with tiny n (`TEST_NA`, `TEST_NB`, `TEST_NC` raise the n of parts (a), (b), (c)+(d)) into a scratch directory and scores it with the tag `TEST_`; its verdict lines are plumbing checks, not results. `TESTS=0` skips the pytest step in either mode.

## Stage 6: the reader heads and the second hop; the exchange on Prakash et al.'s intervention (preregistered P-2026-10-05-H; paper v3)

**What to rent.**
- Main run: 1× 80GB GPU (1× A100 80GB ≈ $1.5–2/h on Vast, or 1× H100), PyTorch template, host RAM ≥ 64 GB, disk ≥ 100 GB (Qwen2.5-7B, Mistral-7B and Qwen2.5-14B are downloaded one after the other; after its steps each model this run downloaded is removed from the HF cache, while models cached before the run are left alone). Runtime is about 6 h: about 0.3 h of setup and unit tests (pytest about 10 min), 2.5 h for part (a) (the reader heads at the two 7B models, eager attention) and 2.5–3.5 h for part (b) (Qwen2.5-14B, about 162k row-forwards, plus the exploratory FP32 re-check, about 25 min).
- Optional Llama-3-70B run (`MODEL=llama70`, prediction H12): 2× 80GB GPUs, disk ≥ 200 GB (the 70B weights are 140 GB). It is a separate invocation and runs only what H12 needs (the three confirmatory formats, the NO-MENTION sweeps, the exchange and the preregistered clamp onsets; about 56k row-forwards). The two GPUs run one after the other (`device_map=auto`), so it takes about 4–5 h including the download and pytest: 8–10 GPU-hours, about $15–20. Rent a separate 2-GPU box for it and unpack the main archive there first (see below), so that H12 is scored against the 14B results; renting 2 GPUs for the main run would leave one idle for 6 h.

**Gated weights.** Only the 70B run needs a Hugging Face token, from an account that has accepted the Meta Llama 3 licence on the model's Hub page. Type it into the box's own terminal with `read -rs HF_TOKEN && export HF_TOKEN` (the token is neither shown nor kept in the shell history). Never paste it anywhere else: not in a chat, a file in the repository, a notebook cell that is saved, or the Vast on-start command. The script never asks for it. Without `HF_TOKEN` it stops at once, and with a token that has no access it stops before any download. The main run does not need a token.

Run it in the instance's **Jupyter → Terminal**:
```bash
git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
H=$(git log --format=%H -1 --grep='^Finalise preregistration H') && [ -n "$H" ] && git checkout "$H"
bash scripts/gpu_stage6.sh
# optional, on a 2-GPU box: clone, check out $H, then tar xzf gpu_stage6_results.tgz first
read -rs HF_TOKEN && export HF_TOKEN   # paste the token, then Enter; not echoed, not saved in the history
MODEL=llama70 bash scripts/gpu_stage6.sh
```
The second line checks out the commit that finalises preregistration P-2026-10-05-H, its scorer `analysis/stage6_score.py` and the stage-6 code. The script refuses to start if any of these holds:
- the entry is still marked DRAFT;
- tracked files have local changes;
- the code at HEAD (`ckeys experiments analysis scripts tests` and the entry) differs from that commit.

The script:
- pins transformers 5.18.0 and records the environment (`ENV.txt`, `PIP_FREEZE.txt`) and the commit (`COMMIT.txt`). It stops if no GPU with ≥ 75 GiB is visible (two such GPUs for `MODEL=llama70`; `MINGIB=<GiB>` lowers the floor).
- fetches Prakash et al.'s release (https://github.com/Nix07/mind) at the commit pinned in `ckeys/causaltom.py`. This is a sparse fetch of the data files and the two code files the port follows, about 1 MB, into `~/.cache/causal-keys/`, outside the repository and the archive. It then checks the sha256 of every file used and the hash of the seed-10 pool (`RELEASE.txt`). Nothing of the release is copied into the repository or the results.
- runs a fixed set of FP32 unit tests on the CPU before loading any model, chosen by the code stage 6 runs, with no reruns: first the stage-6 tests (`log_pytest_stage6.txt`; `tests/test_head_splice.py` is Gate a1, and the scorer reads the last run in that log; `tests/test_clamp.py` also holds the `format_factorial.run_item` and `RowSplice` regressions), then the tests of the other shared modules stage 6 calls (`tests/test_interventions.py`, `test_encoding.py`, `test_row_restricted.py`; `log_pytest.txt`), about 8 min in all on a 4-core CPU. Any failure stops the script before any model. The other test files cover stage 1–5 code only and are run locally before the finalising commit. Without a passing Gate a1 the scorer marks H1–H6 NOT EVALUABLE and the score step fails with exit 2.
- runs the tokenizer-only preflight of part (b): positions, lengths and single-token checks on all 320 pairs in every format.
- loops over the models, downloading each once: `experiments/stage6_heads.py` at Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (`heads/<model>.json`), then `experiments/prakash_swap.py` at Qwen2.5-14B-Instruct, one step each for the LM filter, the sweeps (l*, l*_ID), the exchange, the clamp and the exploratory FP32 re-check of the sweeps at l* ± 2 (`prakash/<model>/`; the FP32 files are archived but not scored).
  - `PART=heads|prakash` runs one part.
  - Every model's Hub revision is resolved once, pinned in `REVISIONS.txt` and passed to each step.
  - A step whose results file exists and is complete is kept unless `FORCE=1`. A file written by a failed step is moved aside to `<file>.failed.<UTC time>`, and the step is run again.
- scores with `analysis/stage6_score.py` into `results/gpu_stage6/STAGE6_SCORE.txt`, in this order: provenance (including the release commit and hashes), population checks, gates a1–a3 and b0–b3, then one verdict line per prediction H1–H12. H12 reads NOT RUN until the 70B run has been done.
- writes `gpu_stage6_results.tgz` (`gpu_stage6_llama70_results.tgz` for `MODEL=llama70`): results, logs and score, no model weights (tens of MB).

**What to upload.** Put `gpu_stage6_results.tgz` (and `gpu_stage6_llama70_results.tgz` if the 70B run was done) in the Google Drive folder `causal-keys-results` and tell Claude. Then destroy the instance (destroy, do not just stop it: a stopped instance keeps its disk).

**Reading the result.** A step failed if the script prints a line starting with `FAILED <step>` (the same lines are in `FAILED.txt`). The other steps still run, the archive is still written, and the script exits with status 1. Upload the archive anyway: the logs are inside it. `NOT EVALUABLE`, `NOT MET` or a failed gate in the score are results, not step failures. A `provenance MISMATCH` or `population MISMATCH` makes the score step fail with exit 2 (the score file is still written).

**Running it anywhere else.** Outside `TEST_MODE`, set `KEEP_CACHE=1` for any invocation on a machine other than the rented box (a dry run, a refusal check): otherwise each model the run downloads is deleted from the Hugging Face cache after its steps.

**Reruns.** To redo a failed step on the same box, run the script again with the same `OUT`, `TESTS=0` (accepted only when `OUT/PYTEST_OK.txt` records a passing pytest of the same commit on the same host) and `PART=<that part>`. Kept files stay, revisions are read back from `REVISIONS.txt`, and everything in `OUT` is re-scored. On a new box, clone and check out the same commit and unpack the earlier archive first (`tar xzf gpu_stage6_results.tgz`).

To re-score an archive off the box: `PYTHONPATH=. python analysis/stage6_score.py --root results/gpu_stage6`.

**CPU plumbing test.** `TEST_MODE=1 bash scripts/gpu_stage6.sh` runs pytest, then both parts at Qwen2.5-0.5B-Instruct in FP32 on the CPU into a scratch directory (heads: n_rank = n_eval = 2 on a reduced grid; prakash: n = 2 pairs, no LM filter), then the scorer with the tag `TEST_`. It takes about 8 min of pytest plus about 11 min of runs on a 4-core box. Its verdict lines are plumbing checks, not results.

## Stage 7: blocking the reader heads while applying the released remap at Mistral-Small-24B (preregistered P-2026-10-08-I; paper v3)

**What to rent.** 1× 80GB GPU (1× A100 80GB ≈ $1.5–2/h on Vast, or 1× H100), PyTorch template, host RAM ≥ 64 GB, disk ≥ 100 GB (the 47 GB Mistral-Small-24B weights are downloaded once and removed from the HF cache at the end unless they were cached before the run). Runtime is about 2–2.5 h: about 0.8 h of setup (pip and pytest about 20–25 min, the download about 6–10 min, five model loads about 8 min, preflight, scoring and archive about 5 min), about 1.0–1.2 h of sdpa passes (about 128,000 row-forwards: the family run, the head gate and the link batches) and about 0.15 h of eager passes (the ranking and the exploratory remap ranking). That is about $4–6, at most about $10. No Hugging Face token is needed.

Run it in the instance's **Jupyter → Terminal**:
```bash
git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
I=$(git log --format=%H -1 --grep='^Finalise preregistration I') && [ -n "$I" ] && git checkout "$I"
bash scripts/gpu_stage7.sh
```
The second line checks out the commit that finalises preregistration P-2026-10-08-I, its scorer `analysis/stage7_score.py` and the stage-7 code. The script refuses to start if any of these holds:
- the entry is still marked DRAFT;
- tracked files have local changes;
- the code at HEAD (`ckeys experiments analysis scripts tests` and the entry) differs from that commit.

The script:
- pins transformers 5.18.0 and records the environment (`ENV.txt`, `PIP_FREEZE.txt`) and the commit (`COMMIT.txt`). It stops if no GPU with ≥ 75 GiB is visible (`MINGIB=<GiB>` lowers the floor).
- fetches our predecessor's released reviewer repository, Anonymous (2026), into `~/paper1` as stage 4 does (or uses `P1R=<path>`, which must exist: a missing `P1R` stops the script instead of fetching a copy elsewhere), and checks it before any test or model: the release manifest with `refit_remap.tolerant_verify` (which pins `RELEASE.json` itself), the nine Mistral bases against the stage-4 provenance and the stories file against the manifest (`RELEASE.txt`, `log_release.txt`).
- runs a fixed set of FP32 unit tests on the CPU before loading any model, with no reruns: the stage-7 tests (`log_pytest_stage7.txt`; `tests/test_stage7_link.py` is Gate I-G0, and the scorer reads the last run in that log), then the shared modules (`log_pytest.txt`), about 20 min on a 4-core CPU (14 min and 6.5 min in our TEST_MODE run). Any failure stops the script.
- runs the tokenizer-only preflight (`preflight.json`), downloads the model at the pinned revision (`REVISIONS.txt`), then the steps `rank` (eager), `family` (`experiments/paper1_frames.py`, the stage-3b reproduction), `gate`, `link` (sdpa) and the exploratory `remaprank` (eager), in that order. A step whose results file exists and is complete is kept unless `FORCE=1`; a file written by a failed step is moved aside to `<file>.failed.<UTC time>` and the step is run again. After `DEADLINE_H` hours (default 3.5; a positive number) the link step skips the exploratory parts (the curve batch and the exploratory conditions) of each format it starts after that time, for every core of that format, and `remaprank` writes a stub instead of running (each skip is recorded in the provenance and listed by the scorer), so the run stays within the 4 h cap. No confirmatory batch is skipped. A link step that fails is rerun for all five formats on the next invocation (about 1 GPU-hour), since steps resume at step granularity.
- scores with `analysis/stage7_score.py` into `results/gpu_stage7/STAGE7_SCORE.txt`: provenance, population checks, Gates I-G0 to I-G4, one verdict line per prediction I1–I7, the reported lines (SENTENCE-AFTER, the LETTERS-AFTER I5/I6 lines, the knockout contrast N(H*)), a summary, then the exploratory report.
- writes `gpu_stage7_results.tgz`: results, logs and score, no model weights (tens of MB; `heads/mu.pt`, the means used by the ablation, is included so that its hash can be checked).

**What to upload.** Put `gpu_stage7_results.tgz` in the Google Drive folder `causal-keys-results` and tell Claude. Then destroy the instance (destroy, do not just stop it).

**Reading the result.** A step failed if the script prints a line starting with `FAILED <step>` (the same lines are in `FAILED.txt`); the other steps still run, the archive is still written, and the script exits with status 1. Upload the archive anyway. A failure of the exploratory `remaprank` goes to `FAILED_EXPLORATORY.txt` and does not make the run fail. `NOT EVALUABLE`, `NOT MET` or a failed gate in the score are results, not step failures. A `provenance MISMATCH` or `population MISMATCH`, or link results without a passing Gate I-G0, make the score step fail with exit 2 (the score file is still written).

**Reruns.** To redo a failed step on the same box, run the script again with the same `OUT` and `TESTS=0` (accepted only when `OUT/PYTEST_OK.txt` records a passing pytest of the same commit on the same host). To fill exploratory parts skipped at the deadline, run it again the same way: a `remaprank` stub is redone automatically while the new session's own deadline has not passed, and `FORCE_STEPS=link` redoes the link step alone (about 1 GPU-hour; the other steps are kept). `FORCE_STEPS` takes a comma-separated list of steps (`preflight,rank,family,gate,link,remaprank`); `FORCE=1` redoes all of them. On a new box, clone and check out the same commit and unpack the earlier archive first (`tar xzf gpu_stage7_results.tgz`). Outside `TEST_MODE`, set `KEEP_CACHE=1` for any invocation on a machine other than the rented box.

To re-score an archive off the box: `PYTHONPATH=. python analysis/stage7_score.py --root results/gpu_stage7` (no copy of the predecessor's release is needed; the stage-3b reference is the committed `results/gpu_stage3b/paper1_frames_v/mistral.json`).

**CPU plumbing test.** `TEST_MODE=1 P1R=<local copy of the release> bash scripts/gpu_stage7.sh` runs pytest, then every step at Qwen2.5-0.5B-Instruct in FP32 on the CPU with random bases (two ranking stories, two evaluation cores) into a scratch directory, then the scorer with the tag `TEST_` (I-G1 (a) has no reference there and is NOT EVALUABLE, so every verdict is NOT EVALUABLE). It takes about 20 min of pytest plus about 15 min of runs on a 4-core box (peak host RAM about 10 GB). Its verdict lines are plumbing checks, not results.
