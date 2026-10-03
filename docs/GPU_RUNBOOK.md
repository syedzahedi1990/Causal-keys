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
bash scripts/gpu_stage4.sh
```

The script:
- needs Python ≥ 3.11 (otherwise it makes a Python 3.12 venv `.venv-s4` with uv, torch 2.11.0+cu128 as in stages 2/3b), pins transformers 5.9.0 (stops if that fails) and accelerate 1.13.0, stops if no GPU is visible, and records the environment and the commit;
- downloads Paper 1's released repository (checked against the reviewed copy's sha256 pins) and the Mistral-Small-24B weights (once), then runs a preflight (release pins, prompt checks, tokenizer identity with and without `fix_mistral_regex`) and stops before the fits if it fails;
- runs Paper 1's unmodified `gpu/train.py` twice through `experiments/refit_remap.py`: `fit_none` (no later mention) and `fit_p1` (Paper 1's own format), in parallel on GPUs 0 and 1. The wrapper replaces at import time the prompt function (fit_none only), the release-integrity check (README.md tolerated) and, by default (`LOADER=ours`), the model loader (same load arguments; Paper 1's exact Python/package pins waived); every replacement is listed in `fit_*/FRAME.json`;
- validates the bases, evaluates both refits next to the released bases in Paper 1's frames (with and without the "Answer:" prefill), and scores with `analysis/stage4_score.py`;
- writes `gpu_stage4_results.tgz` (fit run directories included, no model weights).

Put `gpu_stage4_results.tgz` in the Google Drive folder `causal-keys-results` and tell Claude. Then destroy the instance. If the script prints `FAILED`, upload the archive anyway: the logs are inside it.

To redo only the evaluation and scoring (for example after a failed frames step), run `bash scripts/gpu_stage4.sh` again with the same `OUT`: finished fits (`fit_*/FRAME.json` status COMPLETE) are reused, logs and `COMMIT.txt` are appended to, and an old `FAILED.txt` is moved aside. To re-score an archive: `PYTHONPATH=. python analysis/stage4_score.py --root results/gpu_stage4` (the Paper 1 root is taken from the frames file unless `--p1-root` is given).
