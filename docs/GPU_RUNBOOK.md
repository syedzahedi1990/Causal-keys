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
