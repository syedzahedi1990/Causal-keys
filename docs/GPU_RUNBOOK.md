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
