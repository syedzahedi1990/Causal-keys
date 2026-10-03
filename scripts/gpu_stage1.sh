#!/usr/bin/env bash
# GPU stage 1 (preregistered as P-2026-10-03-B in docs/PREREGISTRATION.md):
# format x key/value factorial on open 1.5-14B models. Forward passes only, no training.
# Usage on a GPU box with >= 40 GB GPU memory and >= 80 GB disk:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   bash scripts/gpu_stage1.sh            # optional: export HF_TOKEN=... first to add Llama/Gemma
# Output: results/gpu_stage1/*.json, *_summary.txt and gpu_stage1_results.tgz (send this back).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/gpu_stage1}
N=${N:-150}
mkdir -p "$OUT"
python -m pip install -q 'transformers==5.18.0' accelerate numpy 2>&1 | tail -2
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
git log --oneline -1 | tee "$OUT/COMMIT.txt"
MODELS=(Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B
        mistralai/Mistral-7B-Instruct-v0.3 allenai/OLMo-2-1124-7B-Instruct Qwen/Qwen2.5-14B-Instruct)
if [ -n "${HF_TOKEN:-}" ]; then MODELS+=(meta-llama/Llama-3.1-8B-Instruct google/gemma-2-9b-it); fi
if [ -n "${MODELS_OVERRIDE:-}" ]; then read -ra MODELS <<< "$MODELS_OVERRIDE"; fi   # testing only
for m in "${MODELS[@]}"; do
  echo "==================== $m  $(date -u +%H:%M:%S)"
  if ! PYTHONPATH=. python experiments/format_factorial.py --model "$m" --n "$N" --dtype bfloat16 \
       --arms P1,NONE,BEFORE,POST,LETTER --out "$OUT" > "$OUT/log_${m##*/}.txt" 2>&1; then
    echo "FAILED $m (see $OUT/log_${m##*/}.txt)" | tee -a "$OUT/FAILED.txt"
  fi
  tail -n 4 "$OUT/log_${m##*/}.txt"
  [ -z "${KEEP_CACHE:-}" ] && rm -rf ~/.cache/huggingface/hub   # free disk between models
done
tar czf gpu_stage1_results.tgz "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(pwd)/gpu_stage1_results.tgz"
cat "$OUT"/FAILED.txt 2>/dev/null || echo "no failures"
