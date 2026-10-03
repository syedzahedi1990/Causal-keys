#!/usr/bin/env bash
# GPU stage 3 (preregistered as P-2026-10-03-D in docs/PREREGISTRATION.md):
#   (a) generality: key/value factorial on two new state tasks (paint colour, schedule day) in 5 open 7-14B models
#   (b) localisation at scale: exact row-restricted key swaps (belief task, Paper 1 encoder) in 3 models
# Needs 1 x 80 GB GPU (A100/H100) and >= 120 GB disk. Runtime is about 2 hours.
# Usage:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   bash scripts/gpu_stage3.sh
# Output: gpu_stage3_results.tgz (send this back).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/gpu_stage3}
N=${N:-150}
NLOC=${NLOC:-60}
mkdir -p "$OUT"
if [ -z "${TEST_MODE:-}" ]; then python -m pip install -q 'transformers==5.18.0' accelerate numpy 2>&1 | tail -2; fi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee "$OUT/ENV.txt"
git log --oneline -1 | tee "$OUT/COMMIT.txt"
run() {
  local name=$1; shift
  echo "==================== $name  $(date -u +%H:%M:%S)"
  if ! PYTHONPATH=. "$@" > "$OUT/log_$name.txt" 2>&1; then echo "FAILED $name (see $OUT/log_$name.txt)" | tee -a "$OUT/FAILED.txt"; fi
  tail -n 2 "$OUT/log_$name.txt"
}
clean() { [ -z "${KEEP_CACHE:-}" ] && rm -rf ~/.cache/huggingface/hub; }
DT=bfloat16
MODELS=(Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen3-8B mistralai/Mistral-7B-Instruct-v0.3 allenai/OLMo-2-1124-7B-Instruct)
LOCALISE="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct mistralai/Mistral-7B-Instruct-v0.3"
if [ -n "${TEST_MODE:-}" ]; then MODELS=(Qwen/Qwen2.5-0.5B-Instruct); LOCALISE="Qwen/Qwen2.5-0.5B-Instruct"; N=3; NLOC=2; DT=float32; fi
for m in "${MODELS[@]}"; do
  s=${m##*/}
  for task in paint schedule; do
    run "tf_${task}_$s" python experiments/task_factorial.py --model "$m" --task "$task" --n "$N" --dtype "$DT" --out "$OUT/task_factorial"
  done
  if [[ " $LOCALISE " == *" $m "* ]]; then
    run "rr_$s" python experiments/row_restricted_keys.py --model "$m" --n "$NLOC" --arms P1,POST,LETTER --windows 4 \
        --dtype "$DT" --out "$OUT/row_restricted/$s"
  fi
  clean
done
tar czf gpu_stage3_results.tgz "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(pwd)/gpu_stage3_results.tgz"
cat "$OUT"/FAILED.txt 2>/dev/null || echo "no failures"
