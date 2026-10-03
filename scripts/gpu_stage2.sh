#!/usr/bin/env bash
# GPU stage 2 (preregistered as P-2026-10-03-C in docs/PREREGISTRATION.md):
#   (a) Paper 1's released DAS/PCA bases at Mistral-Small-24B and Qwen2.5-72B across answer formats
#   (b) natural key/value format factorial at Mistral-Small-24B, Qwen2.5-32B and Qwen2.5-72B
# Needs 2 x 80 GB GPUs (A100 or H100) and >= 300 GB disk. With 1 GPU, the 72B runs are skipped.
# Usage:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   bash scripts/gpu_stage2.sh
# Output: gpu_stage2_results.tgz (send this back).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/gpu_stage2}
N=${N:-150}
mkdir -p "$OUT"
if [ -z "${TEST_MODE:-}" ]; then python -m pip install -q 'transformers==5.9.0' accelerate numpy 2>&1 | tail -2; fi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee "$OUT/ENV.txt"
git log --oneline -1 | tee "$OUT/COMMIT.txt"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

# Paper 1 released data (bases, story cores, saved outputs for the reproduction check)
P1=${P1_ROOT:-$HOME/paper1}
P1R="$P1/v5.5-reviewer-repository"
if [ ! -d "$P1R" ]; then
  mkdir -p "$P1"
  curl -fsSL -o "$P1/p1.zip" "https://anonymous.4open.science/api/repo/Beyond-the-fitted-Scope---Causal-keys-230B/zip"
  python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$P1/p1.zip" "$P1"
fi
ls "$P1R/gpu/component_data/bases" | wc -l | xargs echo "Paper 1 bases:"

run() {
  local name=$1; shift
  echo "==================== $name  $(date -u +%H:%M:%S)"
  if ! PYTHONPATH=. "$@" > "$OUT/log_$name.txt" 2>&1; then echo "FAILED $name (see $OUT/log_$name.txt)" | tee -a "$OUT/FAILED.txt"; fi
  tail -n 3 "$OUT/log_$name.txt"
}
clean() { [ -z "${KEEP_CACHE:-}" ] && rm -rf ~/.cache/huggingface/hub; }
ARMS=P1,NONE,BEFORE,POST,LETTER
if [ -n "${TEST_MODE:-}" ]; then   # CPU plumbing test: tiny model, random bases, 2 items
  T="--model-override Qwen/Qwen2.5-0.5B-Instruct --bases-override 1 --n 2"
  run frames_mistral python experiments/paper1_frames.py --model mistral --p1-root "$P1R" $T --out "$OUT/paper1_frames"
  run ff_test python experiments/format_factorial.py --model Qwen/Qwen2.5-0.5B-Instruct --n 2 --arms P1 --out "$OUT/format_factorial"
else
  run frames_mistral python experiments/paper1_frames.py --model mistral --p1-root "$P1R" --arms $ARMS --out "$OUT/paper1_frames"
  run ff_mistral24 python experiments/format_factorial.py --model mistralai/Mistral-Small-24B-Instruct-2501 \
      --revision 9527884be6e5616bdd54de542f9ae13384489724 --n "$N" --dtype bfloat16 --arms $ARMS --device-map auto --out "$OUT/format_factorial"
  clean
  run ff_qwen32 python experiments/format_factorial.py --model Qwen/Qwen2.5-32B-Instruct --n "$N" --dtype bfloat16 \
      --arms $ARMS --device-map auto --out "$OUT/format_factorial"
  clean
  if [ "$NGPU" -ge 2 ]; then
    run frames_qwen python experiments/paper1_frames.py --model qwen --p1-root "$P1R" --arms $ARMS --out "$OUT/paper1_frames"
    run ff_qwen72 python experiments/format_factorial.py --model Qwen/Qwen2.5-72B-Instruct \
        --revision 495f39366efef23836d0cfae4fbe635880d2be31 --n "$N" --dtype bfloat16 --arms $ARMS --device-map auto --out "$OUT/format_factorial"
    clean
  else
    echo "Only $NGPU GPU: skipping Qwen2.5-72B runs" | tee -a "$OUT/FAILED.txt"
  fi
fi
tar czf gpu_stage2_results.tgz "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(pwd)/gpu_stage2_results.tgz"
cat "$OUT"/FAILED.txt 2>/dev/null || echo "no failures"
