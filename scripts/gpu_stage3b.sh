#!/usr/bin/env bash
# GPU stage 3b (preregistered as P-2026-10-04-E in docs/PREREGISTRATION.md): reviewer-driven controls.
#   (a) instruction-matched 2x2: {option list, neutral sentence} x {before, after the story}, same instruction
#   (b) identity vs state: clamp the critical token from a role-swapped story with the same location word
#   (c) environment deconfound for the scale curve (Qwen2.5-14B under stage-2 software; Qwen2.5-32B under stage-1)
#   (d) value-only exchange in Paper 1's frames (Mistral-24B, Qwen-72B), alongside the key-only exchange
# Needs 2 x 80 GB GPUs and >= 300 GB disk. Runtime about 3 hours (mostly downloads).
# Usage:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   bash scripts/gpu_stage3b.sh
# Output: gpu_stage3b_results.tgz
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/gpu_stage3b}
N=${N:-150}
mkdir -p "$OUT"
git log --oneline -1 | tee "$OUT/COMMIT.txt"
run() {
  local name=$1; shift
  echo "==================== $name  $(date -u +%H:%M:%S)"
  if ! PYTHONPATH=. "$@" > "$OUT/log_$name.txt" 2>&1; then echo "FAILED $name (see $OUT/log_$name.txt)" | tee -a "$OUT/FAILED.txt"; fi
  tail -n 2 "$OUT/log_$name.txt"
}
clean() { [ -z "${KEEP_CACHE:-}" ] && rm -rf ~/.cache/huggingface/hub; }
env_info() { python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee -a "$OUT/ENV.txt"; }
DT=bfloat16
SMALL=(Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct mistralai/Mistral-7B-Instruct-v0.3 allenai/OLMo-2-1124-7B-Instruct)
if [ -n "${TEST_MODE:-}" ]; then SMALL=(Qwen/Qwen2.5-0.5B-Instruct); N=3; DT=float32; fi

# ---- software environment of stage 1
[ -z "${TEST_MODE:-}" ] && python -m pip install -q 'transformers==5.18.0' accelerate numpy 2>&1 | tail -1
env_info
for m in "${SMALL[@]}"; do
  s=${m##*/}
  run "2x2_$s" python experiments/format_factorial.py --model "$m" --n "$N" --dtype "$DT" --arms AFTER,BEFORE,POST,PRE,NONE --out "$OUT/format_2x2"
  run "role_$s" python experiments/role_factorial.py --model "$m" --n "$N" --dtype "$DT" --arms P1,NONE,LETTER --views direct,world --out "$OUT/role_factorial"
  clean
done
if [ -z "${TEST_MODE:-}" ]; then
  # (c) Qwen2.5-32B on ONE GPU under the stage-1 environment (stage 2 ran it sharded under 5.9.0)
  run env_qwen32_tf518_1gpu env CUDA_VISIBLE_DEVICES=0 python experiments/format_factorial.py --model Qwen/Qwen2.5-32B-Instruct \
      --n "$N" --dtype "$DT" --arms P1,NONE,BEFORE --out "$OUT/env_check/tf518_1gpu"
  clean
  # ---- software environment of stage 2
  python -m pip install -q 'transformers==5.9.0' 2>&1 | tail -1
  env_info
  # (c) Qwen2.5-14B sharded over 2 GPUs under the stage-2 environment (stage 1 ran it on one GPU under 5.18.0)
  run env_qwen14_tf59_2gpu python experiments/format_factorial.py --model Qwen/Qwen2.5-14B-Instruct --n "$N" --dtype "$DT" \
      --arms P1,NONE,BEFORE --device-map auto --out "$OUT/env_check/tf59_2gpu"
  clean
  # (d) Paper 1 frames with key-only and value-only exchanges
  P1=${P1_ROOT:-$HOME/paper1}; P1R="$P1/v5.5-reviewer-repository"
  if [ ! -d "$P1R" ]; then
    mkdir -p "$P1"
    curl -fsSL -o "$P1/p1.zip" "https://anonymous.4open.science/api/repo/Beyond-the-fitted-Scope---Causal-keys-230B/zip"
    python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$P1/p1.zip" "$P1"
  fi
  run frames_v_mistral python experiments/paper1_frames.py --model mistral --p1-root "$P1R" --out "$OUT/paper1_frames_v"
  clean
  run frames_v_qwen python experiments/paper1_frames.py --model qwen --p1-root "$P1R" --out "$OUT/paper1_frames_v"
  run role_qwen72 python experiments/role_factorial.py --model Qwen/Qwen2.5-72B-Instruct --n "$N" --dtype "$DT" \
      --arms P1,NONE,LETTER --views direct,world --device-map auto --out "$OUT/role_factorial"
  clean
fi
tar czf gpu_stage3b_results.tgz "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(pwd)/gpu_stage3b_results.tgz"
cat "$OUT"/FAILED.txt 2>/dev/null || echo "no failures"
