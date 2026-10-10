#!/usr/bin/env bash
# GPU stage 8, part D (preregistered as P-2026-10-10-J part D in docs/PREREGISTRATION.md): what the reader heads write
# (the identity flag), the sign of the key read, and the 1.5B / 3B route. Models: qwen7 and mistral7 (the stage-6 models;
# H* and the random sets from results/gpu_stage6/heads, sha256 asserted), then qwen1.5 and qwen3b (H* ranked here, eager).
# No fallback model applies: part D uses no fresh family (rule G2); a model whose files are refused leaves its lines
# NOT EVALUABLE. Steps (scripts/stage8_common.sh: each kept once done, FORCE=1 / FORCE_STEPS=<names or globs> redo, logs in
# $OUT/logs/<step>.log, a failed step goes to FAILED.txt and the pipeline goes on), all experiments/stage8_flag.py:
#   pytest            J-D-G0 before any model: tests/test_flag.py (FP32 hooks at Qwen2.5-0.5B), tests/test_questions.py
#                     (prompt layouts, Qwen and Mistral tokenizers), tests/test_stage8d_score.py, and the tests of the
#                     shared HeadSplice / HopSplice that part D composes with (tests/test_head_splice.py)
#   the core steps per model, in the order qwen7, mistral7, qwen1.5, qwen3b (the next model's weights prefetched):
#     preflight_<key> tokenizer only: populations, hashes, disjointness, every prompt layout, IOI validity, forms
#     sets_<key>      eager: H* and the random / active sets (stage 6 at 7B; ranked at 1.5B / 3B), the hop-2 ranking
#     fit_<key>       the flags on R (P1, KV, active, LOO, POST; at 7B also the initial-state flags on R' and the IOI flag)
#     inject_<key>    qwen7, mistral7: E8, P1, the injection battery and the hop-2 route batch (J-D1, D2, D4, ADDR, KN)
#     ablate_<key>    qwen7, mistral7: the directional ablations under format_factorial.run_item (J-D3)
#     bind_<key>      qwen7, mistral7: the initial-state flags under three queries (J-D5)
#     sign_<key>      qwen7, mistral7: Q_IN, Q_OUT, P1, IOI INLINE, INLINE_CHAT (+ AFTER at qwen7) (J-D7, SIGN, D8, ROUTE, HOP2)
#     diss_<key>      qwen1.5, qwen3b, qwen7: P1 and POST, case-marginalised (J-D6a, J-D6, J-D6-ROUTE)
#   then the exploratory steps before_qwen7, xtask_qwen7, xtask_mistral7, each only if 8 minutes remain before the deadline
#   (the 7B weights stay on disk until then, so no model is downloaded twice)
#   score             analysis/stage8d_score.py -> $OUT/STAGE8D_SCORE.txt; MANIFEST.sha256; the archive
# Deadline (DEADLINE_H, default 3.5 h; STAGE8_DEADLINE): the core steps always run; the exploratory steps are skipped as
# above (dropped first); the exploratory conditions inside ablate are skipped once the deadline has passed (provenance).
# Batch sizes (A100-80GB, BF16, prompts of 100-140 tokens at 7B): at most 29 rows per forward (the injection battery),
# 24 rows with a second attention pass in every layer (the HeadSplice transfer batch), 11 rows with two passes (the IOI
# route batch); format_factorial.run_item's 13-row batch; trie scoring adds <= ~10 positions. Each is < 5 GB of activations
# beyond the weights. Compute estimate (entry, Compute): about 2.3 GPU-h for the core, about 2.5 h with the exploratory
# steps. Disk: >= 50 GB (both 7B models and a prefetch).
# Usage:
#   J=$(git log --format=%H -1 --grep='^Finalise preregistration J$') && git checkout "$J"
#   bash scripts/gpu_stage8d.sh                  # no HF token needed (no gated model)
#   TEST_MODE=1 bash scripts/gpu_stage8d.sh      # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2): every step of qwen7 and
#                                                # of qwen1.5 (TEST_ALL=1: mistral7 and qwen3b too)
# Optional: OUT, DEADLINE_H, FORCE, FORCE_STEPS, KEEP_CACHE, TESTS=0, PY, S8_MODELS, MINGIB.
cd "$(dirname "$0")/.." || exit 2
PART=d
S8_DEADLINE_H_DEFAULT=3.5
# shellcheck source=scripts/stage8_common.sh
source scripts/stage8_common.sh
s8_init

# ---- FP32 gates before any model (J-D-G0): part D's tests, the scorer's tests, the shared HeadSplice / HopSplice tests
s8_pytest tests/test_flag.py tests/test_questions.py tests/test_stage8d_score.py tests/test_head_splice.py

TAGP=; on TEST_MODE && TAGP=TEST_
EX=(experiments/stage8_flag.py --out "$OUT")
KEYS=(qwen7 mistral7 qwen1.5 qwen3b)
if on TEST_MODE && ! on TEST_ALL; then
  KEYS=(qwen7 qwen1.5)
  echo "$(s8_utc) TEST_MODE: keys ${KEYS[*]} (every step of each role; TEST_ALL=1 runs mistral7 and qwen3b too)" | tee -a "$OUT/COMMIT.txt"
fi
declare -A STEPS=([qwen7]="preflight sets fit inject ablate bind sign diss" [mistral7]="preflight sets fit inject ablate bind sign"
                  [qwen1.5]="preflight sets fit diss" [qwen3b]="preflight sets fit diss")
declare -A EXPLO=([qwen7]="before xtask" [mistral7]="xtask" [qwen1.5]="" [qwen3b]="")
declare -A DIRS=()
EXPLO_MIN=8      # one exploratory step, its model load included, plus the score

core_done() { local k=$1 s; for s in ${STEPS[$k]}; do s8_done "${s}_$k" || return 1; done; }

run_key() {  # run_key <key>: the core steps; 0 done (failed steps recorded), 1 files refused, 2 preflight failed
  local key=$1 dir t="${TAGP}$1" s
  if core_done "$key"; then echo "==================== $key: every core step kept"; return 0; fi
  dir=$(s8_fetch "$key") || { s8_check; return 1; }   # s8_check dies on a fetch failure that is not a refusal (disk, network)
  DIRS[$key]=$dir
  local M=(--model "$dir" --key "$key")
  S8_OUTPUTS="$OUT/preflight/$t.json" s8_step "preflight_$key" $PY "${EX[@]}" --stage preflight "${M[@]}" || { s8_drop "$key"; return 2; }
  S8_OUTPUTS="$OUT/sets/$t.json" s8_step "sets_$key" $PY "${EX[@]}" --stage sets "${M[@]}"
  if [ -f "$OUT/sets/$t.json" ]; then
    S8_OUTPUTS="$OUT/fit/$t.json $OUT/fit/$t.pt" s8_step "fit_$key" $PY "${EX[@]}" --stage fit "${M[@]}"
  else
    echo "FAILED fit_$key not run: no sets file" | tee -a "$OUT/FAILED.txt"
  fi
  for s in ${STEPS[$key]}; do
    case $s in preflight|sets|fit) continue;; esac
    if [ -f "$OUT/fit/$t.json" ]; then
      S8_OUTPUTS="$OUT/$s/$t.json" s8_step "${s}_$key" $PY "${EX[@]}" --stage "$s" "${M[@]}"
    else
      echo "FAILED ${s}_$key not run: no flags file" | tee -a "$OUT/FAILED.txt"
    fi
  done
  [ -n "${EXPLO[$key]}" ] || s8_drop "$key"   # a 7B model's weights stay for its exploratory steps (after every core)
  return 0
}

# ---- the core steps, model by model (the next model's weights prefetched in the background)
for i in "${!KEYS[@]}"; do
  key=${KEYS[$i]}
  next=${KEYS[$((i + 1))]:-}
  if [ -n "$next" ] && ! on TEST_MODE && ! core_done "$next"; then s8_prefetch "$next"; fi
  rc=0; run_key "$key" || rc=$?
  case $rc in
    0) ;;
    1) echo "FAILED $key: its files failed verification (no fallback applies in part D; its lines are NOT EVALUABLE)" | tee -a "$OUT/FAILED.txt";;
    *) echo "FAILED $key: its preflight failed (no fallback applies in part D)" | tee -a "$OUT/FAILED.txt";;
  esac
done
wait   # a prefetch still running

# ---- exploratory steps, after every core step, while time remains before the deadline (dropped first)
for key in "${KEYS[@]}"; do
  [ -n "${EXPLO[$key]}" ] || continue
  t="${TAGP}$key"
  for s in ${EXPLO[$key]}; do
    s8_done "${s}_$key" && { echo "==================== ${s}_$key kept"; continue; }
    if ! s8_time_left "$EXPLO_MIN"; then
      s8_skip "${s}_$key" "exploratory: less than $EXPLO_MIN min to the deadline"; continue
    fi
    if [ ! -f "$OUT/fit/$t.json" ]; then
      s8_skip "${s}_$key" "exploratory: no flags file of $key"; continue
    fi
    if [ -z "${DIRS[$key]:-}" ]; then   # a later session whose core steps were all kept: fetch (or re-verify) once
      DIRS[$key]=$(s8_fetch "$key") || { DIRS[$key]=; s8_skip "${s}_$key" "exploratory: the files of $key were refused"; continue; }
    fi
    S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/$s/$t.json" s8_step "${s}_$key" $PY "${EX[@]}" --stage "$s" --model "${DIRS[$key]}" --key "$key"
  done
  s8_drop "$key"
done

# ---- score, manifest, archive
s8_finish analysis/stage8d_score.py
