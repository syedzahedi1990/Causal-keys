#!/usr/bin/env bash
# GPU stage 8, part A (preregistered as P-2026-10-10-J part A in docs/PREREGISTRATION.md): the key/value format law on
# natural reading comprehension. Counterfactual SQuAD v1.1 dev passages (data/stage8a_items.json, rebuilt and compared
# byte for byte in the preflight) at Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3
# (official weights, sha256-verified by scripts/fetch_verified.py through s8_fetch); reader heads on natural text at
# Qwen2.5-7B and Mistral-7B. Steps (scripts/stage8_common.sh: each kept once done, FORCE=1 / FORCE_STEPS=<names or globs>
# redo, logs in $OUT/logs/<step>.log, a failed step goes to FAILED.txt and the pipeline goes on):
#   pytest       before any model: the FP32 gates J-A-G0 (tests/test_natural_clamp.py, tests/test_kvquant.py) and
#                J-A-HA-G0 (tests/test_natural_heads.py), the scorer's tests, and the shared modules part A calls
#   preflight    SQuAD v1.1 dev (downloaded once into $OUT/squad, or SQUAD=<a local copy>; sha256 asserted), the item
#                rebuild with the build's tokenizers, byte-identical to data/stage8a_items.json; stops the run otherwise
#   per model, in the order llama8, gemma9, qwen7, mistral7 (yi9 replaces llama8 or gemma9 when its files fail
#   verification, s8_fetch status 1, before any output of it exists; at most one replacement, recorded in
#   $OUT/FALLBACK.txt and kept by every later session; any other fetch failure stops the run with no fallback):
#     frames_<key>     the answer frame (greedy ID generations on the first 30 R items in NOM and OPTA)
#     factorial_<key>  every valid E item in the six formats, the closed-book prompts, KIVI 2-bit (J-A7)
#     heads_<key>      qwen7 and mistral7: rank (60 R), curves (80 E), ablation at Q+ (every E item), explore
#     explore_<key>    exploratory; run only when the time left covers it and the rest of the core
#   score        analysis/stage8a_score.py -> $OUT/STAGE8A_SCORE.txt; MANIFEST.sha256; the archive
# Deadline (DEADLINE_H, default 4.5 h; STAGE8_DEADLINE): each factorial gets --reserve-min = the core minutes of the
# models after it (plus 10, plus HEADS_MIN for the same model's heads step at qwen7 and mistral7); before LETA and MENB it projects that format's time from OPTA / MENA and, if it would end past
# the deadline minus the reserve, runs LETA with the reduced rows and skips MENB (recorded; exit 3 = partial, the next
# session redoes the step). An explore step runs only when the time left exceeds the reserve plus its estimate; its own
# parts check the deadline. Exploratory passes are dropped first, then the LETA rows, then MENB (entry, A-13).
# Batch sizes (A100-80GB, prompts up to ~780 tokens): one forward per item and format for each of the capture batch
# (4 rows), the scoring batch (<= 14 rows) and the generation batch (<= 15 rows; 16 new tokens at most); the heads'
# splice batches have <= 10 rows (2 passes in masked layers), the ablation batches 7 rows; all well inside 80 GB at 9B.
# Compute estimate (entry, Compute): about 3.4 GPU-h for the core, about 4.2 h with every exploratory pass.
# Usage:
#   J=$(git log --format=%H -1 --grep='^Finalise preregistration J') && git checkout "$J"
#   bash scripts/gpu_stage8a.sh                  # HF_TOKEN optional (official repos for the gated models; else the
#                                                # byte-identical copies scripts/stage8_models.json lists)
#   TEST_MODE=1 bash scripts/gpu_stage8a.sh      # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2, outputs TEST_<key>)
# Optional: OUT, SQUAD=<dev-v1.1.json>, DEADLINE_H, FORCE, FORCE_STEPS, KEEP_CACHE, TESTS=0, PY, S8_MODELS, MINGIB.
cd "$(dirname "$0")/.." || exit 2
PART=a
S8_DEADLINE_H_DEFAULT=4.5
# shellcheck source=scripts/stage8_common.sh
source scripts/stage8_common.sh
s8_init

# ---- FP32 gates before any model (J-A-G0, J-A-HA-G0), the scorer's tests, the shared modules part A calls
s8_pytest tests/test_natural_clamp.py tests/test_kvquant.py tests/test_natural_heads.py tests/test_stage8a_score.py \
  tests/test_generate.py tests/test_clamp.py tests/test_head_splice.py

# ---- SQuAD v1.1 dev and the item rebuild
SQ=${SQUAD:-$OUT/squad/dev-v1.1.json}
if [ ! -f "$SQ" ]; then
  mkdir -p "$(dirname "$SQ")"
  curl -fsSL --retry 3 --retry-delay 20 -o "$SQ.tmp" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" \
    && mv "$SQ.tmp" "$SQ" || die "download of SQuAD v1.1 dev (set SQUAD=<a local copy of dev-v1.1.json>)"
fi
S8_OUTPUTS="$OUT/preflight.json" s8_step preflight $PY experiments/natural_factorial.py --stage preflight --squad "$SQ" --out "$OUT" \
  || die "preflight: SQuAD hash or the item rebuild (see $OUT/logs/preflight.log)"

# ---- the models
TAGP=; on TEST_MODE && TAGP=TEST_
declare -A CORE_MIN=([llama8]=32 [gemma9]=48 [yi9]=34 [qwen7]=58 [mistral7]=58)   # fetch, frames, factorial (+ heads)
HEADS_MIN=20     # the heads step of qwen7 / mistral7 (in CORE_MIN), reserved in the same model's factorial
EXPLORE_MIN=14
KEYS=(llama8 gemma9 qwen7 mistral7)
# the fallback decided in an earlier session of this OUT stays: the replaced key does not run again (no mixing of a
# replaced model's later output with the fallback's)
FALLBACK=$(head -n 1 "$OUT/FALLBACK.txt" 2> /dev/null | awk '{print $1}')
reserve_after() {  # minutes of core the models after position $1 still need, plus the score
  local i s=10
  for ((i = $1 + 1; i < ${#KEYS[@]}; i++)); do s=$((s + CORE_MIN[${KEYS[$i]}])); done
  echo "$s"
}
has_output() { compgen -G "$OUT/frames/${TAGP}$1.json" > /dev/null || compgen -G "$OUT/factorial/${TAGP}$1.json" > /dev/null; }

run_key() {  # run_key <key> <reserve minutes>; status 0, or s8_fetch's status (1 = the files failed verification)
  local key=$1 reserve=$2 dir t="${TAGP}$1" rc=0 own=0
  dir=$(s8_fetch "$key") || rc=$?
  [ $rc = 0 ] || return $rc
  { [ "$key" = qwen7 ] || [ "$key" = mistral7 ]; } && own=$HEADS_MIN
  local M=(--model "$dir" --key "$key" --squad "$SQ" --out "$OUT")
  S8_OUTPUTS="$OUT/frames/$t.json" s8_step "frames_$key" $PY experiments/natural_factorial.py --stage frames "${M[@]}"
  if [ -f "$OUT/frames/$t.json" ]; then
    S8_OUTPUTS="$OUT/factorial/$t.json" s8_step "factorial_$key" $PY experiments/natural_factorial.py --stage factorial "${M[@]}" --reserve-min "$((reserve + own))"
  else
    echo "FAILED factorial_$key not run: no frames file" | tee -a "$OUT/FAILED.txt"
  fi
  if [ "$key" = qwen7 ] || [ "$key" = mistral7 ]; then
    S8_OUTPUTS="$OUT/heads/$t.json $OUT/heads/mu_$t.pt" s8_step "heads_$key" $PY experiments/natural_heads.py "${M[@]}"
  fi
  if s8_time_left $((reserve + EXPLORE_MIN)); then
    S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/explore/$t.json" s8_step "explore_$key" $PY experiments/natural_factorial.py --stage explore "${M[@]}"
  else
    s8_skip "explore_$key" "less than $((reserve + EXPLORE_MIN)) min to the deadline (the core of the later models needs $reserve)"
  fi
  s8_drop "$key"
  return 0
}

run_fallback() {  # run_fallback <position>: yi9 in the place of KEYS[position]
  local rc=0
  run_key yi9 "$(reserve_after "$1")" || rc=$?
  [ $rc = 0 ] && return 0
  [ $rc = 1 ] || s8_check   # a disk or manifest error stops the run here with its reason
  echo "FAILED fetch of the fallback yi9 (exit $rc)" | tee -a "$OUT/FAILED.txt"
}

for i in "${!KEYS[@]}"; do
  key=${KEYS[$i]}
  if [ -n "$FALLBACK" ] && [ "$key" = "$FALLBACK" ]; then   # replaced in an earlier session: yi9 runs in its place
    echo "$(s8_utc) $key replaced by yi9 in an earlier session ($OUT/FALLBACK.txt); $key is not run" | tee -a "$OUT/COMMIT.txt"
    run_fallback "$i"
    continue
  fi
  rc=0
  run_key "$key" "$(reserve_after "$i")" || rc=$?
  [ $rc = 0 ] && continue
  # status 1 only: the files of $key failed verification (FETCH_FAILED.txt), so the entry's single fallback applies,
  # before any output of $key exists; any other status (not enough disk, a manifest error) is not a verification
  # failure: s8_check stops the run with the reason and no fallback runs
  [ $rc = 1 ] || s8_check
  if [ $rc = 1 ] && { [ "$key" = llama8 ] || [ "$key" = gemma9 ]; } && [ -z "$FALLBACK" ] && ! has_output "$key"; then
    FALLBACK=$key
    echo "$key $(s8_utc)" > "$OUT/FALLBACK.txt"
    echo "$(s8_utc) fallback: yi9 replaces $key (its files failed verification; no output of $key exists)" | tee -a "$OUT/COMMIT.txt"
    run_fallback "$i"
  else
    echo "FAILED fetch of $key (exit $rc; no fallback applies)" | tee -a "$OUT/FAILED.txt"
  fi
done

# ---- score, manifest, archive
s8_finish analysis/stage8a_score.py
