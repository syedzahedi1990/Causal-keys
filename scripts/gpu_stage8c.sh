#!/usr/bin/env bash
# GPU stage 8, part C (preregistered as P-2026-10-10-J part C in docs/PREREGISTRATION.md): independently obtained
# identity edits at the writing token and the channel-ratio law. Families E1 (CAA-in), E2 (CAA-out from the 24 neutral
# sentences of ckeys/neutral.py), E3 (the BatchTopK dictionaries of andyrdt/saes-qwen2.5-7b-instruct, Qwen2.5-7B only), E4
# (rank-16 DAS at p, seeds 101 PCA / 102 random), E5 (French, German, synonym forms; NONLEX of E1), controls T and R, the
# PAR / PERP and LEX / NONLEX components, the synthetic rows of J-C-G8; models qwen7 (E1-E5), mistral7 (E1, E2, E4, E5),
# llama8 (E1, E2, E5; yi9 takes its place when its files fail verification, s8_fetch status 1, or its preflight fails,
# before any output of it exists; the decision is kept by every later session of the same OUT), depths
# l in {3, 7, 11, 15}; then the overlap screen on Prakash et al.'s material (qwen7, llama8) and J-C6 at qwen14.
# Steps (scripts/stage8_common.sh: each kept once done, FORCE=1 / FORCE_STEPS=<names or globs> redo, logs in
# $OUT/logs/<step>.log, a failed step goes to FAILED.txt and the pipeline goes on), experiments/stage8_edits.py unless named:
#   release        Prakash et al.'s release at the pinned commit (sparse fetch as gpu_stage6.sh; every file hash and the
#                  pool hash asserted: python -m ckeys.causaltom); needed by the FP32 tests and the screen
#   pytest         J-C-G0 before any model (tests/test_stage8_edits.py, tests/test_sae.py, tests/test_stage8c_score.py and
#                  the shared modules part C calls: surface, clamp, head_splice, prakash, generate; the population check
#                  across the parts and the Holm helper: stage8_populations, stage8_holm)
#   per model, in the order qwen7, mistral7, llama8:
#     preflight_<key>  tokenizer only: populations, prompts, neutral sentences, form sets, the SAE pins on the Hub
#     calib_<key>      frames, means, spans, random directions, the SAE gate J-C-G3 and k_F / beta rule, the E5 alpha rule
#     dasfit_<key>     qwen7, mistral7: the E4 fits (8 per model) and their held-out flip rates
#     eval_<key>       every E story: clamp-row batches in four formats at four depths (resumable; exit 3 at the deadline)
#     readers_<key>    qwen7, mistral7: J-C-READ (HeadSplice x_S rows at P1, l = 7)
#     overlap_<key>    qwen7, llama8: experiments/stage8_overlap.py (LM filter, BIND sweep, s_ID onsets, the window)
#     explore_<key>    exploratory (greedy-generation flip rates); run only when its time and every later core step fit
#   jc6            experiments/prakash_caa.py at qwen14 (filter, CAA means, exchange), deadline-gated
#   score          analysis/stage8c_score.py -> $OUT/STAGE8C_SCORE.txt; MANIFEST.sha256; the archive
# Deadline (DEADLINE_H, default 5.0 h; STAGE8_DEADLINE): every eval step gets --reserve-min = the core minutes of the
# models after it plus 10 (plus READERS_MIN = 8 for its own readers step at qwen7 and mistral7) and stops between stories
# (exit 3: partial, the next session goes on from the stories done); an overlap step runs only if its minutes plus that
# reserve remain; jc6 only if 25 minutes remain; explore steps last in priority (dropped first). Exploratory first, then
# jc6, then the overlap screens are what the deadline drops.
# Batch sizes (A100-80GB, prompts <= ~170 tokens plus <= ~60 trie nodes): scoring forwards of 64 rows (--chunk), each led
# by the in-batch self row (about 13k tokens, < 10 GB of activations and logits at 7-8B); HeadSplice reader forwards of 26
# rows (two attention passes in the masked layers); prefix captures of <= 70 tokens (eval: the edit rows in 5-row
# forwards, the natural pass's shape, so T's tables equal the natural ones bitwise); DAS training at batch 1 (one
# forward and backward of ~130 tokens); the Prakash steps use prakash_swap's batches (16 rows of ~200 tokens;
# 10-row exchange batches at 14B). Compute estimate (entry, Compute): about 3.5 GPU-h for the core, about 4.8 h with the
# screens, J-C6 and the exploratory steps (DEADLINE_H 5.0 fits them in one session). Disk: >= 80 GB (two models, the four dictionaries of 3.76 GB each).
# Usage:
#   J=$(git log --format=%H -1 --grep='^Finalise preregistration J$') && git checkout "$J"
#   bash scripts/gpu_stage8c.sh                  # HF_TOKEN optional; PRAKASH_REPO=<checkout of the release> optional
#   TEST_MODE=1 bash scripts/gpu_stage8c.sh      # CPU plumbing run at Qwen2.5-0.5B (FP32, n = 2, one depth): every step of
#                                                # qwen7, the screen and jc6 (TEST_ALL=1: mistral7 and llama8 too)
# Optional: OUT, PRAKASH_REPO, DEADLINE_H, FORCE, FORCE_STEPS, KEEP_CACHE, TESTS=0, PY, S8_MODELS, MINGIB.
cd "$(dirname "$0")/.." || exit 2
PART=c
S8_DEADLINE_H_DEFAULT=5.0
# shellcheck source=scripts/stage8_common.sh
source scripts/stage8_common.sh
s8_init

# ---- Prakash et al.'s release at the pinned commit (sparse: the data and the two ported code files), hashes asserted
SHA=$(PYTHONPATH=. $PY -c "from ckeys.causaltom import RELEASE_SHA; print(RELEASE_SHA)") || die "ckeys.causaltom not importable"
URL=$(PYTHONPATH=. $PY -c "from ckeys.causaltom import RELEASE_URL; print(RELEASE_URL)")
DEFREL=$HOME/.cache/causal-keys/mind-$SHA
REL=${PRAKASH_REPO:-}
if [ -z "$REL" ]; then
  if on TEST_MODE && [ -d /home/user/nix07/mind/.git ]; then REL=/home/user/nix07/mind; else REL=$DEFREL; fi
fi
if [ ! -d "$REL/.git" ]; then   # fetched only into the default cache path (an existing other directory is left untouched)
  if [ -e "$REL" ]; then [ "$REL" = "$DEFREL" ] && rm -rf "$REL" || die "PRAKASH_REPO=$REL exists but is not a git checkout of $SHA"; fi
  { mkdir -p "$REL" && git init -q "$REL" && git -C "$REL" remote add origin "$URL" \
      && git -C "$REL" sparse-checkout set --no-cone /data/ /src/dataset.py /notebooks/causalToM_novis/utils.py \
      && git -C "$REL" fetch -q --depth 1 --filter=blob:none origin "$SHA" && git -C "$REL" checkout -q FETCH_HEAD; } >> "$OUT/logs/release_fetch.log" 2>&1 \
    || { rm -rf "$REL"; die "could not fetch $URL at $SHA into $REL (see $OUT/logs/release_fetch.log)"; }
fi
[ "$(git -C "$REL" rev-parse HEAD 2>/dev/null)" = "$SHA" ] || die "the release checkout $REL is not at $SHA"
export PRAKASH_REPO=$REL
echo "$(s8_utc) Prakash et al.'s release $URL at $SHA in $REL" >> "$OUT/COMMIT.txt"
s8_run release $PY -m ckeys.causaltom || die "the release or pool hash check (see $OUT/logs/release.log)"

# ---- FP32 gates before any model (J-C-G0): part C's tests, the scorer's tests, the shared modules part C calls, the
# population check across the parts (G6) and the Holm helper (D2)
s8_pytest tests/test_stage8_edits.py tests/test_sae.py tests/test_stage8c_score.py tests/test_surface.py tests/test_clamp.py \
  tests/test_head_splice.py tests/test_prakash.py tests/test_generate.py tests/test_stage8_populations.py tests/test_stage8_holm.py

TAGP=; on TEST_MODE && TAGP=TEST_
EV=(experiments/stage8_edits.py --out "$OUT")
SAE_DIR=$S8_MODELS/sae_qwen7
KEYS=(qwen7 mistral7 llama8)
if on TEST_MODE && ! on TEST_ALL; then
  KEYS=(qwen7)
  echo "$(s8_utc) TEST_MODE: keys ${KEYS[*]} (every family and step; TEST_ALL=1 runs mistral7 and llama8 too)" | tee -a "$OUT/COMMIT.txt"
fi
declare -A CORE_MIN=([qwen7]=80 [mistral7]=62 [llama8]=42 [yi9]=48)    # fetch, preflight, calib, dasfit, eval, readers
OVERLAP_MIN=15; JC6_MIN=25; EXPLORE_MIN=8; READERS_MIN=8   # READERS_MIN: the readers step after eval (qwen7, mistral7)

reserve_after() {  # reserve_after <index>: core minutes of the models after KEYS[index], plus the score
  local i s=10
  for ((i = $1 + 1; i < ${#KEYS[@]}; i++)); do s=$((s + CORE_MIN[${KEYS[$i]}])); done
  echo "$s"
}
has_output() { compgen -G "$OUT/calib/${TAGP}$1.json" > /dev/null || compgen -G "$OUT/eval/${TAGP}$1.json" > /dev/null; }
core_done() { local k=$1 s; for s in preflight calib eval; do s8_done "${s}_$k" || return 1; done
              case $k in qwen7|mistral7) s8_done "dasfit_$k" && s8_done "readers_$k" || return 1;; esac; }

run_key() {  # run_key <key> <reserve minutes>: 0 done, 1 files refused, 2 preflight failed (both before any output)
  local key=$1 reserve=$2 dir t="${TAGP}$1"
  if core_done "$key" && { [ "$key" = mistral7 ] || [ "$key" = yi9 ] || s8_done "overlap_$key"; } && s8_done "explore_$key"; then
    echo "==================== $key: every step kept"; return 0
  fi
  dir=$(s8_fetch "$key") || { s8_check; return 1; }   # s8_check dies on a fetch failure that is not a refusal (disk, network)
  local M=(--model "$dir" --key "$key" --sae-dir "$SAE_DIR")
  S8_OUTPUTS="$OUT/preflight/$t.json" s8_step "preflight_$key" $PY "${EV[@]}" --stage preflight "${M[@]}" || { s8_drop "$key"; return 2; }
  S8_OUTPUTS="$OUT/calib/$t.json $OUT/calib/$t.pt" s8_step "calib_$key" $PY "${EV[@]}" --stage calib "${M[@]}"
  if [ "$key" = qwen7 ] || [ "$key" = mistral7 ]; then
    S8_OUTPUTS="$OUT/das/$t.json $OUT/das/$t.pt" s8_step "dasfit_$key" $PY "${EV[@]}" --stage dasfit "${M[@]}"
  fi
  if [ -f "$OUT/calib/$t.json" ]; then
    local evres=$reserve
    case $key in qwen7|mistral7) evres=$((reserve + READERS_MIN));; esac   # the readers step of this model comes after eval
    s8_step "eval_$key" $PY "${EV[@]}" --stage eval "${M[@]}" --reserve-min "$evres"
    if [ "$key" = qwen7 ] || [ "$key" = mistral7 ]; then
      S8_OUTPUTS="$OUT/readers/$t.json" s8_step "readers_$key" $PY "${EV[@]}" --stage readers "${M[@]}"
    fi
  else
    echo "FAILED eval of $key not run: no calibration file" | tee -a "$OUT/FAILED.txt"
  fi
  if [ "$key" = qwen7 ] || [ "$key" = llama8 ]; then   # the overlap screen (C-5) while the weights are here
    if s8_done "overlap_$key" || s8_time_left $((reserve + OVERLAP_MIN)); then
      S8_OUTPUTS="$OUT/overlap/$t/screen.json $OUT/overlap/$t/window.json" s8_step "overlap_$key" $PY experiments/stage8_overlap.py \
        --model "$dir" --key "$key" --out "$OUT"
    else
      s8_skip "overlap_$key" "less than $((reserve + OVERLAP_MIN)) min to the deadline (the later cores need $reserve)"
    fi
  fi
  if s8_done "explore_$key" || s8_time_left $((reserve + JC6_MIN + EXPLORE_MIN)); then
    S8_EXPLORATORY=1 S8_OUTPUTS="$OUT/explore/$t.json" s8_step "explore_$key" $PY "${EV[@]}" --stage explore "${M[@]}" \
      --reserve-min $((reserve + JC6_MIN))
  else
    s8_skip "explore_$key" "exploratory: less than $((reserve + JC6_MIN + EXPLORE_MIN)) min to the deadline"
  fi
  if [ "$key" = qwen7 ] && ! on KEEP_CACHE && ! on TEST_MODE; then rm -rf "$SAE_DIR"; echo "$(s8_utc) $SAE_DIR removed" >> "$OUT/ENV.txt"; fi
  s8_drop "$key"
  return 0
}

# a fallback decided in an earlier session of this OUT stands (COMMIT.txt): the replaced model is not run again, so the
# fallback's outputs are never mixed with outputs of the replaced model from a later session
FALLBACK=$(grep -o 'fallback: yi9 replaces [A-Za-z0-9._-]*' "$OUT/COMMIT.txt" 2>/dev/null | head -n 1 | awk '{print $4}')
for i in "${!KEYS[@]}"; do
  key=${KEYS[$i]}
  next=${KEYS[$((i + 1))]:-qwen14}
  [ -n "$FALLBACK" ] && [ "$next" = "$FALLBACK" ] && next=yi9
  if ! on TEST_MODE && ! core_done "$next"; then s8_prefetch "$next"; fi   # background prefetch (the fetcher locks)
  if [ -n "$FALLBACK" ] && [ "$key" = "$FALLBACK" ]; then
    echo "$(s8_utc) $key was replaced by yi9 in an earlier session (COMMIT.txt); yi9 runs in its slot" | tee -a "$OUT/COMMIT.txt"
    run_key yi9 "$(reserve_after "$i")" || echo "FAILED the fallback yi9 (exit $?)" | tee -a "$OUT/FAILED.txt"
    continue
  fi
  rc=0; run_key "$key" "$(reserve_after "$i")" || rc=$?
  [ $rc = 0 ] && continue
  why=$([ $rc = 1 ] && echo "its files failed verification" || echo "its preflight failed")
  if [ "$key" = llama8 ] && [ -z "$FALLBACK" ] && ! has_output "$key"; then
    FALLBACK=$key   # entry rule G2: the one technical fallback for the new family, before any output of llama8 exists
    echo "$(s8_utc) fallback: yi9 replaces $key ($why; no output of $key exists)" | tee -a "$OUT/COMMIT.txt"
    run_key yi9 "$(reserve_after "$i")" || echo "FAILED the fallback yi9 (exit $?)" | tee -a "$OUT/FAILED.txt"
  else
    echo "FAILED $key: $why (no fallback applies)" | tee -a "$OUT/FAILED.txt"
  fi
done
wait   # a prefetch still running

# ---- J-C6 at Qwen2.5-14B (deadline-gated)
if s8_done jc6 || s8_time_left "$JC6_MIN"; then
  if dir=$(s8_fetch qwen14); then
    S8_OUTPUTS="$OUT/jc6/${TAGP}qwen14/exchange.json" s8_step jc6 $PY experiments/prakash_caa.py --model "$dir" --key qwen14 --out "$OUT"
    s8_drop qwen14
  else
    echo "FAILED fetch of qwen14 (J-C6 not run)" | tee -a "$OUT/FAILED.txt"
  fi
else
  s8_skip jc6 "less than $JC6_MIN min to the deadline"
fi

# ---- score, manifest, archive
s8_finish analysis/stage8c_score.py
