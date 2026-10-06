#!/usr/bin/env bash
# GPU stage 5 (preregistered as P-2026-10-05-G in docs/PREREGISTRATION.md; paper v3): what a later mention must be, and
# do, to read the writing token's key. Five parts on the seed-0 story cores, one per-model loop:
#   (a) re-mention attention (eager), experiments/remention_attention.py:    Qwen2.5-1.5B/3B/7B/14B (+EXTRA anchors)
#   (b) attention knockout, experiments/attention_knockout.py:              Qwen2.5-7B/14B, Mistral-7B (+EXTRA exploratory)
#   (c) membership and dose: format_factorial.py with the 13 subset arms (seeds 0 and 1) + row_restricted_keys.py splice
#   (d) non-identical re-mentions: the same seed-0 factorial with the 17 variant arms, form_competence.py, form_attention.py
#       (c, d) Qwen2.5-7B/14B, Mistral-7B, OLMo-2-7B (+EXTRA Qwen3-8B); seed 1 and the splice at the three primary models;
#       the seed-0 factorial always carries both the subset and the variant arms (one file per model serves both parts)
#   (e) IOI: ioi_factorial.py at GPT-2 small/XL (FP32), Qwen2.5-7B-Instruct, Mistral-7B, Qwen2.5-7B base, Qwen2.5-14B;
#       ioi_attention.py at GPT-2 small (its head labels); the AFTER row splice at the 7B pair
# Needs one 80 GB GPU (A100/H100) and >= 200 GB disk (the HF cache is cleaned between models). Runtime about 5 h.
# Usage:
#   git clone -b claude/paper2-research https://github.com/syedzahedi1990/Causal-keys.git && cd Causal-keys
#   G=$(git log --format=%H -1 --grep='^Finalise preregistration G') && [ -n "$G" ] && git checkout "$G"
#   bash scripts/gpu_stage5.sh
#   TEST_MODE=1 bash scripts/gpu_stage5.sh      # CPU plumbing test: pytest, then every part at Qwen2.5-0.5B (FP32) and GPT-2
#                                               # small into a scratch OUT, then the scorer; 18 min + pytest on a 4-core CPU
# Optional: PARTS=a,b,c,d,e (default all; a part not listed is skipped everywhere), ONLY=<comma-separated model names,
# e.g. Qwen2.5-7B-Instruct,gpt2> (the loop runs those models only; an unlisted name is an error), EXTRA=1 (Qwen3-8B, and
# the exploratory models of (a) and (b)), N=150, OUT=<dir>, KEEP_CACHE=1, TESTS=0 (skip pytest, 22 min on a 4-core CPU),
# PY=<python>, FORCE=1 (redo a step whose results file already exists in OUT; without it such a step is kept, which makes
# a rerun with the same OUT resume where it failed: a file left by a FAILED step, or one that is not readable JSON, is
# moved aside to <file>.failed.<UTC> and the step is run again; part (a) writes its .npz before its .json, so a readable
# .json certifies both), MINGIB=<GiB> (GPU memory floor, default 75; the 7B parts fit a 40 GB card), IOI_EXACT=no (rerun
# only: keep an ioi_factorial.py file whose FP32 batch-noise floor exceeds 1e-3 instead of failing the step; the floors and
# exact_violations stay in the file and the score, the default 'auto' asserts exactness in FP32 as the entry says). Each model's Hub revision is resolved before its steps, passed to every step as --revision
# (so every results file's provenance carries it) and appended to ENV.txt with the snapshot that was loaded.
# The output of a step is one results file; the scorer reads whatever OUT holds, so the archive of a partial rerun on a
# fresh clone holds only that part unless the earlier archive was unpacked into OUT first (docs/GPU_RUNBOOK.md).
# Outside TEST_MODE it pins transformers 5.18.0 (the stage-1/3b environment), refuses a DRAFT P-2026-10-05-G or modified
# tracked files, runs pytest and the preflights (ckeys.subsets.check_rules, the IOI entry facts) before any model.
# A failed step is recorded in $OUT/FAILED.txt and the pipeline goes on; exit status 1 if any step FAILED (the score step
# FAILS with exit 2 when the provenance or population check reports MISMATCH outside TEST_MODE; the score is still written).
# Output: gpu_stage5_results.tgz (TEST_MODE: $OUT.tgz), a few hundred MB (part (a) stores per-head attention arrays);
# the score is $OUT/STAGE5_SCORE.txt (analysis/stage5_score.py).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
PARTS=${PARTS:-a,b,c,d,e}; PARTS=${PARTS// /}; ONLY=${ONLY:-}; ONLY=${ONLY// /}
for p in ${PARTS//,/ }; do case $p in a|b|c|d|e) ;; *) echo "bad PARTS item '$p' (a,b,c,d,e)"; exit 2;; esac; done
has() { case ",$PARTS," in *",$1,"*) return 0;; esac; return 1; }
among() { local x=$1; shift; for y in "$@"; do [ "$x" = "$y" ] && return 0; done; return 1; }
selected() { [ -z "${ONLY:-}" ] && return 0; case ",$ONLY," in *",$1,"*|*",${1##*/},"*) return 0;; esac; return 1; }
SUB=S2,S3,S3out,S3half,S4,S4out,S6,L2,L3,L3out,L4,L4out,L6
VAR=POST_THE,POST_MODIF,POST_TITLE,POST_UPPER,POST_PLURAL,POST_SYN,POST_FRMIX,POST_DEMIX,POST_FR,POST_DE,POST_OTHER,POST_FR_OTHER,POST_DE_OTHER,AFTER_SYN,AFTER_FR,AFTER_DE,AFTER_OTHER
SPLICE=S2,S3,L2,L3,S3out,L3out,S6,L6
Q7=Qwen/Qwen2.5-7B-Instruct; Q14=Qwen/Qwen2.5-14B-Instruct; MI=mistralai/Mistral-7B-Instruct-v0.3; OL=allenai/OLMo-2-1124-7B-Instruct
Q15=Qwen/Qwen2.5-1.5B-Instruct; Q3=Qwen/Qwen2.5-3B-Instruct; Q3_8=Qwen/Qwen3-8B; Q7B=Qwen/Qwen2.5-7B
if [ -n "${TEST_MODE:-}" ]; then
  OUT=${OUT:-$(mktemp -d)/gpu_stage5}; TGZ=${TGZ:-$OUT.tgz}
  TINY=Qwen/Qwen2.5-0.5B-Instruct; DT=float32; TAG=TEST_${TINY##*/}
  export TEST_MODE=1 KEEP_CACHE=1   # the experiments then force Qwen2.5-0.5B, FP32, CPU and keep --n
  NA=${TEST_NA:-8}; NB=${TEST_NB:-2}; NC=${TEST_NC:-3}; NS=2; ND=${TEST_NC:-3}; NE=3; NP=2
  MODELS=(gpt2 "$TINY")
  A=("$TINY"); B=("$TINY"); CD=("$TINY"); C1=("$TINY"); CS=("$TINY"); D=("$TINY"); E=(gpt2 "$TINY"); EA=(gpt2); ES=("$TINY")
else
  OUT=${OUT:-results/gpu_stage5}; TGZ=${TGZ:-gpu_stage5_results.tgz}
  N=${N:-150}; DT=bfloat16; TAG=stage5
  NA=$N; NB=$N; NC=$N; NS=60; ND=60; NE=200; NP=200
  MODELS=(gpt2 gpt2-xl "$Q15" "$Q3" "$Q7" "$Q14" "$MI" "$OL" "$Q7B")
  A=("$Q15" "$Q3" "$Q7" "$Q14"); B=("$Q7" "$Q14" "$MI"); CD=("$Q7" "$Q14" "$MI" "$OL"); C1=("$Q7" "$Q14" "$MI"); CS=("$Q7" "$Q14" "$MI" "$OL")
  D=("$Q7" "$Q14" "$MI" "$OL"); E=(gpt2 gpt2-xl "$Q7" "$MI" "$Q7B" "$Q14"); EA=(gpt2); ES=("$Q7" "$MI")
  if [ -n "${EXTRA:-}" ]; then MODELS+=("$Q3_8"); A+=("$MI" "$OL" "$Q3_8"); B+=("$Q15" "$Q3" "$OL"); CD+=("$Q3_8"); CS+=("$Q3_8"); fi
fi
for o in ${ONLY//,/ }; do among "$o" "${MODELS[@]}" "${MODELS[@]##*/}" || { echo "bad ONLY item '$o' (not among ${MODELS[*]})"; exit 2; }; done
ARMS=NONE,POST,AFTER,$SUB,$VAR   # parts (c) and (d) share the seed-0 factorial file: both arm sets whenever either part runs
mkdir -p "$OUT"
echo "PARTS=$PARTS ONLY=${ONLY:-all} EXTRA=${EXTRA:-} FORCE=${FORCE:-} models: ${MODELS[*]}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {
  local name=$1 rc=0; shift
  echo "==================== $name  $(date -u +%H:%M:%S)"
  echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$OUT/log_$name.txt"
  PYTHONPATH=. "$@" >> "$OUT/log_$name.txt" 2>&1 || {
    rc=$?; local why="exit $rc"; [ $rc -gt 128 ] && why+=" (killed by signal $((rc - 128)); 9 = SIGKILL, often the host OOM killer)"
    echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) FAILED $why" >> "$OUT/log_$name.txt"
    echo "FAILED $name $why (see $OUT/log_$name.txt)" | tee -a "$OUT/FAILED.txt"; }
  tail -n 2 "$OUT/log_$name.txt"
  return $rc
}
aside() { local old="$1.failed.$(date -u +%Y%m%dT%H%M%SZ)"; mv "$1" "$old"; echo "moved aside $old: $2" | tee -a "$OUT/COMMIT.txt"; }
STEPS=0
keep() {  # keep <results file> <name> <command...>: a step whose results file exists and is readable JSON is kept unless
  local f=$1 name=$2 rc=0; shift   # FORCE=1; a file left by a FAILED step, or unreadable, is moved aside and the step run again
  STEPS=$((STEPS + 1))
  if [ -e "$f" ] && [ -z "${FORCE:-}" ]; then
    if $PY -c "import json, sys; json.load(open(sys.argv[1]))" "$f" 2>/dev/null; then echo "==================== $name kept: $f exists (FORCE=1 to redo)"; return 0; fi
    aside "$f" "not readable JSON (a step interrupted while writing); $name is run again"
  fi
  run "$@" || { rc=$?; [ -e "$f" ] && aside "$f" "left by the FAILED step $name (exit $rc); the scorer does not read it"; }
  return $rc
}
die() {  # fatal before the models: record, archive the logs, stop
  echo "FAILED $1" | tee -a "$OUT/FAILED.txt"
  tar czf "$TGZ" "$OUT"; echo "Results archive (logs): $TGZ"; exit 1
}
HUB=$($PY -c "from huggingface_hub.constants import HF_HUB_CACHE; print(HF_HUB_CACHE)" 2>/dev/null) || HUB=~/.cache/huggingface/hub   # honours HF_HOME / HF_HUB_CACHE
clean() { [ -z "${KEEP_CACHE:-}" ] && rm -rf "$HUB"; }
[ -f "$OUT/FAILED.txt" ] && mv "$OUT/FAILED.txt" "$OUT/FAILED.$(date -u +%Y%m%dT%H%M%SZ).txt"
{ echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) PARTS=$PARTS EXTRA=${EXTRA:-} TEST_MODE=${TEST_MODE:-}"; git rev-parse HEAD; git status --short; } | tee -a "$OUT/COMMIT.txt"
if [ -z "${TEST_MODE:-}" ]; then  # the preregistered code only: the finalised entry, no local changes
  awk '/^## /{f = ($0 ~ /P-2026-10-05-G/)} f && /DRAFT, not yet final/{d = 1} END{exit !d}' docs/PREREGISTRATION.md \
    && die "preregistration P-2026-10-05-G is still a DRAFT: check out the commit 'Finalise preregistration G' (docs/GPU_RUNBOOK.md)"
  # (awk decides alone: an awk | grep -q pipeline under pipefail gets status 141 when grep exits first, and the refusal is skipped)
  git rev-parse HEAD > /dev/null 2>&1 && [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || die "not a clean git checkout (modified tracked files above): run from a clean checkout of 'Finalise preregistration G'"
fi

# ---- software environment of stages 1 and 3b
LOGENV="$OUT/log_env.txt"
if [ -z "${TEST_MODE:-}" ]; then
  $PY -m pip install 'transformers==5.18.0' accelerate numpy pytest >> "$LOGENV" 2>&1; tail -n 1 "$LOGENV"
  $PY -c "import transformers as t; assert t.__version__ == '5.18.0', t.__version__" >> "$LOGENV" 2>&1 || die "transformers is not 5.18.0 (see $LOGENV)"
fi
$PY -m pip freeze 2>/dev/null > "$OUT/PIP_FREEZE.txt"
echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) PARTS=$PARTS ONLY=${ONLY:-all} TEST_MODE=${TEST_MODE:-} HF hub cache $HUB" >> "$OUT/ENV.txt"   # appended, like COMMIT.txt: a rerun keeps the earlier models' lines
$PY -c "import sys, torch, transformers, numpy; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__, 'gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" | tee -a "$OUT/ENV.txt"
command -v nvidia-smi > /dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >> "$OUT/ENV.txt"
MINGIB=${MINGIB:-75}
[ -n "${TEST_MODE:-}" ] || $PY -c "import torch; assert torch.cuda.device_count() >= 1 and torch.cuda.get_device_properties(0).total_memory / 2**30 >= $MINGIB" \
  || die "no CUDA device with >= $MINGIB GiB visible to torch (MINGIB=<GiB> lowers the floor for a rerun of the 7B parts)"

# ---- unit tests (FP32 exactness at Qwen2.5-0.5B and GPT-2 small) and preflights, before any 7B model
[ "${TESTS:-1}" = 0 ] || run pytest $PY -m pytest tests/ -q || die "pytest (see $OUT/log_pytest.txt)"
run check_rules $PY -c "from ckeys.subsets import check_rules; print(check_rules())" || die "check_rules (see $OUT/log_check_rules.txt)"
run entry_facts $PY analysis/ioi_entry_facts.py --out "$OUT/ENTRY_FACTS.preflight.txt" || die "entry_facts (see $OUT/log_entry_facts.txt)"
[ -f results/gpu_stage5/ioi/ENTRY_FACTS.txt ] \
  || die "results/gpu_stage5/ioi/ENTRY_FACTS.txt is missing: it belongs to the commit 'Finalise preregistration G' (the IOI facts quoted by part (e) of the entry)"
diff -q "$OUT/ENTRY_FACTS.preflight.txt" results/gpu_stage5/ioi/ENTRY_FACTS.txt > /dev/null \
  || die "the regenerated IOI entry facts differ from the committed results/gpu_stage5/ioi/ENTRY_FACTS.txt (the facts quoted by part (e) of the entry)"

# ---- per-model loop (each model downloaded once, used by every part that lists it, then the cache is cleaned)
for m in "${MODELS[@]}"; do
  selected "$m" || continue
  { has a && among "$m" "${A[@]}"; } || { has b && among "$m" "${B[@]}"; } || { { has c || has d; } && among "$m" "${CD[@]}"; } \
    || { has e && among "$m" "${E[@]}"; } || continue   # no step of PARTS lists this model: no revision or snapshot line in ENV.txt
  s=${m##*/}; echo "######## $m  $(date -u +%H:%M:%S)"
  case $m in gpt2*) EDT=float32;; *) EDT=$DT;; esac   # GPT-2 in FP32 (exactness asserted), the rest BF16
  # the Hub revision, resolved before the steps and pinned in every step (--revision) and results file; the loaded snapshot recorded after them
  REV=$($PY -c "from huggingface_hub import model_info; print(model_info('$m').sha)" 2>/dev/null) && [ -n "$REV" ] && RV="--revision $REV" || { REV=""; RV=""; }
  echo "$m revision ${REV:-unavailable (offline: the steps load the cached snapshot, recorded below)}" | tee -a "$OUT/ENV.txt"
  has a && among "$m" "${A[@]}" && keep "$OUT/attention/$s.json" "attention_$s" $PY experiments/remention_attention.py --model "$m" --n "$NA" --dtype "$DT" --out "$OUT/attention" $RV
  has b && among "$m" "${B[@]}" && keep "$OUT/knockout/${s}_s0.json" "knockout_$s" $PY experiments/attention_knockout.py --model "$m" --n "$NB" --dtype "$DT" --out "$OUT/knockout" $RV
  { has c || has d; } && among "$m" "${CD[@]}" && keep "$OUT/factorial/${s}_s0.json" "factorial_$s" $PY experiments/format_factorial.py --model "$m" --n "$NC" --seed 0 --dtype "$DT" \
      --arms "$ARMS" --arm-modules ckeys.subsets,ckeys.variants --out "$OUT/factorial" $RV
  has c && among "$m" "${C1[@]}" && keep "$OUT/factorial/${s}_s1.json" "factorial_s1_$s" $PY experiments/format_factorial.py --model "$m" --n "$NC" --seed 1 --dtype "$DT" \
      --arms "NONE,$SUB" --arm-modules ckeys.subsets --out "$OUT/factorial" $RV
  has c && among "$m" "${CS[@]}" && keep "$OUT/row_restricted/${s}_direct.json" "splice_$s" $PY experiments/row_restricted_keys.py --model "$m" --n "$NS" --dtype "$DT" --arms "$SPLICE" \
      --arm-modules ckeys.subsets --out "$OUT/row_restricted" $RV
  has d && among "$m" "${D[@]}" && keep "$OUT/competence/$s.json" "competence_$s" $PY experiments/form_competence.py --model "$m" --dtype "$DT" --out "$OUT/competence" $RV
  has d && among "$m" "${D[@]}" && keep "$OUT/form_attention/$s.json" "form_attention_$s" $PY experiments/form_attention.py --model "$m" --n "$ND" --dtype "$DT" --out "$OUT/form_attention" $RV
  has e && among "$m" "${E[@]}" && keep "$OUT/ioi/${s}_s0.json" "ioi_$s" $PY experiments/ioi_factorial.py --model "$m" --n "$NE" --dtype "$EDT" --out "$OUT/ioi" --assert-exact "${IOI_EXACT:-auto}" $RV
  has e && among "$m" "${EA[@]}" && keep "$OUT/ioi_attention/$s.json" "ioi_attention_$s" $PY experiments/ioi_attention.py --model "$m" --n "$NP" --dtype "$EDT" --out "$OUT/ioi_attention" $RV
  has e && among "$m" "${ES[@]}" && keep "$OUT/row_restricted/${s}_ioi.json" "ioi_splice_$s" $PY experiments/row_restricted_keys.py --task ioi --arms AFTER --model "$m" --n "$NS" --dtype "$DT" --out "$OUT/row_restricted" $RV
  echo "$m loaded snapshot(s): $(ls "$HUB/models--${m//\//--}/snapshots" 2>/dev/null | tr '\n' ' ')" >> "$OUT/ENV.txt"
  clean
done
[ "$STEPS" -gt 0 ] || die "no step selected: PARTS=$PARTS with ONLY=${ONLY:-all} names no model x part"

# ---- score (gates first, one verdict line per prediction G1-G22, provenance, population, then every part's report)
run score $PY analysis/stage5_score.py --root "$OUT" --tag "$TAG"
grep -E "^(GATES|  Gate |VERDICTS|  G[0-9]+ |SUMMARY|  commits:|  population:)" "$OUT/STAGE5_SCORE.txt" 2>/dev/null

tar czf "$TGZ" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' "$OUT"
echo "==================== DONE  $(date -u +%H:%M:%S)"
echo "Results archive: $(cd "$(dirname "$TGZ")" && pwd)/$(basename "$TGZ")"
if [ -f "$OUT/FAILED.txt" ]; then cat "$OUT/FAILED.txt"; exit 1; fi
echo "no failures (no step FAILED; gate and prediction verdicts above are results, see $OUT/STAGE5_SCORE.txt)"
