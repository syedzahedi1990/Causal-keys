# Stage 9 build spec: the listed candidates set the edit window (format x depth), and the key threshold

Binding build specification for the five stage-9 builders. Written 2026-10-14. It is frozen together with the prediction
note `docs/STAGE9_PREDICTIONS.md` in the commit titled exactly `Freeze stage-9 predictions` (section 12.1); the GPU
script refuses to run when either file, or the code, differs from that commit. Stage 9 is not a preregistration in the
sense of entries A-J of `docs/PREREGISTRATION.md`: there is no entry, no class or prior per line and no Holm family.
The prediction note fixes the predictions, thresholds, falsifiers and the primary analysis; this file fixes everything
the code does. Where the two disagree, the note decides what a prediction means and this file decides what is computed;
report the disagreement to the lead instead of resolving it in code.

Design source: the stage-9 design review (development notes outside the repository): its Direction A (format x depth
double dissociation, sections 3.1-3.5) and the GPU part of its Direction B (section 4.3). Its optional composition test
(4.4) and the KV-cache compression arm (5) are NOT built.

Contents: 1 Design in one page. 2 House rules. 3 File ownership and delivery order. 4 Models and sourcing.
5 Populations. 6 Formats and prompt builders. 7 Window rows (stories and passages). 8 Edit and dose arm.
9 Measures, parsers and generation. 10 Statistics. 11 Output layout (JSON schemas). 12 GPU script, guard, steps,
TEST_MODE. 13 Exactness tests (FP32) per builder. 14 The scorer. 15 Zero-GPU paper analyses (stage-9-pre group).
16 Compute estimate. 17 Decisions fixed here, and what is not built.

---

## 1. Design in one page

At the token that writes a value in context (the writing token, position p; for passages the answer span P), the exact
K/V of the source run S are written into the base run B's cache over a set of layers, and the generated answer is read.

| Arm | Population | Models | Formats | Rows (per item and format) |
|---|---|---|---|---|
| Stories | N9: 150 new fresh cores (seed 9101); C: the 30 stage-8b calibration cores (l_b only) | mistral7, llama8, qwen7 | NONE, BEFORE, POST-NULL, POST, AFTER, LETTER | none, none2, KV@0, late@l and early@l for every second block l, K@0, V@0, Klate@l and Vlate@l at l in {4, 8, 12, 16} |
| Passages | the 185 stage-8a E items (decisive) and the 71 R items (held-out check) | same | NOM, OPTB, OPTA (+ LETA, deadline-gated, E only) | none, none2, KV@0, late@l and early@l at l = round(x L), x in {0.15, 0.3, 0.45, 0.6}, K@0, V@0, Klate@l, Vlate@l at the same four l |
| Edits and dose | E9: 80 new canonical-lexicon cores satisfying Part C's cond_E (seed 9102) | same (E4b at qwen7, mistral7 only) | NONE, AFTER, LETTER, POST | blocks 3, 7, 11, 15: nat, E1 (single site), CAA (E1 at every position >= p), E4b; blocks 3 and 7 also: joint interpolation J at alpha in {0.25, 0.5, 0.65, 0.8, 0.9, 1.25} (alpha = 1 is nat), K-only and V-only at alpha in {0.5, 0.8} (+ the natural K-only and V-only rows), the cross rows K(alpha)+V and K+V(alpha), E1 at 0.5, 0.75, 1.5 x |

Measures: greedy generation (stories and edits 16 tokens with the candidate stop and a letter parser; passages 48 tokens to
newline/EOS, containment and strict match); flip rate (answer = S, or the edit's target t), stale rate (answer = B); the
six-way E-scored probabilities and the absolute p(B) from the trie scorer (stories, edits) or the decision-token
log-probabilities (passages) for the same rows; l_c per format; l_b from C; r_K, psi_K, psi_V for the edit arm;
core-cluster (stories, edits) or article-cluster (passages) bootstrap. The predictions that these numbers decide are in
`docs/STAGE9_PREDICTIONS.md`.

## 2. House rules (binding)

1. Do not `git commit` or push. Leave the working tree for the lead.
2. Write only the files you own (section 3). Every existing module is read-only: `ckeys/*.py` (except the new
   `ckeys/stage9.py`), the `experiments/*.py` of earlier stages, `analysis/*` of earlier stages, `scripts/stage8_common.sh`,
   `scripts/fetch_verified.py`, `scripts/stage8_models.json`, every existing test. Import from them. If you need other
   behaviour, wrap it in a module you own. A real bug in shared code: report it to the lead with a proposed patch.
3. Idioms (follow `experiments/fresh_factorial.py`, `experiments/stage8_edits.py`, `experiments/natural_factorial.py`,
   `analysis/stage8b_score.py`, `analysis/stage8c_score.py`, `scripts/gpu_stage8b.sh` and their tests):
   - atomic JSON writes (`experiments.stage6_heads.write_atomic`; a `.pt` file through a `.tmp` and `os.replace`);
   - a `provenance` block in every results file (section 11.1);
   - one process per stage, `--stage <name>`; `--test` or `TEST_MODE=1` switches to TEST_MODE (section 12.5);
   - exit status 3 = complete up to the deadline, parts not run (recorded in provenance; `s8_step` redoes the step in a
     later session); exit 1 = a check failed; anything else = failure;
   - BF16, sdpa attention (the three models), `use_cache=False` in every scoring pass, transformers 5.18.0;
   - model files only from `scripts/fetch_verified.py` via `s8_fetch` (manifest `scripts/stage8_models.json`); keys
     `mistral7`, `llama8`, `qwen7` only. HF_TOKEN may be unset: Llama-3.1-8B comes from the manifest's verified mirrors.
4. Exactness first: every new hooked row equals an independent reference in FP32 (1e-4 in log-probabilities, 1e-5 in
   K/V tables), tested before any model is loaded on the GPU (section 13).
5. Anonymity: refer to the authors' earlier work only as "Anonymous (2026)" (never by an ordinal name, as an earlier
   paper of ours, or through its code release's name), and write no personal names, no e-mail addresses, no scratch or
   branch paths and no name of any AI assistant in any file (code, comments, logs, docs). The scan of
   `scripts/make_anonymous_release.py` lists the patterns; run your files through it before handing them over. The
   study models are named as in the manifest.
6. Every number a file prints is computed from results files by code; nothing is typed in by hand (the prediction note's
   seen-before numbers are re-derived by `analysis/stage9_pre.py`, section 15).

## 3. File ownership and delivery order

| Builder | Writes (owns) | Imports read-only (main ones) |
|---|---|---|
| (a) stories | `ckeys/stage9.py`; `experiments/stage9_stories.py`; `tests/test_stage9_pops.py`; `tests/test_stage9_stories.py` | `ckeys.fresh`, `ckeys.encoding`, `ckeys.clamp`, `ckeys.surface`, `ckeys.generate`, `ckeys.edits` (cond_E, LETTER_FRAMES, `_take`, `pop_hash`), `ckeys.story`, `experiments.fresh_factorial` (`encode_item`, `Invalid`, `unpack`), `experiments.format_factorial.provenance`, `experiments.stage6_heads.write_atomic`, `experiments.ioi_factorial.device_name`; in tests also `experiments.stage8_flag` (populations) |
| (b) passages | `experiments/stage9_passages.py`; `tests/test_stage9_passages.py` | `ckeys.natural_formats`, `ckeys.natural_rows` (`prep`, `capture`, `tables`, `score_rows`, `generate_rows`, `valid_all`, `end_ids`, `cut`), `ckeys.squad_items`, `experiments.natural_factorial` (`load_items`, `split`, `window`, `sha_ids`), `ckeys.stage9` (grids, `run_meta`, sha helpers) |
| (c) edits and dose | `experiments/stage9_edits.py`; `tests/test_stage9_edits.py` | `ckeys.edits`, `ckeys.interventions` (`blocks`, `hooks`, `_out_tensor`, `_with_tensor`, `fixed_subspace_interchange`), `ckeys.clamp`, `ckeys.surface`, `ckeys.generate`, `ckeys.stage9`, `experiments.stage8_edits` (`natural_pass`, `edit_pass`, `split_tables`, `NAT_ROWS`) |
| (d) scorer and pipeline | `analysis/stage9_score.py` (and optionally `analysis/stage9_parts/*.py`); `tests/test_stage9_score.py`; `tests/test_stage9_guard.py`; `scripts/gpu_stage9.sh`; `scripts/stage9_common.sh` (the stage-9 guard and init); three lines appended to `.gitignore`; a "Stage 9" section appended to `docs/GPU_RUNBOOK.md` | `analysis/stage8b_parts/stats.py` (`Boot`), `scripts/stage8_common.sh` (sourced for its helpers; `s8_init` and `s8_finish` are NOT called), `scripts/fetch_verified.py` |
| (e) paper pre-analyses | `analysis/stage9_pre.py`; `tests/test_stage9_pre.py`; `paper/make_figures.py` (only a new "stage-9-pre" block placed before the final numbers.tex write, plus the one adjustment of the stage-8 macro-group check named in 15.4); `paper/figures/fig_window_pre.pdf`; `paper/numbers.tex` (regenerated, new S9P macros only); `docs/stage9/S9P_MACROS.md` | `results/gpu_stage5/knockout`, `results/gpu_stage8a`, `results/gpu_stage8b`, `results/gpu_stage8c`, `results/gpu_stage6` (through make_figures' existing `SW`), the stage-8 scorer parts |

The spec writer owns `docs/stage9/SPEC.md` and `docs/STAGE9_PREDICTIONS.md`; the lead owns every commit.

**Delivery order.** (a) delivers `ckeys/stage9.py` first (its API is fixed in 5.5 and must not change). Until it lands,
(b) and (c) code against the API in 5.5 and (d) builds its synthetic fixtures from section 11. The pins in 5.2 and 5.4 let
every builder check its own copy of the population functions against the same hashes.

## 4. Models and sourcing

| Key | Model | Revision (manifest) | L (blocks) | Attention | Arms |
|---|---|---|---|---|---|
| `mistral7` | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 | 32 | sdpa | all; E4b |
| `llama8` | meta-llama/Llama-3.1-8B-Instruct (verified mirror when HF_TOKEN is unset) | 0e9e39f249a16976918f6564b8830bc894c89659 | 32 | sdpa | all; no E4b (no stored basis) |
| `qwen7` | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 | 28 | sdpa | all; E4b |

Order of the run: mistral7, llama8, qwen7 (the two early-horizon models first). No fallback model: a key whose files
are refused is recorded in `FETCH_FAILED.txt` and its predictions are NOT EVALUABLE. Chat wrapper:
`ckeys.encoding.chat_text` (system "You are a helpful assistant.", generation prompt, thinking disabled, assistant
prefill "Answer:").

Stage-8 artefacts reused (committed; each sha256 is asserted at load and recorded in provenance):

| File | Use | sha256 |
|---|---|---|
| `results/gpu_stage8b/frames/mistral7.json` | stories E frames (discovered: none) | 9434618bf9d10464c1daafa9fe8b6fd45cb11596b8cb668e75eebdfdb6ccb953 |
| `results/gpu_stage8b/frames/llama8.json` | same | da3daabc1b59219f4484fee43a2c81e19dbbc4419ad4bc622d3206fe94570486 |
| `results/gpu_stage8b/frames/qwen7.json` | same | ae18b810014aed61658ed99b73c9cd429e144ca8a65a3135502fe213a96a3f4a |
| `results/gpu_stage8b/g3/mistral7.json` | the trie passed J-B-G3 (`"scorer": "trie"`; the scorer checks it) | dc70f763b0d8f36c65fa7cc24a833a6f2beb2e65d32dee6dce518b48aed67261 |
| `results/gpu_stage8b/g3/llama8.json` | same | da2d4c8a88cf1d394519cf1e57d60751cc13fe88ed9342010e069d651f4431b7 |
| `results/gpu_stage8b/g3/qwen7.json` | same | 9b0810b4b73f65d5ba54fb4a86eec06b9c0533375e285296c49fac816778d08e |
| `results/gpu_stage8a/frames/mistral7.json` | passage answer frame (" ") | e3b5491224639310f539aa19638a81219a33f9985ac99da2492b7dc810919290 |
| `results/gpu_stage8a/frames/llama8.json` | same (" ") | 10a02e134ee9fecf84bbffcd2a83e0c2c529678c90a9eb122d5beed7c8228e75 |
| `results/gpu_stage8a/frames/qwen7.json` | same (" ") | cb8d855f94db4179060fc1ab8b6eca80e8079ed09453469b86b91925f43fa86c |
| `results/gpu_stage8a/factorial/<key>.json` | `provenance.population_sha256` of the valid E items (all three: 185 valid, `57fb5fa1556dc807...`) | (read, not pinned) |
| `results/gpu_stage8c/calib/qwen7.pt` | E1 means `mu1[l]` [6, D], l in {3, 7, 11, 15} | 5b379fcb2ab35a9de1a4012d77ff450d64a43bb321a290d6b1878ce9da218b07 |
| `results/gpu_stage8c/calib/mistral7.pt` | same | 351d52871ec83c04738072c2a3dfddf4ef13697776410fcb2496fe119434cb1b |
| `results/gpu_stage8c/calib/llama8.pt` | same | 26085156a0f1ff6c657a1bb3c647675586c16d6740402166223a48fc10565bcb |
| `results/gpu_stage8c/calib/<key>.json` | discovered frames (none at all three) | (read; its `calib_pt_sha256` must equal the .pt pin) |
| `results/gpu_stage8c/das/qwen7.pt` | E4b bases `{(l, 102): U [16, 3584]}` FP32, l in {3, 7, 11, 15} | aab335615a7d75bc74331a6681c79cff2241d94ea0c57a859c28fa94df80cc00 |
| `results/gpu_stage8c/das/mistral7.pt` | E4b bases `{(l, 102): U [16, 4096]}` FP32 | 0c2069fd7180d9aed727c60ab30579c281f01db115007b10a311df02caa3f912 |
| `results/gpu_stage8a/squad/dev-v1.1.json` | SQuAD v1.1 dev (default `--squad`; `ckeys.squad_items.load_squad` asserts its sha256) | 95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9 |

The das `.pt` files are dicts keyed by `(block, seed)` with seeds 101 (PCA init, "E4a") and 102 (random init, "E4b");
stage 9 uses seed 102 only. Their stage-8c fits: THOLD flip 1.00 / 1.00 / 1.00 / 0.97 (qwen7, blocks 3/7/11/15) and
1.00 / 1.00 / 0.97 / 0.84 (mistral7).

## 5. Populations

### 5.1 The exclusion set X9

A story core is compared by its full tuple `ckeys.fresh.FIELDS` = (agent, other, object, distractor, initial,
distractor_location, base, source), always in the canonical lexicon (`ckeys.story.LOCATIONS`).

    X9 = U  ∪  ⋃_{s ∈ STREAM_SEEDS} { tuple(c) : c ∈ make_cores(3000, Random(s)) }
    STREAM_SEEDS = (7, 8, 9, 99, 101, 202, 20261011, 20261013, 20261014, 8101, 8102, 8103, 8104, 81, 84)

U = `ckeys.fresh.used_cores()` (seeds 0-3, 1000 each; 3,981 tuples, sha256 abd1f053...). The streams cover every seed
that ever drew story cores: the pilot seeds of stage 8 (7, 8, 9, 99, 101, 202, 20261011), Part B's F and C (20261013,
20261014), Part C's E, H and TSET/THOLD (8101-8103; TSET/THOLD used 1,410 draws; 8104 seeded only Part C's random
directions and is excluded as Part D excluded it) and Part D's E8 and BIND (81, 84). `make_cores(3000, rng)` equals 3000
successive `make_cores(1, rng)` draws, so every core any of those streams produced in its first 3000 draws is excluded.
|X9| = 45,832 tuples (U 3,981, streams 42,296, overlap 445). Every
stage-8 story population is a subset of X9 (asserted by `tests/test_stage9_pops.py` from the functions the stage-8 GPU
steps call: `fresh.population("F"|"C"|"S0")`, `edits.populations()`, `experiments.stage8_flag.populations()` R, R', E8,
BIND). The IOI lists (F_IOI_CAND, E_IOI_CAND) and the task cores (XFIT/XEVAL paint, schedule) are cores of other
templates (`ckeys.ioi`, `ckeys.tasks`); a story core cannot equal one, and the test asserts the type difference by
showing the tuple sets are disjoint.

### 5.2 N9: 150 new fresh story cores (stories arm)

- Rule: `N9 = ckeys.fresh.fresh_cores(150, 9101, X9)` (the first 150 cores of the `make_cores(1, Random(9101))` stream
  in neither U nor X9 nor earlier in the stream). 161 draws.
- Attributes as F: `ckeys.fresh.attributes(N9)`: lexicon 2 for the half with the smallest core hash (75 / 75), sentence
  `SENTENCES[rank % 8]` (19, 19, 19, 19, 19, 19, 18, 18), order seeded by the hash. Items:
  `fresh.Item("N9", i, core, lex, order, sent)`.
- Pins: `fresh.pop_hash(items)` = **0cf411ab8fe99ec495adfb30944fad69c6b65964fabde849b848cc6358d38da3**; the tuples alone
  (`sha256(json.dumps([list(tuple) for core in N9]))`) = 5b87f7732bd94b7143f3dd94fe65d58cb59884e0fe0e5fe4b3432a765500a2aa.
- Coverage: all 30 ordered (base, source) pairs and all 20 objects; 53 clusters (lexicon, base, source).
- First two tuples (for debugging): (Alice, Bob, ring, phone, drawer, drawer, basket, box) with attributes
  (lex 1, sentence 7, order (5, 0, 4, 1, 3, 2)); (Bob, Alice, book, watch, closet, closet, shelf, box) with
  (2, 5, (0, 4, 2, 3, 5, 1)).

### 5.3 C: the 30 stage-8b calibration cores (l_b only)

`ckeys.fresh.population("C")` (pop_hash a625fd13..., pinned in `ckeys/fresh.py`). Stage 8b used C only for frame
discovery (clean-run generations, no clamp rows), so no window row was ever seen on C. C is used for one thing: fixing
l_b per model before any N9 item is scored (section 9.6). C enters no prediction statistic.

### 5.4 E9: 80 new canonical-lexicon cores (edit and dose arm)

- Rule: `E9 = ckeys.edits._take(9102, 80, U ∪ X9 ∪ tuples(N9), ckeys.edits.cond_E)`: the first 80 cores of the
  `make_cores(1, Random(9102))` stream in none of those sets, not repeated, with π(S) ≠ B and π(X) ≠ B (`cond_E`;
  X = `story.pick_x`, π = `story.PAIR_SWAP`). 173 draws; 22 of the 24 admissible ordered (base, source) clusters.
- Pin: `ckeys.edits.pop_hash(E9)` = **e4daeebd5227804ed0931302ab93d72b42b2a2d7af6222d922d8e555945c2453**.
- Rendering: canonical lexicon, canonical order, ROOM sentence, i.e. `fresh.Item("E9", i, core, 1, (0, 1, 2, 3, 4, 5),
  None)`, whose `fresh.raw_prompt` equals `ckeys.encoding.raw_prompt` byte for byte (asserted for every E9 core and
  format). The edit vectors are defined on this lexicon (E1 means and E4b bases were fitted on it).
- First two tuples: (Alice, Bob, card, hat, box, box, shelf, cabinet); (Alice, Bob, card, bag, cabinet, basket, box, shelf).

Seeds 9101 and 9102 appear nowhere in the repository's code, pilots or results before stage 9, are not stage-8 pilot
seeds, and are not U seeds. The tokenizer check of 2026-10-14 (spec writer, cached tokenizers of the three models): every
N9 and C core is valid in all six story formats (B, S, X prompts of one length differing at one position), every E9 core
is valid in NONE, AFTER, LETTER, POST and P1 with the B, S, X, π(S), π(X) prompts sharing one p and an identical prefix
through p (`ckeys.edits.prep` passes), and every form set builds: 0 skipped items.

### 5.5 `ckeys/stage9.py` (builder a; fixed API)

```python
SEEDS = {"N9": 9101, "E9": 9102}
STREAM_SEEDS = (7, 8, 9, 99, 101, 202, 20261011, 20261013, 20261014, 8101, 8102, 8103, 8104, 81, 84)
N_STREAM = 3000
POP_SHA256 = {"N9": "0cf411ab8fe99ec495adfb30944fad69c6b65964fabde849b848cc6358d38da3",          # fresh.pop_hash(items)
              "N9_tuples": "5b87f7732bd94b7143f3dd94fe65d58cb59884e0fe0e5fe4b3432a765500a2aa",
              "E9": "e4daeebd5227804ed0931302ab93d72b42b2a2d7af6222d922d8e555945c2453",          # edits.pop_hash(cores)
              "C": fresh.POP_SHA256["C"]}
STORY_FORMATS = ("NONE", "BEFORE", "POST-NULL", "POST", "AFTER", "LETTER")   # run order
EDIT_FORMATS = ("NONE", "AFTER", "LETTER", "POST")
PASSAGE_FORMATS = ("NOM", "OPTB", "OPTA")            # LETA: the deadline-gated extra step
EDIT_BLOCKS, DOSE_BLOCKS = (3, 7, 11, 15), (3, 7)
KV_ONLY_ONSETS = (4, 8, 12, 16)                      # stories: the stage-8c onsets l + 1 of blocks 3, 7, 11, 15
PASSAGE_X = (0.15, 0.3, 0.45, 0.6)
BOOT_SEED, NB = 20261015, 10000

def excluded() -> frozenset            # X9 (cached)
def story_cores() -> list[dict]        # N9 cores, drawn order
def stories() -> tuple                 # N9 fresh.Items (hash asserted)
def calibration() -> tuple             # fresh.population("C") (hash asserted)
def edit_cores() -> list[dict]         # E9 cores (hash asserted)
def edit_items() -> tuple              # E9 as fresh.Items (lex 1, canonical order, sent None)
def story_onsets(nL: int) -> list[int] # list(range(2, nL - 1, 2)): 32 -> 2..30 (15), 28 -> 2..26 (13), 24 -> 2..22 (11)
def passage_onsets(nL: int) -> list[int]  # [round(x * nL) for x in PASSAGE_X]: 32 -> [5, 10, 14, 19]; 28 -> [4, 8, 13, 17]; 24 -> [4, 7, 11, 14]
def parse_letter(text: str, it) -> tuple[str, str]   # (word or "other", how in {"letter", "word", "none"}), 9.2
def letter_stop(tok, it)               # stop(r, ids) for ckeys.generate.greedy under LETTER, 9.2
def file_sha(path) -> str
def run_meta() -> dict                 # 11.1: git commit, dirty flag, freeze commit, note/spec sha256, UTC time
```

Python's `round` is used (banker's rounding never applies to these products). The functions are pure apart from the
cached exclusion set; no torch import is needed except where an Item is built.

### 5.6 Passages

`data/stage8a_items.json` (sha256 40874aa1...; passages from the SQuAD file, `experiments.natural_factorial.load_items`).
- E: the 185 items of split E that are valid for the model's tokenizer and frame in every one of the six stage-8a formats
  (`ckeys.natural_rows.valid_all(tok, it, ckeys.natural_formats.FORMATS, frame)`), the stage-8a rule. All 185 were valid
  at the three models in stage 8a; the stage-9 file records `sha_ids(valid ids)` and the scorer compares it with
  `results/gpu_stage8a/factorial/<key>.json` `provenance.population_sha256` (a difference is a MISMATCH).
- R: the 71 items of split R, the same validity rule, recorded the same way (R was used in stage 8a only for the frame
  (first 30) and the heads' ranking; no window or K/V row was ever run on it). R is a held-out check: it is scored and
  reported, it decides nothing.
- Clusters: the article (`it["art"]`): E 34 articles, R 12.

## 6. Formats and prompt builders

| Arm | Format | Builder (exact call) | Answer alphabet |
|---|---|---|---|
| Stories | NONE, BEFORE, POST-NULL, POST, AFTER, LETTER | `ckeys.encoding.encode(tok, ckeys.fresh.raw_prompt(it, fmt, loc))`; the B, S (and X) ids and p via `experiments.fresh_factorial.encode_item(tok, it, fmt)` | the item's six words; LETTER: letters A-F in the item's order |
| Edits | NONE, AFTER, LETTER, POST | `ckeys.edits.prompt_ids(tok, core, fmt, loc)` (= `encode(raw_prompt(fmt, story, query))`, canonical lexicon, order and ROOM) | LOCATIONS; LETTER: letters A-F in canonical order |
| Passages | NOM, OPTB, OPTA, LETA | `ckeys.natural_formats.encode_item(tok, it, fmt, ent)` through `ckeys.natural_rows.prep(tok, it, fmt, frame)` | entities B, S, X, Z, D; LETA: letters A-D |

Texts (for reference): AFTER = PREFIX + story + "\nQuestion: " + q + "\nChoices: <six in order>\nAnswer with one
word.\nAnswer:"; BEFORE = "Read the story and answer the question.\nChoices: ...\n\nStory: " + story + ... + "Answer with
one word."; NONE without the list; POST / POST-NULL: the core's sentence (null nouns) after the story; LETTER: "Choices:
A) w, B) w, ..." + "\nAnswer with one letter.". Stories' AFTER is stage 8b's AFTER ("one word"), not stage 8c's P1
("exactly one choice"); the in-hand onset curves of stage 8c are P1 (reported as such). OPTB has OPTA's answer space
with the options before the passage; NOM is free form ("Answer with the exact words from the passage.").

Per-format captures: BEFORE (and OPTB) changes the prefix, so the S and B K/V at p (at P) are captured in each format's
own prompts; never reuse another format's tables.

Edits: p and the prefix through p must be identical in NONE, AFTER, LETTER and POST for B, S, X, π(S), π(X) (assert per
core, as `edits.prep` does for its four formats; builder (c) writes `prep9` with `EDIT_FORMATS`).

## 7. Window rows (stories and passages)

### 7.1 Definition

Every row is on the B prompt and overwrites K and V at p (passages: at every position of P) in **every** layer: in each
layer and channel it writes the captured table of a donor run, B's own where the donor is B (identity) and S's where
it is S. `donor(row, l', ch)` for layer l' and channel ch in {k, v}:

| Row | donor = S when | Purpose |
|---|---|---|
| `none`, `none2` | never | identity; two copies (first and last row of the batch): the BF16 floor |
| `KV@0` | always | the full patch (= the S run exactly; control "listed answers are sticky") |
| `late@l` | l' >= l | late window [l, L) |
| `early@l` | l' < l | early window [0, l) |
| `K@0` / `V@0` | ch = k / ch = v | single channel, every layer (stage 8b's K_S@0 / V_S@0) |
| `Klate@l` / `Vlate@l` | ch = k and l' >= l / ch = v and l' >= l | single-channel late windows (diagnostic) |

Onsets: stories `late@l`, `early@l` for l in `story_onsets(L)` (every second block from 2 to L - 2; with `KV@0` = late@0
and `none` = early@0 the grid starts at 0); `Klate@l`, `Vlate@l` for l in `KV_ONLY_ONSETS` = (4, 8, 12, 16).
Passages: `late@l`, `early@l`, `Klate@l`, `Vlate@l` for l in `passage_onsets(L)`.

Row order (stored in the file's `"rows"` list and used for every item): `none`, `KV@0`, the late rows by l, the early
rows by l, `K@0`, `V@0`, the Klate rows by l, the Vlate rows by l, `none2`. Row counts per item and format: stories 43
(mistral7, llama8), 39 (qwen7); passages 21.

### 7.2 Tables and clamps

- Capture: one forward of the B and S prompts (stories: B, S, X as `fresh_factorial.run_item` does; passages: B, S, X, Z as
  `natural_rows.capture`) with `ckeys.clamp.capture_kv(model, [p] (or P), range(L))`.
- Tables: `ckeys.clamp.stack_rows(kv, [lambda l, ch, r=r: donor(r, l, ch) for r in rows], range(L))` (stories) or a
  duck-typed row object with `.donor(l, ch, i)` passed to `ckeys.natural_rows.tables` (passages; `i` the span index,
  ignored). Clamp with `ckeys.clamp.clamp_kv(model, positions, tables, range(L))`.
- Complementarity (FP32-tested, 13.1): writing S on [0, l) and S on [l, L) by two nested `clamp_kv` contexts equals
  `KV@0`; `late@0` equals `KV@0`; `early@L` equals `none` equals the clean B run; `early@l` equals the nesting of an S clamp
  on [0, l) and a B clamp on [l, L). Mixed-depth states are therefore exact table assignments, not broken states.

## 8. Edit and dose arm

### 8.1 Natural pass, vectors and tables (per E9 core)

- `d = prep9(tok, core)`: ids of every format for B, S, X, π(S), π(X); p; the shared prefix (section 6).
- Natural prefix pass: `experiments.stage8_edits.natural_pass(model, d, EDIT_BLOCKS, L)` -> residuals h_v,l(p) at the
  block-l outputs for v in (B, S, X, πS, πX) and K/V at p from block 4 (= min block + 1).
- Targets t in {S, X} (both, as stage 8c; iS, iX, iB are LOCATIONS indices).
- Residual edits written at (p, l) (output of block l), FP32 vectors, applied as tables from one edit prefix pass per
  block (`experiments.stage8_edits.edit_pass(model, d, l, vecs, range(l + 1, L))`: forwards of the natural pass's shape,
  5 rows, last one padded):
  - `E1|t` = h_B + mu1[l][t] − mu1[l][B]; `E1x{c}|t` = h_B + c (mu1[l][t] − mu1[l][B]) for c in (0.5, 0.75, 1.5)
    (dose blocks only). mu1 from `results/gpu_stage8c/calib/<key>.pt` (`torch.load(...)["mu1"][l]`, [6, D] FP32,
    LOCATIONS order; the pinned sha256 asserted before load).
  - `E4b|t` (qwen7, mistral7) = `fixed_subspace_interchange(U)(h_B[None], h_{π(t)}[None])[0]` with
    U = `torch.load(das_pt)[(l, 102)]` ([16, D] FP32; the pinned sha256 asserted). π(t) is the source because π is an
    involution and the remap was trained to write π(source).
  - `T|S` = h_S,l(p) (pipeline check only): its tables must equal the natural S tables (`max |Δ|` recorded per block in
    `T_check`; 0 expected in BF16 because the edit pass has the natural pass's shape).
- Table rows (K/V at p in layers >= l + 1 only; below, p's K/V are B's natural ones), Δ = t − B per layer and channel:

| Row (at block l) | K table | V table | Blocks |
|---|---|---|---|
| `self`, `self2` | B | B | all |
| `nat|t` | t | t | all |
| `E1|t`, `E4b|t` | the edit's | the edit's | all (E4b: qwen7, mistral7) |
| `E1x0.5|t`, `E1x0.75|t`, `E1x1.5|t` | the edit's | the edit's | 3, 7 |
| `J{a}|t`, a in 0.25, 0.5, 0.65, 0.8, 0.9, 1.25 | B + a Δ_K | B + a Δ_V | 3, 7 (`nat` is J1) |
| `K1|t` / `V1|t` | t / B | B / t | 3, 7 |
| `K{a}|t` / `V{a}|t`, a in 0.5, 0.8 | B + a Δ_K / B | B / B + a Δ_V | 3, 7 |
| `K{a}V1|t` / `K1V{a}|t`, a in 0.5, 0.8 | B + a Δ_K / t | t / B + a Δ_V | 3, 7 |

  Interpolated tables: `ckeys.edits.interp`. Names print a with Python's `str(float)`: `J0.25`, `J0.5`, `J0.65`, `J0.8`,
  `J0.9`, `J1.25`, `K0.5`, `V0.8`, `K0.5V1`, `K1V0.8`.
- The all-position CAA row `CAA|t` (every block): not a table row. A forward hook on `blocks(model)[l]` adds
  `vec[r] = mu1[l][t] − mu1[l][B]` (per batch row; zero for the leading `self` row) to the block output at every sequence
  index >= p when the forward's length exceeds p (the prompt pass, the trie pass, the cache-free reference), and at every
  index when it does not (a cached one-token decode step, whose absolute position is > p). At index p this is exactly the
  single-site E1. The CAA batch (rows `self`, `CAA|S`, `CAA|X`) is scored and generated separately from the table rows.

Rows per core and format: blocks 11 and 15: 8 table rows (6 at llama8) + the 3-row CAA batch; blocks 3 and 7: 46 table
rows (44 at llama8) + the CAA batch; about 120 rows per core and format in all. Each cell's table rows are scored in ONE
forward led by `self` (assert the count is at most the chunk, 64), so every row and its `self` share a batch.

### 8.2 Per-row table statistics (`stats`)

For every table row Z and target t at block l, `ckeys.edits.kv_stats(tab_Z, tab_t, tab_B, range(l + 1, L))` (all KV
heads and layers >= l + 1 pooled): `proj_k`, `proj_v` (the projections ⟨Δ_Z, Δ_t⟩ / ⟨Δ_t, Δ_t⟩), `cos_k`, `cos_v`,
`ratio_k`, `ratio_v`, `nu`; and for residual edits `cos_resid`, `ratio_resid` (the residual displacement against
h_t − h_B at (p, l)). **The delivered key ratio r_K = proj_k** (for J{a}: a exactly; K{a}: a; V{a}: 0; CAA: E1's, because
its K/V at p are E1's). Recorded in `stats[str(l)][name]` with `name` the row name without the format.

## 9. Measures, parsers and generation

### 9.1 Trie scoring (stories, edits)

`ckeys.surface.score(model, ids, fs)` under the row clamps, one forward over prompt + trie (G3 chose the trie at all three
models). Word formats: `fs = ckeys.fresh.form_set(tok, it, frames)[0]` (stories; frames from the stage-8b frames file,
none discovered) or `ckeys.edits.form_set(tok, fmt, core, frames)` (edits; frames from the stage-8c calib json, none);
the 32 fixed E frames (`FRAMES_E_FIXED`) with the item's names. LETTER: `FormSet(tok, LETTERS, {"E": edits.LETTER_FRAMES})`
(forms " X", "X", " **X", "**X"). Stored per row: `E` = the six log-probabilities summed over forms, `L` = log p(" w")
(letters: log p(" X")), both as six-vectors in **canonical word order**: index i is `it.words[i]` (edits: `LOCATIONS[i]`);
under LETTER the entry for word i is the score of the letter that names it in the listing, `LETTERS[it.order.index(i)]`
(edits: letter i). The scorer derives: six-way p = softmax(E); absolute p(w) = exp(E_w) (not renormalised); E-argmax.

### 9.2 Generation (stories, edits)

`ckeys.generate.greedy(model, tok, ids, max_new=16, stop=...)` with the row clamps active (stories: one batch with the
rows on the B prompt and the clean S prompt as an unclamped extra row, `clamp_kv(..., per_row=mask)`; edits: one batch per
cell for the table rows, one for the CAA rows). A row ends at EOS or a chat end-of-turn token, at a token containing a
newline (excluded), at 16 tokens, or by the stop rule:
- word formats: `ckeys.fresh.stop_fn(tok, it.words)` (edits: the same with LOCATIONS);
- LETTER: `stage9.letter_stop(tok, it)`: with s = the decoded text after `lstrip()` and then the skip of leading
  markdown, brackets and quotes (the skip set of `ckeys.natural_formats.letter_of`: asterisk, underscore, backtick, `(`,
  `[`, double and single quote, space), stop when `len(s) >= 2`, no U+FFFD in `s[:2]`, `s[0] in "ABCDEF"` and
  `not s[1].isalpha()`; or when the word stop fires.
Text: `ckeys.fresh.gen_text(tok, ids, gen)`. Answer:
- word formats: `ckeys.fresh.parse(text, it.words)` (first candidate named, lower case, plurals allowed) or "other";
- LETTER: `stage9.parse_letter(text, it)`: s as above; if `s[0] in "ABCDEF"` and (`len(s) == 1` or `not s[1].isalpha()`):
  the word `it.words[it.order["ABCDEF".index(s[0])]]`, how "letter"; else `fresh.parse` (how "word" if a candidate is
  named, else "none", answer "other").

### 9.3 Passages: decision tokens and generation

- Teacher-forced scoring: `ckeys.natural_rows.score_rows(model, d, kv, rows, L, dev, "S")` with the window rows (duck
  typed): per row the decision-token log-probabilities `lp` of B, S, X, Z, D (LETA: the letters of B, S, X, D) after the
  frame tokens w, the option mass, the full-vocabulary argmax. Absolute p(B) = exp(lp_B).
- Generation: `ckeys.natural_rows.generate_rows(model, tok, it, d, kv, rows, L, max_new=48, extra={"S_run": "S"},
  stop=fmt in LETTER_FORMATS)`: no W + 2-word cut in the entity formats (rows end at EOS or end-of-turn, at a newline, or
  at 48 tokens; the text is read up to the first special token); LETA keeps `decided_stop`'s two-character letter rule.
- Two answer classes per row (stored by builder b):
  - `who_strict` = `ckeys.natural_formats.answer_of(text, it, fmt)` (normalised first line equals the entity or starts
    with it followed by a space; LETA: `letter_of`);
  - `who_contain` (entity formats): with g = `norm(first line)` and e_Y = `norm(entity Y)` for Y in (B, S, X, Z, D), the
    entities whose e_Y occurs in g as a whole-word substring (`re.search(r"(?<!\S)" + re.escape(e_Y) + r"(?!\S)", g)`);
    the one with the smallest start index; ties to the longer e_Y; none -> "other". LETA: equal to `who_strict`.
  - **Primary is containment**; strict is the sensitivity analysis (both printed by the scorer).

### 9.4 Competence (per item and format)

Stories: the `none` row's answer is B and the clean S run's answer is S. Edits: the `self` row's answer is B, and, per
target, the `nat|t` row at block 3 under NONE answers t (reported; the edit arm's flips use all cores, competence is
reported). Passages: `who_contain(none) = B` and `who_contain(S_run) = S` (LETA: the letters). A format's statistics use
its competent items (C9(f)); the share competent is reported per format. In stories `KV@0` reproduces the S run, so
`flip(KV@0)` on competent items is ~1.00 by construction; on all items it is the control "KV@0 = 1.00".

### 9.5 Flip, stale, confidence, inconsistency

- flip(f, row) = mean over C9(f) of 1[answer = S] (edits: answer = t, mean over (core, t)); stale = 1[answer = B].
- E-argmax flip: the same with the E-argmax in place of the generated answer (reported; agreement printed).
- Silent failure: among C9(f) items with answer ≠ S, the share with answer = B; the median absolute p(B) = exp(E_B) of the
  items whose answer is B; the six-way p(B) likewise (reported).
- Inconsistency at onset l: share of items competent in NONE and AFTER whose NONE `late@l` answer is S while the AFTER
  `late@l` answer is B.

### 9.6 l_c and l_b

- l_c(f) = the smallest l in `story_onsets(L)` (ascending) with flip(f, `late@l`) < 0.5 on C9(f) of N9; if none,
  l_c(f) = L (censored; printed as "> last onset"). Reported as l_c / L. Recomputed in every bootstrap resample.
- l_b = the smallest l in `story_onsets(L)` with flip(AFTER, `late@l`) <= 0.2 over all 30 C cores (no competence filter);
  None when no onset qualifies (then every statistic at l_b is NOT EVALUABLE in that model). Written to `lb/<tag>.json` by
  the stories `cal` stage before the `eval` stage starts; the eval stage reads it, records its sha256 and its value, and
  the scorer recomputes l_b from the stored C rows and checks equality, file order and that the eval read the same file.
  l_b is never computed from N9.

### 9.7 Edit-arm measures

- flip_Z(f, l) over (core, t) from the generated answers; transfer ratio τ_Z(l) = flip_Z(AFTER, l) / flip_Z(NONE, l).
- ID contrasts on E (nats), with d_w(row) = E_w(row) − E_w(self) of the same forward:
  ID(R) = ½[(d_S(R|S) − d_S(R|X)) + (d_X(R|X) − d_X(R|S))] per core (indices iS, iX).
- ψ_K(f, l, a) = mean ID(K{a}) / mean ID(K1); ψ_V(f, l, a) = mean ID(V{a}) / mean ID(V1); ψ_J(f, l, a) = mean ID(J{a}) /
  mean ID(nat). Logit version (no saturation): the same with m_w(row) = L_w(row) − L_B(row) in place of E_w (a margin
  inside one row is a logit difference): ID^logit(R) = ½[(m_S(R|S) − m_S(R|X)) + (m_X(R|X) − m_X(R|S))] with m taken
  relative to `self`.
- AUC of r_K for listed-format transfer (Direction B, B3): units u = (core, t, Z) with Z in {E1x0.5, E1x0.75, E1, E1x1.5,
  E4b (qwen7, mistral7), J0.25, J0.5, J0.65, J0.8, J0.9, nat, J1.25} at the dose blocks l in (3, 7) where the natural
  late patch transfers (flip_nat(AFTER, l) >= 0.9, point, in that model), restricted to units whose NONE answer is t;
  y = 1[AFTER answer = t]; predictor r_K(u). AUC = P(r_K of a positive > r_K of a negative) + ½ P(tie). Evaluable with at
  least 10 positives and 10 negatives. Reported beside it: the accuracy of the fixed rule "r_K >= 0.8 predicts transfer".

## 10. Statistics

- Bootstrap: two-stage cluster bootstrap (resample clusters with replacement, then items within each drawn cluster with
  replacement), 10,000 resamples, seed 20261015, implemented with `analysis.stage8b_parts.stats.Boot(clusters, nb=10000,
  seed=20261015)`. Clusters: N9 stories (lexicon, base, source); E9 edits (base, source) (both targets of a core move
  together); passages the article. One index set per population and model (over all valid items), shared by every format
  and row, so contrasts are paired. A rate over a subset (the competent items C9(f), the units of the AUC) is the ratio
  of weighted sums, `(W @ (x * m)) / (W @ m)` with m the subset mask; a resample whose subset weight is zero is undefined
  and dropped (more than 5 % dropped: the statistic is NOT EVALUABLE). Every statistic (rates, l_c, ψ, AUC, differences)
  is recomputed in every resample from the resampled items.
- Intervals: 95 % percentile intervals (numpy linear interpolation).
- Verdict levels per model (the note fixes which number and threshold): **SUPPORTED** = the point estimate is on the
  predicted side and the 95 % interval lies entirely on the predicted side of the threshold; **SUPPORTED AT THE POINT** =
  the point is on the predicted side, the interval crosses; **NOT SUPPORTED** = the point is on the wrong side;
  **NOT EVALUABLE** = the model's files are missing, a needed format was not run, l_b is None (for statistics at l_b), or
  a population or provenance check failed for that model.
- Combination over the three models (unless the prediction says "k of 3"): SUPPORTED if SUPPORTED in all three;
  SUPPORTED AT THE POINT if every model is at least that; NOT SUPPORTED as soon as one model is NOT SUPPORTED (named);
  otherwise NOT EVALUABLE (k of 3 evaluable). "k of 3" predictions count models at the point level and print each.
- Falsifiers are read at the point in each model (TRIGGERED / not), with the interval printed.
- No multiplicity correction (not a preregistration); every number is printed.

## 11. Output layout

### 11.1 Common provenance (every results file)

`provenance` = `experiments.format_factorial.provenance(a)` (args, git_commit, torch, transformers, python, device,
wrapper) | `stage9.run_meta()` | the following:

| Field | Value |
|---|---|
| `stage9` | `true` |
| `arm`, `stage` | "stories" / "passages" / "edits"; "check" / "cal" / "eval" |
| `git_dirty` | tracked files modified (`git status --porcelain --untracked-files=no` non-empty) |
| `freeze_commit` | `$STAGE9_FREEZE` (exported by the guard; `null` in TEST_MODE) |
| `predictions_sha256`, `spec_sha256` | sha256 of `docs/STAGE9_PREDICTIONS.md`, `docs/stage9/SPEC.md` in the working tree |
| `started_utc`, `finished_utc` | ISO times |
| `model`, `model_key`, `revision`, `verified` | as `experiments.stage8_edits.prov` (`verified` = sha256 and fields of VERIFIED.json) |
| `dtype`, `attn_implementation`, `device`, `n_layers`, `hidden`, `n_heads`, `n_kv_heads` | from the loaded model |
| `test_mode` | bool |
| `population`, `population_sha256`, `n_items`, `skipped_items` | pinned names and values; skipped items with reasons |
| `formats`, `formats_not_run`, `onsets`, `kv_only_onsets`, `max_new`, `reserve_min`, `timings` | as run |
| arm-specific | stories: `frames_file`, `frames_sha256`, `frames`, `g3_file`, `g3_sha256`, `scorer`; eval also `lb_file`, `lb_sha256`, `l_b`. passages: `items_sha256`, `squad_sha256`, `frame`, `frames_file`, `frames_sha256`, `valid_sha256` {E, R}, `stage8a_population_sha256`. edits: `calib_file`, `calib_sha256`, `das_file`, `das_sha256`, `das_seed` 102, `blocks`, `dose_blocks`, `frames` |

### 11.2 Directory

```
results/gpu_stage9/                      (TEST_MODE: results/gpu_stage9_test/, tags TEST_<key>)
  COMMIT.txt ENV.txt PIP_FREEZE.txt REVISIONS.txt PYTEST_OK.txt MANIFEST.sha256 [FAILED.txt SKIPPED.txt FETCH_FAILED.txt]
  logs/<step>.log  logs/pytest.log  logs/score.log      steps/<step>.done      verified/<key>.json
  check/<tag>_stories.json  check/<tag>_passages.json  check/<tag>_edits.json
  lb/<tag>.json                stories/<tag>.json
  edits/<tag>.json             passages/<tag>.json  passages/<tag>_LETA.json
  STAGE9_SCORE.txt  stage9_summary.json  fig_window.csv  fig_edits.csv  fig_dose.csv
```

### 11.3 `check/<tag>_<arm>.json` (tokenizer only, no weights)

`{"provenance": ..., "pass": bool, "fails": [str], "skipped": {pop/fmt: [[key, reason]]}, ...}`. Stories: N9 and C in the
six formats (`encode_item` valid), the form sets of N9 and C (word sets build and decode back; the letter set builds).
Passages: the valid E and R ids and their sha256; outside TEST_MODE the E sha256 equals stage 8a's. Edits: `prep9` of every E9 core in
the four formats (one p, identical prefix through p), the form sets, the pinned calib/das sha256 of the files on disk.
A failed check is exit 1 and the pipeline does not run that arm for the model. (The byte-equality of the E9 prompts with
`encoding.raw_prompt` is a unit test, 13.1.2.)

### 11.4 `lb/<tag>.json` and `stories/<tag>.json`

```
{"provenance": {...},
 "population": "C" | "N9",
 "rows": [row names in batch order, section 7.1],
 "l_b": int | null,                                   (lb file only)
 "late_after_flip": {"<l>": share, ...},              (lb file only; over all C cores, generated answers)
 "rule": "l_b = first onset with late AFTER flip <= 0.2 over all C cores",   (lb file only)
 "items": [ {                                        one record per item x format
   "key": "N9007", "pop": "N9", "idx": 7, "core": {...}, "lex": 1, "order": [6 ints], "sent": int|null,
   "words": [6], "track": {"S", "B", "X", "init"}, "cluster": [lex, base, source],
   "fmt": "AFTER", "p": int, "T": int,
   "rows": {"<row>": {"E": [6], "L": [6], "gen": str, "ans": word|"other", "how": "word"|"letter"|"none", "n_tok": int}},
   "clean_S": {"E": [6], "L": [6], "gen": str, "ans": ..., "how": ..., "n_tok": int} } ] }
```
Floats rounded to 5 decimals. The file is rewritten atomically after every format; a rerun keeps the formats already in
an existing file with the same `git_commit`, `revision` and `lb_sha256` (resume by format) and exits 3 if the deadline
stops it before the last format.

### 11.5 `edits/<tag>.json`

```
{"provenance": {...}, "population": "E9",
 "cores": [ {"index": i, "core": {...}, "X": loc, "p": int, "iB", "iS", "iX", "iPiS", "iPiX": int, "cluster": [base, source],
   "T_check": {"<l>": max_abs_diff},
   "stats": {"<l>": {"<row>|<t>": {"proj_k", "proj_v", "cos_k", "cos_v", "ratio_k", "ratio_v", "nu", "cos_resid", "ratio_resid"}}},
   "cells": {"<fmt>@<l>": {"<row>": {"E": [6], "L": [6], "gen": str, "ans": loc|"other", "how": str},
                           "caa": {"self": {...}, "CAA|S": {...}, "CAA|X": {...}}}} } ] }
```
Row names as in 8.1 including the target suffix (`nat|S`, `J0.5|X`, ...); `self` and `self2` without one. E and L in
LOCATIONS order (LETTER: letter i = location i). Resume by core (as stage 8c's eval), written every 5 cores; exit 3 at the
deadline.

### 11.6 `passages/<tag>.json` and `passages/<tag>_LETA.json`

```
{"provenance": {...}, "rows": [...], "frame": " ",
 "splits": {"E": {"NOM": [items], "OPTB": [...], "OPTA": [...]}, "R": {...}},          (_LETA file: {"E": {"LETA": [...]}})
 item: {"id", "art", "title", "sub", "D_in", "stratum", "split", "fmt", "nP", "P", "T", "j", "w", "dec": {Y: id},
        "rows": {"<row>": {"lp": {Y: float}, "mass": float, "argmax": int, "text": str, "ids": [int], "n_tok": int,
                           "who_strict": Y|"other", "who_contain": Y|"other"}},
        "S_run": {"text", "ids", "n_tok", "who_strict", "who_contain"}} }
```
Resume by format and split; exit 3 at the deadline.

### 11.7 Scorer outputs

`STAGE9_SCORE.txt` (sections: PROVENANCE, POPULATION, CHECKS, CONTROLS, PREDICTIONS, FALSIFIERS, DIRECTION B,
REPORTED, EXPLORATORY, SUMMARY; each section header on its own line starting with "#### "), `stage9_summary.json` (every
number printed, keyed by `<arm>/<model>/<statistic>`, each with `pt`, `lo`, `hi`, `n`), `fig_window.csv` (columns
`model, arm, split, format, window, onset, onset_frac, n, flip, flip_lo, flip_hi, stale, stale_lo, stale_hi, eflip`),
`fig_edits.csv` (`model, format, block, row, n, flip, lo, hi, stale, r_K`), `fig_dose.csv` (`model, format, block, kind
(J|K|V), alpha, psi_nats, lo, hi, psi_logit, lo, hi`).

## 12. GPU script, guard, steps and TEST_MODE (builder d)

### 12.1 The guard (`scripts/stage9_common.sh`, function `s9_guard`, called by `s9_init` outside TEST_MODE)

`scripts/gpu_stage9.sh` sources `scripts/stage8_common.sh` for its helpers (`s8_step`, `s8_run`, `s8_fetch`,
`s8_prefetch`, `s8_drop`, `s8_pytest`, `s8_time_left`, `s8_skip`, `s8_done`, `s8_check`, `die`, `s8_tar`) and then
`scripts/stage9_common.sh`, whose `s9_init` replaces `s8_init` (which hard-codes parts a-d and the J guard) and sets the
variables the helpers need (`OUT`, `TGZ`, `PY`, `DEV`, `DTYPE`, `S8_MODELS`, `S8_STATE`, `STAGE8_DEADLINE` exported,
`KEEP_CACHE`, `FORCE`, `FORCE_STEPS`, `TESTS`, `PART=9`, `PART_UC=9`). `s9_guard` refuses (die, exit 1, logs archived):
1. a tree with modified tracked files;
2. no commit in the history of HEAD whose subject is exactly `Freeze stage-9 predictions` (the most recent one is F);
3. `docs/STAGE9_PREDICTIONS.md` or `docs/stage9/SPEC.md` absent at HEAD, or different from their text at F;
4. code (`ckeys experiments analysis scripts tests data`) that differs between F and HEAD;
5. transformers other than 5.18.0 (pip install as `s8_init` does), or no CUDA device with >= MINGIB GiB (default 75).
It appends to `COMMIT.txt`: "stage-9 predictions frozen at F (commit date); note sha256 ...; spec sha256 ...; HEAD ...",
and exports `STAGE9_FREEZE=F`, `STAGE9_NOTE_SHA256`, `STAGE9_SPEC_SHA256`. The git checks are a separate function so
`tests/test_stage9_guard.py` can run them in a temporary clone without CUDA. `s9_finish` = `s8_finish` with the score
file `STAGE9_SCORE.txt` (sections printed: CHECKS, PREDICTIONS, FALSIFIERS, SUMMARY).

### 12.2 Steps (`scripts/gpu_stage9.sh`)

```
s8_pytest  tests/test_stage9_pops.py tests/test_stage9_stories.py tests/test_stage9_passages.py tests/test_stage9_edits.py
           tests/test_stage9_score.py tests/test_stage9_guard.py tests/test_stage9_pre.py
           tests/test_fresh.py tests/test_fresh_factorial.py tests/test_generate.py tests/test_surface.py tests/test_clamp.py
           tests/test_natural_clamp.py tests/test_stage8_edits.py tests/test_stage8_populations.py
SQ=${SQUAD:-results/gpu_stage8a/squad/dev-v1.1.json}
for key in mistral7 llama8 qwen7 (TEST_MODE: qwen7 llama8; TEST_ALL=1: all three):
  prefetch the next key in the background
  dir=$(s8_fetch key)         refused -> record, skip the key (no fallback)
  chk_st_<key>   stories  --stage check                         -> check/<tag>_stories.json
  chk_ed_<key>   edits    --stage check                         -> check/<tag>_edits.json
  chk_ps_<key>   passages --stage check --squad $SQ             -> check/<tag>_passages.json
  cal_<key>      stories  --stage cal   (if chk_st passed)      -> lb/<tag>.json
  st_<key>       stories  --stage eval  (if the lb file exists) -> stories/<tag>.json
  ed_<key>       edits    --stage eval  (if chk_ed passed)      -> edits/<tag>.json
  ps_<key>       passages --stage eval --formats NOM,OPTB,OPTA --splits E,R (if chk_ps passed) -> passages/<tag>.json
  leta_<key>     passages --stage eval --formats LETA --splits E (S8_EXPLORATORY=1; only if LETA_MIN minutes remain
                 after the reserve) -> passages/<tag>_LETA.json
  s8_drop key
s9_finish analysis/stage9_score.py
```
Every step: `S8_OUTPUTS=<its file>`, `--model "$dir" --key "$key" --out "$OUT" --reserve-min <minutes>`. Step order within
a model is fixed: the decisive stories first, then edits, then passages. A step exit 3 is "partial" (SKIPPED.txt) and is
redone in the next session.

### 12.3 Deadline

`DEADLINE_H` default 4.5. `CORE_MIN` per model (stories + edits + passages, load included): mistral7 60, llama8 55, qwen7
50; within a model a step's reserve is the minutes of its later steps (stories 12, edits 10, passages 30) plus the later
models' CORE_MIN plus 10 (score). `LETA_MIN` = 10.

### 12.4 Options

`OUT`, `SQUAD`, `DEADLINE_H`, `FORCE`, `FORCE_STEPS`, `KEEP_CACHE`, `TESTS`, `PY`, `S8_MODELS`, `MINGIB`, `TEST_ALL`
(as stage 8). `.gitignore` gets `results/gpu_stage9_test/`, `gpu_stage9_results.tgz`, `gpu_stage9_test_results.tgz`.

### 12.5 TEST_MODE (`TEST_MODE=1 bash scripts/gpu_stage9.sh`)

No guard, no pip; CPU, FP32, `Qwen/Qwen2.5-0.5B-Instruct` (24 blocks) for every key (`s8_fetch` prints the hub id);
`OUT=results/gpu_stage9_test`, archive `gpu_stage9_test_results.tgz`, tags `TEST_<key>`; every step of section 12.2 runs:
- stories: N9 and C n = 2 each, all six formats, grid `story_onsets(24)`, `KV_ONLY_ONSETS` (4, 8, 12, 16), frames `[]`
  (no frames file; `frames_file: null`), l_b from the two C cores (may be None: the eval then records `l_b: null`);
- passages: n = 2 E and 2 R items with windowed passages (`experiments.natural_factorial.window`), frame " ", formats
  NOM, OPTB, OPTA and the LETA step;
- edits: n = 2 E9 cores, blocks (3, 7, 11, 15), dose at (3, 7); mu1 = `ckeys.edits.class_means(model, tok, H_fit[:3],
  blocks)` with H_fit from `ckeys.edits.populations()["H"][:150]`; E4b basis (qwen7 and mistral7 keys) = the transpose of
  the Q factor of `torch.linalg.qr(torch.randn(hidden, 16, generator=torch.Generator().manual_seed(102)))`, i.e. U
  [16, hidden] with orthonormal rows; both recorded in provenance as TEST stand-ins (`calib_file`, `das_file`: "TEST");
- the scorer with `--test` (population sizes and pins, the stage-8a E-set comparison, BF16, sdpa, the freeze and the
  pytest-log checks waived; verdicts are plumbing).
Expected wall time on a 4-core CPU: under 1 h including the pytest run.

## 13. Exactness tests (FP32, CPU; Qwen2.5-0.5B-Instruct or tiny random models; 1e-4 log-prob, 1e-5 K/V)

Every test file runs in the pipeline's pytest step before any model. Tests that need a cached study tokenizer (Qwen2.5-7B,
Mistral-7B, the Llama-3.1-8B mirror) carry "cached_tokenizer" in their name and skip when it is absent.

### 13.1 Builder (a): `tests/test_stage9_pops.py`, `tests/test_stage9_stories.py`

1. N9, E9, C: sizes, the pins of 5.2-5.4, draws (161, 173), determinism (two calls equal), X9 size 45,832; N9 covers 30
   pairs and 20 objects, 53 clusters, lexicon 75/75, sentence counts; E9 satisfies cond_E; N9, E9, C pairwise disjoint;
   N9 and E9 disjoint from U, from X9 and from every stage-8 population taken from the functions the stage-8 steps call
   (`fresh.population`, `edits.populations`, `stage8_flag.populations`, IOI and task lists by type); seeds 9101, 9102 not in
   the stage-8 pilot seeds nor any stage-8 population seed; `story_onsets` and `passage_onsets` values for L = 24, 28, 32.
2. E9 items: `fresh.raw_prompt(item, f, loc) == encoding.raw_prompt(f, story, query)` for every core, f in EDIT_FORMATS
   and P1, loc in B, S, X.
3. Every row of `run_item9` (stories, one N9 item of each lexicon, formats NONE, BEFORE, AFTER, LETTER) equals
   `ckeys.surface.score_reference` under a separately built single-row clamp (E and L, 1e-4).
4. `late@0` = `KV@0` = the clean S run, and `early@L` = `none` = the clean B run (scores 1e-4; the tables captured under
   the clamp 1e-5); complementarity: S on [0, l) and S on [l, L) by two nested `clamp_kv` contexts equals `KV@0`, and
   S on [0, l) nested with B on [l, L) equals `early@l`, for l in (2, 12, 22).
5. The generation batch equals `ckeys.generate.greedy_reference` of each row alone under its own clamp, with the word
   stop and with `letter_stop` (LETTER), 16 tokens.
6. `parse_letter` and `letter_stop` on hand cases: "A", " B)", "**C**", "(D) jar", "E. box", "F", "Box", "Bob", "b", "",
   a word answer naming a candidate, an incomplete multi-byte character; the LETTER E/L mapping to canonical word order on
   a constructed item whose order is not the identity.
7. l_b and the lb-file contract on synthetic flips (first onset <= 0.2; None); resume by format; a TEST_MODE subprocess
   run of `check`, `cal`, `eval` writes files with every key of 11.3-11.4.

### 13.2 Builder (b): `tests/test_stage9_passages.py`

1. Window rows on windowed passages (NOM, OPTB, OPTA, LETA): each row's teacher-forced decision log-probabilities equal a
   reference forward under a separately built clamp (1e-4); `late@0` = `KV@0` = the S prompt's own; `early@L` = `none` =
   the B prompt's own; the nested complementarity of 13.1.4 at span positions.
2. The 48-token generation of the batch equals `greedy_reference` of each row alone under its own clamp; no W + 2 cut in
   NOM (a row continues past W + 2 words); LETA stops at the letter rule.
3. `who_contain` and `who_strict` on hand cases: entity first; entity after a preamble ("The answer is S."); two entities
   (earliest wins; tie to the longer); entity as a sub-word (not counted); empty text; LETA letters.
4. With the cached Qwen2.5-7B tokenizer and SQuAD (skip when absent; name contains "cached_tokenizer"): the valid E set
   at frame " " has 185 items and `sha_ids` = stage 8a's `population_sha256`; the R set is computed and stable.

### 13.3 Builder (c): `tests/test_stage9_edits.py`

1. The lemma at Qwen2.5-0.5B: B with an E1 vector written at (p, l) (`ckeys.edits.write_resid`, full forward) equals B
   with the tables of the edit prefix pass clamped from l + 1 (E and L 1e-4; tables 1e-5), l in (3, 7), formats NONE,
   AFTER, LETTER, POST; `T|S` tables equal the natural S tables bitwise with the padded 5-row shape.
2. Dose tables: `J{a}` = B + a(t − B) exactly; r_K of J{a} and K{a} equals a and of V{a} equals 0 (1e-6); J1 = nat.
3. E4b: the vector equals `fixed_subspace_interchange(U)` with a random orthonormal U; the stored-basis loader asserts the
   pinned sha256 and the shape (on the committed files, no model), and refuses a file with another hash.
4. CAA: greedy decoding with the KV cache under the all-position hook equals `greedy_reference` with the hook active in
   every cache-free forward; the trie scores under the hook equal `score_reference` with the hook; the hook at index p
   alone equals single-site E1 (scores 1e-4).
5. `prep9`: identical prefix through p in NONE, AFTER, LETTER, POST for every E9 core (cached tokenizers; skip when
   absent); the calib and das pins on the committed files.
6. A TEST_MODE subprocess run of `check` and `eval` (n = 1) writes every key of 11.5.

### 13.4 Builder (d): `tests/test_stage9_score.py`, `tests/test_stage9_guard.py`

Synthetic results in the layouts of section 11 with known answers: every prediction's SUPPORTED, AT THE POINT, NOT
SUPPORTED and NOT EVALUABLE paths; l_c with censoring; l_b recomputation and its mismatch; the "2 of 3" rule; falsifier
triggers; AUC with ties and the evaluability floor; ψ in nats and in logits; containment primary with strict printed;
competence filtering; the bootstrap (two-stage structure, determinism, paired across formats); provenance checks
(MISMATCH exit 2: two commits, a TEST_ file outside TEST, a population hash, a frames/calib/das sha, a freeze commit or
note sha differing from COMMIT.txt, an eval that read another lb file); the pytest-log gate (every stage-9 test file in
the last pytest session with at least one pass, no failure or error; outside TEST a missing log prints "FP32 tests not
passed: no outcome" and every verdict is NOT EVALUABLE); a missing model. Guard: in a temporary clone, the git checks
refuse a dirty tree, a history without the freeze commit, a note or spec changed after it, code changed after it, and
pass on a clean frozen clone.

### 13.5 Builder (e): `tests/test_stage9_pre.py`

The functions of `analysis/stage9_pre.py` reproduce, at printed precision, every seen-before number in
`docs/STAGE9_PREDICTIONS.md` section 1 (they read only committed results files).

## 14. The scorer (`analysis/stage9_score.py --results DIR [--test]`)

- CHECKS: provenance (one commit across files; freeze commit and note/spec sha equal to COMMIT.txt's; BF16, sdpa, model
  revisions equal to the manifest's; no TEST_ files outside TEST), population pins and skipped counts (0 expected),
  stage-8 artefact sha256s, the passage E set against stage 8a, l_b recomputed and read in order, the `T_check` maxima,
  the pytest-log gate. A failed check prints MISMATCH and the scorer exits 2 after writing every output.
- CONTROLS (per model and format): competence share; flip(KV@0) on all items (the "listed answers are sticky" control);
  the BF16 floor (mean |E(none) − E(none2)| and the answer disagreement rate); E-argmax/generation agreement; the
  complementarity reminder (FP32-tested; no GPU check).
- PREDICTIONS P1-P5 and POST, DIRECTION B (B1-B3), FALSIFIERS (F1-F3, B-F1, B-F2): exactly as the note defines them,
  with per-model point, interval, verdict, the combined verdict, and the pre-written reading the outcome selects (note,
  section 6).
- REPORTED: the full late and early curves per format and model (stories and passages, E and R), l_c per format, l_b,
  silent-failure shares and confidences at l_b and at round(0.3 L), the inconsistency rate per onset, the K-only and
  V-only late windows, all edit rows (flip, stale, τ, r_K), CAA at every block, ψ curves in nats and logits, the AUC and
  the fixed-rule accuracy, R items' versions of the passage predictions, strict-match versions of the passage
  predictions, the LETA file when present.
- EXPLORATORY: E-argmax versions; all items instead of competent items; per lexicon and per list order (B before S
  versus S before B) at l_b; the reproduction check of stage 8a's OPTA K_S@0.3L decision-token share by `Klate@round(0.3L)`.

## 15. Zero-GPU paper analyses (builder e; group "S9P", stage-9-pre)

### 15.1 Scope and labels

Items 1-7 of the design review's section 7, computed from committed results only, registered as macros of a new group
`G9P = "S9P"` with `rmac` (every description ends in "[recomputed; label exploratory/post hoc]"), plus one figure
`paper/figures/fig_window_pre.pdf`. The computations live in `analysis/stage9_pre.py` (pure functions returning
numbers); `paper/make_figures.py` calls them and registers the macros. Names: `Np` + statistic + format + block or onset
word + the model tag of make_figures' `K8` map (e.g. `NpStaleAfterMistralSeven`, `NpStaleAfterMistralSevenLo`).

### 15.2 Items, sources and computations

1. Behavioural format gap, 8 fresh-story models (`results/gpu_stage8b/eval/<k>_F.json`, k in mistral7, llama8, phi4,
   olmo7, falcon7, gemma9, qwen7, qwen14; the row `KV_S@l0` with l0 the third onset = round(0.3 L)): E-argmax share
   stale (= B) under AFTER with the stage-8b two-stage bootstrap (`analysis.stage8b_parts.stats.Boot`, clusters =
   `cluster`, seed 20261013) -> Mistral 0.90 [0.83, 0.96], Llama 0.82 [0.74, 0.90], Phi-4 0.67 [0.56, 0.78], OLMo 0.61
   [0.49, 0.73], Falcon 0.25 [0.15, 0.36], Gemma-2-9B 0.19 [0.11, 0.29], Qwen-7B 0.09 [0.03, 0.17], Qwen-14B 0.01
   [0.00, 0.04]; flips under NONE, BEFORE, POST-NULL, PRE (min 0.98); flip of `KV_S@0` (min over arms 0.98); median
   absolute p(B) = exp(E_B) among stale items (Mistral 0.999, Phi-4 0.997, Qwen-7B 0.997, Gemma 0.994, Falcon 0.981,
   OLMo 0.914, Qwen-14B 0.908, Llama 0.846) and the six-way normalised version (Mistral 1.000, Phi-4 0.998, Gemma 0.996,
   OLMo 0.918, Llama 0.855); POST flip (Mistral 0.87, others 0.99-1.00); AFTER generated V_S@0 flip (0.00-0.15). Label:
   "E-scored argmax, not generation" for the 0.3 L rows.
2. Onset curves (`results/gpu_stage8c/eval/<k>.json`, k in qwen7, mistral7, llama8): E-argmax flip of the natural KV,
   K and V rows (`nat|KV|t`, t in S, X) by format (NONE, POST, P1, LETTER) at blocks 3, 7, 11, 15 (onsets 4, 8, 12, 16);
   stage-8c bootstrap (`analysis.stage8c_parts.stats.Boot`, (base, source) clusters, seed 20261012). P1 KV: Qwen
   0.98/0.94/0.69/0.00, Mistral 1.00/0.36/0.01/0.00, Llama 1.00/0.68/0.01/0.00; NONE KV 1.00 throughout; V-only NONE
   0.84-0.99, P1 0.00 (max 0.01).
3. Passages (`results/gpu_stage8a/explore/<k>.json`, part `onset_piece`, k in qwen7, mistral7, llama8, gemma9): share of
   the 185 E items whose argmax over the five decision tokens (B, S, X, Z, D; `rows[r]["lp"]`) is S, for NOM `V_S`,
   `V_S@on` and OPTA `K_S`, `K_S@on` (onset round(0.3 L)): NOM V 0.93/0.88/0.93/0.95 from 0 and 0.92/0.87/0.94/0.94
   from 0.3 L; OPTA K 0.65/0.41/0.25/0.60 from 0 and 0.19/0.01/0.04/0.03 from 0.3 L (Qwen/Mistral/Llama/Gemma); the
   ratio of mean margin shifts (m = lp_S − lp_B relative to ID) from 0.3 L to from 0: NOM V 0.987-1.000, OPTA K
   0.516/0.014/0.273/0.116; article bootstrap (`analysis.stage8a_parts.common`, seed 20261010).
4. Edit transfer in generation (`results/gpu_stage8c/explore/<k>.json`, block 7, first 40 E stories): flip per instance
   under NONE and P1 (Mistral E1 1.00 -> 0.35, E4b 0.97 -> 0.03; Qwen E3 1.00 -> 0.50, E4b 0.99 -> 0.25), the share B
   among P1 failures (Mistral 0.99-1.00 for nat, E1, E4b), and the Spearman correlation of κ/σ against the transfer ratio
   (0.92 over 9 edit x model pairs). Flips are means over the stories of the mean over t in (S, X) of 1[answer = t]. The
   pairs are the (model, instance) of qwen7 and mistral7 among E1, E2, E3, E4a, E4b, E5FR, E5DE, E5SYN, E5NL whose NONE
   flip is >= 0.5 (9 pairs). The transfer ratio is (flip(P1) / flip_nat(P1)) / flip(NONE). κ and σ are parsed from the
   printed `P1@7: sigma ...; kappa ...` lines of `results/gpu_stage8c/STAGE8C_SCORE.txt` (lines 368 and 414 at the
   current commit; located by content, never hand-copied).
5. The binding swap (Qwen2.5-14B; stage 6, already registered as the Pk macros): no new number unless needed; the figure
   inset redraws BIND IIA by block under NO-MENTION against the named formats from make_figures' `SW` (0.993 at block 28
   under NO-MENTION; at most 0.01 at every block 18-40 under QNAMES, OPTIONS-AFTER and LETTERS).
6. Dose curves (`results/gpu_stage8c/eval`): ψ_K at 0.5 and 0.8 (SK50, SK80 K rows) by format and ψ_V (SV50, SV80 V
   rows) at blocks 3 and 7 (as `analysis.stage8c_parts.law.Model.psi`): ψ_K(0.5) in P1 at block 3 0.10/0.18/0.13 and
   ψ_V(0.5) in NONE 0.60/0.58/0.56 (Qwen/Mistral/Llama). Behavioural E-argmax flips over (story, t): SK50|KV in P1 at block 3
   0.31/0.32/0.66 and SV50|KV in NONE 0.93/0.92/0.94. Per-story ratio ID(P1, 3, SK50, K) / ID(P1, 3, nat, K) at Qwen over
   the stories with natural ID_K > 2 nats: 10th/50th/90th percentile 0.02/0.06/0.20 (SK80 median 0.48).
   The post hoc law diagnostics (138 λ statistics in 49 units) follow this exact rule:
   - Per story, ID(f, l, Z, C) = ½(r_S[iS] − r_S[iX] + r_X[iX] − r_X[iS]), where r_t are the first six entries (dE) of the
     cell row `Z|C|t`. SK rows read with C = V take the `nat|V` row; SV rows read with C = K take `nat|K`.
   - Instances Z: E1, E2, E3, E4a, E4b, E5FR, E5DE, E5SYN, E5NL, PERP:{E1, E2, E3, E4a, E4b}, NONLEX:{E1, E2, E3, E4a,
     E4b}, NONLEX:nat, at every depth where φ = ID(NONE, l, Z, KV) / ID(NONE, l, nat, KV) >= 0.3.
   - p_k and p_v are the means over stories and t of `stats[l][Z|t]["proj_k"]` and `["proj_v"]`.
   - Codes (f_K, f_V): W(P1) = (P1, P1), W(POST) = (POST, POST), A_L = (LETTER, NONE), A_P = (P1, NONE). A code is kept
     when both rows exist and the natural ID of K in f_K and of V in f_V are each >= 2 nats and >= 0.1 of the natural
     ID_KV of that cell.
   - ψ_K = ID(f_K, l, Z, K) / ID(f_K, l, nat, K) and ψ_V likewise. c50 and c80 are the SK50 and SK80 (K) or SV50 and
     SV80 (V) ratios of the same cell.
   - curve(a) = 0 for a <= 0; linear interpolation through (0, 0), (0.5, c50), (0.8, c80), (1, 1) for a <= 1; and
     1 + (a − 1)(1 − c80)/0.2 above 1. Lg(x) = log max(x, 0.01).
   - λ = Lg(ψ_K) − Lg(ψ_V). The curve predictor is Lg(curve_K(p_k)) − Lg(curve_V(p_v)), O1 = Lg(p_k) − Lg(p_v), and
     the baseline is Lg(p_k).
   The results: Spearman with λ 0.862 (curve) and 0.855 (log proj_k); MAE 0.369 (curve) against 0.868 (O1). The
   difference in Spearman is 0.007; its 95 % interval from 2,000 unit resamples (units = (model, Z, depth),
   `numpy.random.default_rng(1)`) is [−0.060, 0.090], with a resample mean of 0.012. The partial rank correlation of the
   curve with λ given log proj_k is 0.53 (ranks residualised on [1, rank(log p_k)]); Spearman(p_k, p_v) is 0.92.
7. Knockout (`results/gpu_stage5/knockout/<model>_s0.json`, Qwen2.5-7B, Mistral-7B, Qwen2.5-14B): the share of
   listed answers kept at B under M3 (the answer's attention to the writing token cut) and M7 (every non-candidate later
   row's), AFTER and P1 (1.00 under M3; 0.97-1.00 under M7), and under NONE M3 the share sent to the initial or the
   distractor location (Qwen-7B 1.00, Mistral 0.85, Qwen-14B 0.79); item bootstrap, 10,000, seed 20261015.

### 15.3 Figure `fig_window_pre.pdf` (one row of four panels, width `\textwidth`)

(a) item 1: AFTER stale share with intervals for the 8 models, beside the minimum flip of the free-form arms;
(b) item 2: flip by onset block for P1 and LETTER (solid) against NONE and POST (dashed), three models;
(c) item 3: OPTA K from 0 and from 0.3 L, NOM V from 0 and from 0.3 L, four models;
(d) item 6: ψ_K(α) under P1 and ψ_V(α) under NONE at block 3 (α = 0, 0.5, 0.8, 1), three models, identity line dotted.
Caption text ends with "Post hoc: rows of stages 5-8 re-analysed before the stage-9 run." Item 4, 5 and 7 numbers are
macros only (the main figure of stage 9 comes after the GPU run).

### 15.4 make_figures integration

Place the block after the stage-8 block and before the index and numbers.tex writes. Record `N_PRE9 = len(REG)` before
the block; change the stage-8 group check from `REG[N_PRE8:]` to `REG[N_PRE8:N_PRE9]`; add `assert all(r[0] == G9P for r
in REG[N_PRE9:])` and assert that every macro defined before the block keeps its value. Regenerate `paper/numbers.tex`
(the diff must contain only added `\Np...` lines) and write `docs/stage9/S9P_MACROS.md` with `MACRO_INDEX=...
MACRO_INDEX_GROUPS=S9P`. Do not edit the paper's .tex sections.

## 16. Compute estimate (A100-80GB, BF16)

From stage-8 timings (stage 8b: 40-60 s per 150-item arm with 24 rows; stage 8a: 180-225 s per 185-item format with
16-token generation; stage 8c: 25-40 min per model for 494 rows per story-cell):

| Step (per model) | Work | Minutes |
|---|---|---|
| fetch and verify (prefetched) | ~15 GB | 0-10 |
| checks | tokenizer only | 1-2 |
| `cal` (C) | 30 cores x 6 formats x ~45 rows (trie + generation) | 2-3 |
| `st` (N9) | 150 x 6 x ~45 rows | 8-12 |
| `ed` (E9) | 80 x 4 formats x ~120 rows (table rows and CAA batches) | 8-12 |
| `ps` (E + R) | 256 items x 3 formats x 21 rows (48-token generation) | 20-30 |
| `leta` (optional) | 185 x 21 rows | 6-9 |
| model load | | 1-2 |
| **per model** | | **40-65** |

Three models: 2.0-3.3 GPU-hours, plus pytest (~10 min) and the scorer (~5 min): about **3-4 A100-hours**, within the
target of about 4 (DEADLINE_H default 4.5). Memory: at most ~48 rows x ~560 tokens per trie forward (stories), ~21 rows x
~830 tokens per passage forward: under 12 GB beyond the weights. Disk: >= 50 GB (one model at a time plus a prefetch).

## 17. Decisions fixed here, and what is not built

1. Seeds 9101 (N9) and 9102 (E9); the exclusion streams of 5.1 (3000 draws per seed, stricter than populations alone).
2. Stories use the S donor only (no X rows): one row per window keeps 6 formats x ~45 rows affordable; the identity
   contrasts with X belong to the edit arm, which keeps both targets as stage 8c did.
3. "Every second block" = onsets 2, 4, ..., L − 2; the even onsets contain the stage-8c onsets 4, 8, 12, 16.
4. Every row clamps every layer (B's own tables where the donor is B): rows are exact table assignments; the early
   window is not left to recompute above l.
5. Containment is the primary passage answer rule, strict the sensitivity (the 48-token generation is meant to catch
   verbose correct answers that the stage-8a W + 2 cut ended early; at the stage-8a cut, strict NOM competence was 45 of
   185 items at Mistral).
6. Primary populations are the items competent in the format (section 9.4); all-items versions are exploratory.
7. Dose rows only at blocks 3 and 7 (Direction B's GPU part); nat, E1, CAA and E4b at all four blocks.
8. E4b = the stored seed-102 bases; no retraining; Llama has none.
9. AUC units and the transfer-cell restriction of 9.7 (the window must be open for transfer to be at stake).
10. Not built: the composition test (design review 4.4), any KV-cache compression or truncation arm (5), new models,
    a frame rediscovery (the stage-8b and 8a frames are reused; none was discovered at the three models).
