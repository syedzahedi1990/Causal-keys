LABEL design:natural
### part
A: natural data. Counterfactual SQuAD v1.1 reading comprehension with multi-token, open-vocabulary answers; emitted-form scoring; two new model families; behavioural consequences; reader heads on natural text

### objections_answered
[
 "(2) Generality: the evidence was templated tasks with single-token values. Part A tests the account on natural Wikipedia passages: 697 SQuAD v1.1 dev questions whose answer is a PERSON, PLACE or NUMBER entity named exactly once in the passage, with multi-token answer spans (2-4 tokens is typical; only 9-25% are single tokens, depending on tokenizer). It covers four families, two of them new (Llama-3.1-8B, Gemma-2-9B), with intervals from a cluster bootstrap over articles.",
 "(4) Scores on lowercase tokens that get almost no probability: every score is on the emitted form. The scored token is the first content token of the entity the model actually generates. Competence is checked by greedy generation of all three clean runs. Gate A-G3 requires the scored token to carry the clean probability (median p \u2265 0.5) and to equal the first generated content token in \u2265 95% of competent items. Behaviour is also measured directly by greedy generation.",
 "(5) The sentence effect was confirmed on the same cores and models where it was found: A5 is a fresh-sample test of the later-mention (sentence) effect, with new data, new natural wording ('Related articles mention ...'), two families never used in stages 1-7, an instruction-matched before/after control, and margins on the scale-free s_ID rather than absolute nats.",
 "Generalist and stats reviewers asked for a practical, behavioural consequence. There are three. A6, a double dissociation by generation: swapping only the answer span's cached keys flips multiple-choice answers, while swapping only its values flips free-form answers. A7, cache corruption: MCQ answers survive corruption of the span's values, which breaks free-form answers. HA3: the reader heads found in MCQ format are needed for the MCQ key read but not for free-form reading (random heads give no change).",
 "(6) H3 vs G7b and 'ablation compresses all margins': HA4 tests the copy fallback directly on natural data, by accuracy and ID_V. The unablated model's own option margin is reported under every ablation, including three size-matched random sets.",
 "(1) Novelty, partly: A8 is a non-obvious, risky prediction of the account that 'QK/OV plus duplicate heads exist' does not state. In free-form answers, every piece after the first is looked up through the span's key from the model's own answer prefix (a later mention) and copied from its value. So it depends jointly on K and V, and the key alone reads negatively. A CPU pilot already shows this (continuation interaction share 1.41 vs 0.04 at the decision token).",
 "(2b) Negative key reads, partly: A8's secondary line shows a principled mechanism for a negative key identity: a key match that retrieves another candidate's value. It is offered as a testable hypothesis for the IOI-inline and LIST-BEFORE negatives, not claimed to explain them.",
 "Mechinterp reviewer's 'have you tried non-templated data ... values that span several tokens?': HA1 and HA2 run the stage-6 head analysis on natural MCQ items, including a transfer test of the template-found reader heads."
]

### design

# Part A of preregistration J: the format law on natural reading comprehension

## 0. In one paragraph
We take SQuAD v1.1 dev questions whose answer entity e_B occurs **exactly once** in the passage and never in the question. We build counterfactual passages that differ **only inside the entity's token span P**: e_B is replaced by a same-type entity e_S, e_X or e_Z of the **same token length in each of the four tokenizers**. We clamp the cached key and/or value at **all positions of P, in every layer from 0**. The clamp is exact: KV_S from layer 0 reproduces the S run, as verified in FP32 to 3e-5 on multi-token spans.

We vary where the candidates are mentioned again: an options list after or before the passage, a natural "Related articles mention ..." sentence after or before, no mention, or lettered MCQ. We score the token the model actually emits (verified by greedy generation) and also measure behaviour by generation.

Confirmatory predictions:
- A1–A5 replicate the law: a key lookup with later options, a copy without, position matters, and a fresh test of the mention-sentence effect.
- A6–A7 give its behavioural consequences.
- A8 is a new prediction for multi-token answers.
- HA1–HA4 locate the natural-text readers, test whether the template-found heads transfer, and test whether these MCQ readers matter for free-form behaviour.

## 1. Items (built on CPU before finalisation; no model involved)

### 1.1 Source
- SQuAD v1.1 dev, https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json, sha256 `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`. It has 48 articles and 10,570 questions; licence CC BY-SA 4.0 (cite Rajpurkar et al. 2016).
- The counterfactual substitution follows Longpre et al. (2021, entity-based knowledge conflicts); cite it.

### 1.2 Base item rules (all must hold)
1. Majority answer a = the most common answer string, given by ≥ 2 of the 3 annotators.
2. a occurs **exactly once** in the passage (regex `(?<![\w])a(?![\w])`, case-sensitive) and **not** in the question (case-insensitive substring).
3. a is typed (§1.3) as PERSON, PLACE or NUMBER (primary) or YEAR (exploratory stratum). ORG and other answers are dropped; ORG was heterogeneous in the pilot audit.

### 1.3 Entity typing (regex plus fixed word lists, no NLP library)
- **wh-word**: the first match of `\b(who|whom|whose|where|when|what|which|how|why)\b` in the question.
- **head noun**: for what/which, the last token before the first token in the stop list. The stop list is the auxiliaries {is, was, are, were, did, does, do, has, had, have, can, could, will, would, should, may, might, became, become, becomes}, the prepositions {of, to, in, for, on, at, by, from, with} and {that, which, who}. At most 5 tokens are read.
- **YEAR**: `^(1[0-9]{3}|20[0-2][0-9])$`.
- **NUMBER**: `^(\d{1,3}(,\d{3})+|\d+)$` and not a year.
- **PERSON**: the wh-word is who/whom/whose, and a is person-shaped:
  - 2–3 words, each matching `^[A-Z][a-z]+(-[A-Z][a-z]+)?$`, `^[A-Z][a-z]*[A-Z][a-z]+$` or the initial `^[A-Z]\.$`;
  - the last word is not an initial;
  - no word is in the fixed NONPERSON list (the organisation, geography, demonym and title words listed in the reference builder).
- **PLACE**: the wh-word is where, or the head noun (or its singular) is in PLACE_HEADS = {country, city, town, state, county, region, continent, river, lake, island, province, capital, nation, area, village, district}.
  - a is 1–4 capitalised words with connectors {of, de, the, and, von, van, da, du, la, le, del, di, y, al, for, on, in}.
  - a has no NONPERSON organisation word and does not start with an ordinal or a pronoun.
- The rules are implemented in `ckeys/squad_items.py`; the reference implementation is `build_items5.py` (scratchpad).

### 1.4 Substitutes and distractor
- **e_S, e_X, e_Z** (Z is used only for corruption rows and never appears in any prompt). Each item has its own random stream, `random.Random(int(sha256(id)[:8], 16))`.
  - PERSON/PLACE: drawn from majority answers of the **same subtype and word count** from **other articles**. For PLACE, the pool is limited to answers of questions with an explicit place head noun.
  - NUMBER: the leading digit is replaced by another digit, in seeded order.
  - YEAR: same century, 3 ≤ |Δ| ≤ 60, different decade from e_B and from each other.
  - Every substitute is absent from the passage and the question (case-insensitive) and shares no word (`\w+`, lowercased) with e_B, the other substitutes or e_D.
- **e_D** (the fourth option):
  - First choice: the majority answer of another question on the same paragraph, of the same subtype, that occurs in the passage and not in the question (D_in = True).
  - Otherwise an absent pool entity (D_in = False).
  - Option order: `random.Random(seed_of(id + "order")).shuffle([e_B, e_S, e_X, e_D])`. The order is fixed per item and identical in the B, S, X and Z prompts.
- **Tokenizer validity**, required in every format of §2 and in each of the four model tokenizers:
  - The chat-wrapped B, S, X and Z prompts have equal length and differ only at positions inside P, where P is the set of tokens whose offsets overlap the entity's characters in the B prompt.
  - The continuation of `" " + e` after the `Answer:` prefill is prefix-stable.
  - Its tokens before the first non-whitespace token (w, e.g. the space token of a number) are identical for B, S, X, Z and D.
  - The **decision tokens** dec_Y (the first non-whitespace token) are **pairwise distinct among B, S, X, Z and D** ("first-token" stratum, FT).
  - YEAR items are exempt from distinctness (shared-prefix stratum, SP).

### 1.5 Type audit (pre-finalisation, blind to any model output)
- A reader checks every item in the R and E splits (about 330), from a printed list of question, e_B, e_S, e_X, e_Z and e_D.
- An item is flagged if any of the five entities is not of the item's type, for example "Political" as a PLACE or "Digital Spy" as a PERSON. Flagged items are removed.
- The audit file `data/stage8a_type_audit.tsv` is committed before finalisation, and removal counts are reported.
- If more than 15% of a type's items are flagged, the type's word lists are corrected and the build rerun; the audit is then redone and committed.

### 1.6 Splits and counts (computed in the pilot with the four tokenizers)
- **Valid FT items:** 697 in all four tokenizers × six formats (PERSON 373, NUMBER 191, PLACE 133; 47 articles). The YEAR SP stratum has 529 items.
- **Article split:** `random.Random(20261010).sample(sorted(article titles), 12)` gives the ranking articles R; the remaining 35 articles are E.
- **Caps:** items are taken in seeded shuffles of id-sorted lists (`Random(1)` for R, `Random(2)` for E), with at most 2 per paragraph and at most 8 (R) or 10 (E) per article. Super_Bowl_50 alone holds 23% of the raw items.
- **Result:** R has 88 items over 12 articles; E has **239 items over 35 articles** (PERSON 110, PLACE 66, NUMBER 63; D_in 54). About 220 are expected after the audit.
- **Uses:**
  - Main factorial: every E item that survives the audit.
  - Head ranking: the first 60 R items.
  - Head evaluation: the first 80 E items.
  - YEAR exploratory stratum: the first 80 YEAR items in E articles.
- The committed `data/stage8a_items.json` holds ids, substitutes, option order, split and audit flag. The preflight rebuilds the items from the downloaded file and asserts equality with it.

## 2. Formats (exact strings; system turn "You are a helpful assistant.", user turn below, generation prompt with thinking disabled, assistant prefill `Answer:`)
`PRE = "Read the passage and answer the question.\n\n"`, `FREE = "Answer with the exact words from the passage."`, `MC = "Answer with exactly one of the options."`, `OL = "Options: o1; o2; o3; o4"`, `MEN = "Related articles mention o1, o2, o3 and o4."`
- **NOM** (no mention): `PRE + "Passage: " + P + "\nQuestion: " + Q + "\n" + FREE`
- **OPT-A** (options after): `PRE + "Passage: " + P + "\nQuestion: " + Q + "\n" + OL + "\n" + MC`
- **OPT-B** (options before, instruction-matched to OPT-A): `PRE + OL + "\n\nPassage: " + P + "\nQuestion: " + Q + "\n" + MC`
- **MEN-A** (natural mention after, instruction-matched to NOM): `PRE + "Passage: " + P + " " + MEN + "\nQuestion: " + Q + "\n" + FREE`
- **MEN-B** (the same sentence before): `PRE + "Passage: " + MEN + " " + P + "\nQuestion: " + Q + "\n" + FREE`
- **LET-A** (MMLU-style letters): `PRE + "Passage: " + P + "\nQuestion: " + Q + "\nOptions:\nA. o1\nB. o2\nC. o3\nD. o4\nAnswer with the letter of the correct option."`. Scored on the letter token, which is a single token in all four tokenizers.
- Gemma-2 rejects a system role. `chat_text` merges the system text into the user turn and records this in WRAPPER_USED.
- The design gives two instruction-matched before/after pairs, {OPT-A, OPT-B} and {MEN-A, MEN-B}, with NOM as the free-instruction reference.

## 3. Interventions (in-batch rows on the B prompt; tables are [|P|, D] per layer and channel, from the B, S, X and Z capture passes)
- **All formats:** ID (= KV_B, the in-batch self-clamp reference), K_S, V_S, KV_S, K_X, V_X and KV_X, from ℓ0 = 0.
- **NOM, OPT-A and LET-A:** add K_Z and V_Z.
- **NOM and OPT-A, exploratory:**
  - onset rows K_S and V_S from ℓ0 = round(0.3 L);
  - piece rows that change K or V at only one piece of the span: K_S^first and K_S^rest in OPT-A, V_S^first and V_S^rest in NOM.
  - Every other position and channel keeps B's value, set per row and position by the `per_row` [R, |P|] mask of `clamp_kv`.
- **Capture passes:** each runs on prompt + w. K and V at P are unaffected by w because of the causal mask.

## 4. Measures (exact)
- **Candidate continuations.** c_Y = the tokens of `" " + e_Y` after the prefill; j_Y = the index of the first non-whitespace token; w = c_Y[:j_Y] (shared by construction); dec_Y = c_Y[j_Y].
- **Decision log-probability.** ℓ_Y(r) = log softmax at the last position of prompt + w, evaluated at dec_Y, under row r (full vocabulary). For LET-A, dec_Y is the letter of e_Y's option.
- **Shifts.** Δ_Y(r) = ℓ_Y(r) − ℓ_Y(ID).
- **Identity in each channel.**
  - ID_K = ½[(Δ_S(K_S) − Δ_S(K_X)) + (Δ_X(K_X) − Δ_X(K_S))]; ID_V is the same with the V rows.
  - s_ID = mean ID_K / (mean ID_K + mean ID_V), a ratio of means over the population, recomputed in every bootstrap resample.
- **Continuation (NOM, OPT-A).**
  - L_S^cont(r) = Σ_{t > j_S} log p(c_S[t] | prompt + c_S[:t]; r), teacher-forced.
  - d_C^cont = mean[L_S^cont(C_S) − L_S^cont(ID)] for C ∈ {K, V, KV}; d_C^dec = mean[ℓ_S(C_S) − ℓ_S(ID)].
  - Interaction shares: I_cont = (d_KV^cont − d_K^cont − d_V^cont)/d_KV^cont, and I_dec likewise.
- **Full string (secondary).** L^full = the sum over all tokens of c_Y; ID_K^full and s_ID^full are defined as above.
- **Generation.**
  - Greedy, at most 16 new tokens, stopping at newline or EOS. The clamp acts in prefill only, with a fresh cache per call.
  - g1 = the first generated token that decodes to non-whitespace.
  - match(g, e) holds if norm(first line of g) equals norm(e) or starts with norm(e) + " ". norm is SQuAD normalisation: lowercase, strip punctuation and articles, collapse whitespace.
  - Decision flip: φ_S^dec(r) = 1[g1 = dec_S]. Full flip: φ_S^full(r) = match(g, e_S).
  - Accuracy: a^dec(r) = 1[g1 = dec_B]; a^full(r) = match(g, e_B). For LET-A, the first non-space character must be e_B's letter.
  - Hybrid rate: h(r, f) = P(¬match(g, e_B) | g1 = dec_B).
  - Under head ablation (HeadSplice), generation is replaced by its exact equivalent: the argmax chain under teacher forcing (unit-tested equal to greedy generation).
- **Option margin**, reported under every ablation: m_B = ℓ_B − max_{Y ∈ {S, X, D}} ℓ_Y.
- **Competence.** competent(i, f) holds if match(g(ID), e_B), match(g(KV_S), e_S) and match(g(KV_X), e_X) all hold (letters for LET-A).
- **Candidate mass.** Σ_Y p(dec_Y | ID) over the four options, reported per cell.

## 5. Models (BF16; sdpa except Gemma-2, which uses eager attention because of soft-capping; use_cache=False for scoring)
- **Fresh families, never run in stages 1–7:**
  - meta-llama/Llama-3.1-8B-Instruct; ungated mirror unsloth/Meta-Llama-3.1-8B-Instruct;
  - google/gemma-2-9b-it; ungated mirror unsloth/gemma-2-9b-it.
  - The pilot validity check used the mirrors' tokenizers.
  - Ungated fresh-family fallback if neither repository is accessible: ibm-granite/granite-3.1-8b-instruct. Its tokenizer validity is rerun in the preflight, and failing items are dropped and counted.
- **Continuity models:** Qwen/Qwen2.5-7B-Instruct and mistralai/Mistral-7B-Instruct-v0.3, which have the stage-6 template reader rankings.
- Hub revisions are recorded in the preflight. A mirror, if used, is a disclosed deviation.

## 6. Gates
- **A-G0 (FP32 exactness, CPU, Qwen2.5-0.5B, unit tests, 1e-4):**
  - The KV_S row equals the unclamped S run at the decision position and on every continuation token.
  - The ID row equals clean B.
  - Batched rows equal single runs.
  - Onset and piece rows equal hand-built clamps, and first plus rest recombine to KV_S.
  - Generation under KV_S equals greedy generation of the S prompt.
  - The teacher-forced argmax chain equals greedy tokens.
  - w is identical across candidates.
  - Multi-position HeadSplice/HopSplice behave as in §9.
- **A-G1 (BF16 floor, per model × format):**
  - mean_i max_Y |ℓ_Y(KV_S row) − ℓ_Y(S capture pass)| ≤ 0.3 nats, and likewise for ID versus B;
  - g(KV_S) = g(S run) in ≥ 97% of items.
- **A-G2 (competence):** n_comp(m, f) ≥ 80 in each format a prediction uses.
- **A-G3 (emitted form, competent items):** median p(dec_B | ID) ≥ 0.5, and g1(ID) = dec_B in ≥ 95% of items.
- **A-G4 (s_ID defined):** mean ID_K + mean ID_V ≥ 2 nats in the cell.
- **Evaluability.** A model is evaluable for a prediction if every gate passes in every format the prediction uses. A-G0 failing makes all predictions not evaluable.

## 7. Populations and statistics
- **Population.** For model m and the set of formats F that a prediction compares, the population is C_m(F) = the items competent in every f ∈ F (an intersection, fixed per prediction below). Results on all valid items are reported as secondary.
- **Intervals.** Two-stage cluster bootstrap: resample the 35 E-articles with replacement, then items within each sampled article. 10,000 resamples, seed 20261010, 95% percentile intervals. Every ratio and paired difference is recomputed within each resample.
- **Verdicts** use point estimates and the bounds named.
  - **Combination across models:** a prediction is MET if it is met in **every evaluable model**, with ≥ 3 models evaluable including ≥ 1 fresh family. It is NOT MET if it fails in any evaluable model, and NOT EVALUABLE otherwise. The count k/4 is reported.
- **Multiplicity.** Holm–Bonferroni over every "CI excludes" or bound component in Part A is reported as a sensitivity analysis, using one-sided bootstrap p = 2 × the fraction of resamples beyond the bound. Any verdict that changes under Holm is flagged.
- **Risk labels.** Each prediction is labelled **risky** (not implied by data in hand), **moderate**, or **low-risk control**. The paper reports the met rate separately for risky predictions, as the stats reviewer asked.

## 8. Confirmatory predictions (per model; combined as in §7)
- **A1, lookup with later options. Risky: no natural-text data in hand; 0.5B pilot s_ID 0.43.**
  - On C(OPT-A): s_ID(OPT-A) ≥ 0.5 with lower bound ≥ 0.35, and ID_K > 0 with CI excluding 0.
  - **A1b, multi-token spans (risky):** the same restricted to items with |P| ≥ 3 in that tokenizer (needs ≥ 40 items): s_ID ≥ 0.4.
  - **A1c, letters (moderate):** s_ID(LET-A) ≥ 0.5 with lower bound ≥ 0.35.
- **A2, copy without a later mention. Moderate; 0.5B pilot s_ID −0.01.**
  - s_ID(NOM) ≤ 0.2 with upper bound ≤ 0.3, and ID_V(NOM) > 0 with CI excluding 0.
- **A3, crossover (headline number):** on C(OPT-A, NOM), paired s_ID(OPT-A) − s_ID(NOM) ≥ 0.4 with CI excluding 0.
- **A4, position (low-risk control):** on C(OPT-A, OPT-B), ID_K(OPT-B)/ID_K(OPT-A) ≤ 0.15 with upper bound ≤ 0.25, and paired ID_K(OPT-A) − ID_K(OPT-B) > 0 with CI excluding 0. A negative sign is allowed and reported.
- **A5, fresh test of the mention effect. Risky: the one effect found post hoc; new data, wording and families.** On C(MEN-A, MEN-B, NOM):
  - (i) paired ID_K(MEN-A) − ID_K(MEN-B) > 0 with CI excluding 0;
  - (ii) paired s_ID(MEN-A) − s_ID(NOM) ≥ 0.10 with CI excluding 0.
- **A6, behavioural double dissociation by generation (risky).** On C(OPT-A, NOM):
  - OPT-A: φ^dec(K_S) ≥ 0.5, and φ^dec(K_S) − φ^dec(V_S) ≥ 0.3 with paired CI excluding 0.
  - NOM: φ^dec(V_S) ≥ 0.5, and φ^dec(V_S) − φ^dec(K_S) ≥ 0.3 with CI excluding 0.
  - The LET-A line φ(K_S) ≥ 0.5 is reported with the same rule.
- **A7, MCQ answers survive value corruption that breaks free form (risky, practical).**
  - OPT-A: a^full(V_Z) ≥ 0.8. LET-A: a(V_Z) ≥ 0.8. NOM: a^dec(V_Z) ≤ 0.5.
- **A8, multi-token answers: the answer's own prefix is a later mention (risky, novel).**
  - Population: C(NOM, OPT-A) items whose c_S has ≥ 1 token after dec_S in that tokenizer.
  - (a) NOM continuation is conjunctive: I_cont ≥ 0.5 with lower bound ≥ 0.3, given d_KV^cont ≥ 2 nats.
  - (b) The NOM decision token is additive: I_dec ≤ 0.25.
  - (c) The OPT-A continuation is read from the option words (low-risk): |d_KV^cont(OPT-A)| ≤ 0.2 × d_KV^cont(NOM).
  - (d) Behaviour: h(K_Z, NOM) ≥ 0.2 with lower bound > 0.1, and h(K_Z, OPT-A) ≤ 0.05. Each needs ≥ 40 items with g1 = dec_B. In words, key corruption leaves the first piece right and the rest wrong only in free form.
  - Secondary, its own verdict line: d_K^cont(NOM) < 0 with CI excluding 0. This is a key match that retrieves the base value, a principled negative key read.

## 9. Reader heads on natural text and the behavioural dissociation (Qwen2.5-7B, Mistral-7B; eager)
### 9.1 Definitions
- **Option rows.** G = rows overlapping the four option strings in OPT-A; G_Y = rows of option Y.
- **a3.** a3(l,h) = ½[(A^{K_S} − A^{ID})[G_S→P] + (A^{ID} − A^{K_S})[G_B→P]], where A[G_Y→P] = mean_{r ∈ G_Y} Σ_{p ∈ P} A_{lh}[r, p]. It is averaged over the 60 R items and ranked; N* = top k* (k* = 40 for Qwen, 52 for Mistral).
- **Template set.** T* = the first k* of `arms.P1.rankings.a3` in the committed `results/gpu_stage6/heads/<model>.json` (sha256 asserted).
- **Random sets.** The first k* heads of three `numpy.default_rng(2)` permutations, as in H.
- **Effects.** m = ℓ_S − ℓ_B. d_full = m(K_S) − m(ID). d_G = the effect of RowSplice when only G sees K_S at P.
- **R(k) and KO(k).** Defined as in H, on the 80 evaluation items, with an in-batch none row and an all-G row.
- **Region mean-ablation.**
  - The region Q+ is every row after the question's last character, including the decision position and teacher-forced continuation rows. This makes the ablation non-vacuous in NOM, where it hits the instruction and answer rows.
  - Each head's o_proj-input slice is replaced by μ_f(l,h), its mean over the R items and the Q+ rows of format f.
  - ρ_K = mean ID_K(ablated)/mean ID_K(none); the ID_V ratio is defined likewise.
- **Hop 2.** HopSplice r_ans over rows G, with K_S clamped at P.

### 9.2 Gates and predictions
- **Gates:**
  - **HA-G0:** FP32 tests (§10).
  - **HA-G1:** the BF16 floor as Gate a2 of H.
  - **HA-G2** (row-restricted analogue on natural text): mean d_full ≥ 3 nats and d_G/d_full ≥ 0.6. Otherwise the option rows are not the readers, and HA1–HA4 are not evaluable (reported).
- **HA1, sparse natural readers (moderate):** R_N(k*) ≥ 0.7 with lower bound ≥ 0.6 and KO_N(k*) ≥ 0.7; the random sets give R and KO ≤ 0.15 (means over the three draws).
- **HA2, template readers transfer to natural text (risky):** R_T(k*) ≥ 0.5 and KO_T(k*) ≥ 0.5. |T* ∩ N*| is reported with its hypergeometric P.
- **HA3, the MCQ readers do not carry free-form reading (risky; the requested dissociation).** With N* mean-ablated over Q+:
  - (a) OPT-A: ρ_K ≤ 0.5 with upper bound ≤ 0.6;
  - (b) NOM: ID_V ratio ≥ 0.8 with lower bound ≥ 0.7, and a^dec(abl) − a^dec(none) ≥ −0.05 with lower bound ≥ −0.10;
  - (c) each random set: OPT-A ρ_K ≥ 0.8, NOM ID_V ratio ≥ 0.9.
- **HA4, copy fallback (moderate; reconciles H3 with G7b on natural data).** With N* ablated in OPT-A: ID_V rises (paired CI excluding 0) and a^full ≥ 0.85.
- The option margin m_B of the unablated model is reported under every condition, so that general damage (I6's margin compression) is visible against the random sets.
- **Optional:** the same at Llama-3.1-8B with N* only (HA2 not applicable; exploratory).

## 10. Exploratory (labelled; no verdicts)
- **E1.** Onset 0.3 L: the fraction of d_K and ID_K that survives a mid-depth onset on natural text. This answers 'the key channel = layer-0 token matching'.
- **E2.** Piece rows: which span pieces carry the key read in OPT-A and the copy in NOM.
- **E3.** Moderators: type (PERSON/PLACE/NUMBER), D_in, |P| bins, and templated against natural s_ID(OPT-A) at Qwen-7B and Mistral-7B.
- **E4.** YEAR stratum (80 items; NOM and OPT-A). Written expectation: digit-splitting tokenizers (Qwen, Mistral, Gemma) put the decision digit after a shared content prefix that re-mentions the span. So their I_dec ≥ 0.5 (conjunctive), while Llama-3 (3-digit chunks, the decision is the first piece) gives I_dec ≤ 0.25. The same items, with tokenizer-dependent predictions.
- **E5.** Practical cache compression: KIVI-style fake quantization of the passage tokens' pre-RoPE keys (per-channel, groups of 32 tokens) or values (per-token, groups of 32 channels) at 2 and 3 bits, in every layer. Accuracy a^full is reported in NOM, OPT-A and LET-A. Expected, not scored: value quantization hurts NOM more than the MCQ formats.
- **E6.** Full-string s_ID and X-double-difference continuation measures.
- **E7.** Heads extras: T* ablation; option-rows-only ablation (the H3 analogue); r_ans; copy heads ranked by direct logit attribution on NOM R items, ablated at the answer row (accuracy in NOM against OPT-A); LET-A under ablation.
- **E8.** The Holm sensitivity analysis.
- **Optional add-on for whichever part owns objection 5 (about 5 GPU-min per model):** the unchanged `experiments/format_factorial.py` at seed 8 (fresh templated cores) with arms P1, AFTER, BEFORE, POST, PRE, NONE and LETTER at Llama-3.1-8B and Gemma-2-9B, giving the templated 2×2 in the new families.

## 11. Power (simulated: 35 article clusters, ICC 0.15, coefficients of variation from the pilot; `power.py`)
- **A1** (s ≥ 0.5 and lower bound ≥ 0.35): per model, SE ≈ 0.03.
  - Power 0.92–0.95 at a true s = 0.55 and ≈ 1.00 at 0.60 (n = 120–200).
  - All four models together: 0.72–0.82 at 0.55 and ≥ 0.98 at 0.60.
  - At a true 0.50, power is 0.5 by construction, so this is the risk point. Templated 7B values were 0.79–0.87.
- **A2:** power ≈ 1.0 if the true s ≤ 0.10, and 0.62–0.68 (all four) at 0.15.
- **A5:** power 0.99 per model (0.95–0.98 for all four) if the true s_ID(MEN-A) = 0.20; 0.77–0.80 per model at 0.15. The templated sentence s_ID at 7B+ was 0.19–0.45.
- **A6:** power 0.97–0.98 per model (0.90–0.94 for all four) at a true flip difference of 0.40; 0.85–0.87 at 0.35.
- The expected n_comp is 150–200 per model (E ≈ 220 after the audit; the pilot competence was 5/7 and 6/7 at 0.5B).

## 12. Compute (A100-80GB, BF16; calibrated on stage 1: Qwen2.5-7B scored about 10k prompt tokens/s)
- **Per item, all six formats:** about 32k prompt tokens plus about 60 batched decode steps, roughly 4.5–5 s. Over about 220 items that is **15–18 min per model with sdpa (Qwen, Mistral, Llama) and about 23–27 min for Gemma-2 (eager)**.
- **Heads:** about 15 min per model (Qwen, Mistral; ranking about 2 min).
- **Exploratory passes** (E1, E2, E4, E5): about 8 min per model.
- **Overheads:** downloads and loads about 3 min per model; preflight and FP32 tests about 15 min.
- **Totals:**
  - Core (A1–A8, HA1–HA4): **about 2.1 GPU-h**.
  - With exploratory passes: **about 2.6 GPU-h**.
  - With the optional Llama heads and the templated add-on: about 2.9 GPU-h.
  - Lean fallback (n = 160, no LET-A, no exploratory): about 1.6 GPU-h.
- **Run order and deadline:** preflight → fresh families (Llama, Gemma) → Qwen, Mistral → heads → exploratory. A deadline (DEADLINE_H, default 3.0) skips the exploratory passes first and records each skip.

## 13. What would count against the account, and the pre-declared consequence for the paper
- **A1 or A3 not met** (natural OPT-A s_ID < 0.5, or crossover < 0.4): the key lookup is a property of short templated prompts. The title claim is restricted to templates, and the natural-text result is reported as a boundary.
- **A2 not met:** free-form reading of natural spans is not a copy at the decision token, and the 'copy' half of the dichotomy is withdrawn for natural text.
- **A4 not met:** position does not decide on natural text; the 2×2 claim is restricted to templates.
- **A5 not met:** the sentence effect is template-specific. It moves from 'later mentions' to 'option lists' in the abstract.
- **A6 or A7 not met:** channel attributions do not translate into behaviour, and the practical claim is dropped.
- **A8 not met:** multi-token continuations are read positionally (successor heads), not by key lookup, and the 'answer prefix as a later mention' refinement is wrong.
- **HA2 not met:** the template readers do not generalise to natural text.
- **HA3 not met:** the MCQ readers also drive free-form reading, so 'mechanisms localised in MCQ format do not explain free-form behaviour' is withdrawn.
- **In every case:** all verdicts, including failures, enter the claim-status table, and the abstract states the met rate among risky predictions.


### code_plan

All new code reuses `ckeys/clamp.py`, `ckeys/encoding.py`, `ckeys/headsplice.py`, `experiments/stage6_heads.py`, `experiments/format_factorial.provenance`, `experiments/ioi_factorial.device_name` and `experiments/row_restricted_keys.encode_with_offsets`/`RowSplice`. Reference implementations from the pilot are in /tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA/:
- `build_items5.py`: item rules, typing, substitutes S/X/Z ("Y" in the file), D, option order;
- `check_prompts2.py`: four-tokenizer, six-format validity;
- `pilot_natural.py`: the format strings `parts()`, `encode()`, `cont_ids()`, `norm()`/`matches()`, and full-string plus generation scoring;
- `pilot_fast.py` and `pilot_cont.py`: the decision-token batch and the continuation interaction;
- `power.py`.

**New files**

1. `ckeys/squad_items.py`, porting `build_items5.py`.
   - Functions: `load_squad(path)` (asserts sha256 95aa6a52…), `etype`, `wh_head`, `person_shaped`, `subtype`, `pools`, `substitutes(item, pools)`, `distractor(item)`, `first_content(tok, s)`, `span_ok(tok, ctx, start, a, b)`, `build_items(path, tokenizers)`, `split_articles(items, seed=20261010, n_rank=12)`, `cap(items, per_article, per_paragraph, seed)`.
   - The fixed word lists live in the module.
   - The output is a list of dicts {id, art, par, title, context, question, answer, start, sub, S, X, Z, D, D_in, options, stratum, split}.
   - `data/stage8a_items.json` (ids, substitutes, order, split, audit flag) and `data/stage8a_type_audit.tsv` are committed before finalisation; the preflight rebuild must reproduce them.
2. `ckeys/natural_formats.py`.
   - `FORMATS = ("NOM", "OPTA", "OPTB", "MENA", "MENB", "LETA")` and `parts(fmt, item) -> (pre, post)` (exact strings of §2).
   - `encode_item(tok, item, fmt, ent) -> (text, ids, offsets, (c0, c1))` via `chat_text`, and `span_positions(offsets, c0, c1)`.
   - `cont_ids(tok, text, ids, ent) -> (c, j)` asserts prefix stability; `decision_ids(tok, text, ids, item, fmt)` asserts that w is shared and the decision tokens are distinct; `letter_ids`.
   - `option_rows(text, offsets, item)` returns G and G_Y; `qplus_rows(text, offsets, item)`; `norm`; `matches`.
3. `experiments/natural_factorial.py`.
   - `run_item(model, tok, item, fmt, dev, part)`:
     - validates equal lengths and diffs ⊆ P;
     - runs the capture passes on prompt + w with `capture_kv(model, P, layers)`;
     - builds row tables with `stack_rows`; the onset and piece rows use per-row, per-layer lambdas and the `per_row` [R, |P|] mask;
     - scores the decision batch (`logits_to_keep=1`);
     - for NOM and OPT-A, scores full strings with one batch per target S/X over rows ID, K_S, V_S, KV_S, K_X, V_X, KV_X;
     - generates under `clamp_kv` (HF `generate`, `max_new_tokens=16`, a stop on newline via a StoppingCriteria, a fresh cache);
     - stores per-row ℓ for B/S/X/Z/D, continuation sums, generations, g1 and the candidate mass.
   - `main()` takes `--model --revision --formats --items data/stage8a_items.json --split E --part core|explore --dtype --test`, writes atomically and records provenance (commit, versions, WRAPPER_USED, attention implementation, revisions, skipped items with reasons).
   - TEST_MODE: Qwen2.5-0.5B FP32 CPU, 2 items, passages windowed to ≤ 320 characters (`window()` from `pilot_fast.py`).
4. `ckeys/kvquant.py` (exploratory E5): `fake_quant(x, bits, axis, group)` and a context manager hooking `k_proj`/`v_proj` at the passage positions in every layer.
5. `experiments/natural_heads.py`.
   - `class NaturalStage(Stage6)` overrides:
     - `prep` (P list, G and G_Y, Q+ rows, decision ids via `natural_formats`);
     - `ks_of` (returns {l: [|P|, D]});
     - `base_runs` (a3 over G_Y→P sums, the o_proj-input means over Q+ for NOM and OPT-A on R).
   - It reuses `splice`/`curves` (KS = {1, 2, 5, 10, 20, k*, 2k*}; no single-head or LOO grids), `hop_cache`/`hop_batch` (rows G) and RowSplice for d_G.
   - New `region_ablation(d, conds, MU)` runs the natural decision batch and the teacher-forced c_B/c_S argmax chains under HeadSplice "ablate" with masks over Q+ (means broadcast from [H, hd] per head via `mean_table` with repeated rows). Conditions: none, N*, T*, rand0–2 (+ option-rows-only N* and DLA copy heads, exploratory).
   - T* is loaded from the committed stage-6 JSON (sha256 asserted).
6. `analysis/stage8a_score.py` (with `analysis/stage8a_parts/`).
   - Gates A-G0–G4 and HA-G0–G2, populations C_m(F), and `cluster_boot(df, stat_fn, clusters='art', n=10000, seed=20261010)` (two-stage).
   - Verdicts A1–A8 and HA1–HA4 with the cross-model rule, risk labels and Holm sensitivity.
   - Prints `results/gpu_stage8/STAGE8A_SCORE.txt`; TEST_ tags in TEST_MODE.
7. `scripts/gpu_stage8.sh`, PART=A steps: preflight (download SQuAD, assert hash, rebuild and compare items with the run's tokenizers, Hub revisions), factorial per model (Llama, Gemma, Qwen, Mistral), heads (Qwen, Mistral, optional Llama), explore, score. Guards are copied from `gpu_stage7.sh`:
   - refuse a DRAFT entry J, a modified tree, or code different from the 'Finalise preregistration J' commit;
   - pin transformers 5.18.0;
   - FP32 pytest first; DEADLINE_H; FORCE_STEPS; KEEP_CACHE.
   - HF_TOKEN optional: without it, the unsloth mirrors are used and recorded.

**Changed file**

- `ckeys/headsplice.py`, minimal:
  - `HeadSplice._khook` must accept a span table `ks[l]` of shape [|P|, D] when `self.pos` is a list (assert `len(pos) == k.shape[-2]`; write `out[:, pos] = k[None]`) and never route the P axis through `_bcast`.
  - Today a [|P|, D] table is passed to `_bcast` as if P were the batch axis. That fails when |P| ∉ {1, B} and silently mis-assigns when |P| equals the batch size B. The 1-D (single-position) path stays byte-identical in behaviour, and the stage-6/7 tests must still pass.

**Tests (FP32, CPU, Qwen2.5-0.5B, windowed passages; about 3–5 min on this 11-GFLOPS CPU)**

- `tests/test_squad_items.py`:
  - rules (exactly once, absent from the question, majority, substitutes absent and word-disjoint, same subtype, decision tokens distinct per cached tokenizer, else skip with a reason);
  - diffs only inside P in every format;
  - split and cap determinism; the builder's output hash equals the committed items.
- `tests/test_natural_clamp.py` (A-G0):
  - KV_S row == S run at the decision position and on every continuation token (≤ 1e-4); ID == clean B;
  - batched == single; onset/piece rows == hand clamps; first + rest == KV_S;
  - `generate` under KV_S == greedy S; teacher-forced argmax chain == greedy tokens; w shared.
- `tests/test_headsplice_span.py`:
  - multi-position HeadSplice: all heads in all rows == `clamp_kv` K_S at P; empty == clean; rows G == RowSplice(G) with a [|P|, D] table;
  - the |P| == B case;
  - region ablation with each head's own outputs as means == clean;
  - HopSplice answer-only with the K_S-run cache == full.
- `tests/test_stage8a_score.py`: synthetic data with known s_ID, I_cont, flips and hybrid rates; cluster-bootstrap reproducibility; evaluability and cross-model verdict logic; Holm.


### pilot_results

Everything ran on this container's CPU (4 cores, measured at about 11 GFLOPS FP32, so the model pilots are small). It is disclosed as pilot data; nothing came from a GPU, and no 7B+ model was run. Scripts and outputs are in /tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partA/.

1. **Data scan** (`scan_squad.py`): SQuAD v1.1 dev, 10,570 questions, sha256 95aa6a52….
   - 3,081 questions have a typed majority answer that occurs exactly once in the passage and not in the question.
   - With the tightened typing and the S, X, Z substitutes (`build_items5.py`), there are 706 first-token items (PERSON 381, NUMBER 191, PLACE 134; ORG dropped) and 529 YEAR shared-prefix items.
   - A first version typed only by wh-word paired 'Groton School' with 'Richard Lindzen'. Head-noun typing, the PLACE pool rule and title stoplists fixed most cases. Residual noise ('Political' as PLACE, 'Digital Spy' as PERSON) motivated the complete blind audit (§1.5).
2. **Tokenizer validity** (`check_prompts2.py`; tokenizers of Qwen2.5-7B, Mistral-7B-v0.3, and the unsloth mirrors of Llama-3.1-8B and Gemma-2-9B; all six chat-wrapped formats):
   - **697 items are valid in every tokenizer and format.** The B/S/X/Z prompts have equal length and differ only inside the span, w is shared, and the five decision tokens are pairwise distinct.
   - Median prompt length is 207–255 tokens (maximum about 770).
   - Span lengths are mostly 2–4 tokens; single-token spans are 14% (Qwen), 11% (Mistral), 36% (Llama) and 16% (Gemma).
   - The letters are single tokens in all four tokenizers. Gemma-2 rejects the system role, which `chat_text` merges into the user turn.
   - Seeded article split: R = 88 items over 12 articles; E = **239 items over 35 articles** (PERSON 110, PLACE 66, NUMBER 63; D_in 54).
3. **Exactness** (Qwen2.5-0.5B, FP32): over multi-token spans, the KV_S clamp row reproduces the unclamped S run at the decision token within 1e-5 to 3e-5 nats (4 item-format checks).
4. **Effect pilot** (`pilot_fast.py`/`pilot_fast2.py`; Qwen2.5-0.5B; passages windowed to ≤ 320 characters; 7 items in NOM and OPT-A, 1 item in OPT-B and MEN-A; BF16 for 6 of the items; about 25 CPU-minutes):
   - **NOM:** ID_K −0.13 (sd 0.58), ID_V +11.01 (sd 4.94), **s_ID −0.01**. Competent 5/7; median p(dec_B) 0.96.
   - **OPT-A:** ID_K +2.64 (sd 1.78), ID_V +3.56 (sd 1.39), **s_ID 0.43**. Competent 6/7; median p(dec_B) 0.99; candidate mass 0.996.
   - **The single item in OPT-B and MEN-A:** OPT-B ID_K −0.52 against ID_V 6.23; MEN-A ID_K +0.61 against ID_V 5.89.
   - **Decision flips:** NOM K_S 0.00 / V_S 0.86; OPT-A K_S 0.14 / V_S 0.57. At 0.5B the value still wins in OPT-A, consistent with the small-model templated s_ID of 0.36 at 1.5B.
   - **Full-match flips:** NOM V_S 0.43, half the decision flips. The generations were hybrids such as 'Robert M. Newton' and 'Wade Gertner'. This motivated A8.
5. **Continuation pilot** (`pilot_cont.py`; FP32; 10 multi-token PERSON/PLACE items; rows ID, K_S, V_S, KV_S teacher-forced on e_S):
   - **NOM decision token:** dK +1.07, dV +13.90, dKV +15.62, **interaction share 0.04**.
   - **NOM continuation:** dK **−10.24**, dV +3.95, dKV +15.34, **interaction share 1.41**. The pieces after the first need the span's key and value jointly, and the key alone reads negatively.
   - **OPT-A decision token:** dK +9.21, dV +8.49, dKV +11.39 (redundant).
   - **OPT-A continuation:** |d| ≤ 0.31 in every row, dKV 0.00. It is read from the option words.
6. **GPU calibration:** the stage-1 log shows Qwen2.5-7B on an A100 scoring about 10k prompt tokens/s, which was used for §12.
7. **Power simulation** (`power.py`): the numbers are in §11.
8. **Template rankings:** the stage-6 a3 rankings T* were extracted (k* = 40 for Qwen2.5-7B, 52 for Mistral-7B); see `tmpl_rank_*.json`.


### paper_payoff

**Main-text figure (new; replaces part of the ledger).** "Later mentions decide on natural reading comprehension". The data are SQuAD passages with multi-token answers, in 4 models from 4 families, 2 of them new.
- (a) s_ID by format: OPT-A/LET-A high; NOM and OPT-B about 0; MEN-A intermediate. Templated values are shown as ghost markers.
- (b) Generation flip rates under key-only against value-only swaps, in MCQ against free form: the behavioural double dissociation.
- (c) Accuracy under value corruption of the answer span: MCQ intact, free form broken.
- (d) d_K, d_V and d_KV on the decision token against the continuation, in NOM against OPT-A: the answer's own prefix is a later mention.

**One table** gives the head transfer and the dissociation (HA1–HA4). Appendix tables cover the per-model detail, gates, the audit and the moderators.

**Abstract sentence.** "On natural reading-comprehension passages with multi-token answers, in four model families including two not used before, swapping only the cached keys of the answer span flips multiple-choice answers and swapping only its values flips free-form answers. Within a free-form answer, every piece after the first is looked up through the span's key from the model's own prefix."

**Practitioner sentence.** "Mechanisms localised and KV-cache methods evaluated in multiple-choice format can miss what free-form answering uses: MCQ answers survive corruption of the passage entity's cached values that breaks free-form answers."

**Score movement.**
- Contribution, all four reviewers: generality was each reviewer's first or second 'would raise score' item (+1 claimed by the generalist, skeptic and mechinterp reviewer; +1 'fresh-sample' and +0.5 'emitted form' by the stats reviewer).
- The practical consequence answers the generalist's +1 request.
- A8 and HA2 give the mechinterp reviewer a mechanistic result that QK/OV does not state.
- Soundness: emitted-form scoring, competence by generation, a cluster bootstrap over articles, new families, s_ID-relative margins and risk-labelled predictions answer the stats and skeptic concerns.
- If A1–A6 are met, a realistic move is from 4 to 6 for three reviewers. Met or not, the risky preregistered outcome is reported either way, which the stats reviewer explicitly values.


### risks

- **Counterfactual compliance.** Models may answer the original entity from memory under the S, X and Z passages (knowledge conflict). This lowers n_comp. Mitigations: 239 E items, Gate A-G2 (n_comp ≥ 80), and reporting all-item results. The pilot had 5/7 and 6/7 competent at 0.5B, and 7–9B models usually follow context better.
- **A1 boundary.** If the true natural s_ID(OPT-A) is about 0.5, A1 fails half the time. Accepted: it is the risky prediction, the 0.5B pilot gave 0.43, and the templated 7B values were 0.79–0.87.
- **Type noise in substitutes** makes some counterfactual passages odd. Mitigated by the complete blind audit (§1.5) and type moderators (E3).
- **Hybrid generations** make full-match flips undercount; they are predicted (A8). The flip and NOM-accuracy criteria are therefore set at the decision token, full-match is reported, and OPT-A accuracy uses full match (continuation copied from the options).
- **Multi-token key read concentrated on the first piece.** The first-token rule could make the 'multi-token' key read effectively a first-piece read. E2 (piece rows) and A1b report this, and E4 (YEAR) tests a stratum where the decision token is not the first piece.
- **Gated weights.** Without HF_TOKEN, the unsloth mirrors are used and disclosed, after the preflight confirms the tokenizer and item rebuild. If they are unavailable too, granite-3.1-8b-instruct is the ungated fresh family (tokenizer check in the preflight; some items may drop).
- **Gemma-2.** Eager attention is about 1.5× slower; the system turn is merged; the sliding window is irrelevant at < 1k tokens.
- **HeadSplice span extension** has a silent mis-broadcast hazard when |P| equals the batch size. It is covered by a dedicated test, and stage-6/7 tests must still pass.
- **BF16 noise in the interaction shares** (A8). Mitigated by in-batch reference rows, Gate A-G1 and an evaluability floor of d_KV ≥ 2 nats. The 0.5B pilot interaction (1.41 vs 0.04) is far from the bounds.
- **HA3(b) could fail for a real reason.** Duplicate-token heads may help free-form QA locate the passage region from question-passage word overlap. Q+ starts after the question to limit this, and a failure would be reported as such ('the readers also serve free form').
- **Contamination.** SQuAD is likely in pretraining data. Every identity measure uses the counterfactual S and X targets against the in-batch B reference, so memorising B affects only the baseline.
- **Budget.** The core is about 2.1 GPU-h and about 2.6 with exploratory passes, against 8–12 h for all of stage 8. The deadline drops exploratory passes first; the lean fallback is about 1.6 h.
- **Not covered by Part A:** the IOI negative reads and LIST-BEFORE sign (A8's secondary only proposes a mechanism), the intervention-family objection (3) and the anonymous-submission objection (8). Those belong to other parts.


