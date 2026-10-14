# V5 plan (final): sections, floats, notation, bibliography and checks

This is the single plan that the v5 writers follow. Writers work in parallel, one per file, so each section below is
self-contained: what the file must say, in which order, with which macros, labels, floats and word budget. Where a
sentence is binding (pre-written in the preregistration, or decided here), its exact wording is given.

Sources: `docs/PREREGISTRATION.md` (entry P-2026-10-10-J: common part, Parts A–D with their pre-written tables, C-6,
C-12, D-13, and the four outcome records ending with "The entry in total"); `docs/stage8_design/partE/*` and
`SYNTHESIS.md` §E (E-1..E-3), `critic_rewrite.md` (RC1–RC11) and `design_rewrite.md`; `paper/BRIEF.md` (hard rules 1–6);
the macro index `ALL_MACROS.md` (groups S8A–S8D, S8X); the v4 sections; the three planner reports (reviewer, story,
rigor). A layout test with the floats specified here and realistic filler text was compiled with tectonic in a scratch
copy of `paper/` (Section 10): with the binding texts in place and 4,150 words of running text, the main text ends on
page 8 with about 0.3 page to spare; the budget here is 4,075 words.

---------------------------------------------------------------------------------------------------------------------

## 0. How the three plans were judged, and what this plan takes from each

| Criterion | Reviewer plan | Story plan | Rigor plan |
|---|---|---|---|
| Fidelity to outcome records (veto items) | 9/10. No veto. Small defects: keeps "every published verdict" in the J-B6c text (it misdescribes the discovery-sample verdicts and hints at an earlier publication); "a learned remap" in the abstract is ambiguous (the remap that is read through the value is the one trained at the writing token, not the released one); counts "185 items from 48 articles" (48 is the whole dev set; the items come from 34 articles); one sentence leans on the "match without the read at 1.5B/3B" framing that J-D6-ROUTE undercuts | 7.5/10. Puts a sign clause in the abstract although the designated primary line of the sign clause (J-D8-IOI) is not evaluable and no other sign line was planned for the abstract (E-2); uses "partly", which E-2 reserves for MET IN PART; needs a new macro for the family count; Table 1 row on scale says "read near zero" at 1.5B, which J-D6-ROUTE contradicts for the injected flag | 8/10 with one veto: its novelty paragraph says the channel attribution is not predicted by the model's own read "except for steering vectors close to the natural write", which contradicts outcome (e) ("Even steering vectors close to the natural edit depart"). Otherwise the most exhaustive (failure map, forbidden list, claim map) |
| Clarity of the main story | 8/10 | 9/10 (composite Fig. 1 puts Claim 1's evidence on page 2; one-sentence takeaway) | 6.5/10 (dense; Section 3 at 1,170 words; status codes in Table 1) |
| Coverage of reviewer objections | 10/10 (explicit map) | 7/10 | 8/10 |
| Feasibility in 8 pages | 9/10 (measured abstract and floats) | 9/10 (compiled with filler) | 7/10 (no compile; largest budget) |

**Base: the reviewer plan.** Grafted: from the story plan, the composite Figure 1 (schematic + fresh stories + generated
answers), the four-column Table 1 that E-3 asked for, the J-B6c wording fix (here "re-scored"), the "unless marked,
preregistered and held" status convention, the main-text crossover variant; from the rigor plan, the map of every
failed or not-evaluable prediction to a main-text location, the forbidden-text list, the claim map (crosswalk), the
abstract's "in every evaluable model" hedge, the sentence on what is taken from Anonymous (2026), and the ψ_K/ψ_V
depth caveat for the trained remap.

## 1. Decisions in one place

1. **Title kept verbatim** (the common part's G4 title; Part A's table keeps it). Running title "Looked Up or Copied?".
   No shorter variant: changing a preregistered title after the data would need its own disclosure. The conflict between
   the common part's title rule and Part A's table is disclosed in Appendix A.6.
2. **Abstract: four numbers** ("seven", a design count equal to `\JxFamiliesABCD`, written as a word; and the G4 risk
   sentence `\JxRiskyMetABCD` of `\JxRiskyVerdABCD` ... expected `\JxRiskyExpABCD`). No sign clause (E-2: its primary
   line J-D8-IOI is not evaluable). No passage-behaviour clause (J-A6a/c not met). No quantization, no head claim on
   passages, no small-model clause, no released remap, no post hoc ν.
3. **Claim 1 leads**: Figure 1 is a composite (a: schematic; b: fresh stories, eight models; c: generated answers), so
   the paper's main result is visible on page 2. Section 3 carries the most text.
4. **Claim 2 with its limits in the same sentences**: sufficient and read by key, not specific and not necessary as one
   direction; the question sets the sign through the same readers (body only); IOI's key/value sign dissociation is the
   novelty item (ii) (reduced per D-13); "the same frozen heads" is never said of IOI.
5. **Claim 3 = outcome (e)**: the C-12 (e) text is verbatim in Section 5; the released remap is the motivating case in
   text only (its figure moves to Appendix G); the ν pattern is labelled post hoc in its first words.
6. **Main floats**: Fig. 1 composite (`figure*`, p. 2), Table 1 QK/OV (`table*`, p. 3), Fig. 2 `fig_natural`
   (`figure*`), Fig. 3 `fig_flag` (`figure*`), Fig. 4 `fig_law` (`figure*`), Table 2 calibration (column). Everything
   else (fig_frames, fig_readers, fig_nu, tab_2x2, all new tables) goes to the appendix.
7. **Status convention**: unmarked = preregistered in the final round and held; tags in words otherwise (Section 2.2).
   No prediction or gate codes in the main text.
8. **Every prediction that did not hold (21) is named in words where it belongs in Sections 3–5** (map in Section 12);
   the not-evaluable ones (12) are named there or in one listing sentence; codes only in Appendix A.
9. **Discovery-sample family count is three** (Qwen, Mistral, OLMo, counted by developer as `\JxFamiliesABCD` does);
   v4's "four families" is not repeated.
10. **Prerequisites before writing starts** (Section 15): bibliography rebuild with about 20 new keys; `\fmtNull`,
    `\pct`, `\pctn` and `xfp` in `main.tex`; `tables/tab_status.tex` and three figure variants generated; the A–I post
    hoc risk labels built or their omission disclosed.

---------------------------------------------------------------------------------------------------------------------

## 2. Rules for every writer

### 2.1 Numbers
- Every result number is a macro from `paper/numbers.tex`, a cell of a generated table, or a string that appears
  verbatim in `docs/PREREGISTRATION.md` (then add `% verbatim: outcome J part X` at the end of the line). Never type a
  result from memory. Design counts may be words (eight models, four new families, 150 stories, 185 items, 34 articles,
  80 stories, 60/100 stories); they are verbatim in the entry.
- Percentages: `\pct{\Macro}` prints `round(100 × value) %`; `\pctn{\Macro}` prints the number without the sign, for
  ranges: `\pctn{\JbRangeBetaKAfterEightMin}--\pct{\JbRangeBetaKAfterEightMax}` gives "76–100 %". Use them only on
  non-negative share macros. Percentages that are verbatim in the record and have no macro ("31 % and 44 %",
  "4.5–12 %", "1.0 %", "12–17 %") are typed with the verbatim comment.
- A number appears in at most two places of the main text (Introduction to Conclusion; captions count; the abstract does
  not). Section 5.x below says where each number lives; do not add numbers elsewhere.
- Captions carry no result numbers: only design counts, interval levels and preregistered thresholds (±log 1.25, 0.25,
  0.7).
- Aim at three numbers per paragraph or fewer, except inside a pre-written sentence, which keeps its numbers.
- Never quote a macro marked NEGATIVE ZERO in the index (`\JbSixbSidAfterQwenSevenLo`, `\JbSixbSidBeforeQwenSevenLo`,
  `\JbSixbRBeforeQwenSevenLo`, `\JbSixbSidPreQwenSevenHi`, `\JbSixbSidPreQwenFourteen`, `\JbSixbRPreQwenFourteen`,
  `\JbSixbSidAfterOlmoSevenLo`, `\JbSixbSidNoneOlmoSevenLo`, `\JbSzeroPctIDKOptQwenSeven`, `\JbSzeroPctIDVOptQwenSeven`,
  `\JbSzeroPctIDKOptQwenFourteen`, `\JbSzeroPctIDVOptQwenFourteen`, `\JbLevelRPostGemmaTwoMin`, the
  Part-C entries `\JcLamEoneWpostThreeQwenSevenHi`, `\JcLamEoneAletterThreeQwenSeven`,
  `\JcLamEoneAletterThreeMistralSeven`, `\JcLamEtwoAletterThreeQwenSeven`, `\JcLamEtwoAoptThreeQwenSevenLo`,
  `\JcLamEtwoWpostThreeMistralSevenLo`, `\JcLamEtwoAletterThreeMistralSevenLo`, `\JcKappaEfiveDeNoneThreeLlamaEight`,
  `\JcLexNonlexEtwoThreeLlamaEight`, and the v4 macro
  `\psiBeforeMistral`); quote the range or the interval instead.
- A macro marked `[recomputed]` in the index (for example every `\JcNu*`, `\JcOneNe*`, `\JcOneLamThreeAbsMax`,
  `\JcFourDepth*`, `\JcFourPsi*`, `\JcReadaRandNat*`, `\JcThreePosthoc*`, `\JdSixNat*`, `\JdSixDiff*`, `\JaSidMena`,
  `\JaSidOptb`, `\JaSidMenb`, `\IbMarginU`, `\IbMarginA`, `\HfourDkstar*`) is printed only in a sentence labelled
  "(exploratory)" or "(post hoc)".
- Do not headline `\twoBeforeMax` (+0.01); say "every interval before the story contains zero or lies below it".

### 2.2 Status vocabulary (main text)
- No tag: preregistered in the final round (entry J) as a replication or extrapolation, and held.
- "(risky; held)": class R (prior below 0.8) and met. Measurement-validity lines are all class R.
- "(did not hold)" for a non-risky line not met; "(risky; did not hold)" for an R line not met.
- "(not evaluable: reason in words)".
- "(discovery sample)": a result of entries A–I (the v4 results); keep v4's "exploratory" or "post hoc" label where v4
  had one: "(discovery sample; exploratory)".
- "(exploratory)", "(post hoc)", "(post hoc reading)" for an interpretation that an outcome record gives after the data.
- The convention is stated once, at the end of Section 2 (text in 5.2 ¶5). Appendix A.7 maps each main-text sentence to
  its prediction code, verdict and class.

### 2.3 Terms and symbols
Terms (use exactly): **writing token** (the token that states the value; "the critical token" of
`\citet{anonymous2026fitted}`, said once); **later mention** (any token after the writing token that names a candidate);
**looked up** (a later token that already holds a candidate matches the writing token's key, so the key carries which
value was written) and **copied** (the identity travels in the writing token's value); **discovery sample** (the
\nModels{} models and stories of entries A–I); **fresh stories** (Part B's 150 new cores); **passages** (Part A's
counterfactual SQuAD items); **reader heads** (the 5 % of heads that read the writing token's key at the later mention);
**hop 1 / hop 2** (the later mention reads the writing token; the answer reads the later mention); **flag** (the reader
heads' summed write at the mentioned word's row, one vector per layer); **emitted-form score**; **key-only / value-only
clamp**; **channel-ratio law**.

Symbols allowed in the main text, each defined where first used:
- `$\mathrm{ID}_K$`, `$\mathrm{ID}_V$`, `$s_{\mathrm{ID}}$` (§2), `$r$` (§2; the sentence read relative to the list
  read), `$\beta_K$`, `$\beta_V$` (§2; Fig. 1c axes);
- `$\rho_K$` and `$\iota$`, `$\omega$`, `$\rho$`, `$\rho_{\mathrm{nat}}$` only as figure labels of Fig. 3, defined in
  its caption; `$\rho_K$` may appear once in §4 beside its words;
- `$\psi_K$`, `$\psi_V$`, `$\lambda$` (§5 ¶2, Eq. `eq:lambda`), `$\bar\lambda$` (the pooled mean of a test's
  statistics), `$\kappa$`, `$\sigma$` (Fig. 4b caption and §5 ¶4/¶6), `$\nu$` (§5 ¶5, post hoc).
- Everything else (φ, a3, R(k), KO(k), r_ans, π, Ψ, d_C, s_K, W, A_L, A_P, I_cont) is said in words or lives in the
  appendix. Model names: "Qwen2.5-7B", "Mistral-7B", "Llama-3.1-8B", "Gemma-2-9B", "Phi-4", "Falcon3-7B",
  "OLMo-2-7B", "Qwen2.5-14B", "Qwen2.5-72B", "Mistral-Small-24B", "Gemma-2-2B", "Qwen2.5-1.5B", "Qwen2.5-3B".

### 2.4 Format notation (`main.tex`)
Story formats are small-caps macros: `\fmtOpt` (options after the question, "Answer with exactly one choice"),
`\fmtListA` (the same list, "Answer with one word"), `\fmtLetter` (lettered options after), `\fmtPost` (a neutral
sentence naming every candidate after the story), **`\fmtNull` (new: `\textsc{null-sentence}`, a sentence of the same
position and length naming no candidate)**, `\fmtNone`, `\fmtSentB` (the sentence before the story), `\fmtBefore` (the
list before the story). The fresh stories use `\fmtListA`, `\fmtPost`, `\fmtNull`, `\fmtNone`, `\fmtSentB`,
`\fmtBefore`; `\fmtOpt` and `\fmtLetter` appear for the discovery sample, the interventions and the mechanism.
**Passage formats are words**, as in Fig. 2: "options after" (answered by name), "lettered options", "free form",
"options before", "a related-articles sentence after (before) the passage". Never print the internal arm codes (OPTA,
NOM, MENA, POST, Q_OUT, the P4/N4 slot codes, ...) in the main text.

### 2.5 Citations
Cite only keys in `paper/references.bib` after the rebuild of Section 9. Describe a cited work only as its abstract
supports. Wording fixed by checks against the abstracts:
- `prakash2025lookbacks`: lookbacks built on ordering IDs, found with causal mediation and abstraction (binding, answer
  and visibility lookbacks). Never "address via QK, payload via OV".
- `oh2026rebinding`: a compact attention-head circuit propagates binding; the binding signature is in query/key
  subspaces in Gemma and "carried primarily in key vectors" in Llama. Never "edited on the key side".
- `wu2026record`: operation-span transplants redirect which state is read (a routing record), and the writing model
  reads mainly the address. Never "keys and values moved together".
- `gurarieh2025mixing`: positional, lexical (through the bound counterpart) and reflexive (a direct pointer)
  mechanisms; the positional one is unreliable in middle positions of long lists. We do not test which of them our
  lookup corresponds to.
- `steele2026routing`: a generic value slot binds the value; a router at the query position selects belief or reality.
- `cheng2026steering`: steering vectors act mainly through the OV circuit, largely ignoring QK (refusal).
- `ok2026lost`: with options before the context, the causal mask keeps option tokens from attending to the context.
- `tulchinskii2024wise`: our second hop "resembles" their query–key selection; never "is".
- `mueller2025mib`: a benchmark of circuit and causal-variable localisation methods; do not say its tasks are
  multiple-choice.
- `dai2024binding`: a low-rank subspace encodes the order (ordering ID) of entity and attribute.
- `prakash2024finetuning`: entity tracking by the position of the correct entity.
Prior work of `\citet{anonymous2026fitted}` is "Anonymous (2026)", always in the third person.

### 2.6 Anonymity (absolute)
- Never link the authors to Anonymous (2026): third person only; never any wording of BRIEF rule 3; never the leak
  patterns of `scripts/make_anonymous_release.py`; never "independently confirm" (use "we reproduce"); never a
  "correcting earlier work" frame. Describe what the released analysis does, not what that paper concluded.
- Never print the name of the Part-B frames script, the release-check status string from the Part-B log, local paths,
  user names, or model identifiers of any writing assistant (also in comments, figure metadata, table generators).
- The format label that generated tables print for options-after is rendered `\fmtOpt{}` (Section 8).
- The development record (`paper/versions/`, `paper/sections/old/`, `paper/BRIEF.md`, `docs/stage8_design/`, this plan,
  and `docs/PREREGISTRATION.md` as it stands in the repository) never ships; the anonymous release contains only the
  files that `scripts/make_anonymous_release.py` scrubs and checks, and its scan must pass on every file before release.
- Citation of Anonymous (2026) at submission: decide on the deadline day (accepted and public → names, third person;
  under review → "Submitted to ICLR 2027" with the public forum URL if visible; withdrawn → the public release of its
  bases only). Do not upload its manuscript to the supplement.

### 2.7 Forbidden texts (outcome-binding)
Never write: "in four model families including two not used before"; "later mentions" as the scope of the passage claim
(say "a later options list"); "and so does a natural sentence that mentions the candidates after the passage"; the J-A6
MET sentence ("the same cache answers the key's entity under a later options list and the value's entity in free form");
any behavioural claim about passages in the abstract; any quantization claim or "KV-cache methods" anywhere in the main
text; any head claim on passages as met; "no sentence read at 2B"; the unqualified J-B5b MET clause; "E-scored numbers
replace the lower-case ones as primary throughout"; "intervening text alone opens part of the key read" about Llama;
"does not mediate" (of the flag); "small models make the match without using it"; "not read at 1.5B/3B"; "the same frozen
reader heads" for IOI in the abstract, novelty or Table 1; polarity as a discovery; "sufficient and necessary"; the J-D3
MET sentence; "30–70 % away"; "the law's boundary is the kind of edit, not its depth"; the C-12 paragraphs (a), (a′),
(b), (c), (d), (f); "predicted before the edit is run"; "equals" for the law; "dictionary features" in the R1 sentence;
"misaligned" (of the autoencoder); the J-C-BOUND MET or NOT MET template sentences; any claim that J-C6 separates kind
from depth; J-C1 "met" under the post hoc channel floor; "The Prompt, Not the Token"; "the key share equals the unpatched
model's own share"; "a binding swap departs from it" as evidence; "except for steering vectors close to the natural write";
"a question-blind flag sufficient and necessary"; "the key channel reads only the token-form part ... natural or edited".

### 2.8 Labels (fixed now; use `\cref`)
| Object | Label | File |
|---|---|---|
| §1 Introduction | `sec:intro` | intro.tex |
| §2 Setup | `sec:method` | method.tex |
| Eq. ID_K | `eq:idk` | method.tex |
| §3 Claim 1 | `sec:claim1`; subsections `sec:fresh`, `sec:natural`, `sec:scope` | results_read.tex |
| §4 Claim 2 | `sec:claim2` | results_mechanism.tex |
| §5 Claim 3 | `sec:claim3`; Eq. `eq:lambda` | results_intervention.tex |
| §6 Related Work | `sec:related` | related.tex |
| §7 Discussion | `sec:discussion`; paragraph `sec:limitations` (label after the \paragraph) | discussion.tex |
| Fig. 1 composite | `fig:schematic` | intro.tex |
| Table 1 | `tab:qkov` | intro.tex |
| Fig. 2 | `fig:natural` | results_read.tex |
| Fig. 3 | `fig:flag` | results_mechanism.tex |
| Fig. 4 | `fig:law` | results_intervention.tex |
| Table 2 | `tab:status` | discussion.tex |
| Appendix labels | `app:prereg` (A) with `app:prereg-how`, `app:prereg-j`, `app:failures-j`, `app:table1-rows`, `app:prereg-ai`, `app:decisions`, `app:crosswalk`; `app:prompts` (B); `app:impl` (C); `app:claim1` (D) with `app:discovery`, `app:fresh`, `app:measure`, `app:generated`, `app:natural`; `app:claim2` (E) with `app:readers`, `app:heads`, `app:flag`, `app:sign`, `app:scale`; `app:claim3` (F) with `app:law`, `app:edits`, `app:lawresults`, `app:nu`; `app:remap` (G); `app:binding` (H); `app:accounts` (I); `app:compute` (J) | app_*.tex |

**Legacy labels.** The v4 appendix text refers to v4 main-text labels. Keep every v4 label alive by placing it where its
content now lives (one extra `\label` after the new heading):
`sec:read`, `sec:mentions`, `sec:general`, `app:additional`, `tab:twobytwo` → D.1; `sec:readers`, `sec:forms`,
`app:stage5`, `app:g-attention`, `app:g-knockout`, `app:g-membership`, `app:g-forms`, `fig:readers` → E.1/E.2 as the content
moves; `sec:heads`, `sec:role`, `app:stage6`, `app:heads` → E.2; `app:g-ioi` → E.4; `sec:intervention`, `sec:link`,
`sec:refit`, `app:refit`, `app:stage7`, `fig:frames`, `eq:patch`, `eq:phi`, `eq:psik`, `eq:psiv` → G; `sec:prakash`,
`app:prakash` → H; `sec:failed` → A.5; `app:discussion` → I; `eq:dc` → C. After moving text, re-read each sentence
that cites a legacy label (it now prints "Appendix X.y") and retarget the v4 references to `app:stage6` that concern the
binding swap to `app:binding`.

---------------------------------------------------------------------------------------------------------------------

## 3. `main.tex`

Preamble (replace the v4 lines; keep the rest):
```latex
\usepackage{xfp}                                   % \fpeval for \pct (tested under tectonic 0.15)
\newcommand{\pct}[1]{\fpeval{round(100*(#1),0)}\,\%}
\newcommand{\pctn}[1]{\fpeval{round(100*(#1),0)}}
\newcommand{\fmtOpt}{\textsc{options-after}}
\newcommand{\fmtListA}{\textsc{list-after}}
\newcommand{\fmtLetter}{\textsc{letters-after}}
\newcommand{\fmtPost}{\textsc{sentence-after}}
\newcommand{\fmtNull}{\textsc{null-sentence}}
\newcommand{\fmtSentB}{\textsc{sentence-before}}
\newcommand{\fmtNone}{\textsc{no-mention}}
\newcommand{\fmtBefore}{\textsc{list-before}}
\icmltitlerunning{Looked Up or Copied?}
...
\icmltitle{Looked Up or Copied? Later Mentions Decide Whether a Model Reads\texorpdfstring{\\}{ }an In-Context Value Through Its Key or Its Value}
```
- Remove the format comments after the v4 `\newcommand`s, and `\usepackage{todonotes}` (no `\todo` remains).
- Keywords: mechanistic interpretability, attention, key-value cache, interventions, preregistration.
- Input order: `abstract` (inside the abstract environment), `intro`, `method`, `results_read`, `results_mechanism`,
  `results_intervention`, `related`, `discussion`, `impact`; `\bibliography{references}`; `\newpage\appendix\onecolumn`;
  `\input{sections/appendix}` (a wrapper that inputs `app_a_prereg` … `app_j_compute`).
- Before overwriting, copy the v4 section files to `paper/versions/v4_sections/` (not shipped).

## 4. Title and abstract (`sections/abstract.tex`, final text)

Title: **Looked Up or Copied? Later Mentions Decide Whether a Model Reads an In-Context Value Through Its Key or Its
Value** (two lines in icml2026, checked).

```latex
A token that states a value in context, such as \emph{shelf} in ``the candle is moved to the shelf'', can reach a later answer in two ways: a later token can match its cached key, so the value is looked up, or the answer can copy its cached value. With exact cache clamps at this writing token, we find that later mentions decide the route. When the candidates are named again after the token, its key carries the value's identity, most of it after a list and part of it after a neutral sentence; a sentence that names no candidate opens no lookup. Without a later mention, or with the mention placed before the token, the identity is copied. On fresh stories in seven model families, scored on the forms the models emit, the route decides generated answers: clamping only the key changes them when the candidates are listed after the story, and clamping only the value does the reverse. On reading-comprehension passages with multi-token answers, a later options list likewise makes models read the answer entity through its keys, and free-form answers read it through its values, in every evaluable model. At each later mention, heads with a duplicate-token core write a flag that names no candidate; injected alone, it moves the answer and is read by key, but it is neither specific nor, as one direction, necessary. A preregistered law that edits of the written value are read in the model's own key-to-value ratio failed, even for steering vectors close to the natural edit, and a remap trained at the writing token is read mainly through its value. In total, \JxRiskyMetABCD{} of \JxRiskyVerdABCD{} predictions we judged risky before the run held (expected \JxRiskyExpABCD{}); we report every failure.
```
Measured: fills column 1 of page 1 (same length as the reviewer plan's tested abstract). "seven" = `\JxFamiliesABCD`
(families with a stage-8 verdict: Qwen, Mistral, OLMo, Llama, Gemma, Phi, Falcon, which are exactly the fresh-story
families); add `% seven = \JxFamiliesABCD (design count)`.

Licence per clause:
| Clause | Primary line(s), verdict | Source of wording |
|---|---|---|
| list after: most of the identity | J-B1-P4f (L) and J-B1-N4 (M), met 4/4 + 4/4 | J-B1-N4 MET clause, numbers dropped |
| neutral sentence: part of it | J-B3-P4f (M), J-B3-N4 (R), met | J-B3-N4 MET ("intermediate key read") |
| null sentence: no lookup | J-B-NULL-P4f (M), J-B-NULL-N4 (R, 3 of 4), met | J-B-NULL MET clause |
| no mention / mention before: copied | J-B4 (L, M), J-B2 (L, M), met | J-B4-N4 MET clause |
| fresh stories, seven families, emitted forms | J-B1/2/4 in 8/8; J-B6c, J-B7 met | J-B6c MET clause |
| generated answers, key vs value | J-B5-P4f, J-B5-N4 (M), met | J-B5 MET clause; "the reverse" as written there |
| passages, options list, keys vs values | J-A1 (M, 3/3 evaluable), J-A2 (M, 3/3) | J-A1 MET with the scope correction; "a later options list" (J-A5 NOT MET replacement) |
| flag: names no candidate, moves answer, read by key | J-D1 (R), J-D-ADDR (M), J-D4 (M), met | J-D1, J-D4 MET clauses condensed |
| not specific, not necessary as one direction | J-D2, J-D3 (R), not met | J-D3 NOT MET replacement; J-D2 per record |
| law failed; trained remap via value | J-C1 (M) not met, HEADLINE (e); J-C4 (R) R1 | C-12 (e) and the R1 replacement (remap only) |
| risk sentence | G4 | "k of n predictions we judged risky before the run held (expected k*)" |

---------------------------------------------------------------------------------------------------------------------

## 5. Main-text files: paragraph plans

Word budgets are running text (floats excluded). Total 4,075 words (Section 10); the binding texts given in full count towards their section. The section total is binding; if a paragraph's required content does not fit its share, move words between paragraphs of the same section, never across sections. "Numbers" lists every macro the
paragraph may use; a paragraph may use fewer. Text in quotation marks or code blocks is binding wording; the rest is
content to write in the writer's own words, in the plain style of v4.

### 5.1 `sections/intro.tex` — §1 Introduction (645 words; Fig. 1 and Table 1 environments)

Place the `figure*` of Fig. 1 and then the `table*` of Table 1 at the top of the file, before ¶1, so that Fig. 1 lands
at the top of page 2 and Table 1 at the top of page 3 (tested; defining Table 1 after the text pushed it to page 4).
Final LaTeX of both floats: Section 6.

**¶1 — The question (95 w).** No numbers. Content: mechanistic accounts credit specific cache channels with carrying
information or mediating an intervention: keys act as sparse routers and values as dense payloads
`\citep{ma2025addressbook}`; binding information is carried mainly in key vectors in Llama models
`\citep{oh2026rebinding}`; an operation's cache is read mainly as an address `\citep{wu2026record}`; refusal steering
acts mainly through the OV circuit `\citep{cheng2026steering}`. A later token sees an earlier one only through its cached
key and value, so a written value can reach the answer in two ways (`\cref{fig:schematic}a`). Definitions (they also
answer the sceptic who says "the identity is already in the option word": that is what a lookup is): "It is \emph{looked
up} when a later token that already holds a candidate matches the writing token's key, so that the key itself carries
which value was written; it is \emph{copied} when the identity travels in the writing token's value." End: "What decides
which route a model takes, and what follows for interventions that are credited to one channel?"

**¶2 — What we do (100 w).** Numbers: `\nModels`, `\JxLinesABCD`. Content, in this order:
- "In the belief stories of \citet{anonymous2026fitted}, one token, which they call the critical token and we call the
  \emph{writing token}, states where an object was moved." We clamp its cached key and value separately to those of a
  run in which another value was written; clamping both from the first layer reproduces that run exactly
  (`\cref{sec:method}`). Prompts differ only in whether and where the candidates are named again.
- Three tiers of evidence: a discovery sample of \nModels{} models from three families (1.5B–72B); a preregistered
  confirmation on 150 new stories in eight models from seven families, four never examined before, scored on the forms
  the models emit and on generated answers; counterfactual reading-comprehension passages with multi-token answers in
  four families.
- "Before any output of the final round we recorded, for each of its \JxLinesABCD{} predictions, a probability of
  holding, a risk class and the text the paper would print for each outcome; we name every failure."
- Reliance sentence (answers the objection): "From \citet{anonymous2026fitted} we take only the story template and, in
  \cref{sec:claim3}, one released remap and its training code; every other edit is built from public material."

**¶3 — Claim 1 (95 w).** Bold lead "\textbf{Later mentions decide the route} (\cref{sec:claim1})." Numbers:
`\pctn{\JbRangeBetaKAfterEightMin}--\pct{\JbRangeBetaKAfterEightMax}` (its second and last main-text place is §3.1 ¶4).
Content: with the candidates listed after the story, the writing token's key carries most of the value's identity in all
eight models of the confirmation, and a key-only clamp changes 76–100 % of generated answers to the clamped value;
without a later mention the identity is copied from the value and a value-only clamp changes the answers instead
(`\cref{fig:schematic}b,c`). A neutral sentence naming the candidates opens part of the read; a sentence of the same
length naming none opens nothing; the same list or sentence before the writing token opens no read. "On passages a later
options list opens the key read too; that the channel decides the answer there we could show only for lettered options."

**¶4 — Claim 2 (80 w).** Bold lead "\textbf{A match at the later mention, read by key} (\cref{sec:claim2})." No
numbers. Content: heads whose core is canonical duplicate-token heads match the writing token's key at each later
mention and write into that mention's row a flag that names no candidate; injected alone into the row of an option absent
from the story, it moves the answer there and the answer reads it by key; but structured controls also move the answer and
removing the flag leaves about half of the read: "sufficient, but neither specific nor necessary as one direction". Asked
which option was not mentioned, the same reader heads carry a negative read.

**¶5 — Claim 3 (90 w).** Bold lead "\textbf{Where an intervention appears to act} (\cref{sec:claim3})." No numbers.
Content: a released learned remap `\citep{anonymous2026fitted}` has nearly the same behavioural effect in every format,
yet the channel that appears to carry it flips from the keys, with lettered options, to the values, without a later
mention. We preregistered a law that would make such attributions predictable from the model's own key-to-value ratio. It
failed: even steering vectors close to the natural edit depart from that ratio in some cells beyond the shallowest depth,
and a remap trained at the writing token is read mainly through its value. Post hoc, departures grow with how far an edit
writes from the natural state.

**¶6 — Novelty paragraph (185 w), final text:**
```latex
\paragraph{What the architecture implies, and what it does not.}
Every attention head factors into a QK circuit, which decides where it attends, and an OV circuit, which decides what it moves \citep{elhage2021framework}. With the causal mask, a writing token's key can tell which value was written only to a later token that already holds a candidate, so a mention placed before it cannot read the key \citep[cf.][]{ok2026lost}. Duplicate-token and induction heads attend from later tokens to earlier copies \citep{wang2022ioi,olsson2022context}, option selection resembles query--key alignment \citep{tulchinskii2024wise}, and bound entities are retrieved through binding and ordering IDs, positions, lookbacks or a mix of these \citep{feng2023binding,dai2024binding,prakash2024finetuning,prakash2025lookbacks,gurarieh2025mixing}. None of this says which route a model takes, because the copy route is always open: the answer can attend to the writing token and copy its value. We find, each by a preregistered test, what the architecture allows but does not imply (\cref{tab:qkov}): the tokens after the writing token choose the route, in every family we tested and in generated answers; after an earlier list no later token looks the key up; the match writes a flag that names no candidate, sufficient to move the answer and read by key, though neither specific nor necessary as one direction; and the key read can take the opposite sign of the value read (\cref{sec:claim2}). We claim no new head type.
```
Licence: (i) J-B1, J-B4, J-B5, J-A1; (ii) J-B-LB-N4 (R, met), J-B2, J-A4; (iii) J-D1, J-D-ADDR, J-D4 met, J-D2,
J-D3 not met (D-13 (i) with J-D2 beside it); (iv) J-D-SIGN-IOI (D-13 (ii) reduced to the sign dissociation, no "same
frozen heads"); D-13 (iii) dropped. Claim 3 is not a novelty item; Table 1's last row carries it. No contributions list.

### 5.2 `sections/method.tex` — §2 Setup (505 words; Eq. 1; no float)

**¶1 — Stories, runs and formats (135 w).** No result numbers. Template in one sentence (v4 wording, credited to
`\citet{anonymous2026fitted}`; full template `\cref{app:prompts}`). B and S differ only at the writing token
{\textsc{loc}} at position $p$; X is a third value absent from the story. "Answer:" prefill. Formats (Section 2.4 of this
plan), all fresh-story formats with "Answer with one word". "A \emph{later mention} is any token after $p$ that names a
candidate." Fresh stories: 150 new cores, eight new neutral sentences (each with a null version of the same length that
names no candidate), a second six-word lexicon in half of the cores, a per-core order of the candidates
(`\cref{app:fresh}`).

**¶2 — Passages (80 w).** "From the SQuAD v1.1 development set \citep{rajpurkar2016squad} we take questions whose answer
is a person, place or number named once in the passage, and write a counterfactual entity in its place
\citep{longpre2021conflicts}: 185 items from 34 articles, with answers of mostly two to four tokens." The writing token is
the answer span; the clamps act at every position of it. Formats in words (Section 2.4). Each model's answer frame was
fixed from its own generations before any test; the first answer token after it is scored (`\cref{app:natural}`). A
closed-book prior for the original answer cancels from the identity measures (one clause).

**¶3 — Clamps (50 w).** `$\mathcal{C}_K(Z)$` sets the writing token's pre-RoPE key, in every layer from $\ell_0$ (0
unless stated), to the one cached in run $Z$; `$\mathcal{C}_V(Z)$` the value. Later tokens see the writing token only
through these, so clamping both from layer 0 reproduces run $Z$ exactly; each part checked this in FP32 before any model
output (`\cref{app:impl}`). "A single-channel clamp creates a cache state that no prompt produces, which is why we also
measure generated answers."

**¶4 — Measures (130 w + Eq. `eq:idk`).** Equation, v4 form:
```latex
\begin{equation}\label{eq:idk}
\mathrm{ID}_K = \tfrac{1}{2}\big[\Delta_{\mathrm{S}}(\mathrm{S}) - \Delta_{\mathrm{S}}(\mathrm{X}) + \Delta_{\mathrm{X}}(\mathrm{X}) - \Delta_{\mathrm{X}}(\mathrm{S})\big]
\end{equation}
```
with $\Delta_Y(Z)$ the change in $\log p(Y)$ at the answer under $\mathcal{C}_K(Z)$ against a self-clamp row in the same
batch. Effects shared by all foreign keys, and a prior for B, cancel. $\mathrm{ID}_V$ likewise with values;
$s_{\mathrm{ID}} = \overline{\mathrm{ID}}_K/(\overline{\mathrm{ID}}_K + \overline{\mathrm{ID}}_V)$, a ratio of means; $r$
= a format's $\overline{\mathrm{ID}}_K$ divided by that with the list after the story. Emitted-form score: "a
candidate's probability is summed over the surface forms the model emits, found from its own generations on held-out
stories, and coverage of at least 0.8 is required in every cell"; the discovery sample keeps its lower-case score, whose
verdicts the emitted-form score leaves unchanged (§3.1). Generated answers: greedy decoding under each clamp;
$\beta_K$ ($\beta_V$) is the share of answers that move to the clamped value under the key-only (value-only) clamp, net
of moves that a clamp to the third value also causes; the flip rate is the share that move, without that correction.

**¶5 — Models, statistics and preregistration (110 w).** Numbers: `\nModels`. Content: discovery sample: Qwen2.5
`\citep{yang2024qwen25}` at 1.5B–72B and Qwen3-8B (optional cite `yang2025qwen3`), Mistral-7B (optional
`jiang2023mistral`) and Mistral-Small-24B `\citep{mistral2025small}`, OLMo-2-7B `\citep{olmo2025two}`; added: Llama-3.1-8B
`\citep{grattafiori2024llama3}`, Gemma-2-9B and Gemma-2-2B `\citep{gemma2024gemma2}`, Phi-4 `\citep{abdin2024phi4}`,
Falcon3-7B `\citep{tii2024falcon3}`; all instruction-tuned, "official weights (sha256-verified)" (G2's phrase), BF16
(models per test: `\cref{app:compute}`). Statistics: 95 % two-stage cluster-bootstrap intervals with 10,000 resamples
(clusters: lexicon and ordered location pair for fresh stories; articles for passages; stories otherwise); a prediction
over several models holds only if it holds in every evaluable model; predictions over the four new families need three
of four at 98.75 %; equivalence tests for the law use 90 % intervals; a Holm correction changes no verdict
(`\cref{app:prereg-j}`). Ten preregistrations, each committed with its scoring code before its run; the last, in four
parts, recorded for each prediction a probability, a class (replication, extrapolation, or risky: probability below 0.8)
and the text to print for each outcome. Close with the convention:
"Unless marked otherwise, a result in \cref{sec:claim1,sec:claim2,sec:claim3} was preregistered in this final round and
held. We mark risky predictions, say in words where a prediction did not hold or could not be evaluated, and label
results of the discovery sample, exploratory analyses and post hoc readings; \cref{app:crosswalk} maps every sentence to
its prediction and verdict."

### 5.3 `sections/results_read.tex` — §3 Later mentions decide whether a value is looked up or copied (820 words; Fig. 2)

`\section{Later mentions decide whether a value is looked up or copied}\label{sec:claim1}`. Place the `figure*` of
Fig. 2 (`fig_natural`) at the top of the file. Opening (two sentences, no tag):
"\textbf{When the candidates are named again after the writing token, later tokens read its key and the value's identity
is looked up; without such a mention it is copied from the value.} The position of the mention decides, not the
instruction, and the decision reaches generated answers."

`\subsection{Fresh stories in seven families}\label{sec:fresh}`

**¶1 — Discovery sample (55 w).** Numbers: `\nModels`, `\twoAfterMin`, `\twoAfterMax`. In \nModels{} models from three
families (1.5B–72B), a list after the story opened a large key read and no later mention left the identity to the value;
an instruction-matched 2×2 then showed that the position of the list or sentence, not the instruction, decides
($\mathrm{ID}_K$ \twoAfterMin–\twoAfterMax nats with the list after the story; every interval before it contains zero or
lies below it). The sentence effect was found there without having been predicted (discovery sample;
`\cref{app:discovery}`).

**¶2 — Lists, no mention, mention before (130 w).** Numbers: `\JbOneNfMet`, `\JbOneNfEval`,
`\JbRangeSidAfterEightMin`, `\JbRangeSidAfterEightMax`. Design sentence (150 new cores, new sentences and lexicon; the
four discovery-sample models and four families never examined; `\cref{fig:schematic}b`). Then, verbatim with macros:
"In four model families never examined before, a list of the candidates after the story makes the writing token's key
carry most of the value's identity ($s_{\mathrm{ID}} \geq 0.5$ in \JbOneNfMet{} of \JbOneNfEval{})", continuing "as in
the four discovery-sample models ($s_{\mathrm{ID}}$ \JbRangeSidAfterEightMin--\JbRangeSidAfterEightMax{} over all
eight)." Verbatim: "Without a later mention, the identity is copied through the value in every family
($s_{\mathrm{ID}} \leq 0.1$ under \fmtNone{})." Then: the same list or sentence placed before the story opens no key read
in any of the eight models; "in all four new families a list before even lowers the key read below its no-mention level
(risky; held), which we cannot explain."

**¶3 — The sentence and its null control (110 w).** Numbers: `\JbThreeNfMet`, `\JbThreeNfEval`,
`\JbRangeRPostNfourMin`, `\JbRangeRPostNfourMax` (second place: Table 1), `\JbRNoneLlamaEight`. Verbatim: "A neutral
sentence naming the candidates after the story opens an intermediate key read ($r$ =
\JbRangeRPostNfourMin--\JbRangeRPostNfourMax{} of the list read) in \JbThreeNfMet{} of \JbThreeNfEval{} new families"
+ "(risky; held), as in the four discovery-sample models". One clause: the read appears for each of the eight sentences
and both lexicons (exploratory). Verbatim: "A sentence of the same length naming no candidate opens no key read, so the
read needs the re-mention." Llama: "In Llama-3.1-8B, one of the four new families, the null sentence missed the preregistered threshold only
because the model shows a small key read without any mention ($r$ = \JbRNoneLlamaEight{}), large as a ratio because its list read is the smallest
of the eight; the null sentence adds nothing to it (post hoc reading; \cref{app:fresh})." Then: "The test in a 2B model
(Gemma-2-2B) was not evaluable: the model failed the competence gate." Never "no sentence read at 2B".

**¶4 — Generated answers (125 w).** `\paragraph{Generated answers.}` Numbers: `\pctn{\JbRangeBetaKAfterEightMin}`–
`\pct{\JbRangeBetaKAfterEightMax}`, `\pct{\JbRangeBetaKNoneEightMax}` (the list-before and sentence-before maxima,
`\JbRangeBetaKBeforeEightMax` and `\JbRangeBetaKPreEightMax`, are also 0.000),
`\pctn{\JbRangeBetaVNoneEightMin}`–`\pct{\JbRangeBetaVNoneEightMax}`, `\pctn{\JbFivebFlipPostMetMin}`–
`\pct{\JbFivebFlipPostMetMax}`, `\pct{\JbRangeFlipPreEightMax}`. Verbatim: "Clamping only the writing token's key
changes the generated answer to the clamped value in \pctn{...}--\pct{...} of stories when the candidates are listed
after it and in at most \pct{\JbRangeBetaKNoneEightMax} when they are not; clamping only its value does the reverse"
(`\cref{fig:schematic}c`). Then: "Here `not' means no mention, a list before or a sentence before, and the reverse is a
value-only change in \pctn{\JbRangeBetaVNoneEightMin}--\pct{\JbRangeBetaVNoneEightMax} of stories without a mention."
Disclosed decision, verbatim: "the sentence read changes generated answers in seven of the eight models (flip rate
\pctn{\JbFivebFlipPostMetMin}--\pct{\JbFivebFlipPostMetMax} after the sentence, \pct{\JbRangeFlipPreEightMax} before
it)" + "(risky; held in the four new families)"; "at OLMo-2-7B it is visible in log-probabilities but not in generated
answers (flip rate 1.0\,\% after, 0\,\% before; risky; did not hold)" `% verbatim: outcome J part B`. Add: "The
preregistration had no text for this split; the wording is a disclosed decision (\cref{app:decisions})."

**¶5 — The measurement (80 w).** `\paragraph{The measurement.}` Numbers: `\JbSixbRPostMistralSeven`,
`\JbSixbRPostQwenFourteen`, `\JbSevenAMin`. Text (J-B6c with the one-word change "published" → "re-scored", disclosed):
"Scored on the forms the models actually emit (coverage $\geq 0.8$ in every cell), every re-scored verdict on the
original stories is unchanged (risky; held)." Then the disclosed correction of the J-B6b template, verbatim: "$s_{\mathrm{ID}}$
moves by at most 0.05, while the sentence read relative to the list read, $r$, rises under emitted forms by
$+$\JbSixbRPostMistralSeven{} at Mistral-7B and by $+$\JbSixbRPostQwenFourteen{} at Qwen2.5-14B, where equivalence
within 0.05 is not shown (risky; did not hold)": the lower-case score of the discovery sample understated the sentence
read. The emitted-form choice agrees with the generated answer in at least \JbSevenAMin{} of items in every cell (risky;
held), and the released remap's frames of §5 are unchanged under it (`\cref{app:measure}`).

`\subsection{Natural passages}\label{sec:natural}`

**¶6 — Identity on passages (120 w).** Numbers: `\JaSidOptaMistralSeven`, `\JaSidOptaQwenSeven`,
`\JaSidNomLlamaEight`, `\JaFiveDiffQwenSeven`, `\JaFiveDiffGemmaNine`. Verbatim (J-A1 with the corrected scope): "On
natural reading-comprehension passages with multi-token answers, a later options list makes the model read the passage
entity through its cached keys ($s_{\mathrm{ID}}$ \JaSidOptaMistralSeven--\JaSidOptaQwenSeven{} in Gemma-2-9B,
Qwen2.5-7B and Mistral-7B), and without it through its values ($s_{\mathrm{ID}}$ at most \JaSidNomLlamaEight{} in
Llama-3.1-8B, Gemma-2-9B and Qwen2.5-7B; \cref{fig:natural}a)." Then: also for answers of three or more tokens (risky;
held); with lettered options it holds in all four models. Evaluability in one sentence: Llama-3.1-8B's options-after
cell missed a generation-agreement floor by one item, Mistral-7B's free-form cell failed because it answered in full
sentences (both preregistered gates; `\cref{app:natural}`). Options before the passage leave a small negative key read.
Verbatim (J-A5 replacement): "On natural passages a related-articles sentence opens only a small key read ($s_{\mathrm{ID}}$
gain \JaFiveDiffQwenSeven--\JaFiveDiffGemmaNine{} in Gemma and Qwen, the evaluable models)" + "(risky; did not hold), so
on passages our claim concerns a later options list."

**¶7 — Behaviour on passages is a boundary (130 w).** `\paragraph{Behaviour on passages is a boundary.}` Numbers:
`\JaKeySrcLetaMistralSeven`, `\JaKeySrcLetaGemmaNine`. Content: the cache gets two sources, the key from one
counterfactual entity and the value from another. "With lettered options the generated answer follows the key's entity
in all four models" (key-source rate \JaKeySrcLetaMistralSeven–\JaKeySrcLetaGemmaNine; risky; held), and when the key
matches no option the value is copied (risky; held). "With options answered by name the key wins in two of the three
evaluable models (Gemma, Qwen) and not in Mistral", and "in free form the decision token follows the value while the
continuation breaks (hybrids)" (both risky; did not hold; `\cref{fig:natural}b`; the hybrids are an observation).
Verbatim: "The channel attributions are therefore not shown to decide behaviour on natural text." Listing sentence:
"Two predictions on 2-bit quantization of the cached values and one on the rate of hybrid answers did not hold either,
and the head tests on passages and three further passage tests were not evaluable (\cref{app:natural})." Nothing else on
quantization.

`\subsection{Scope}\label{sec:scope}`

**¶8 (40 w).** Numbers: `\roleFracMax`. On two further templated tasks (paint, schedule) the same contrast holds in five
7–14B models each; the channels carry which value was written, not its belief role (at most \roleFracMax{} of a role
swap); with the clamp started at about a third of the depth the key read persists in most models (all discovery sample;
the last exploratory; `\cref{app:discovery}`).

### 5.4 `sections/results_mechanism.tex` — §4 A match at the later mention, read by key (645 words; Fig. 3)

New file. `\section{A match at the later mention, read by key}\label{sec:claim2}`; `figure*` of Fig. 3 (`fig_flag`) at
the top. Opening: "\textbf{The lookup has two hops: heads whose core is canonical duplicate-token heads match the writing
token's key at each later mention and write a flag into that mention's row, which the answer then reads by key.} The flag
suffices to move the answer but is neither specific nor the only carrier, and the question sets the sign of the read."

**¶1 — Two hops (90 w).** `\paragraph{Two hops.}` Numbers: `\HoneRQwenSeven`, `\HoneRMistralSeven`. At Qwen2.5-7B and
Mistral-7B, 5 % of the attention heads read the writing token's key at the option words: when only they see the clamped
key they recover \HoneRQwenSeven{} and \HoneRMistralSeven{} of the option rows' read, random sets of the same size almost
none. Their core is canonical duplicate-token heads `\citep{wang2022ioi}`, although the full set is not (it depends on
how many heads are counted). The answer then reads the option words rather than the writing token, mainly through their
keys (an unscored contrast there; preregistered below for the flag). All discovery sample; "We claim no new head type"
(`\cref{app:heads}`).

**¶2 — The flag: sufficient, names no candidate, not specific (140 w).** `\paragraph{What the readers write.}` Numbers:
`\JdOneRatioQwenSeven`, `\JdOneRatioMistralSeven`, `\pct{\JdOnePiXQwenSeven}`, `\pct{\JdOnePiXMistralSeven}`,
`\JdTwoRatioMeanHQwenSeven`, `\JdTwoRatioOrthMistralSeven`. Design clause: the flag is fit on 60 discovery-sample stories
and tested on 100 fresh ones. Verbatim (J-D1): "The reader heads write one vector per layer into the re-mentioned word's
row. Injected with no key clamp into the row of an option absent from the story, a version fit without that word moves
the answer there (\JdOneRatioQwenSeven{} / \JdOneRatioMistralSeven{} of the key read's identity effect; the answer
changes in \pct{\JdOnePiXQwenSeven} / \pct{\JdOnePiXMistralSeven} of stories)" + "(risky; held; \cref{fig:flag}a)". It
names no candidate: fit with or without the value's identity, it is the same direction with the same effect. J-D2 beside
it (record wording): "specificity against structured controls failed: at Qwen2.5-7B the mean reader output moves the
answer \JdTwoRatioMeanHQwenSeven{} as much, and at Mistral-7B the story's own write with its flag component removed,
norm-matched, \JdTwoRatioOrthMistralSeven{} as much (risky; did not hold)"; the mean reader output contains the flag by
construction, which explains part but not all of this (`\cref{app:flag}`).

**¶3 — Read by key; not necessary as one direction (100 w).** `\paragraph{Read by key, not carried by one direction.}`
Numbers: `\JdFourKQwenSeven`, `\JdFourKMistralSeven`, `\JdFourVQwenSeven`, `\JdFourVMistralSeven`,
`\JdThreeRhoFlagQwenSeven`, `\JdThreeRhoFlagMistralSeven`. Verbatim (J-D4) main clause: "The answer reads the injected
flag by key, as it reads the natural one" with the numbers in words: "(giving only the answer row the option rows'
original keys removes \JdFourKQwenSeven{} / \JdFourKMistralSeven{} of its effect, their original values
\JdFourVQwenSeven{} / \JdFourVMistralSeven{})". Verbatim (J-D3 replacement): "One direction per layer does not carry the
key read ($\rho_K$ = \JdThreeRhoFlagQwenSeven{} / \JdThreeRhoFlagMistralSeven{} of it is left when the direction is
removed): other carriers exist; the flag is sufficient but not necessary" + "(risky; did not hold; \cref{fig:flag}b)".
Failing criterion, verbatim: "removing the flag direction leaves about half of the key read in both models, and at Qwen
the control directions, which overlap the flag, remove 31\,\% and 44\,\%" `% verbatim: outcome J part D`. One clause:
ablating the whole reader heads removes far more of the read (discovery sample), so they write more than this direction.

**¶4 — The sign of the read (170 w).** `\paragraph{The sign of the read.}` Numbers: `\JdSevenQoutQwenSeven`,
`\JdSevenQoutMistralSeven`, `\JdSignIoiIDKQwenSeven`, `\JdSignIoiIDVQwenSeven`, `\JdRouteKVQwenSeven`,
`\JdRouteKVMistralSeven`, `\JdRouteKQwenSeven`, `\JdRouteKMistralSeven`, `\JdRouteVQwenSeven`, `\JdRouteVMistralSeven`.
Content in order: (1) asked which option was \emph{not} mentioned, the key read turns negative ($\mathrm{ID}_K$
\JdSevenQoutQwenSeven{} and \JdSevenQoutMistralSeven{} nats), and so does the value read, so the sign follows the task;
the same frozen reader heads carry the reads under both questions (`\cref{fig:flag}c`). "A negative use of duplicate
detection is known \citep{wang2022ioi,mcdougall2023copy}; we do not present the polarity as a discovery." (2) What
separates the channels: in the in-sentence re-mentions of indirect-object identification (IOI) the key read is negative
while the value read stays positive ($\mathrm{ID}_K$ \JdSignIoiIDKQwenSeven{} against $\mathrm{ID}_V$
\JdSignIoiIDVQwenSeven{} nats at Qwen2.5-7B; the same signs at Mistral-7B). (3) Verbatim (J-D-ROUTE-IOI): "The negative
IOI read is applied at the answer's key read of the listed names" with the numbers in words: "(restoring the listed names'
keys and values for the answer row alone removes \JdRouteKVQwenSeven{} / \JdRouteKVMistralSeven{} of it, their keys alone
\JdRouteKQwenSeven{} / \JdRouteKMistralSeven{}, their values alone \JdRouteVQwenSeven{} / \JdRouteVMistralSeven{};
risky; held)". (4) Verbatim (J-D-HOP2): "a different (or a distributed) reader applies the negative read": the ten
answer-side heads ranked as the flag's top readers in lists carry neither the IOI nor the `not mentioned' read (risky;
did not hold; the scorer's reading: a different reader), although even in lists those ten carry only part of the read. (5) "Whether the belief reader heads also carry the IOI read was met at Qwen2.5-7B and
is not evaluable at Mistral-7B (a transfer gate missed)." Never "the same frozen heads" for IOI as a result.

**¶5 — Scale, binding and a non-candidate key (110 w).** `\paragraph{Scale, binding and a non-candidate key.}` No
numbers. "The flag is written in sentences as in lists at 1.5B, 3B and 7B. Whether it carries the size of the sentence
read came out mixed, the preregistered outcome: the test failed at 1.5B and 7B (3B not evaluable), and at 1.5B the
injected flag was read by key, not by value as predicted (both risky; did not hold; \cref{fig:flag}d). Post hoc, the
preregistered natural ratio contains a gain that every non-candidate receives; on a contrast specific to the absent
option, the injected flag reproduces the natural ratio in sign and size (\cref{app:scale})." Then: "Two further risky
predictions did not hold: the initial-state flag does not carry binding in a way separable from order, and a key that
names no candidate does not act as removing the base value's flag (\cref{app:flag})."

### 5.5 `sections/results_intervention.tex` — §5 Where an intervention appears to act (655 words; Fig. 4)

`\section{Where an intervention appears to act}\label{sec:claim3}`; `figure*` of Fig. 4 (`fig_law`) at the top. Opening
(general claim of the section, C-12 (e) in its summary form): "\textbf{Which cache channel an intervention appears to act
through depends on the prompt and on the edit; a preregistered law that would have predicted it from the model's own read
failed.}"

**¶1 — A motivating case (90 w).** `\paragraph{A motivating case.}` Numbers: `\psiLetterQwen`, `\psiNoneQwen`. Text:
channel credits for interventions are common (`\cref{sec:intro}`). The released remap of `\citet{anonymous2026fitted}`,
in our words: a rank-16 distributed alignment search subspace `\citep{geiger2024finding}` at the output of block 4 over
the event span, trained so that the patched model answers a fixed permutation of the source location; with \fmtOpt{}, a
key-only exchange at the writing token attributes part of its effect, relative to a PCA control, to the keys, which we
reproduce (`\cref{app:remap}`). Changing only the format barely changes its behavioural effect, yet the channel that
appears to carry it flips: at Qwen2.5-72B its keys carry \psiLetterQwen{} of the effect with lettered options and
\psiNoneQwen{} without a later mention, and its values the reverse (discovery sample; a refit without later mentions shows
the same crossover). Never "concluded", never "independently confirm".

**¶2 — The law and its test (120 w).** `\paragraph{The law and its test.}` No result numbers. Content: the law for any
edit that writes a value at the writing token at depth $l$: it is read through key and value in the same ratio as the
model's own written value at that depth and format. With $\psi_C$ the edit's identity effect through channel $C$ as a
share of the natural write's, from block $l{+}1$ on:
```latex
\begin{equation}\label{eq:lambda}
\lambda = \log\left(\psi_K/\psi_V\right),
\end{equation}
```
"and the law holds in a cell when the 90\,\% interval of $\lambda$ lies inside $\pm\log 1.25$." Lemma in one sentence:
because the prompts differ only at the writing token, an edit there reaches later tokens only through its keys and values,
so writing the source run's residual is the natural clamp itself; it served as the equivalence control, and synthetic
rows with half the key effect were classified as departures (`\cref{app:law}`). Families, three models (Qwen2.5-7B,
Mistral-7B, Llama-3.1-8B), blocks 3 to 15, four formats: steering vectors from held-out stories and from unrelated
sentences `\citep{turner2023actadd,panickssery2023caa}`; features of a public sparse autoencoder
`\citep{bussmann2024batchtopk,andyrdt2025saes}`; a rank-16 remap trained at the writing token with the released training
code of \citet{anonymous2026fitted} (the only place this is said); steering vectors that write the value in another
language, as a synonym, or without its English lexical component.

**¶3 — The outcome (120 w).** `\paragraph{The outcome.}` Numbers: `\JcOnePoolEoneMistralSeven`,
`\JcOnePoolEtwoMistralSeven`, `\JcOneLamThreeAbsMax` (exploratory), `\JcTwoPoolEoneLlamaEight`,
`\JcTwoPoolEtwoLlamaEight`, `\JcOneNeIDKMin`, `\JcOneNeIDKMax` (post hoc). First sentence, C-12 outcome (e), verbatim:
"Even steering vectors close to the natural edit depart from the natural channel ratio; the format law describes the
model's own written value and does not predict how an edit is attributed." Then J-C1's numbers and its second clause:
"In the preregistered core test the steering vectors' departures are small but graded ($\bar\lambda$ =
\JcOnePoolEoneMistralSeven{} to \JcOnePoolEtwoMistralSeven{}; every pooled value inside the margin, not every
statistic): the format law does not transfer to edits as a channel-ratio law (did not hold; in Llama-3.1-8B for the
vectors from unrelated sentences too)." Allowed context, each
labelled: at the shallowest depth every statistic is equivalent ($|\lambda| \leq$ \JcOneLamThreeAbsMax; exploratory);
pooled over depths, the keys under-read the steering vectors by "4.5--12\,\%" ($\bar\lambda$ \JcTwoPoolEoneLlamaEight{} to
\JcTwoPoolEtwoLlamaEight{}) `% verbatim: outcome J part C`; the failing statistics sit in cells whose natural key read is
small (\JcOneNeIDKMin–\JcOneNeIDKMax{} nats; post hoc). Fig. 4a. No mention of the post hoc 4.8-nat channel floor.

**¶4 — Edits far from the natural write (145 w).** `\paragraph{Edits far from the natural write.}` Numbers:
`\JcFourPoolEfourQwenSeven`, `\JcFourPoolEfourMistralSeven`, `\JcFourPsiKWpostSevenQwenSeven`,
`\JcFourPsiVWpostSevenQwenSeven` (post hoc). Verbatim (R1, remap only): "Edits that change the value through a learned
remap are read mainly through the value ($\bar\lambda$ = \JcFourPoolEfourQwenSeven{} / \JcFourPoolEfourMistralSeven{}):
the key channel does not read them as it reads the written word" + "(risky; did not hold)". Its size depends on depth, so
quote the shares as the record asks, in one clause: "(under \fmtPost{} at block 7, $\psi_K$
\JcFourPsiKWpostSevenQwenSeven{} against $\psi_V$ \JcFourPsiVWpostSevenQwenSeven{} at Qwen2.5-7B; post hoc)". Then, as
compactly as possible (about 85 words for all of these): the autoencoder test was not evaluable, because
its alignment gate measured reconstruction over positions that the dictionary's published fit excludes (`\cref{app:edits}`);
identity-carrying components orthogonal to the natural write or outside the lexical span are read mainly through the
value (risky; did not hold); the predicted lexical boundary was not isolated: non-lexical edits are read less through the
key than lexical ones, but the effective ones also write far from the natural state, "so we claim neither that the
token-form account holds for edits nor that it is contradicted" (risky; did not hold); the reader heads that read the
natural value also read the steering vectors, but that test failed through a random-set control that the natural value
fails as well (did not hold; post hoc reading); the reader-head test for the autoencoder and remap edits was not
evaluable (Fig. 4c, d; `\cref{app:lawresults}`).

**¶5 — A post hoc pattern and a practical rule (85 w).** `\paragraph{Post hoc: distance from the natural write.}`
Numbers (all post hoc): `\JcNuEqLow`, `\JcNuNLow`, `\JcNuEqMid`, `\JcNuNMid`, `\JcNuEqHigh`, `\JcNuNHigh`,
`\JcNuRhoNu`. "Post hoc, the departures follow how far an edit's cached keys and values lie from the natural ones
($\nu$, a relative distance): of the $\lambda$ statistics of identity-carrying edits, \JcNuEqLow{} of \JcNuNLow{} with
$\nu < 0.1$ are equivalent, \JcNuEqMid{} of \JcNuNMid{} with $\nu$ between 0.1 and 0.3, and \JcNuEqHigh{} of
\JcNuNHigh{} beyond (Spearman \JcNuRhoNu{}; \cref{app:nu})." $\nu$ is collinear with the key cosine, so which matters is
open. Rule: report an intervention's channel only with the model's own read at the same depth, in more than one prompt,
and with the edit's distance from the natural write.

**¶6 — The binding swap (70 w).** `\paragraph{The binding swap.}` Numbers: `\JcSixKappaCaaNoMention`,
`\JcSixKappaBindNoMention`. Verbatim (J-C6): "At block 28 of Qwen2.5-14B an identity edit is copy-read like the natural
value ($\kappa \leq 0.30$), while the binding swap there is key-read; key-flat reading is not a property of depth for
identity edits" with "($\kappa$ \JcSixKappaCaaNoMention{} against \JcSixKappaBindNoMention{} without a mention)" and the
record's caveat in the same paragraph: "This does not separate kind from depth, because the natural read there is itself
copy-only, and the swap changes no answer in two of the three formats." Screen (C-12 text with words for the codes):
"Binding forms only after the identity key route closes in every model screened, so a binding swap cannot be compared
with an identity edit at a depth where the key route is open (risky; held); the four earlier preregistered predictions
on this swap, which failed, stand (\cref{app:binding})." The binding swap is that of `\citet{prakash2025lookbacks}`
(cite once here).

### 5.6 `sections/related.tex` — §6 Related Work (300 words), final draft

```latex
\section{Related Work}\label{sec:related}

\paragraph{Where attention looks and what it moves.}
Every head factors into a QK circuit, which computes where it attends, and an OV circuit, which computes what the attended tokens contribute \citep{elhage2021framework}; attention weights alone do not determine a head's output \citep{kobayashi2020norm}. Duplicate-token and induction heads attend to earlier copies of the current token \citep{wang2022ioi,olsson2022context}, induction can copy tokens or whole concepts \citep{feucht2025dualroute}, a copy-suppression head suppresses earlier occurrences of a predicted token \citep{mcdougall2023copy}, and circuit components are reused across tasks \citep{merullo2024reuse}. Our first hop is a duplicate-token match; we ask which cache channel it opens, what it writes and what sets the sign of its use.

\paragraph{Binding, options and position.}
Bound entities are retrieved through binding IDs \citep{feng2023binding}, an ordering subspace \citep{dai2024binding}, position \citep{prakash2024finetuning}, lookbacks \citep{prakash2025lookbacks} or a mix of positional, lexical and reflexive mechanisms \citep{gurarieh2025mixing}; belief and reality share a value slot read out under a router at the query \citep{steele2026routing}. Specific heads select among options \citep{lieberum2023chinchilla,wiegreffe2025answer}, a winner is chosen before it is bound to a symbol \citep{wong2026decide}, and middle-layer heads select by query--key alignment \citep{tulchinskii2024wise}, which our second hop resembles; options placed before a context cannot attend to it \citep{ok2026lost}. We separate the two cache channels of the token that states a value and show that the placement of the candidates decides which one carries its identity, under a free-answer instruction and on passages; which of the mechanisms of \citet{gurarieh2025mixing} our lookup corresponds to, if any, we do not test.

\paragraph{Attributing interventions.}
Interchange interventions \citep{geiger2021inducing,geiger2024finding} may act through pathways the model does not use naturally \citep{makelov2024subspace,wu2024reply,liu2026wrong} and need not reproduce natural activations \citep{shih2026steering}; steering vectors \citep{turner2023actadd,panickssery2023caa} are brittle to prompt changes \citep{tan2024steering}, and prompt format changes what is measured \citep{sclar2023quantifying,opielka2026causality}. Cache-level accounts place routing or binding information in keys \citep{ma2025addressbook,oh2026rebinding} or find an operation's cache read mainly as an address \citep{wu2026record}. We find that the channel credited with an edit of a written value depends on the prompt and on the edit, and that a preregistered law predicting it from the model's own read fails.
```
Before submission the writer re-checks against the abstracts (Section 9 lists the cached ones): `kobayashi2020norm`,
`feucht2025dualroute`, `mcdougall2023copy`, `merullo2024reuse`, `ma2025addressbook`, `wong2026decide`,
`turner2023actadd`, `panickssery2023caa`, `tan2024steering`, `opielka2026causality`. Keys moved out of the main text
(v4 or the drafts had them) go to Appendix I ("Further related work", same abstract-supported wording as v4):
`vig2020causal`, `li2026bucketing`, `zou2026introspection`, `pustovit2026packs`, `bojieli2026notes`, `steele2026spaces`,
`hewitt2019control`, `huang2024ravel`, `arora2024causalgym`, `zhou2026ordinal`, `sun2026persuaded`,
`venkatesh2026steering`, `zhang2023patching`, `heimersheim2024patching`, `gao2026encodings`, `robinson2023mcsb`
(answering with option symbols requires binding options to symbols). `cheng2026steering` is cited in §1 and Table 1;
`kamath2025attention` and `mueller2025mib` in the Discussion. Measured: the citation-heavy text takes more space than its
word count suggests (author-year citations), which is why this section is held at about 300 words.

### 5.7 `sections/discussion.tex` — §7 Discussion (505 words; Table 2)

`\section{Discussion}\label{sec:discussion}`; `table` (column) of Table 2 at the top (LaTeX in Section 6).

**¶1 — What to do differently (105 w), final text:**
```latex
\paragraph{What to do differently.}
(i)~State where the candidates are mentioned relative to the token that writes a value, and test a channel claim with and without a later mention; moving the mention before the token keeps the words and removes the readers. (ii)~A mechanism localised with listed options can use another channel than the free-form behaviour it should explain; localisation methods \citep{mueller2025mib} should be checked in that behaviour's format. (iii)~Do not infer an intervention's channel from the model's own read: measure it with key-only and value-only exchanges, in more than one prompt and at matched depth, and report how far the edit writes from the natural state. Attributions that freeze attention may miss the key channel (untested; cf.\ \citealt{kamath2025attention}).
```

**¶2 — How the predictions fared (105 w), final text:**
```latex
\paragraph{How the predictions fared.}
Before any output of the final round we recorded, for each of its \JxLinesABCD{} predictions, a class and a probability of holding (\cref{tab:status}). Replications and extrapolations held about as often as these probabilities implied; risky predictions held less often. The shortfall is informative: on fresh stories the risky predictions held more often than expected, while the specifics of the mechanism, behaviour on natural passages and the intervention law fell short. In the three entries before this one, \GHIcountFailed{} of \GHIcountPred{} predictions failed (\cref{app:prereg-ai}). Every prediction that did not hold or could not be evaluated is named in \cref{sec:claim1,sec:claim2,sec:claim3} and listed with its reason in \cref{app:failures-j}.
```
(`\JxLinesABCD` appears also in §1 ¶2: two places.)

**¶3 — Limitations (205 w), final text** (each item answers a known objection; numbers: `\idKBeforeQwenSeventytwo`,
`\IbMarginU`, `\IbMarginA` (exploratory), "12–17 %" verbatim):
```latex
\paragraph{Limitations.}\label{sec:limitations}
(i)~Our natural data are counterfactual passages with added options; there only a list opens the key read, and answers follow the key only with lettered options. (ii)~A list before the writing token gives a small negative key read (to \idKBeforeQwenSeventytwo{} nats at Qwen2.5-72B), and in-sentence IOI re-mentions are read negatively by heads we did not identify; we explain neither. (iii)~A single-channel clamp creates a cache state no prompt produces. (iv)~With the clamp started at a third of the depth part of the key read disappears, most in OLMo-2-7B and on passages (exploratory). (v)~The flag is neither specific nor necessary as one direction; the readers' duplicate-token core depends on how many heads are counted; ablating the readers and cutting their attention disagree on whether the copy route takes over, and the passage tests that would have settled this were not evaluable. (vi)~For interventions the law failed, the autoencoder test was not evaluable, the distance pattern is post hoc and collinear with the key cosine, and the binding swap changes no answer when the candidates are named; at 24B, blocking the reader heads compresses the model's own margin (\IbMarginU{} to \IbMarginA{} nats; exploratory). (vii)~The mechanism was tested in two 7B models (and at 1.5B and 3B), the interventions in three 7--8B models, in BF16; eager attention lowers the key read by 12--17\,\% against the fused kernel in one model (exploratory). (viii)~Values span one to four tokens; all prompts are English.
```
`% verbatim: "12 % (AFTER) and 17 % (POST)", outcome J part B` on the line of (vii).

**¶4 — Conclusion (90 w), final text:**
```latex
\paragraph{Conclusion.}
Across seven model families, on fresh stories and, for an options list, on natural passages, the tokens after a writing token decide whether its value is looked up through its key or copied from its value, and on fresh stories the decision reaches generated answers. The lookup is a duplicate-token match that writes a flag the answer reads by key, sufficient but not the only carrier, with a sign the question sets. Where an intervention appears to act depends on the prompt and on the edit; a preregistered law that would have predicted it from the model's own read failed.
```

### 5.8 `sections/impact.tex` (outside the 8 pages), final text
```latex
\section*{Impact Statement}
This work aims to make explanations of language-model internals more reliable, by showing that the cache channel that appears to carry a value or an intervention depends on the prompt under which it is measured. More reliable component-level explanations could help researchers audit and debug models, and could reduce the risk that a mechanism found under one evaluation format is trusted to explain behaviour under another. Our results also caution against over-reading interpretability findings: an attribution validated under one prompt format, or inferred from the model's own read, should not on its own certify behaviour under others. Methods for locating and editing what a model reads from its context could also be used to steer models in unwanted ways; our experiments use public models, templated stories, public reading-comprehension passages, edits built from public material, a released remap and a released state-token patch.
```

---------------------------------------------------------------------------------------------------------------------

## 6. Main-text floats (final LaTeX; sizes measured in the layout test)

| # | Label | File(s) | Env. | Page | Size |
|---|---|---|---|---|---|
| Fig. 1 | `fig:schematic` | `fig_schematic.pdf` (panel a only), `fig_fresh.pdf` (b), `fig_crossover_main.pdf` (c) | `figure*` in intro.tex | top of p. 2 | 0.55 p |
| Table 1 | `tab:qkov` | inline | `table*` in intro.tex, after Fig. 1 | top of p. 3 | 0.33 p |
| Fig. 2 | `fig:natural` | `fig_natural.pdf` | `figure*` in results_read.tex | p. 4–5 | 0.30 p |
| Fig. 3 | `fig:flag` | `fig_flag.pdf` | `figure*` in results_mechanism.tex | p. 5–6 | 0.30 p |
| Fig. 4 | `fig:law` | `fig_law.pdf` | `figure*` in results_intervention.tex | p. 6–7 | 0.33 p |
| Table 2 | `tab:status` | `tables/tab_status.tex` (generated) | `table` in discussion.tex | p. 7–8 | 0.17 p |

**Figure 1** (if the regenerated files of Section 8 are not ready, use `fig_schematic.pdf` with `trim=0 52 0 0,clip`
and `fig_crossover.pdf` with `trim=0 70 0 0,clip`, both tested, and add "(b)"/"(c)" with `\llap`):
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_schematic.pdf}\\[2pt]
  \includegraphics[width=0.49\textwidth]{figures/fig_fresh.pdf}\hfill
  \includegraphics[width=0.49\textwidth]{figures/fig_crossover_main.pdf}
  \caption{\textbf{Later mentions decide whether a written value is looked up through its key or copied from its value.} (a)~The writing token \emph{shelf} leaves a key $K$ and a value $V$ in the cache at every layer. A later mention of a candidate matches $K$ through heads with a duplicate-token core, which write a flag into the mention's row (hop~1; \cref{sec:claim2}); the answer reads the flagged mention by key (hop~2). Without a later mention the identity reaches the answer in $V$. We clamp $K$ and $V$ separately to another run's. (b)~Fresh stories, eight models: the key share of the value's identity, $s_{\mathrm{ID}}$, by where the candidates are named; emitted-form score; 95\,\% two-stage cluster intervals (98.75\,\% for the new families); grey: Gemma-2-2B, not evaluable. (c)~Generated answers: share moved to the clamped value by the key-only clamp ($\beta_K$) against the value-only clamp ($\beta_V$), per format and model; the four formats at zero key flips are spread out in \cref{app:generated}.}
  \label{fig:schematic}
\end{figure*}
```

**Table 1** (four columns as E-3 asked; tested at 0.33 p; numbers: the J-B1-N4 range, the pre-written "Later list, new
families" content, and the J-B3-N4 range, each in at most one other main-text place):
```latex
\begin{table*}[t]
  \centering
  \caption{\textbf{What the architecture implies, what known circuits supply, and what we find.} ``Architecture alone'': what QK/OV and the causal mask imply \citep{elhage2021framework}. Unmarked findings were preregistered and held; ``risky'' marks predictions given a probability below 0.8 before the run. Every prediction, its prior and its verdict: \cref{app:prereg}.}
  \label{tab:qkov}
  \footnotesize\setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}p{0.17\textwidth}p{0.11\textwidth}p{0.21\textwidth}p{0.47\textwidth}@{}}
    \toprule
    Question & Architecture alone & Known circuits & What we find (status; section) \\
    \midrule
    Can a mention \emph{before} the writing token read its key? & No (causal mask) & Options before a context cannot attend to it \citep{ok2026lost} & Nor does a later token read the key through it, in all eight story models and on passages; a list before even lowers the read below no mention (risky; held; unexplained) (\S\ref{sec:fresh}) \\
    With the candidates named after it, is the value looked up or copied? & Either: the copy route stays open & Duplicate-token and induction heads attend to earlier copies \citep{wang2022ioi,olsson2022context} & Looked up: the key carries most of the identity in all eight story models ($s_{\mathrm{ID}}$ \JbRangeSidAfterNfourMin--\JbRangeSidAfterNfourMax{} in the new families) and in three evaluable passage models; copied without a later mention (\S\ref{sec:claim1}) \\
    Must the later mention be an answer list? & Not decided & -- & Stories: a neutral sentence opens part of the read ($r$ = \JbRangeRPostNfourMin--\JbRangeRPostNfourMax{} in the new families; risky; held), a null sentence none; passages: only a small read (risky; did not hold) (\S\ref{sec:claim1}) \\
    Does the route decide generated answers? & Not decided & -- & Stories: yes in all eight models, the sentence read in seven of eight; passages: with lettered options (risky; held), not with options by name or in free form (risky; did not hold) (\S\ref{sec:claim1}) \\
    What does the matching head write? & Not decided & Binding vectors can be injected \citep{feng2023binding} & A flag naming no candidate that moves the answer when injected and is read by key (risky; held); not specific, not necessary as one direction (risky; did not hold) (\S\ref{sec:claim2}) \\
    What sets the sign of a key read? & Not decided & S-inhibition \citep{wang2022ioi}; copy suppression \citep{mcdougall2023copy} & The question, through the same readers; in IOI the key read is negative and the value read positive, applied at the names' keys (risky; held), not by the answer-side heads ranked in lists (risky; did not hold) (\S\ref{sec:claim2}) \\
    Is an intervention's apparent channel a property of the edit? & Not decided & Steering via OV \citep{cheng2026steering}; keys as routers \citep{ma2025addressbook} & No: a released remap's channel flips with the format (discovery sample); the law that edits keep the model's own key-to-value ratio failed, even for near-natural steering vectors (did not hold) (\S\ref{sec:claim3}) \\
    \bottomrule
  \end{tabular}
\end{table*}
```

**Figure 2:**
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_natural.pdf}
  \caption{\textbf{On passages, a later options list moves the answer entity's identity into the cached keys; answers follow the key only with lettered options.} Counterfactual SQuAD passages, four models, 185 items; 95\,\% two-stage article-cluster intervals. (a)~$s_{\mathrm{ID}}$ by format; hollow: cell not evaluable by a preregistered gate; $\dagger$: computed from the raw results, not printed by the scorer (exploratory). (b)~Cue conflict, with the key from one counterfactual entity and the value from another: share of generated answers naming the key's or the value's entity; crosses: the decision token alone. Panel (b) is a boundary, not a confirmed claim (\cref{sec:natural}).}
  \label{fig:natural}
\end{figure*}
```

**Figure 3:**
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_flag.pdf}
  \caption{\textbf{What the reader heads write.} Qwen2.5-7B (blue) and Mistral-7B (orange); flags fit on 60 discovery-sample stories and tested on 100 fresh ones; 95\,\% story-bootstrap intervals. (a)~$\iota$: share of the natural identity effect moved to an absent option by the flag (added at half, single and double strength, or moved to that option's row) and by structured controls and isotropic directions (\cref{app:flag}). (b)~$\rho_K$: key read left when one direction per layer is removed at the option rows. (c)~$\mathrm{ID}_K$ and $\mathrm{ID}_V$ when asked which option is mentioned, which is not, and in IOI's in-sentence re-mentions. (d)~By scale: $\omega$, how far the flag written in a sentence matches the list's; $\rho$ and $\rho_{\mathrm{nat}}$, the injected and natural sentence-to-list ratios; $\dagger$: recomputed, post hoc or not evaluable (3B).}
  \label{fig:flag}
\end{figure*}
```

**Figure 4** (the pre-written "Figure 4" of Part C, all four panels):
```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_law.pdf}
  \caption{\textbf{The preregistered channel-ratio law failed.} Qwen2.5-7B, Mistral-7B and Llama-3.1-8B, 80 stories. (a)~$\lambda = \log\psi_K - \log\psi_V$ per edit family (E1, E2: steering vectors from held-out stories and from unrelated sentences; E3: features of a public sparse autoencoder; E4: a remap trained at the writing token), depth and format, with 90\,\% intervals and the $\pm\log 1.25$ band; hollow: no verdict (E3's gate failed; values recomputed). (b)~$\kappa$, the edit's key share, against $\sigma$, the natural key share at the same depth and format (descriptive). (c)~Identity effect without a mention of each edit's component along the natural write (PAR), along the English lexical span (LEX) and orthogonal to each. (d)~Share of the key read removed by blocking the reader heads, per family, beside the natural write; open: random head sets, which the natural write also fails to keep under the preregistered 0.25.}
  \label{fig:law}
\end{figure*}
```

**Table 2** (G4's "Table 3"; column float; generated as `tables/tab_status.tex`, Section 8; the values below are what
the generator must print, all from S8X and the per-part `\J?Class*` macros):
```latex
\begin{table}[t]
  \centering
  \caption{\textbf{How the predictions of the final preregistration fared.} Held / with a verdict; in parentheses the expected number, the sum of the probabilities recorded before the run. Risky: probability below 0.8. Not evaluable (n.e.) predictions are left out of the counts. Under the common part's evaluability rule one more risky prediction counts as not held (\JxRiskyMetABCDAlt{} of \JxRiskyVerdABCDAlt{}, expected \JxRiskyExpABCDAlt{}). Brier scores and every prediction: \cref{app:prereg-j}.}
  \label{tab:status}
  \footnotesize\setlength{\tabcolsep}{3pt}
  \input{tables/tab_status.tex}
\end{table}
```
Rows and cells of `tab_status.tex` (`lcccc`; header: Evidence (claim) & Replication & Extrapolation & Risky & n.e.):
Fresh stories (1): `\JbClassLMet`/3, `\JbClassMMet`/7, `\JxRiskyMetB`/`\JxRiskyVerdB` (`\JxRiskyExpB`), `\JbCountNotEval`;
Passages (1): 1/1, 3/3, `\JxRiskyMetA`/`\JxRiskyVerdA` (`\JxRiskyExpA`), `\JaCountNotEval`;
Mechanism (2): 2/2, 5/5, `\JxRiskyMetD`/`\JxRiskyVerdD` (`\JxRiskyExpD`), `\JdCountNotEval`;
Interventions (3): --, 1/4, `\JxRiskyMetC`/`\JxRiskyVerdC` (`\JxRiskyExpC`), `\JcCountNotEval`;
All: `\JxClassLMetABCD`/`\JxClassLVerdABCD` (`\JxClassLExpABCD`), `\JxClassMMetABCD`/`\JxClassMVerdABCD`
(`\JxClassMExpABCD`), `\JxRiskyMetABCD`/`\JxRiskyVerdABCD` (`\JxRiskyExpABCD`), `\JxNotEvalABCD`;
Measurement validity (fresh stories): risky column `\JxClassValMetABCD`/`\JxClassValVerdABCD` (`\JxClassValExpABCD`),
n.e. 0. The per-part "with a verdict" counts of the L and M columns are met + not met of the scorer's SUMMARY (printed
by the generator as `[arithmetic on printed values]` macros `\JaClassLVerd` … `\JdClassMVerd`). Fallback if the
generator is not ready: hand-typeset the same cells from those macros, writing L and M cells as "met of met + not met"
with the two counts from `\J?ClassLMet`/`\J?ClassLNotMet`.

---------------------------------------------------------------------------------------------------------------------

## 7. Appendix files

`sections/appendix.tex` becomes a wrapper: `\input{sections/app_a_prereg}` … `\input{sections/app_j_compute}`. One writer
per file (or per pair). Source lines refer to the v4 `appendix.tex` (A 3–333, B 334–507, C 508–553, refit 554–571,
additional 572–679, G 680–762, H heads 763–804, H binding 805–859, I 860–878, discussion 879–894) and to the v4 main
files. Every table and figure below exists in `paper/tables` or `paper/figures` unless marked "new". Appendix text may
use prediction codes, symbols and many numbers (all macros or generated tables); it keeps the anonymity and citation rules.

### `app_a_prereg.tex` — A. Preregistrations and claim status (`app:prereg`)
- **A.1 How the predictions were made** (`app:prereg-how`): v4 text (lines 3–10) plus entry J: four parts, each with its
  scorer, Holm helper, FP32 gate and populations fixed before any output; the four checks per part (recomputation,
  verdict audit, anomaly hunt, integrity), re-scoring byte for byte; pilots only on CPU at ≤ 1.5B and disclosed; the risk
  rule of G4 (L ≥ 0.9, M ≥ 0.8, R < 0.8; measurement-validity lines tallied apart). Define the line-code suffixes P4f
  (the four discovery-sample models, fresh stories) and N4 (the four new families).
- **A.2 Entry J: every prediction** (`app:prereg-j`): `tab_prereg_j.tex` in a `table*` (`\scriptsize`); a class table
  per part with met / not met / not evaluable, expected and Brier (`\JaClassLMet` … `\JdClassRBrier`, `\JbClassVal*`);
  the R tally under both readings of J-C-READb (`\JxRiskyMetABCD`/`\JxRiskyVerdABCD`/`\JxRiskyExpABCD` and the `Alt`
  macros); Holm sensitivity per part: A one decision (J-A5 at Qwen), no verdict, and J-A6b not robust to Holm; B none
  (Phi-4's J-B5b component narrow); C `\JcHolmChanged` decisions, no verdict; D none.
- **A.3 Every prediction that did not hold or could not be evaluated** (`app:failures-j`): new table `tab:failures-j`
  (hand-typeset; codes, words, reason, where the main text names it), the 21 NOT MET and 12 NOT EVALUABLE lines of
  Section 12 of this plan, plus J-A3 (derived, not counted).
- **A.4 The preregistered Table-1 rows** (`app:table1-rows`): new table `tab:table1-rows` carrying every pre-written
  "Table 1 row" of Parts A–D with its per-model numbers and verdict (Later list, new families: s_ID(AFTER) per family;
  Later sentence, new families: r(POST) per family, with the null control note; No mention; Generated answers;
  Sentence, generated; natural passages rows; What the readers write: ID ratio, ρ_K, r_ans; Sign: ID_K, ID_V, R(H*) in
  \fmtOpt{}, "not mentioned", IOI; Scale: ω, ρ, ρ_nat at 1.5B, 3B (recomputed), 7B; Near-natural edits (E1, E2) with the
  full J-C1 sentence verbatim; Independent edits (E3, E4): E4 R1, E3 not evaluable; Boundary (E5): not met, per
  sub-family with E2 controls (`\JcBoundNlQwenSeven` … `\JcBoundEtwoNlLlamaEight`); the J-C-SCREEN note). Preferably
  generated (Section 8); else hand-typeset from macros.
- **A.5 Entries A–I** (`app:prereg-ai`; legacy `sec:failed`): v4 tables `tab:prereg-ab` … `tab:prereg-i` and
  `tab:failures` unchanged; v4's claim tables (`tab:claims`, `tab:claims2`, `tab:claims3`) replaced by the crosswalk of
  A.7; the post hoc A–I risk labels (G4) as a table, marked post hoc and never pooled with J, or, if not built, the
  disclosure sentence of Section 13 item 17.
- **A.6 Pre-written texts used and decisions taken after the data** (`app:decisions`): one short paragraph per item of
  Section 13, quoting the outcome record.
- **A.7 Crosswalk** (`app:crosswalk`): new table `tab:crosswalk`: main-text claim → lines → verdict → class → where
  (Section 11 of this plan is its content).

### `app_b_prompts.tex` — B. Prompts and items (`app:prompts`)
v4 lines 334–507, plus: the fresh-story arms (exact builders), the eight sentences and their null versions (null nouns),
the two lexicons, the per-core order and the cluster definition; the passage formats with exact strings, the item rules
(majority answer named once; PERSON, PLACE, NUMBER; substitutes S, X, Z; distractor D; no partial mentions; type audit),
the answer frame and closed-book prompts; the mechanism's question formats (which option is mentioned / is not) and the
IOI in-sentence prompts; the intervention population (80 stories) and formats. Credit SQuAD (CC BY-SA 4.0).

### `app_c_impl.tex` — C. Implementation, exactness and measurement (`app:impl`)
v4 lines 508–553, plus moved from the v4 main text: $d_C$ and $s_K$ (Eq. `eq:dc`), the row-restricted splice, HeadSplice
and HopSplice. New: clamps over multi-token spans; each part's FP32 gate (test counts from the records); BF16 floors
(Llama's options-after cell by one item; Part D batch floor); eager against sdpa (X1, exploratory, `\JbXoneEagerPost`,
`\JbXoneFpPost`); the emitted-form scorer (32 fixed forms plus discovered frames, the trie, coverage, the trie floor),
frame discovery and generation parsing; weight verification ("official weights (sha256-verified)"; gated models
assembled from byte-identical copies, each file checked against the official hash). No local paths, no script file names
that reveal authorship.

### `app_d_claim1.tex` — D. Claim 1 details (`app:claim1`)
- **D.1 Discovery sample** (`app:discovery`; legacy `sec:read`, `sec:mentions`, `sec:general`, `app:additional`,
  `tab:twobytwo`): v4 §3.1–3.2 text condensed; `tab_2x2` captioned "discovery sample (stories of stages 1 and 3b)";
  `tab_natural`, `fig_scale`, `tab_idshare`, `fig_formats`, `tab_decomposition`, `tab_onset`, `tab_windows`, `tab_tasks`,
  `tab_role`. Keep v4's status labels (B3 exploratory, E3 met, etc.).
- **D.2 Fresh stories** (`app:fresh`): `tab_fresh.tex`; per-sentence and per-lexicon r (`\JbLevelRPostMin`, exploratory);
  Llama's null-sentence reading with `\JbRNoneLlamaEight`, `\JbRPostNullLlamaEight`, `\JbIDKAfterLlamaEight` and the
  record's recomputed r(POST-NULL) − r(NONE) (post hoc); the null nouns are not inert (record); OLMo-2's three POST flips;
  Gemma-2-2B: not evaluable (competence gate), its r(POST) and s_ID reported "without a scale claim" (never "no sentence
  read at 2B"); X2 at 1.5B and 3B (exploratory).
- **D.3 The measurement** (`app:measure`): `tab_mass.tex`, `tab_invariance.tex`; J-B6a/b/c with the corrected template
  sentence and the capitalised-form explanation from the record; J-B7; J-B8 at Mistral-Small-24B (`\JbEightLoMin`,
  `\JbEightHiMax`).
- **D.4 Generated answers** (`app:generated`): the full `fig_crossover.pdf` (with the zoom strip; label
  `fig:crossover-full`); crossover numbers (exploratory) from the record's "Reported" list.
- **D.5 Natural passages** (`app:natural`): `tab_natural8.tex`; gates and their failures (Llama options-after by one item,
  with the record's sensitivity; Mistral free form, answer form; Qwen's eager heads floor); cue conflict per model with
  decision-token and first-word rates (first-word re-scoring labelled not the preregistered criterion; J-A6c would be met
  on the first word); copy fallback; flag-only cue (not evaluable); new exploratory table `tab:kivi` from `\JaKivi*`
  (drops, low power, the opposite direction on competent items, reconstruction errors `\JaKiviErrKMin`–`\JaKiviErrKMax`,
  `\JaKiviErrVMin`–`\JaKiviErrVMax`; cite `liu2024kivi` here only; no quantization claim); the continuation pattern
  "seen in all four models (I_cont \JaIcontMin–\JaIcontMax, I_dec \JaIdecMin{} to \JaIdecMax) whose line is not
  evaluable", "a channel-level replication of induction-head K-composition" `\citep{elhage2021framework,olsson2022context}`,
  not a novelty; heads on passages: not evaluable, Mistral numbers and the copy-set ablation as exploratory (never
  "met"); closed-book prior; LEAK and YEAR strata; onset and span pieces (exploratory).

### `app_e_claim2.tex` — E. Claim 2 details (`app:claim2`)
- **E.1 Which later tokens read, and what a re-mention must be** (`app:readers`; legacy `sec:readers`, `sec:forms`,
  `app:stage5`, `app:g-attention`, `app:g-knockout`, `app:g-membership`, `app:g-forms`): v4 §3.3 text and v4 App. G
  (a)–(d) with `tab_localisation`, `fig_localisation`, `tab_attention`, `tab_knockout`, `tab_membership`, `tab_variants`.
- **E.2 Reader heads and the second hop** (`app:heads`; legacy `sec:heads`, `sec:role`, `app:stage6`, `fig:readers`): v4
  §3.4–3.5 and App. H (a): `fig_readers`, `tab_heads`, `fig_heads_app`, `tab_ablation`; the duplicate-token account and
  its set-size dependence (`\HfourDkstarQwenSeven`, recomputed); ablation versus knockout not reconciled; G1–G3 (the
  match follows the clamped key at 1.5B and 3B) stated as discovery-sample facts with the D6 context of E.5, never as
  "match without a read".
- **E.3 The flag** (`app:flag`): definitions (ι, ω, κ, ρ, Ψ, π); `tab_flag.tex` (upper block); dose and sanity rows; what
  the flag is (logit-lens cosines; identity share of the per-story flag); the J-D2/J-D3 overlap analysis (meanH contains
  the flag; energy shares; Mistral's controls overlap yet pass); J-D-KN with the six-way-logit β (recomputed); J-D5
  strata and the order effect; cross-task injection (exploratory).
- **E.4 Signs and IOI** (`app:sign`; legacy `app:g-ioi`): J-D7, J-D-SIGN-Q, J-D-SIGN-IOI, J-D8-Q, J-D8-IOI (Mistral's
  transfer gate missed by 0.021; met at Qwen), J-D-ROUTE-IOI with its key share, J-D-HOP2 with the caveat that under
  \fmtOpt{} the top ten carry only `\JdHopOptTopQwenSeven` / `\JdHopOptTopMistralSeven` of hop 2; the Q_OUT answer
  observation; v4's IOI factorial (`tab_ioi`, G18–G22).
- **E.5 Scale** (`app:scale`): `tab_flag.tex` (lower block); the D6 decision table and outcome (d) "Mixed."; the post hoc
  ρ_nat reading with `\JdSixNatX*`, `\JdSixDiffX*` (recomputed); 3B's would-be failure (recomputed); the record's sentence
  governs wording: the paper does not say that the flag does not mediate the natural read.

### `app_f_claim3.tex` — F. Claim 3 details (`app:claim3`)
- **F.1 The law, the lemma and the statistics** (`app:law`): ψ_C, λ, the statistics (W, A_L, A_P), TOST at 90 %, the
  rivals (copy-only, key-flat, graded), the combination rule; the lemma with proof; why T is the natural clamp; the
  sensitivity gate (SK50, SV50).
- **F.2 The edit families and gates** (`app:edits`): E1–E5 construction (means, sentence pools, the BatchTopK dictionaries
  at pinned revisions, the DAS fit with losses and held-out flip rates, the E5 sub-families), PAR/PERP and LEX/NONLEX,
  ν; gates G0–G8 with the G3 specification failure in full: the published fit drops the first tokens and high-norm
  activations, the gate did not; at the writing position the layer-3 dictionary reproduces its published value
  (`\JcGthreeFvePThree` against `\JcGthreePubThree`); the post hoc E3 statistics labelled; "a corrected gate needs a
  rerun". Never "misaligned".
- **F.3 Results per line** (`app:lawresults`): `tab_law.tex`; J-C1/J-C2 per combo; J-C4 per depth and seed with
  `\JcFourDepthThreeQwenSeven`, `\JcFourDepthThreeMistralSeven` and the ψ values (post hoc); J-C5; J-C-BOUND per
  sub-family with its E2 controls and the disclosed decision; J-C-READa with the natural donor's random-set KO
  (`\JcReadaRandNatQwenSeven`, `\JcReadaRandNatMistralSeven`, post hoc) and the set construction; J-C-READb under both
  readings; HEADLINE (e) and why (a), (a′), (c), (b) do not apply.
- **F.4 Post hoc and exploratory** (`app:nu`): `fig_nu.pdf` (label `fig:nu`) with the ν bins and the collinearity
  (`\JcNuCollMin`–`\JcNuCollMax`); where J-C1 fails (small natural key channels); the channel-floor sensitivity (4.8 nats
  would make J-C1 met; stated as not used); κ against σ and depth tracking (14 of 14); the lexical code of E1/E2 versus
  E4 (`\JcLexParShareEfourMax`); generated answers under edits (O4, exploratory).

### `app_g_remap.tex` — G. The released remap (`app:remap`; legacy `sec:intervention`, `sec:link`, `sec:refit`, `app:refit`, `app:stage7`, `fig:frames`, `eq:patch`, `eq:phi`, `eq:psik`, `eq:psiv`)
Self-contained description in our words (objective, rank, block, span, seeds, generator, readout, key-only and
value-only exchanges, φ, ψ_K, ψ_V with v4's equations); reproduction (argmax agreement 0.994–1.000, verbatim from the
earlier record); `fig_frames.pdf` and `tab_frames.tex`; the post hoc tracking (r = `\crossR`); the refit (preregistration
F, `tab_refit`); the 24B reader test (preregistration I, `tab_link`, I6 not met, the margin compression); J-B8 under
emitted forms. What the released analysis does, never what that paper concluded.

### `app_h_binding.tex` — H. The binding swap of Prakash et al. (`app:binding`; legacy `sec:prakash`, `app:prakash`)
v4 App. H (b) with `tab_replication`, `fig_sweeps`, `tab_prakash`, `tab_exchange`, `tab_clamp` and H7–H11 with codes;
J-C6 (κ per format, IIA 0.00 in two formats); J-C-SCREEN with the full C-12 text: "Binding forms only after the identity
key route closes in every model screened, so a binding swap cannot be compared with an identity edit at a depth where the
key route is open; H7–H10 stand." (`\JcScreenWbFirstQwenSeven`–`\JcScreenWbLastQwenSeven`, `\JcScreenSidWbMaxQwenSeven`);
J-C-WIN not evaluable (no window).

### `app_i_accounts.tex` — I. Competing accounts and further related work (`app:accounts`; legacy `app:discussion`)
v4 App. I with `tab_accounts`, updated with the flag (sufficient, not specific, not necessary as one direction), the
sign by task semantics, and the failed law (or labelled "as of the earlier entries"). "Further related work": the v4
keys moved out of §6 (`vig2020causal`, `li2026bucketing`, `zou2026introspection`, `pustovit2026packs`,
`bojieli2026notes`, `steele2026spaces`) with v4's abstract-supported wording.

### `app_j_compute.tex` — J. Models, compute and licences (`app:compute`)
Table of models per test (discovery sample; fresh stories; passages; mechanism; interventions; J-B8 at 24B) with pinned
revisions; BF16, attention implementation per model (Gemma-2 eager); GPU hours per part (from the records' run
sections); SQuAD CC BY-SA 4.0; the dictionaries' pinned hashes; Prakash et al.'s release at its pinned commit; the
anonymous code release (scrubbed). No local paths or user names.

---------------------------------------------------------------------------------------------------------------------

## 8. Figure and table generation (`paper/make_schematic.py`, `paper/make_figures.py`)

**`make_schematic.py` (Figure 1a).** Regenerate `figures/fig_schematic.pdf` with panel (a) only (drop the format strip,
panel (b); the fresh-story formats are the x axis of Fig. 1b and the passage formats that of Fig. 2a):
- `figsize` about (6.8, 1.9); panel title "(a) Two routes from the token that wrote a value".
- Keep the token stream (… moved to the | **shelf** | … Question … | Choices: box, | **shelf** | … Answer: ▢).
- Hop-1 arrow label: "hop 1: a later mention of the candidate matches the writing token's key (heads with a
  duplicate-token core) and writes a flag into its own row".
- Hop-2 arrow label: "hop 2: the answer reads the flagged mention by key".
- Value arrow label unchanged: "value copy: later tokens copy the writing token's value"; "writing token / key K · value V".
- Optional: a faded "Choices: … shelf …" box left of the story with a dotted, crossed arrow labelled "a mention before the
  token cannot attend to it (causal mask)".
- Keep `pdf.fonttype 42`, no metadata beyond matplotlib's defaults.

**`make_figures.py`.**
1. `fig_fresh.pdf`: legend headers "discovery-sample models" and "new families" (not the P4/N4 codes); add a small
   "(b)" panel tag at the top left, or leave it to the LaTeX overlay.
2. New `fig_crossover_main.pdf`: the scatter of `fig_crossover.pdf` without the zoom strip, legend "discovery-sample
   models" / "new families", the annotation "four formats at $\beta_K = 0$ (Appendix D.4)", panel tag "(c)"; keep
   `fig_crossover.pdf` (full, with the strip) for Appendix D.4.
3. Generated tables that print the options-after label as a code (`tab_law`, `tab_invariance`, `tab_mass`, `tab_flag`):
   print `\fmtOpt{}`; replace the P4/N4 codes in table text by words or define them in each caption and in A.1.
4. New `tables/tab_status.tex` (Table 2; spec in Section 6) and the macros `\JaClassLVerd` … `\JdClassMVerd` (met + not
   met per part and class), each labelled `[arithmetic on printed values]` in the index.
5. New (optional, preferred to hand-typesetting) `tables/tab_table1_rows.tex` (Appendix A.4) and `tables/tab_kivi.tex`
   (Appendix D.5, exploratory).
6. Re-run `make_figures.py` and regenerate `ALL_MACROS.md`; check that no existing macro value changed (diff
   `numbers.tex` before/after).

## 9. Bibliography (`paper/build_bib.py` → `paper/references.bib`)

None of the keys below is in `references.bib` or `build_bib.py` today (checked). Merge them from
`docs/stage8_design/partE/v5_bib_additions.py` into `ARXIV`, `AUTHOR_FIX` and the manual block, then rebuild (the arXiv
API rate-limits: run the script once, with its built-in pause, and check the printed titles).

| Key | Source | Used in |
|---|---|---|
| `elhage2021framework` | manual (Transformer Circuits Thread, 2021) | §1 novelty, §6, Table 1 caption, App. D.5 |
| `kobayashi2020norm` | arXiv 2004.10102 | §6 |
| `mcdougall2023copy` | arXiv 2310.04625 | §4, §6, Table 1 |
| `merullo2024reuse` | arXiv 2310.08744 | §6 |
| `dai2024binding` | arXiv 2409.05448 | §1, §6 |
| `prakash2024finetuning` | arXiv 2402.14811 | §1, §6 |
| `gurarieh2025mixing` | arXiv 2510.06182 (ICLR 2026 per the PDF) | §1, §6 |
| `robinson2023mcsb` | arXiv 2210.12353 | App. I |
| `turner2023actadd` | arXiv 2308.10248 | §5, §6 |
| `panickssery2023caa` | arXiv 2312.06681 | §5, §6 |
| `bussmann2024batchtopk` | arXiv 2412.06410 | §5 |
| `andyrdt2025saes` | manual (Hugging Face repository at revision c37e53c4) | §5 |
| `rajpurkar2016squad` | arXiv 1606.05250 | §2, App. B |
| `longpre2021conflicts` | arXiv 2109.05052 | §2 |
| `grattafiori2024llama3` | arXiv 2407.21783 | §2 |
| `gemma2024gemma2` | arXiv 2408.00118 (author fix "Gemma Team") | §2 |
| `abdin2024phi4` | arXiv 2412.08905 | §2 |
| `tii2024falcon3` | manual (model card) | §2 |
| `liu2024kivi` | arXiv 2402.02750 | App. D.5 only |
| `yang2025qwen3` (optional) | arXiv 2505.09388 (Qwen3 Technical Report) | §2 |
| `jiang2023mistral` (optional) | arXiv 2310.06825 (Mistral 7B) | §2 |

Not needed (drop from the merge): `variengien2023leap`, `geva2023dissecting`, `merullo2024talking`, `springer2024echo`,
`lieberum2024gemmascope`, `gould2023successor`.

Fix `prakash2025lookbacks`: the arXiv page (2505.14685, v3) states no venue; move it from the manual block to `ARXIV`
(`"prakash2025lookbacks": "2505.14685"`, an `@article` with year 2025), unless the ICLR 2026 venue is verified on the
proceedings site. Keep `anonymous2026fitted` as it is until the submission-day decision (Section 2.6).

Abstracts already checked for the wording of Section 2.5 (cached in the scratchpad: `lit/2510.06182.txt`,
`ids2.xml` for 2409.05448 and 2402.14811, `lit/2607.11945.txt`, `arx_2505.14685.xml`, `arx_2606.08644.xml`,
`arx_2609.24635.xml`, `arx_2504.13151.xml`, `abs/cheng2026steering.xml`, `abs.xml` for 2601.14152). The others listed in
5.6 are re-checked by the related-work writer before submission (BRIEF rule 2).

## 10. Page budget (measured) and cut order

Layout test (scratch copy of `paper/`, tectonic 0.15, icml2026): this abstract; Fig. 1 composite (trimmed v4 schematic,
fig_fresh, trimmed fig_crossover); Table 1 as in Section 6 (0.33 p); Figs. 2–4 (0.30, 0.30, 0.33 p); Table 2 (0.17 p);
filler text built from v4 prose (with its macros, citations and run-in paragraph heads). First test, all filler at 4,310
words: 8 pages with about 0.35 page to spare; at 4,545 words with a longer Table 1 (0.42 p) it overflowed by five lines.
Second test, with the binding texts of this plan (abstract, novelty paragraph, Related Work, Discussion, Impact) in place:
at 4,415 words the Conclusion ran 12 lines onto page 9, because author-year citations make the Related Work and the
novelty paragraph much longer on the page than their word counts; with the Related Work cut to 300 words and 4,150 words in all the main text ends on page 8 with about 0.3 page
to spare. The budget below (4,075 words) keeps that margin and adds about 75 words of reserve:

| Block | Words | Floats (measured) |
|---|---|---|
| Title + abstract | (column 1 of p. 1) | – |
| §1 Introduction (95, 100, 95, 80, 90, 185) | 645 | Fig. 1 0.55 + Table 1 0.33 |
| §2 Setup (135, 80, 50, 130, 110) | 505 | – |
| §3 Claim 1 (opening 30; 55, 130, 110, 125, 80; 120, 130; 40) | 820 | Fig. 2 0.30 |
| §4 Claim 2 (opening 35; 90, 140, 100, 170, 110) | 645 | Fig. 3 0.30 |
| §5 Claim 3 (opening 25; 90, 120, 120, 145, 85, 70) | 655 | Fig. 4 0.33 |
| §6 Related work (citation-heavy) | 300 | – |
| §7 Discussion (105, 105, 205, 90) | 505 | Table 2 0.17 |
| **Total** | **4,075** (hard cap 4,150) | 1.98 |

Measured slack at 4,150 words (second layout test, with the binding texts of Sections 4–5 in place and v4 prose as
filler elsewhere): the Conclusion ends 40 % down column 2 of page 8, about 0.3 page before the Impact Statement. Keep
that slack for the real text, whose citations and cross-references take more room than filler. Hard checks after the assembled compile: the Conclusion ends on page 8; the Impact
Statement starts after it; Fig. 1 is at the top of page 2 and Table 1 at the top of page 3; no overfull boxes in floats.

Cuts if it overflows, in this order (never cut a pre-written text, a Limitations item, a status tag or the risk
sentence):
1. §3.3 Scope to one sentence (−30 w); §4 ¶5 last sentence to a clause (−20 w).
2. Related Work to 335 words: merge "Options and position" into "Binding and retrieval" and drop its first sentence (−50 w).
3. Table 1 row 3 merged into row 2 (−0.03 p).
4. §5 ¶1 shortened to three sentences, the reproduction clause to App. G (−30 w).
5. Fig. 3 panel (d) to App. E.5 (regenerate fig_flag with (a)–(c); −0.04 p); §4 ¶5 then points to the appendix.

## 11. Claim map (the content of Appendix A.7; writers check each main-text claim here)

| Main-text claim | Lines (class, verdict) | Status words | Where |
|---|---|---|---|
| List after: key carries most of the identity, eight models | J-B1-P4f (L, met), J-B1-N4 (M, met 4/4) | none | §3.1 ¶2, Table 1, Fig. 1b |
| No mention: copied | J-B4-P4f (L), J-B4-N4 (M), met | none | §3.1 ¶2 |
| Before: no key read; list before below no mention in new families | J-B2-P4f (L), J-B2-N4 (M), met; J-B-LB-N4 (R, met) | "(risky; held)" for the second | §3.1 ¶2, Table 1 |
| Sentence after: intermediate | J-B3-P4f (M, met), J-B3-N4 (R, met) | "(risky; held)" | §3.1 ¶3, Table 1 |
| Null sentence: none | J-B-NULL-P4f (M, met), J-B-NULL-N4 (R, met 3/4; Llama reading post hoc) | none / "(post hoc reading)" | §3.1 ¶3 |
| 2B sentence test | J-B-SMALL (R, NE) | "not evaluable" | §3.1 ¶3 |
| Generated answers follow the channel | J-B5-P4f, J-B5-N4 (M, met) | none | §3.1 ¶4, Fig. 1c |
| Sentence read in generated answers | J-B5b-N4 (R, met), J-B5b-P4f (R, not met: OLMo-2) | "(risky; held in the new families)", "(risky; did not hold)" | §3.1 ¶4 |
| Scoring invariance; sentence ratio understated | J-B6a, J-B6c, J-B7, J-B8 (R validity, met); J-B6b (R, not met) | "(risky; held)", "(risky; did not hold)" | §3.1 ¶5 |
| Passages: options after → keys; free form → values | J-A1 (M, 3/3), J-A2 (M, 3/3), J-A1b (R, met), J-A1c (M, 4/4) | "(risky; held)" for J-A1b | §3.2 ¶6, Fig. 2a |
| Passages: options before | J-A4 (L, met) | none | §3.2 ¶6 |
| Passages: sentence after small | J-A5 (R, not met); J-A5B (NE) | "(risky; did not hold)"; NE in the listing | §3.2 ¶6–¶7 |
| Passages: cue conflict | J-A6b (R, met), J-A6e (R, met), J-A6a (R, not met), J-A6c (R, not met), J-A6d (NE) | as written | §3.2 ¶7, Fig. 2b |
| Passages: quantization, hybrids, continuation, heads | J-A7, J-A7b, J-A8d (R, not met); J-A8 (M, NE); J-A-HA1–HA3b (R, NE) | listing sentence | §3.2 ¶7 |
| Scope: tasks, role, onset | discovery D1, D2, E2 (met); onset exploratory | "(discovery sample)" | §3.3 |
| Two hops; duplicate-token core | discovery G, H1–H5 (met; hop-2 key split unscored) | "(discovery sample)" | §4 ¶1 |
| Flag sufficient; names no candidate | J-D1 (R, met), J-D-ADDR (M, met) | "(risky; held)" | §4 ¶2 |
| Not specific | J-D2 (R, not met) | "(risky; did not hold)" | §4 ¶2 |
| Read by key | J-D4 (M, met) | none | §4 ¶3 |
| Not necessary as one direction | J-D3 (R, not met) | "(risky; did not hold)" | §4 ¶3 |
| Question sets the sign; same readers | J-D7, J-D-SIGN-Q, J-D8-Q (M, met) | none | §4 ¶4 |
| IOI: key negative, value positive | J-D-SIGN-IOI (L, met) | none | §4 ¶4 |
| IOI read applied at the names' keys | J-D-ROUTE-IOI (R, met) | "(risky; held)" | §4 ¶4 |
| Not by the same hop-2 heads | J-D-HOP2 (R, not met) | "(risky; did not hold)" | §4 ¶4 |
| Belief readers carry the IOI read | J-D8-IOI (R, NE; met at Qwen) | "not evaluable at Mistral-7B" | §4 ¶4 |
| Written in sentences at every scale | J-D6a (L, met) | none | §4 ¶5 |
| Mediation across scale; value route at 1.5B | J-D6, J-D6-ROUTE (R, not met; outcome (d) "Mixed.") | "(risky; did not hold)"; post hoc reading | §4 ¶5 |
| Binding flag; non-candidate key | J-D5, J-D-KN (R, not met) | "(risky; did not hold)" | §4 ¶5 |
| Released remap crossover | discovery C3, E4, F (met); tracking post hoc | "(discovery sample)" | §5 ¶1 |
| The law, near-natural steering vectors | J-C1 (M, not met), J-C2 (M, not met); HEADLINE (e) | "(did not hold)" | §5 ¶3, Fig. 4a |
| Shallow equivalence; pooled under-read; small channels | reported context; recomputed | "(exploratory)", "(post hoc)" | §5 ¶3 |
| Trained remap via the value | J-C4 (R, not met, R1) | "(risky; did not hold)" | §5 ¶4 |
| Autoencoder features | J-C3 (R, NE) | "not evaluable" | §5 ¶4 |
| Orthogonal components | J-C5 (R, not met) | "(risky; did not hold)" | §5 ¶4 |
| Lexical boundary | J-C-BOUND (R, not met) | "(risky; did not hold)" | §5 ¶4 |
| Reader heads read the steering vectors | J-C-READa (M, not met); J-C-READb (R, NE) | "(did not hold; post hoc reading)"; "not evaluable" | §5 ¶4 |
| ν pattern | none | "(post hoc)" | §5 ¶5 |
| Identity edit at block 28 | J-C6 (M, met) | none | §5 ¶6 |
| No overlap window | J-C-SCREEN (R, met); J-C-WIN (R, NE) | "(risky; held)" | §5 ¶6 |
| Calibration | G4 tallies | – | Table 2, §7 ¶2 |

## 12. Every prediction that did not hold or was not evaluable, and where it is named

NOT MET (21): J-A5 (§3.2 ¶6); J-A6a, J-A6c (¶7); J-A7, J-A7b, J-A8d (¶7 listing); J-B5b-P4f (§3.1 ¶4); J-B6b (§3.1 ¶5);
J-C1, J-C2 (§5 ¶3); J-C4, J-C5, J-C-BOUND, J-C-READa (§5 ¶4); J-D2 (§4 ¶2); J-D3 (§4 ¶3); J-D-HOP2 (§4 ¶4); J-D6,
J-D6-ROUTE, J-D5, J-D-KN (§4 ¶5).
NOT EVALUABLE (12): J-A5B, J-A6d, J-A8, J-A-HA1, J-A-HA2, J-A-HA3a, J-A-HA3b (§3.2 ¶7 listing); J-B-SMALL (§3.1 ¶3);
J-C3, J-C-READb (§5 ¶4); J-C-WIN (§5 ¶6, "no window"); J-D8-IOI (§4 ¶4). J-A3 (derived, not counted): App. A.6 only.
Appendix A.3 lists all of them with codes and reasons.

## 13. Disclosed decisions (Appendix A.6; one paragraph each, quoting the outcome record)

1. **Title.** "The common part lets only the natural-text lines named in Part A change the title and gives the
   templates-only fallback if they are NOT MET or NOT EVALUABLE; Part A's table names J-A1 (with J-A2; J-A3 derived) and
   keeps the title when J-A1 is MET. J-A1 is MET; J-A3, a derived line that is not counted, is NOT EVALUABLE (met in both
   models where it is evaluable). Read literally, the common part would give the fallback through J-A3. We keep the title
   under Part A's table and state the conflict here."
2. **J-A1 scope.** "In four model families including two not used before" overstates the evaluable set and is not used;
   the evaluable models are named.
3. **J-A6.** The row's hybrid clause applies only if J-A6c alone fails; J-A6a also failed, so hybrids are an observation.
4. **Planned figure panels.** Fig. N's panels (c) (KIVI) and (d) (continuation) and the templated ghost markers are given
   as Appendix D.5 tables; the flag figure's planned inset (J-D4) is given in the text.
5. **J-B5b split** (no pre-written text for a split): the record's two sentences, used verbatim.
6. **J-B6b template** named the wrong statistic; the record's corrected sentence is used.
7. **J-B6c wording.** "published" is replaced by "re-scored": the five verdicts are those of this paper's discovery
   sample that the check re-scores.
8. **J-B-NULL at Llama-3.1-8B.** The NOT MET template ("intervening text alone ...") is not applied; the record's reading
   (its no-mention baseline) is reported, labelled post hoc.
9. **J-D-SIGN-IOI with J-D8-IOI.** The joint abstract clause is not used; novelty (ii) is reduced to the sign
   dissociation; (iii) is dropped (D-13).
10. **J-D6.** Outcome (d), "Mixed."; the post hoc ρ_nat reading is reported as exploratory; the paper does not say that
    the flag does not mediate.
11. **J-C-BOUND.** The pre-written sentence is contradicted by the numbers; reported as not met with the per-sub-family
    values, claiming neither that the token-form account holds for edits nor that it is contradicted.
12. **J-C-READb.** Two readings; the official verdict (NOT EVALUABLE) is used; under the common part's reading the risky
    tally is 10 of 28 (expected 12.65).
13. **J-C1 sentence.** The main text opens with the (e) text and gives J-C1's numbers with its second clause; the full
    pre-written J-C1 sentence is in Appendix A.4. The C-12 screen text's "H7–H10 stand" is written in words in the main
    text and with codes in Appendix H. J-C4's size by depth is quoted with ψ_K and ψ_V in one clause beside the pre-written
    λ̄ sentence, as the record's sensitivity note asks; the per-depth values are in Appendix F.3.
14. **Table numbering.** The common part's "Table 3" is Table 2; the pre-written "Table 1 rows" are in Appendix A.4,
    because Table 1 is the QK/OV table of the rewrite rule E-3; each row keeps its verdict.
15. **Abstract.** The pre-written abstract clauses appear without their numbers (four-number cap); the numbers are in
    the body.
16. **J-B5 wording** "in at most 0 % when they are not" is kept, and the record's definition of "not" is added.
17. **A–I post hoc risk labels** (G4): built by a script from each entry's "seen before" record and reported in A.5,
    marked post hoc and not pooled; or, if not built, this sentence: "The common part promised post hoc risk labels for
    the predictions of entries A–I; we did not compute them, because those entries' records of what was seen before
    finalisation are free text that a script cannot classify without judgment; their verdicts are in \cref{tab:prereg-ab} to \cref{tab:prereg-i}."

## 14. Reviewer objections and where v5 answers each

| Objection | Where | What the paper says |
|---|---|---|
| Templated stories only | §3.1 (new wording, lexicon, eight sentences), §3.2 (SQuAD), Limitations (i) | Fresh stories confirm in eight models; on passages a later options list opens the key read; behaviour there only with letters |
| Lower-case scoring | §2 ¶4, §3.1 ¶5, App. D.3 | Emitted-form scoring primary; re-scored verdicts unchanged; the old score understated the sentence ratio; generated answers as a score-free check |
| Novelty vs QK/OV and duplicate-token heads | §1 ¶6 + Table 1, §4 ¶1, §6 | Concede mask, hop 1 and option selection with citations; claim only what the architecture allows but does not imply, each preregistered; no new head type; polarity not a discovery |
| Interventions rest on one remap | §5 ¶2–¶5 | Five families built from public material in three models, an equivalence control and a sensitivity gate; the law failed and is reported as such, with a usable rule |
| 7 of 10 models are Qwen | §2 ¶5, §3 | Eight fresh-story models from seven families (two Qwen); passages in four families; mechanism limits in Limitations (vii) |
| Reliance on an anonymous submission | §1 ¶2 reliance sentence, §5 ¶1, App. G | Only the template, one released remap and its training code are used; the remap is described in our words; every number is ours |
| Circularity of the head ranking | §4 ¶2 | Flags fit on 60 discovery stories, tested on 100 fresh ones |
| Ablation vs knockout | Limitations (v), App. E.2 | Not reconciled; the passage tests were not evaluable |
| Statistics | §2 ¶5, App. A.1–A.2 | Cluster bootstrap; every-evaluable-model rule; 98.75 % for the new families; TOST for the law; Holm changes no verdict |
| Overclaiming / calibration | status tags, Table 2, §7 ¶2, App. A.3 | Risky predictions held less often than expected; every failure named |

## 15. Assembler checklist (in this order)

1. Archive the v4 sections (`paper/versions/v4_sections/`). Do not edit files of another writer.
2. Prerequisites: `main.tex` preamble (Section 3); bibliography rebuild (Section 9); `make_schematic.py` and
   `make_figures.py` changes (Section 8), then rebuild `numbers.tex` and the index; A–I labels script or the disclosure.
3. Collect the section files; compile only in a scratch copy of `paper/`
   (`tectonic -X compile main.tex --keep-logs`); fix undefined references and citations.
4. Page check (Section 10); apply cuts in order if needed.
5. Number check: every number in the main text is a macro, `\pct`/`\pctn` of a macro, a design count, a preregistered
   threshold, or carries a `% verbatim:` comment; no number in more than two main-text places; no NEGATIVE ZERO macro;
   every `[recomputed]` macro sits in a sentence labelled exploratory or post hoc.
6. Text checks (grep the main-text sections and the built PDF's pages 1–8): no prediction or gate codes
   (`J-[A-D]`, `G[0-9]`, `H[0-9]`, `I[0-9]` as codes, `E[1-5]` only inside Fig. 4 and §5's family names if used); no
   internal arm codes (Section 2.4); every phrase of Section 2.7 absent; "published verdict" absent; "partly" absent.
7. Anonymity: the leak patterns of `scripts/make_anonymous_release.py` and BRIEF rule 3 over the main text, appendix,
   tables, figure generators and the PDF text; Anonymous (2026) only in the third person; no local paths, user names or
   assistant model identifiers; PDF metadata of every figure checked (`pdfinfo`).
8. Pre-written texts: each item of Sections 5 and 13 present at its place, verbatim where marked.
9. Citations: only keys in `references.bib`; the descriptions of Section 2.5; the 5.6 re-check list done.
10. No `\todo`, no `todonotes`.
