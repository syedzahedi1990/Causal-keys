# Paper v6: space plan for the main text (2026-10-22)

Scope: the main text, from the Introduction through the Conclusion, including floats, must fit in 8 pages. The new
claim-3 section ("Whether an edit reaches the answer") needs about 1.3-1.6 pages, including a full-width figure
(`fig_window.pdf`) about 2.2 in tall that replaces `fig_law` in the main text. Read with `docs/V6_PLAN.md` and
`paper/BRIEF.md`.

**How this was measured.**
- `paper/` was copied to the scratchpad (`.../scratchpad/v6space/paper/`) and built there with Tectonic. The build is 95
  pages. `paper/main.pdf` was not touched.
- Positions come from `pdftotext -bbox-layout` on pages 1-10.
- A text column is 650 pt tall (y 67 to 717) with a line pitch of 11.95 pt: 54.4 lines per column, 108.8 lines per page.
  So **1 column line is about 0.0092 page**, and 11 lines are about 0.1 page.
- Float heights run from the top of the text block to the first text line after the float. A `figure*` costs twice its
  height.
- Paragraph lengths are counted lines in the PDF, and they agree with a source estimate at 57 characters per line to
  within 1-2 lines.

**v5 has no slack.** The Conclusion ends at the foot of page 8, right column (y 717.2). The Impact Statement opens
page 9.

---

## (a) Current page budget (v5 build)

| Unit | Where in the v5 PDF | Column lines | Pages |
|---|---|---|---|
| Title block, author line, affiliation and notice footnote (fixed) | p1 | - | 0.24 |
| Abstract (narrow abstract column) | p1, left | 31 | 0.31 |
| **§1 Intro, text.** P1 "Mechanistic accounts often credit…" 14; P2 "In the belief stories of…" 14; P3 "What is known, and what we add." 29 (its (2) is about 10 lines, its (3) about 3) | p1-p2 | 57 + heading | 0.56 |
| **Fig 1** `fig:schematic` (schematic + `fig_fresh` + `fig_crossover_main`), figure*, 336 pt | p2, top | - | 0.52 |
| **Table 1** `tab:qkov`, table*, 231 pt | p3, top | - | 0.355 |
| **§2 Setup.** Stories/formats 21; Passages 10; Clamps 8; Measures 22 (with the equation); Models and statistics 19; Preregistration and status marks 14 | p2-p3 | 94 + heading | 0.93 |
| **§3 Claim 1.** Lead 5; §3.1: list 17, sentence 13, Generated answers 14, The measurement 11; §3.2: passages 18, Behaviour 19 | p3-p5 | 97 + 3 headings | 1.03 |
| **Fig 2** `fig:natural`, figure*, 205 pt | p4, top | - | 0.315 |
| **§4 Claim 2.** Lead 7; Two hops 15; What the readers write 31; The sign of the read 28 | p5-p6 | 81 + heading | 0.79 |
| **Fig 3** `fig:flag`, figure*, 207 pt | p6, top | - | 0.32 |
| **§5 Claim 3.** Lead and the released remap 17; The law and its test 9; Steering vectors 19; Edits far from the natural write 42; The binding swap 13 | p6-p7 | 100 + heading | 0.99 |
| **Fig 4** `fig:law`, figure*, 250 pt | p7, top | - | 0.385 |
| **§6 Related.** Options, belief stories and heads 16; Attributing interventions 16 | p7-p8 | 32 + heading | 0.34 |
| **§7 Discussion.** What to do differently 18; How the predictions fared 17; Limitations 33; Conclusion 12 | p8 | 80 + heading | 0.81 |
| **Table 2** `tab:status`, single column, 137 pt | p8, top left | - | 0.105 |
| **Total** | | | **8.00** |

By claim:

| Claim | Text | Floats | Total |
|---|---|---|---|
| Claim 1 | 1.03 | 0.315, plus panels (b) and (c) of Fig 1 | |
| Claim 2 | 0.79 | 0.32 | 1.11 |
| Claim 3 | 0.99 | 0.385 | 1.37 |

The floats take 2.00 pages, a quarter of the budget.

---

## (b) Cuts and moves

### Constraints for every cut

- **Every number stays printed somewhere.** I checked every number macro in the main sections against `sections/app_*.tex`
  and `tables/*.tex`.
  - All claim-2 and claim-3 macros are already in the appendix.
  - Printed **only** in the main text: `\JaGateFailCells`, `\JbRangeSidAfterEightMin/Max` (§3, both kept in Tier A),
    `\GHIcountVerd` (Discussion) and `\JxFamiliesABCD` (a comment).
  - If a sentence that prints one of these is cut, add the macro to the appendix (`app:natural`, `app:fresh`,
    `app:prereg-ai`).
- **Text that may only be printed in words** (entry J, E-2, C-12 and D-13) stays verbatim where it is cited below. Every
  move from the main text to the appendix that the entry placed in the body is disclosed. The disclosure is a new item
  (iv) under `app:decisions` "Deviations from the reporting plan": the flag figure, Figure 4, and the body texts of J-C1,
  J-C4, J-C-SCREEN, J-D-ROUTE-IOI, J-D-HOP2, J-D5 and J-D6.
- **Moving v5 text verbatim is the cheapest safe route.** For §4 and the old §5, put the v5 paragraphs nearly verbatim
  into a new "in brief" subsection at the top of the appendix section for that claim. This keeps every checked sentence,
  number and pre-written text. Then write a short main-text version.

### Tier A (required): frees 2.40 pages

**A1. The old §5 moves out of the main text whole; the new §5 is written in its place. Frees 1.37 pages** (text 0.99, Fig
4 0.385). The compact law paragraph (at most 15 lines, about 0.14 page) is charged to the new section's 1.3-1.6.

| v5 paragraph (first words) | Destination | What stays in the main text |
|---|---|---|
| Lead: "\textbf{Even steering vectors close to the natural edit depart from the natural channel ratio;…" | | The pre-committed C-12 outcome (e) sentence, **verbatim**, as the first sentence of the new law paragraph. Update `app:decisions` item 13, which says it "opens its section". |
| "Qualitatively, the format dependence transfers to near-natural edits…" | New `app_f_claim3.tex` subsection **"The test in brief"** (`\label{app:law-brief}`), verbatim. Its content is already in `app:nu`, "The key share against the natural share". | Nothing. |
| "The motivating case is the released remap of \citet{anonymous2026fitted}…" | Already in `app_g_remap.tex` §`app:remap-crossover`, with `\psiLetterQwen` and `\psiNoneQwen`. | Optionally one clause: the released remap's carrying channel flips with the format (discovery sample; App. G). |
| "\paragraph{The law and its test.}…" | `app:law-brief` (the formal text is in `app:law`). | Two lines that state the law. |
| "\paragraph{Steering vectors.} In the preregistered core test…" | `app:law-brief`. Its numbers are in `app:lawresults`: "Steering vectors (J-C1…)" and "Where the steering vectors depart", including "4.5--12 %" and the power note. | J-C1's pre-written clause with the pooled $\bar\lambda$ range. |
| "\paragraph{Edits far from the natural write.}…" | `app:law-brief`; J-C3/C5/BOUND/READa/READb are in `app:lawresults`; ν is in `app:nu`. The sentence "A hypothesis for a future test: if the key route selects competitively…" is printed **only** in the main text today: add it to `app:nu` as `\paragraph{A hypothesis for a future test (post hoc).}`, and re-point its `tab:crosswalk` row. | J-C4's R1 replacement sentence with its $\bar\lambda$ pair. The block-7 $\psi_K$ / $\psi_V$ clause, which the record asks for in the main text (`app:decisions` 13). One clause naming the other law lines that did not hold or were not evaluable. |
| "\paragraph{The binding swap of \citet{prakash2025lookbacks}.}…" | Already in `app_h_binding.tex` §`app:binding-j`. | Optionally J-C-SCREEN's body sentence "Binding forms only after the identity key route closes in every model screened" (1.5 lines). Otherwise disclose the move. |
| **Fig 4** `fig_law` | `app_f_claim3.tex`, at the top of §`app:lawresults`, as `\begin{figure}[!ht]` at `\textwidth`. Label `fig:law` unchanged. | Nothing. |

**A2. §4 text, 81 → about 33 lines. Frees about 48 lines, 0.44 page.** Move the four v5 paragraphs verbatim to a new
subsection at the top of `app_e_claim2.tex`, **"The mechanism tests in brief"** (`\label{app:claim2-brief}`). Every
detail is also already in `app:heads`, `app:flag`, `app:sign` and `app:scale`.

D-13 limits the main text to three flag numbers: $\mathrm{ID}^{\mathrm{LOO}}_{\mathrm{inj}}/\mathrm{ID}_K$ (`\JdOneRatio*`),
$\rho_K$ (`\JdThreeRhoFlag*`) and $r^{\mathrm{inj}}_{\mathrm{ans}}(K)$ (`\JdFourK*`). The v5 main text gives more than these.

| Paragraph (first words) | Main text keeps | Goes to the appendix only |
|---|---|---|
| Lead: "\textbf{In two 7B models (Qwen2.5-7B, Mistral-7B), the reader heads…" (7 lines) | 4 lines. | - |
| "\paragraph{Two hops.} In the discovery sample, 5\,\% of heads…" (15 lines) | 6 lines: 5 % of heads, `\HoneR*`; enriched for duplicate-token heads (a clause, no numbers); ablation leaves `\HthreeRho*` and the copy route stays; the answer reads the option words by key. | `\HfourKC*`, `\HfourD*` and `\HfourDwhole*`, and the "core that depends on how many heads" clause. |
| "\paragraph{What the readers write.} The reader heads write one vector per layer…" (31 lines) | 14 lines: J-D1's sentence **verbatim** with its two number pairs; "its mean direction names no candidate" (J-D-ADDR); J-D4 with `\JdFourK*`; J-D2 not held, in words only; J-D3's replacement **verbatim** with $\rho_K$. | The add/subtract explanation and half-flag dose, which go with `app:decisions` 18 (update it); the fit variants; `\JdTwoRatio*`; `\JdFourV*`; the 31 % and 44 % control sentence (already in `app:flag` and `app:decisions` 19). |
| "\paragraph{The sign of the read.} Asked which option was \emph{not} mentioned…" (28 lines) | 8 lines: the question sets the sign after hop 1 (exploratory); IOI's key read negative and value read positive (J-D-SIGN-IOI, held); J-D-ROUTE-IOI held with the `\JdRouteKV*` pair; HOP2 not held, with its replacement words "a different (or a distributed) reader applies the negative read"; J-D8-IOI met at Qwen and not evaluable at Mistral; J-D6 ("Mixed."), J-D5 and J-D-KN not held, in one clause. | `\JdSevenQout*`, `\JdRouteK*` and `\JdRouteV*`, and the scale details. |

**A3. Fig 3 `fig_flag` moves to `app_e_claim2.tex`, at the top of §`app:flag`, as `\begin{figure}[!ht]` at
`\textwidth`. Frees 0.32 page.** Label `fig:flag` unchanged. Disclose it with the deviations: D-13 planned the flag figure
for the main text.

**A4. Setup (§2), preregistration bookkeeping. Net -11 lines, 0.10 page** (18 lines cut, 8 added).

| Paragraph (first words) | Change | Lines |
|---|---|---|
| "\paragraph{Preregistration and status marks.} Each of ten preregistrations…" (14) | Keep at 9 lines, including a new 2-line stage-9 sentence: a dated prediction note frozen before the run, not a preregistration entry, no class priors, its own mark. The risk-class detail and "Where two rules of the entry conflicted… both choices favour our account" go to a half-sentence pointer; the detail is already in `app:prereg-how` and `app:decisions`. | -5 |
| "\paragraph{Models and statistics.} The discovery sample has \nModels{}…" (19) | Keep at 14 lines, including stage 9 (three models; 95 % two-stage cluster bootstrap on generated answers). Move "…in the mechanism tests over stories with the flag's fit fixed…" (it goes with the fig_flag caption), "the intervention law uses 90 % intervals…" (`app:law`), "one over the four new families in three of them (98.75 %…)" (`app:prereg-how`) and "Holm's step-down… changes no verdict" (`app:prereg-j`). | -5 |
| "\paragraph{Stories, runs and formats.}…" (21) | Cut the clause "a synonym or a word of another language in its place opens at most \GfifteenRKMax{}…" (already in `app:readers`). Add the stage-9 format mapping: AFTER = `\fmtListA`, LETTER = `\fmtLetter`, NONE, BEFORE and POST. | -2 |
| "\paragraph{Measures.}…", the sentence "Under the \emph{emitted-form score}…" | Becomes a 1-line pointer; the detail is already in `app:emitted`. | -3 |
| "\paragraph{Clamps.}…" (8) | Add the late and early patch (K and V of the writing token from block $l$ to the top, or below $l$), $l_c$ and $l_b$. This can instead go in §5. | +4 |

**A5. Discussion (§7), tally bookkeeping. Net -12 lines, 0.11 page; Table 2 adds about 0.01.**

| Paragraph (first words) | Change | Lines |
|---|---|---|
| "\paragraph{What to do differently.} (i)~State where the candidates are mentioned…" (18) | Becomes the three numbered rules of V6_PLAN in 13 lines. (i) and (ii) fold into rule 1; (iv), the attention knockout, becomes rule 3. (iii)'s "report the edit's distance from the natural state" is dropped: it was post hoc and is superseded by the key-share diagnostic, while ν stays in `app:nu`. (v), the KV-cache pitfall, is already in `app_c` "A KV-cache pitfall". | -5 |
| "\paragraph{How the predictions fared.} The risky predictions (\cref{tab:status}) held…" (17) | Keep at 13 lines, including the 3-line stage-9 sentence (3 of 10, near misses named, F2 at Qwen). Keep the risky tally, its tail probability and the alternative-rule tally (`app:decisions` 12 requires it here). Move "Of the \JxRiskyNotEvalABCD{} risky predictions not evaluable…" and the extrapolation and Brier sentence to `app:prereg-j` ("The tally"; both texts exist there). Keep the G-I clause: `\GHIcountVerd` is printed only here; otherwise add it to `app:prereg-ai`. | -4 |
| "\paragraph{Limitations.} (i)~Our passages are counterfactual SQuAD items…" (33) | Keep at 31 lines, including 4 stage-9 lines. Cut (iv), (v) and (vi) to one clause each, pointing to `app:fresh`, `app_d` "Clamp onset" and `app_e` "Ablation against knockout (not reconciled)". | -2 |
| "\paragraph{Conclusion.} Across seven model families…" (12) | Rewritten in at most 11 lines. | -1 |
| Table 2 `tab:status` | Add one row, "Edit window (stage-9 note)", printing 3 / 10 supported and 0 not evaluable in a `\multicolumn`, with no expected count (no priors). | +0.01 page |

**A6. §3.1, "\paragraph{The measurement.} Scored on the forms the models actually emit…" (11 → 6 lines). Frees 0.05
page.**
- Keep J-B6c's sentence and the failed invariance with `\JbSixbRPost*`.
- Cut the last sentence, "Two further validity checks held\rsk: … the released remap's frames (\cref{sec:claim3}) are
  unchanged at Mistral-Small-24B…". It is already in `app:measure` ("The intervention frames.", J-B7), and its pointer to
  §5 becomes wrong in v6.

**A7. Table 1 `tab:qkov`. Net about -0.02 page.**
- Merge "What does the matching head write?" and "What sets the sign of a key read?" into one row.
- Replace the last row, "Is an edit's apparent channel fixed?", with the window row: "Until when can an edit at the
  writing token change the answer?".
- Trim the "What we find" cells.

**Tier A total: 1.37 + 0.44 + 0.32 + 0.10 + 0.10 + 0.05 + 0.02 = 2.40 pages**, which meets the 1.8 target. Of this, 1.03
pages come from outside the old §5.

### What the freed space must cover

| Addition | Pages |
|---|---|
| New §5, all-in | 1.30-1.60 |
| Abstract (window clauses) | +0.02-0.04 |
| Intro: contribution (3) with numbers, +4 lines; contribution (2) compressed, -6 lines | about 0 |
| Setup additions | already netted in A4 |

**Projected v6 main text:** 8.00 - 2.40 + (1.30 to 1.60) + 0.03 = **6.9-7.2 pages**. That leaves a reserve of 0.8-1.1
page.

Use the reserve in this order:
1. Leave at least 0.25 page unused, for float placement and draft overshoot.
2. Optionally, a single-column table of the window in §5 (`tab_window`: $l_c/L$ by model and format, with the early
   flips), about 0.12 page.
3. Optionally, a column-width `fig_flag` with panels (a) and (c) only, back in §4 (about 0.2 page). This reduces the
   D-13 deviation.
4. Compress §4 less, to about 40 lines.

Do **not** spend the reserve on bringing back the law paragraphs.

**Suggested internal budget for the new §5** (column lines):

| Part | Lines |
|---|---|
| Heading and bold claim | 5 |
| Design, if not in Setup | 5 |
| The window: P2, P1, BEFORE and F2s | 14 |
| Silent failure (P4) | 6 |
| Early window / crossover (P3s; not supported at Mistral) | 8 |
| Scope: BEFORE, POST (not supported at Mistral), passages as an evaluation warning (F2 at Qwen) | 12 |
| Edits: E1 (P5a, one cell), E4b (P5b near miss), CAA transfers, and the scope sentence | 14 |
| Key share (B3) | 6 |
| Dose convexity, with no threshold (B1, B2) | 5 |
| Tally of 3 of 10, with the near misses | 3 |
| Compact failed-law paragraph (A1) | 14 |
| **Total** | **92 lines, about 0.85 page** |

With `fig_window` (about 0.35-0.37 page: 158 pt of graphic, a 4-5 line caption and the float gap) this is about 1.23
pages, or about 1.35 with the optional `tab_window`.

### Tier B (only if the first full v6 build overflows, or §5 exceeds 1.6)

| # | Item (first words) | Change | Saves |
|---|---|---|---|
| B1 | §3.2 "\paragraph{Behaviour on passages is a boundary.} In cue conflict…" (19) | Down to 12 lines: keep the lettered-options key win, the failures by name and in free form, and "not shown to decide behaviour on natural text". Cut "although a key that matches no option leaves the value to be copied" (`app:natural`, J-A6e). Reduce the KIVI, hybrid and not-evaluable sentence to a 2-line clause (`app:natural`). | 0.06 |
| B2 | §3.2 "On counterfactual SQuAD passages, a later options list…" (18) | Down to 15 lines. Keep the gate sentence (`\JaGateFailCells` is printed only here; `app:decisions` 1 requires the J-A3 statement in §3.2). | 0.03 |
| B3 | §3.1 "A neutral sentence naming the candidates after the story…" (13) | Down to 10 lines: the Llama-3.1-8B post hoc reading and the Gemma-2-2B gate as one clause each (`app:fresh` has both). | 0.03 |
| B4 | §2 "\paragraph{Passages.} From the SQuAD v1.1 development set…" (10) | Down to 7 lines; the filters go to `app:prompts-natural`. | 0.03 |
| B5 | Fig 1 caption (6 lines) | Down to 4 lines: drop "(the new families' verdicts use 98.75 % bounds)" (in Setup) and the panel-(c) restatement. | 0.03 |
| B6 | §6 "Options, belief stories and heads." | Down 3-4 lines by trimming the flag/routing-feature comparison. | 0.04 |
| B7 | **Last resort:** Fig 2 `fig_natural` | Moves to `app_d_claim1.tex` §`app:natural`. The text keeps the passage numbers. | 0.315 |

Tier B total: about 0.55 page.

---

## (c) Floats in the v6 main text

**Stay (in order):**

| Float | Kind | Page | Changes |
|---|---|---|---|
| Fig 1 `fig:schematic` (schematic, `fig_fresh`, `fig_crossover_main`) | figure* | top of p2 | Caption trim optional (B5). Panel (a)'s hop-1/hop-2 annotation still fits the compressed claim 2. |
| Table 1 `tab:qkov` | table* | top of p3 | Updated as in A7. |
| Fig 2 `fig:natural` | figure* | top of p3 or p4 | None. Only B7 moves it. |
| **New** Fig 3 `fig:window` (`fig_window.pdf`) | figure*, about 2.2 in | top of the page where §5 opens (p5 in the projection) | Put it at the head of `results_intervention.tex` so that it floats there. |
| Table 2 `tab:status` | single column, in §7 | | Plus the stage-9 row. |
| Optional `tab_window` | single column, in §5 | | Uses the reserve. |
| Optional column-width `fig_flag` (a, c) | | in §4 | Uses the reserve. |

**Move to the appendix:**
- `fig:flag` → `app_e` §`app:flag`.
- `fig:law` → `app_f` §`app:lawresults`.

Both labels are kept, so every `\cref` still resolves. **v6 has 3 figure* floats, 1 table* and 1 single-column table** in
the main text, against 4, 1 and 1 in v5.

---

## (d) What the new story makes obsolete, and hazards

### Intro and title

1. **Title** (`main.tex` `\icmltitle`): becomes the V6_PLAN title. Keep the running title "Looked Up or Copied?".
2. **Intro P3 contribution (3)**, "A preregistered law predicting an edit's apparent channel… failed… an intervention's
   channels must be measured." This is replaced by the window, with the practitioner consequence first. The failed law
   becomes one clause.
3. **Intro P3 contribution (2)**, second half: "In lists (discovery sample), the written value is thus read at its later
   mention… consistent with S-inhibition". Cut it; it already sits in §4 and the appendix. Keep (2) to about 4 lines.
4. **Intro P2.**
   - The final sentence ("From \citet{anonymous2026fitted} we take only the story template and, in \cref{sec:claim3},
     one released remap… and its training code") is obsolete: §5 no longer studies the released remap. The new text is:
     the story template and the remap training recipe (E4/E4b), with the released remap in `app:remap`.
   - Add the stage-9 run: three models; a dated prediction note frozen before the run, not a preregistration entry.
5. **Intro P1 closing pointer** "(\cref{sec:limitations}, ix)": the limitation numbering will change.
6. **Table 1.**
   - The last row "Is an edit's apparent channel fixed?" (released-remap flip; "the law did not hold") is replaced.
   - The caption's legend defines only `\rsk` (a recorded prior below 0.8). Stage-9 lines have no priors, so they need
     their own mark, or words, here, in Setup and in `tab:crosswalk`.
   - Window cells say "model-specific depth" and never a universal depth.
7. **Abstract.**
   - The sentence "A preregistered law that edits keep the model's own key-to-value ratio failed…" becomes one clause.
   - The tally sentence adds stage 9's 3 of 10.
   - The claim-2 sentence ("In two 7B models the matching heads write…") is compressed.

### Setup

8. **"Unless marked, a result in \cref{sec:claim1,sec:claim2,sec:claim3} was preregistered in the final round and
   held"** is false for the new §5, which is stage 9. Restate the convention.
9. **The 90 % TOST intervals of the law** leave the Setup with the law.

### Discussion

10. **"What to do differently" (i)-(v).** All become the three V6_PLAN rules. (iii)'s ν advice and (v)'s KV-cache pitfall
    leave the main text (A5).
11. **"How the predictions fared".**
    - Its singling-out of the intervention extrapolations ("the clearest shortfall… Brier") is v5's claim-3 story.
      Move it to `app:prereg-j` and add the stage-9 sentence.
    - Stage 9 has no priors, so it never enters an "expected" count.
12. **Limitations.**
    - (vii) "the channel-ratio law in three 7-8B models" becomes the window plus the law.
    - Add: one story template for the window; three 7-8B models; the passages at Qwen (F2); a model-specific depth;
      E4b at two models only (no stored basis at Llama).
    - **(v), the onset sweep** ("From a clamp onset at 30 % of the depth, the key share stays at or above one half…"),
      sits next to the new late-window result. That result comes from a K+V patch read on generated answers in N9; the
      onset sweep is a key-only log-probability share in the discovery sample. Do not present them as agreeing or
      conflicting without saying that they measure different things. The simplest course is to move (v) to `app_d`
      "Clamp onset" with that sentence.
13. **Conclusion.** Its last sentence ("Where an intervention appears to act depends on the prompt and the edit; a
    preregistered law… failed") is replaced by the window, the steering scope, the key share and one clause on the law.

### Related work and Impact Statement

14. **Related.** "Attributing interventions" should gain one sentence on KV-cache editing and patching practice. The keys
    `bojieli2026notes`, `pustovit2026packs`, `heimersheim2024patching` and `zhang2023patching` are in `references.bib`;
    check each abstract before describing it.
15. **Impact Statement** (outside the 8 pages): add that whether an edit reaches the answer depends on the format.

### Hazards for the writer

16. **"P1".** The note's line codes P1-P5 collide with the anonymity rule ("P1" must never name a paper) and with stage
    8c's format "P1" (= `\fmtOpt`).
    - Never print a bare "P1".
    - Prefix the codes, for example "S9-P1", consistently in the text and in the auto-generated `tab_stage9.tex`.
17. **"Crossover".** It already names the released remap's channel flip (`app:remap-crossover`) and the generated-answer
    figure (`fig_crossover*`, `app:generated` "The crossover (reported)").
    - If §5 calls the early window "the crossover", say "early window" or rename the older uses.
18. **Symbol clash.**
    - $r_K$, the stage-9 key share, is already used about 11 times in `app_e` (stage-5 G lines).
    - $r$ is the format ratio; $r_{\mathrm{ans}}$ and $r(KV)$ belong to claim 2; $\kappa$ and $\sigma$ are the law's key
      shares.
    - Choose a distinct symbol for the stage-9 key projection and define it once.
19. **Repetition rule** (BRIEF 5: no number in more than two places in the main text).
    - The window's depth fractions are tempting in the abstract, intro (3), Table 1, §5 and the figure caption.
    - Pick two places, for example §5 with its figure and Table 1. Say it in words in the intro, or the reverse.
20. **Appendix knock-ons.**
    - `app_a_prereg.tex` has 40 lines that point to `\cref{sec:claim2}` / `\cref{sec:claim3}`, mostly in `tab:crosswalk`.
      Re-point every moved sentence.
    - `tab:guide` needs stage-9 rows.
    - `app:decisions` needs updates to items 4, 13, 18 and 19, plus the new deviation (iv).
    - `app_f`'s title and opening should say it is the stage-8 channel-ratio law.
    - Add a new section for the stage-9 window (for example `sections/app_f0_window.tex`, input before `app_f_claim3`).
21. **Version B** (`main_estimates.tex`, if written) can use the same float plan. Its verdict bookkeeping in §2, §5 and
    §7 is shorter, so it gains about 0.1-0.2 page more.
