LABEL design:rewrite
### part
E: the rewrite and the novelty argument (objections 1, 7 and 8). It gives the three-claim structure of the v5 main text, the novelty paragraph, the missing related work (verified), the title and abstract, how to make Section 4 (Section 5 in v5) self-contained, and the appendix moves with a page budget.

### objections_answered
[
 "(1) Novelty: 'QK/OV plus duplicate-token heads already imply it; the position effect follows from the causal mask.' The answer concedes in print exactly what the architecture implies. A key can carry identity only to a later token that already holds a candidate, so a mention placed before the writing token cannot read it. It then names the regularities the architecture allows but does not predict, each backed by a preregistered risky test in v5 with a stated rival. (a) The route is a property of the prompt: the value route is open in every prompt, yet with options listed after the story the key carries 0.6-0.9 of the identity, even though the answer word could be copied. (b) A matching query is not sufficient: at 1.5B/3B the match is present but not read. (c) The match has no sign of its own; the question sets the sign (Part D). (d) The channel credited with an intervention can be predicted from the natural read at matched depth, and a binding swap breaks the prediction (Part C). Also (e) the answer's own prefix acts as a later mention for multi-token answers (Part A, A8). Table 1 of the Introduction sets these out ('What QK/OV and the causal mask imply, and what we find'). The paper keeps 'we claim no new head type'.",
 "(1, sceptic) 'Lookup over-reads what the key carries; the identity is already in the option word.' v5 adopts this as the definition. Looked up means a later token that already holds the candidate matches the writing token's key. Copied means the identity is moved out of the writing token's value. Part D's low-rank flag is what the match writes.",
 "(1, generalist) 'No actionable consequence.' v5 adds a 'What to do differently' box with four rules, plus the behavioural results from Parts A and B. Key-only swaps flip MCQ answers and value-only swaps flip free-form answers. MCQ answers survive value corruption that breaks free form. MCQ-found readers do not explain free-form answers. These are tied to KV-cache asymmetry (KIVI) and to MCQ-based localisation (MIB).",
 "(7) Presentation, a ledger of codes. The v4 main text has 111 mentions of prediction codes (61 distinct), about 238 numbers and more than 15 bespoke statistics in 6,141 words; all were measured. v5 rules: zero prediction codes in the main text, with status tags in words; at most four symbols (ID_K, ID_V, s_ID, kappa); at most two numbers per paragraph; one claim statement, then evidence, then a single caveat sentence per section; Table 3 gives evidence status per claim by risk class; an appendix crosswalk maps every main-text sentence to its prediction code. An automated check enforces these rules.",
 "(7, stats) 'Calibrate: separate risky from low-risk, and bring the caveats into the main text.' A risk-label rule is fixed in entry J before any stage-8 output. The abstract reports the met rate among risky predictions. The Limitations paragraph carries the caveats in words: scoring, off-distribution clamps, onset, set-size dependence of the canonical core, the ablation/knockout conflict, 0/150 flips of the binding swap. The seed-0 2x2 (E1c) is relabelled as a discovery-sample re-measurement.",
 "(8) Reliance on an unreadable anonymous submission. Section 5's claim rests on edits anyone can rebuild (Part C: in-task steering, unrelated-text steering, a public third-party SAE, DAS trained in-paper), on public models, with the tautology lemma and controls stated in the paper. The released remap becomes one motivating paragraph. It is described completely in the paper and its appendix, and every number about it is our own reproduction. Its significance is argued from published channel attributions (Prakash et al. 2026; Ma et al.; Oh et al.; Cheng et al.), not from the anonymous paper. Citation format is fixed for the three possible states of that submission. Anonymity checks are automated.",
 "(Missing related work) Elhage et al. 2021; Gur-Arieh et al. 2025; Dai et al. 2024; Prakash et al. 2024; Variengien & Winsor 2023; and Merullo et al. 2024 were verified, with titles and arXiv ids, against arxiv.org/abs pages and the Transformer Circuits page. Eighteen further works the rewrite needs were verified the same way. Each is positioned explicitly in a rewritten Related Work section, written so every description is supported by the cited abstract."
]

### design

# Part E: the v5 rewrite, the novelty argument and self-containment

Drafts ready for the builder (LaTeX, under `/tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partE/`):
- `v5_title_abstract.tex`: title, abstract, and replacement clauses for each failure case.
- `v5_novelty.tex`: the novelty paragraph and Table 1.
- `v5_related.tex`: the new Related Work, about 470 words.
- `v5_skeleton.tex`: section skeleton with paragraph budgets and labels.
- `v5_bib_additions.py`: entries for `build_bib.py`.

The abstract and Table 1 were compiled with the ICML 2026 style (`tex/t.pdf`). The abstract takes about 0.42 page and Table 1 about 0.22 page.

---------------------------------------------------------------------
## (i) Three claims for the 8-page main text (assuming Parts A–D come out as predicted)

Each claim section opens with **one bold sentence stating the claim**. Evidence follows in order of strength: risky preregistered tests first, then replications, then exploratory results. The section closes with **one** caveat sentence; claims and caveats are no longer interleaved.

### Claim 1 (Section 3), "Later mentions decide the route"
Statement: *For a token that writes a value into context, the value's identity reaches the answer through the token's key (looked up by later tokens that already hold the candidates) when the candidates are mentioned after it, and through its value (copied) otherwise. The position of the mention decides, not its content or the instruction. The route decides which cache manipulation changes behaviour.*

| Evidence | Source | Status tag in text |
|---|---|---|
| Discovery: natural factorial, 10 models, 1.5B–72B; sentence effect found post hoc | v4 (A1, B1, B2, C4) | "discovery sample" |
| Fresh stories, instruction-matched 2x2, 4 new families + 4 original, emitted-form scoring, generation flips | Part B (JB1–JB5 risky in new families; JB6 invariance; JB9) | "preregistered, risky; met in k/4 new families" |
| SQuAD passages, multi-token spans, 4 families: options after vs none vs before; natural mention sentence | Part A (A1–A5) | "preregistered, risky" |
| Behaviour: key-only swaps flip MCQ answers and value-only swaps flip free form; MCQ survives value corruption | Part A (A6, A7) + Part B (JB5) | "preregistered, risky" |
| Scope: paint/schedule tasks, role swap, deeper onsets | v4 (D1, D2, E2, onset) | one sentence, appendix |

Main-text artefacts:
- **Fig 2** (full width):
  - (a) s_ID by format on fresh stories, 8 models, with the 10 discovery models as ghost markers;
  - (b) s_ID by format on passages, 4 models;
  - (c) key-flip and value-flip rates, MCQ against free form, templated and passages.
- **Table 2** (single column): the fresh-sample 2x2, giving r = ID_K(format)/ID_K(LIST-AFTER) and s_ID(NO-MENTION) per model, with new families marked.

### Claim 2 (Section 4), "A generic match, read downstream"
Statement: *The lookup has two hops. Duplicate-token heads at each later mention match the writing token's key and write a low-rank flag into that mention's rows. The answer then selects the flagged mention by query–key alignment. The match is generic: whether it is read, and with which sign, is set downstream.*

| Evidence | Source | Status |
|---|---|---|
| 5% of heads recover/remove the read; random sets do not; the core is canonical duplicate-token heads (set-size dependence stated) | v4 (H1, H2, H4); 24B readers (I-G2) | preregistered |
| The flag: a rank-1 direction from the readers' output; injected alone into an absent option's row, it moves the answer there; random direction and non-option row do not | Part D (i) | preregistered, risky |
| Hop 2: the answer reads the option rows (select-and-copy-like); MCQ readers not needed for free-form answers | v4 (H5); Part A (HA3) | preregistered (HA3 risky) |
| Readers on natural text; template-to-passage transfer | Part A (HA1, HA2) | preregistered, risky |
| Present but not read: 1.5B/3B attention follows the key one for one, no read; flag written-but-unread vs never-written | v4 (G1–G3) + Part D (i) | preregistered |
| Sign set by the question: same story and list, a question asking for an unwritten option turns ID_K negative through the same readers; IOI in-sentence re-mentions go through the same readers | Part D (ii) | preregistered, risky |

Main-text artefacts: **Fig 3** (full width):
- (a) R(k)/KO(k) at 7B (template), natural-text markers;
- (b) flag injection;
- (c) scale panel (match attention vs key identity vs flag read);
- (d) sign panel (written-value question vs unwritten-value question vs IOI, with readers blinded).

Part D's own CPU pilot (0.5B/1.5B, read from its logs, to be disclosed by D) already shows:
- the flag "move" injection reaches iota about 0.7–0.8, with random directions at about 0;
- the IOI-inline negative read (−2.75 nats) is reproduced by belief-task readers alone (−2.12) and removed when they are knocked out (−0.07);
- at 1.5B, ID_K is +2.1 when the question asks for the written option and −4.2 when it asks for an unwritten one.

### Claim 3 (Section 5), "Where an intervention appears to act follows the natural read"
Statement: *For edits that change which value is written, the share of the edit's identity effect carried by the writing token's keys (kappa) equals the unpatched model's own identity key share in the same prompt from the next block on (s_ID at matched depth). A channel attribution of such an edit is therefore a property of (edit, prompt, depth), and it can be predicted before the edit is run. Binding swaps do not follow this rule.*

| Evidence | Source | Status |
|---|---|---|
| Tautology lemma (a full-residual patch = the natural clamp); only edits with nu ≥ 0.30 count | Part C | stated, unit-tested |
| Steering from held-out stories, steering from unrelated sentences, third-party SAE features, DAS trained in-paper: transfer, crossover, law, depth tracking, same readers | Part C (JC1–JC5) | preregistered, risky |
| Boundary: the binding swap departs (failed extension); at the same depth an identity edit follows, so kind, not depth | v4 (H7–H10) + Part C (JC6) | failed, then preregistered diagnosis |
| Motivating case: released remap (format flip, refit, 24B readers; I6 not met) | v4 (C3, E4, F, I1–I3, I6) | in-sample / post hoc / preregistered, labelled |

Main-text artefact: **Fig 4** (single column):
- kappa against s_ID at matched depth for every out-of-sample cell, coloured by edit family and shaped by model;
- the equivalence control on the diagonal;
- the binding swap (H7 and its JC6 re-run) off the diagonal;
- the released remap as hollow markers, labelled post hoc;
- an inset for depth tracking.

### What changes if a part fails (pre-written, applied mechanically from the preregistered verdicts)
Rule: a claim enters the abstract only if its primary risky prediction is MET under its preregistered combination rule. MET IN PART goes in the body only, with the word "partly". NOT MET becomes the pre-written boundary statement.

| Outcome | Title | Abstract | Claims / figures | Table 1 |
|---|---|---|---|---|
| A1 or A3 not met (no lookup on passages) | "...Option Lists Decide Which Attention Channel Carries an In-Context Value" | S4 replacement in `v5_title_abstract.tex` | Claim 1 scoped to templated prompts; Fig 2b kept as a boundary | rows 1–3 say "templated" |
| A5 or JB3 not met in ≥2 new families | unchanged | "as answer options (and, in some families, a neutral sentence)" | sentence result reported per family | – |
| A6/A7 not met | unchanged | drop the behaviour clause; keep JB5 if met | drop practitioner bullet 3 | row 3 removed |
| JB1/JB2/JB4 fail in a new family | unchanged | the family count drops to the families where they hold | Table 2 marks failures | – |
| JB6 not met | unchanged | unchanged | emitted-form numbers become primary everywhere, stated in the Setup | – |
| D(i) flag injection not met | unchanged | S5 replacement | Claim 2 becomes routing-level (readers plus hop 2); Fig 3b moves to the appendix | row 5 removed; row 4 keeps "present but not read" |
| D(ii) sign not met | unchanged | drop the sign clause | negative reads listed in Limitations as unexplained | row 6 removed |
| C "restricted to near-natural edits" | unchanged | S6 replacement (attribution follows the natural read only near the natural state) | §5 shrinks to 0.7 page as a scoped negative; the freed 0.3 page goes to Claim 2 | row 7: "Only near the natural state" |
| C "format-only" (JC4 not met) | unchanged | "in the same prompt" (drop "and depth") | Fig 4 inset moves to the appendix | row 7 edited |
| C and D both fail | unchanged | as above | the paper becomes Claim 1 (broad) + mechanism + intervention case study; novelty rests on Table 1 rows 1–3 and the passages | – |

---------------------------------------------------------------------
## (ii) The novelty argument (one paragraph; the last paragraph of the Introduction, followed by Table 1)

> **What the QK/OV picture does not predict.** Every attention head factors into a QK circuit, which decides where it attends, and an OV circuit, which decides what it moves (Elhage et al., 2021). With the causal mask this bounds what is possible: a writing token's key can carry which value it wrote only to a later token that already holds a candidate. Duplicate-token and induction heads attend from such tokens (Wang et al., 2022; Olsson et al., 2022), answers select options by query–key alignment (Lieberum et al., 2023; Wiegreffe et al., 2024; Tulchinskii et al., 2024), and bound entities are retrieved through binding or ordering IDs, positions, lookbacks, or a context-dependent mix of positional, lexical and reflexive routes (Feng & Steinhardt, 2023; Dai et al., 2024; Prakash et al., 2024; 2026; Gur-Arieh et al., 2025). None of this says which channel a model uses, because the value route is open in every prompt: the answer can always attend to the writing token and copy. Measuring that choice, we find four regularities the framework allows but does not predict (Table 1). The choice belongs to the prompt: with options listed after the story, where the answer word could simply be copied, the key carries most of the identity; without a later mention, almost none. A matching query is not enough: at 1.5B and 3B the duplicate-token match follows the clamped key one for one, yet does not reach the answer. The match has no sign of its own: asking for an option that was not written turns the same read negative, as in-sentence re-mentions in IOI do. And the channel credited with an edit of the written value follows the unpatched model's own read at the same depth, which a binding swap does not. We claim no new head type; we show which channel is used, when, and what follows for interventions.

Why a sceptical mechanistic-interpretability reviewer should accept it:
1. It **concedes** what is implied: the necessity of a candidate-holding later token, the mask's ban on reads by earlier list tokens, and the canonical hop-1 heads.
2. Each non-implied regularity has a **risky preregistered test with a named rival**:
   - the route belongs to the prompt: copy-only rival, Parts A and B;
   - insufficiency: G2 plus D's written-vs-read test;
   - sign: D(ii), where the rival is a sign fixed by format;
   - attribution: C's copy-only, key-flat and depth-blind rivals.
3. The binding-swap failure is used as the **proof that the attribution law is not definitional**. A key-flat edit exists, so "kappa = s_ID" can be false.
4. It places the closest prior work (Gur-Arieh's lexical route; Tulchinskii's QK option selection) as the residual-level and hop-2 counterparts of what we measure at the cache level.

Wording rules inside the paragraph: clauses that depend on D or C carry [D-i]/[D-ii]/[C] tags in the source file and are deleted per the contingency table.

**Table 1 ("What QK/OV and the causal mask imply, and what we find"; single column, about 0.22 page; compiled).** Seven rows: question | implied? | finding (status; section).
1. A mention before the writing token reads its key? Implied (no). Finding: no later token reads it instead either.
2. With a later mention, is the key used instead of copying? Not implied.
3. Does the route change behaviour? Not implied.
4. Is a matching query sufficient? Partly implied.
5. What does the reader write? Not implied.
6. What sets the sign? Not implied.
7. Where does an intervention appear to act? Not implied.

---------------------------------------------------------------------
## (iii) Related work the panel named as missing: verified titles and ids

All entries below were checked on 2026-10-10.
- arXiv entries were checked against the arxiv.org/abs page meta tags (title, authors, date).
- Venues come from the arXiv Comments field unless noted.
- The arXiv API returned 503 from this container, so the abs pages were used instead.

| Bib key | Exact title | Authors | arXiv id | Venue | Verification |
|---|---|---|---|---|---|
| elhage2021framework | A Mathematical Framework for Transformer Circuits | Elhage, Nanda, Olsson, Henighan, Joseph, Mann, Askell, Bai, Chen, Conerly, DasSarma, Drain, Ganguli, Hatfield-Dodds, Hernandez, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, Olah | **none (not on arXiv)** | Transformer Circuits Thread, 22 Dec 2021 | verified on transformer-circuits.pub/2021/framework/index.html (title, full author list, date, the QK/OV definitions) |
| gurarieh2025mixing | Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context | Gur-Arieh, Geva, Geiger | 2510.06182 | ICLR 2026 | verified |
| dai2024binding | Representational Analysis of Binding in Language Models (v1 title: "...in Large Language Models") | Dai, Heinzerling, Inui | 2409.05448 | EMNLP 2024 | arXiv verified; venue taken from the ACL Anthology author listing via search, Anthology page **not opened** |
| prakash2024finetuning | Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking | Prakash, Rott Shaham, Haklay, Belinkov, Bau | 2402.14811 | ICLR 2024 | verified. The head names the panel used (value fetcher, position transmitter) are **not in the abstract**; under brief rule 2 cite only "tracks the position of the correct entity", or check the body first |
| variengien2023leap | Look Before You Leap: A Universal Emergent Decomposition of Retrieval Tasks in Language Models | Variengien, Winsor | 2312.10091 | ICLR 2025, as "Look Before You Leap: Universal Emergent Mechanism for Retrieval in Language Models" | verified (arXiv + iclr.cc poster 28935) |
| merullo2024reuse | Circuit Component Reuse Across Tasks in Transformer Language Models | Merullo, Eickhoff, Pavlick | 2310.08744 | ICLR 2024 | verified |

Further works the rewrite cites, verified the same way:
- merullo2024talking: 2406.09519, Talking Heads, NeurIPS 2024
- mcdougall2023copy: 2310.04625, Copy Suppression
- robinson2023mcsb: 2210.12353, MCSB, ICLR 2023
- geva2023dissecting: 2304.14767, EMNLP 2023
- kobayashi2020norm: 2004.10102, EMNLP 2020
- springer2024echo: 2402.15449, ICLR 2025
- turner2023actadd: 2308.10248
- panickssery2023caa: 2312.06681
- bussmann2024batchtopk: 2412.06410
- liu2024kivi: 2402.02750, ICML 2024; its abstract supports "keys per-channel, values per-token"
- rajpurkar2016squad: 1606.05250
- longpre2021conflicts: 2109.05052, EMNLP 2021
- grattafiori2024llama3: 2407.21783
- gemma2024gemma2: 2408.00118
- abdin2024phi4: 2412.08905
- optional: lieberum2024gemmascope (2408.05147), gould2023successor (2312.09230)

Not papers, cited as model cards:
- Falcon3-7B-Instruct: HF repo exists; no arXiv paper.
- andyrdt/saes-qwen2.5-7b-instruct: HF repo exists, revision c37e53c4, tagged arXiv:2412.06410.

Existing keys whose title was re-checked: tulchinskii2024wise ("Listening to the Wise Few: Query-Key Alignment Unlocks Latent Correct Answers in Large Language Models"; now NeurIPS 2026). Its abstract describes select-and-copy heads that select options by QK alignment, pre-RoPE, in middle layers. This is our hop 2 and must be cited as such.

How each named work is positioned (text in `v5_related.tex`):
- **Elhage**: the framework whose two factors we measure.
- **Merullo 2024a (reuse)**: the same components serve tasks with different downstream use. Our sign result is an instance: the same readers, with polarity set downstream.
- **Merullo 2024b**: an inhibition subspace.
- **Dai et al.**: ordering IDs. A binding swap changes order, not identity, which is why it is the boundary of Claim 3.
- **Prakash 2024**: entity tracking by position.
- **Prakash 2026**: lookbacks, address via QK and payload via OV. This is the binding swap we test.
- **Gur-Arieh**: positional/lexical/reflexive mix that depends on context. It is the closest work; we add the cache-channel separation and show that later mentions open the key route.
- **Variengien & Winsor**: request-then-retrieve at the last token. In option formats, retrieval is instead done at the option rows (hop 1) and only selection happens at the answer.

---------------------------------------------------------------------
## (iv) Title and abstract

**Title (primary):** *Looked Up or Copied? The Prompt, Not the Token, Decides Which Attention Channel Carries an In-Context Value*. Running title: *Looked Up or Copied?*
- Fallback if A1/A3 fail: "...Option Lists Decide Which Attention Channel Carries an In-Context Value".
- The title rests only on Claims 1 and 2, so a Part C failure never changes it.

**Abstract (6 sentences, about 243 words; numbers: seven families, 72B, \riskyMet, \riskyTotal, so four):**

> A token that states a value in context, such as *shelf* in "the candle is moved to the shelf", can reach a later answer in two ways: a later query can match its attention key, or its attention value can be copied. Clamping one channel at a time, exactly, we show that the prompt, not the token, decides the route. When the candidates are mentioned again after the writing token, as answer options or even in a neutral sentence, the later mentions match its key and the value's identity is looked up; otherwise it is copied from the value, and the same mention placed before the token opens no lookup. This holds in seven model families up to 72B parameters, on fresh stories and on reading-comprehension passages with multi-token answers, where swapping only the answer span's cached keys changes multiple-choice answers and swapping only its values changes free-form ones. The match is made by duplicate-token heads, which write a low-rank flag that the answer then selects; small models make the match without using it, and the question sets its sign. Because the route depends on the prompt, so does the channel credited with an intervention: for steering vectors, sparse-autoencoder features and learned subspace edits that change the written value, the key share equals the unpatched model's own share in the same prompt and depth, while a binding swap departs from it; N of M risky preregistered predictions held, and we report every failure.

- "Seven families" means Qwen (2.5 and 3 counted as one, answering the generalist's point), Mistral, OLMo, Llama, Gemma, Phi and Falcon.
- If the Yi fallback replaces Falcon or Phi, the family count stays the same.
- The replacement clause for each failure case is in `v5_title_abstract.tex`.

---------------------------------------------------------------------
## (v) Making the intervention section (v4 §4, v5 §5) self-contained without Anonymous (2026)

1. **Re-base the claim.** §5 opens with published channel attributions as the motivation:
   - lookbacks: address via QK, payload via OV (Prakash et al., 2026);
   - keys as routers and values as payloads (Ma et al.);
   - key-side rebinding (Oh et al.);
   - refusal steering acting through OV (Cheng et al.).

   It does **not** open with "correcting" the anonymous paper. Its significance then no longer depends on how widely that paper is relied upon (generalist).
2. **Define everything in the paper.**
   - Edit E at (p, l).
   - The key-only and value-only exchange.
   - kappa = mean ID_K^E / (mean ID_K^E + mean ID_V^E), which is s_ID for an edit. This saves a symbol.
   - The matched-depth target, s_ID from block l+1.
   - The lemma in one sentence: "Because the prompts differ only at p, an edit confined to p at block l reaches the rest of the sequence only through p's keys and values from block l+1 on, so a patch that writes the source residual there is the natural clamp itself; only edits whose cached keys and values differ from the natural ones (nu ≥ 0.30) test the law" (proof in Appendix F).
3. **Primary evidence from rebuildable edits** (Part C, E1–E3; E4 is a DAS remap trained in-paper and labelled secondary). Everything runs on public models and public data, with an equivalence control on the diagonal and the binding swap as the discrepancy anchor.
4. **The released remap becomes one paragraph (about 120 words, §5.3) plus Appendix G.**
   - Describe it in our own words: "a rank-16 distributed alignment search subspace at the output of block 4 over the event span, trained so that the patched model answers a fixed permutation of the source location; the bases are public".
   - Every number about it is from **our** reproduction (argmax agreement 0.994–1.000), our exchanges, our refit and our 24B reader test.
   - Remove sentences that report what the anonymous paper *concluded*, such as "their component-level explanation of the tested effects". Replace them with what the released analysis does: "a key-only exchange at the writing token under options-after attributes part of the remap's effect, relative to a PCA control, to keys; we reproduce this".
   - Appendix G gives the full objective, rank, layer, span, seeds, generator, readout and reproduction table, so no claim requires reading the manuscript.
5. **Citation status at submission** (decide on the ICML deadline day):
   - (a) accepted at ICLR 2027 and de-anonymised: cite by author names in the third person, like any work;
   - (b) still under review: cite "Anonymous. Beyond the Fitted Answer.... Submitted to ICLR 2027", with the public OpenReview forum URL so reviewers can read it, after checking that the forum is publicly visible;
   - (c) withdrawn and not public: cite only the public release of the bases.

   Do **not** upload a copy of that manuscript to the supplement, since it would signal authorship. Rely on (b) and on self-containment.
6. **Anonymity.**
   - Third person throughout; never "our previous", "Paper 1" or "predecessor".
   - Part C's E4 is "trained with the released training code of Anonymous (2026)", stated once.
   - Run `scripts/make_anonymous_release.py` on the stage-8 tree, and extend its grep check to new files.
   - The BRIEF and the planning notes stay excluded.
   - New main-text check (code plan): a regex for `Paper 1|paper1|predecessor|our (own )?(prior|previous)`.

---------------------------------------------------------------------
## (vi) What moves to the appendix, and the page budget

### Presentation rules for v5 (objection 7), enforced by `paper/check_maintext.py`
- No prediction codes (A1–J*, gates) in the main text. Each result sentence ends with a status tag in words: "(preregistered, risky; met in 4/4 new families)", "(replication)", "(exploratory)", "(post hoc)".
- Symbols are limited to ID_K, ID_V, s_ID and kappa. R(k)/KO(k) become words ("recover", "remove"). s_K, phi, psi_K, psi_V, rho_K, r_K, r_ans, a3, t, g_K, f_K, D, E and F/E all go to the appendix.
- At most two numbers per paragraph and about 70 in the whole main text (v4: about 238). Every number is a numbers.tex macro, and no number appears more than twice.
- Each claim section follows the pattern: bold claim, evidence, one caveat sentence.
- **Table 3** (Discussion, about 0.15 page): evidence status per claim, with columns:
  - risky met/total;
  - replication or low-risk met/total;
  - exploratory;
  - not met (named in words).
- **Appendix A crosswalk**: main-text sentence → prediction code → entry → outcome → risk label. Auditability is kept.
- Risk-label rule, fixed in entry J **before any stage-8 output**:
  - RISKY means no data from the same model × format × measure cell had been inspected at commit (per the entry's "seen before" record), and the direction or threshold is not implied by a lemma or the causal mask. Everything else is LOW-RISK.
  - A–I are labelled by this rule in `data/prereg_ledger.csv`, with a one-line justification each. The paper states that these labels were assigned after A–I outcomes were known.
  - MET IN PART counts as not met in \riskyMet. Gates are not predictions.

### Page budget (about 930 words per full page; floats in pages)
| Block | Words | Floats | Pages |
|---|---|---|---|
| Title + abstract | 243 | – | 0.45 |
| 1 Introduction (P1 problem and definitions 110; P2 method 90; P3–P5 one per claim, 80 each; P6 novelty 276) | 716 | Fig 1 0.40 + Table 1 0.22 | 1.39 |
| 2 Setup (formats, clamp and exactness, ID_K/ID_V/s_ID, emitted-form scoring and flip rates, passages, models/stats/risk labels) | 800 | – | 0.86 |
| 3 Claim 1 | 800 | Fig 2 0.45 + Table 2 0.22 | 1.53 |
| 4 Claim 2 | 800 | Fig 3 0.40 | 1.26 |
| 5 Claim 3 | 700 | Fig 4 0.28 | 1.03 |
| 6 Related work | 470 | – | 0.51 |
| 7 Discussion (practice box, Table 3, limitations, conclusion) | 430 | Table 3 0.15 | 0.61 |
| **Total** | **about 4,960** | **2.12** | **7.64 + 0.36 slack = 8.0** |

v4 has 6,141 words. If the page count overflows, cut in this order:
1. Table 2 → dots in Fig 2a, with the table moved to the appendix (−0.22);
2. Fig 3c → appendix (−0.1);
3. Fig 4 inset → appendix;
4. the A8 sentence → appendix.

### Moves (v4 main text → v5)
| v4 item (approx. words) | Destination | Main-text residue |
|---|---|---|
| Abstract (290 w, about 25 numbers) | rewritten | 243 w, 4 numbers |
| Intro contributions list with codes (about 150 w) | deleted | Table 1 + 3 claim paragraphs |
| Intro "Where an intervention appears to act" (anonymous-centred, about 250 w) | §5 rewritten around independent edits | 80 w |
| Setup: formats paragraph (300 w) | App. B (exact strings already there) | 150 w |
| Setup: d_C, s_K (Eq. 1) | App. C | – |
| Setup: reader heads / HopSplice definitions (a3, R(k), KO(k), r_ans) | App. E | 40 w of words in §4 |
| Setup: the remap paragraph with Eqs. 3–6 (phi, psi; 330 w) | App. G | kappa equation in §5 |
| Setup: Prakash exchange paragraph (180 w) | App. H | 2 sentences in §5.3 |
| Fig 2 (frames of the released remap) | App. G | hollow markers in Fig 4 |
| Table 1 (seed-0 2x2) | App. D, relabelled "discovery sample" | replaced by fresh-sample Table 2 |
| Natural-format paragraphs (A1, B1, B2, C4; 250 w) | App. D tables | 2 sentences + Fig 2a |
| Tasks/IOI/scale paragraph (D1, D2, G18–G19, B3, E3, onset) | App. D (IOI factorial → App. E) | 1 sentence (tasks), 1 (onset), IOI → §4.3 |
| Row-restricted splice and knockouts (D3, G5, G6, G9, G10, G12; 200 w) | App. E | 1 sentence |
| "What a re-mention must be" (G13–G17; 150 w) | App. E | 1 clause in Limitations |
| Reader heads detail (single-head, LOO, layers; H3 vs G7; G11) | App. E, with the HA4 reconciliation | 2 sentences |
| Second hop with K/V split (G8a/b, H5 unscored) | App. E | 1 sentence |
| Duplicate-token account (G1–G3 numbers, H4 medians, E2) | App. E | 2 sentences + set-size caveat; E2 one clause in §3 |
| Released remap results (C3, E4, the post hoc r = 0.98) | App. G | 1 paragraph §5.3 |
| Readers at 24B (I1–I7) | App. G | 1 clause |
| Refit (F) | App. G | 1 sentence |
| Binding edit (H7–H11; 330 w) | App. H | boundary paragraph, 100 w |
| Failed predictions list (§4.3) | App. A / I | Table 3 + 1 sentence |
| KV-cache pitfall, eager/sdpa, environment rerun | App. C / J | – |

### Appendix order (reader-first)
- A. Evidence ledger: all entries A–J, predictions, risk labels, outcomes, and the crosswalk.
- B. Prompts and items: templated and SQuAD, including the type audit.
- C. Exactness and implementation: clamps, splices, lemma proofs, the KV pitfall, the BF16 floor, eager vs sdpa, FP32 anchors.
- D. Claim 1 details: all-model tables; the discovery-sample 2x2; scoring invariance (JB6); tasks; role swap; scale and onset.
- E. Claim 2 details: row splice and knockouts; forms; heads; ablation vs knockout; flag; small models; sign; IOI.
- F. Claim 3 details: edit construction, nu, JC tables.
- G. The released remap: full description, reproduction, frames, refit, 24B readers.
- H. The binding swap.
- I. Failed predictions and competing accounts.
- J. Compute, licences, model revisions and hashes.

### Caveats that must appear in the main text (stats reviewer), each in one sentence of Limitations
1. Scoring: emitted forms are primary; the lowercase numbers are invariant (JB6), or the change is stated.
2. Single-channel clamps are off-distribution; the K×V interaction is about 0 at onset 0.
3. At onset 0 the key read includes early token matching, and deeper onsets shrink it.
4. The readers' canonical core depends on set size.
5. Ablation and knockout disagree on templates; they are reconciled only on passages (HA4).
6. The binding swap flips no answer when candidates are named.
7. Values span one to four tokens; English only.


### code_plan
Nothing in the repository was changed; this is the plan for the v5 builder. The drafts are in /tmp/claude-0/-home-user-Causal-keys/7f2a5307-16d9-55d5-8797-c2900bcd44b4/scratchpad/s8design/partE/ (v5_title_abstract.tex, v5_novelty.tex, v5_related.tex, v5_skeleton.tex, v5_bib_additions.py, tex/t.tex compile test).

1. **paper/main.tex**
   - New title and running title.
   - \input order: sections/v5/{abstract, intro, setup, claim1, claim2, claim3, related, discussion}.tex.
   - The v4 section files move to sections/old_v4/. They are excluded from the anonymous release, like sections/old/.
   - Labels sec:claim1/2/3 and tab:qkov, as used by the drafts.

2. **paper/build_bib.py**
   - Merge ARXIV_ADD (22 keys), AUTHOR_FIX_ADD and NON_ARXIV (elhage2021framework, tii2024falcon3, andyrdt2025saes) from v5_bib_additions.py.
   - Run it, and check the printed titles against the table in the design.
   - Add a note to variengien2023leap ('ICLR 2025 as ...Universal Emergent Mechanism for Retrieval...') and to dai2024binding ('EMNLP 2024').
   - anonymous2026fitted: set the note per case (a)/(b)/(c) in (v).

3. **data/prereg_ledger.csv**, committed in entry J before any stage-8 GPU output.
   - Columns: code, entry, claim (1/2/3/none), risk (R/L), justification, outcome (MET / MET IN PART / NOT MET / NOT EVALUABLE / NOT RUN), main_text_sentence_id.
   - A–I outcomes are copied from docs/PREREGISTRATION.md. J rows are filled by the stage-8 scorers.

4. **analysis/stage8_status.py** (new).
   - Reads the ledger and the stage-8 verdict files.
   - Writes to paper/numbers.tex: \nFamilies, \riskyMet, \riskyTotal, \lowMet, \lowTotal and per-claim counts (\cOneRiskyMet ...).
   - Writes paper/tables/tab_status.tex (Table 3) and tables/tab_ledger.tex (the Appendix A crosswalk).
   - Unit test tests/test_stage8_status.py, on a synthetic ledger: MET IN PART counts as not met; gates are excluded.

5. **Figures**
   - paper/make_schematic.py: Fig 1 v5 with panels (a) lookup vs copy with hop 1 / flag / hop 2, (b) format strip with a templated and a passage example, (c) kappa follows s_ID.
   - paper/make_figures.py: fig_claim1.pdf (Fig 2a–c from results/gpu_stage8/B and A), fig_claim2.pdf (Fig 3a–d from v4 heads results + results/gpu_stage8/A heads + D), fig_claim3.pdf (Fig 4 from results/gpu_stage8/C + v4 stage-2/4/6 frames).
   - The v4 figures fig_frames and fig_readers(a, c) move to the appendix.
   - Table 2 tables/tab_fresh2x2.tex is generated by the Part B scorer.

6. **paper/check_maintext.py + tests/test_paper_maintext.py** (new).
   - Extract the text from \section{Introduction} to the start of the Impact Statement, with tables and figures expanded.
   - Assert no code regex `(?<![\w\\])(?:[A-J][0-9]{1,2}[a-c]?|[A-J]-G[0-9]|Gate~?\s?[a-z]?[0-9])(?![\w])`.
   - Distinct math symbols must be a subset of {ID_K, ID_V, s_ID, kappa, nu}.
   - At most 2 numerals per paragraph and at most 70 in total. Allowed literals: model sizes like '7B', and '1.5B/3B' in Table 1.
   - Every decimal must come from a macro.
   - Every \cite key exists in references.bib.
   - Anonymity regex `Paper 1|paper1|predecessor|[Oo]ur (own )?(prior|previous|earlier) (work|paper)`.
   - Page limit: compile with tectonic, run pdftotext per page, and assert 'Impact Statement' starts on page ≤ 9 and the main text ends by page 8.
   - Run it in CI alongside the existing tests.

7. **scripts/make_anonymous_release.py**
   - Add the stage-8 files and data/prereg_ledger.csv.
   - Extend the final grep to the new paper/sections/v5.
   - Exclude paper/BRIEF*.md and the s8 design notes.

8. **paper/BRIEF_v3.md** (internal, excluded from the release): the presentation rules, the contingency table and the citation-status decision from the design.

Exactness tests are not applicable (no new intervention code); the tests are the main-text checker and the status-scorer unit test.

### pilot_results
No GPU runs and no model runs. I ran the following on CPU and the network.

1. Literature verification (2026-10-10). The arXiv API returned HTTP 503 from this container, so I fetched arxiv.org/abs/<id> pages with curl and read their citation_title, citation_author, citation_date and Comments fields.
   - All six panel-named works are verified: 2510.06182, 2409.05448, 2402.14811, 2312.10091, 2310.08744, and Elhage et al. 2021 on transformer-circuits.pub, which has no arXiv id.
   - 18 further citations are verified the same way (listed in the design).
   - Venues: ICLR 2026 (Gur-Arieh), ICLR 2024 (Prakash 2024, Merullo 2024a), NeurIPS 2024 (Talking Heads), ICLR 2025 (Variengien & Winsor, under a revised title, confirmed on iclr.cc), ICLR 2025 (Springer), ICML 2024 (KIVI), EMNLP 2023/2021/2020 (Geva, Longpre, Kobayashi).
   - Dai et al.'s EMNLP 2024 venue comes only from search results citing the ACL Anthology author page; I did not open the Anthology entry itself.
   - Falcon3 and the andyrdt SAE release have no paper. Both HF repos exist (SAE revision c37e53c4, tagged arXiv:2412.06410).
   - Abstracts of the 12 most-cited new works were read, so every description in v5_related.tex is supported by the cited abstract. One correction from this check: tulchinskii2024wise's abstract describes QK-alignment select-and-copy heads at option rows, which is our hop 2, so it is positioned that way.
2. Presentation baseline, measured on v4's main-text sources (abstract through discussion):
   - 111 prediction-code mentions (61 distinct codes);
   - about 238 decimals or number macros;
   - about 6,141 words.
   These are the before-values for the v5 checker targets (0 codes, about 70 numbers, about 4,960 words).
3. Compile test with tectonic and icml2026.sty: the v5 title, abstract and Table 1 compile. The abstract fills about 0.42 page of column 1 on page 1. Table 1 takes about 0.22 page in one column, after the column widths were adjusted. One overfull box of 6 pt remains, fixable by shortening one cell.
4. Read only, from Part D's CPU logs (0.5B/1.5B, FP32; Part D must disclose these as its own pilots):
   - At 1.5B, ID_K is +2.1 when the question asks for a mentioned option and −4.2 when it asks for an unmentioned one.
   - At 0.5B, the IOI-inline negative read (−2.75 nats) is reproduced by belief-task readers alone (−2.12) and removed when they are knocked out (−0.07).
   - The flag 'move' injection reaches iota about 0.7–0.8, against random directions at about 0.
   These make Table 1 rows 5–6 plausible, but they are not evidence for the paper.

### paper_payoff
**What the rewrite adds to the paper**
- A title and abstract stating one claim per sentence, with four numbers.
- An Introduction that ends with the novelty paragraph and Table 1, which separates what QK/OV and the mask imply from what was measured.
- Three claim sections, each with one figure: route, mechanism, attribution.
- A Discussion with four practitioner rules and Table 3 (evidence status by risk class).
- A Related Work section placing the work against Elhage, Gur-Arieh, Dai, both Prakash papers, Variengien & Winsor, Merullo (twice), Tulchinskii and copy suppression.
- A §5 that stands without the anonymous paper.

Key sentences:
- "None of this says which channel a model uses, because the value route is open in every prompt."
- "We claim no new head type; we show which channel is used, when, and what follows for interventions."

**Expected score movement, assuming Parts A–D land**
- Presentation, all four reviewers: 2 → 3. Each listed 'restructure around 3 claims, move codes to the appendix' as +0.5.
- Contribution:
  - mechinterp reviewer and generalist: 2 → 3, because objection 1 is answered by conceding what is implied and isolating four non-implied, preregistered regularities, including one counterexample (the binding swap) showing the attribution law is not definitional;
  - sceptic: 2 → 3, through the self-contained §5 and the 'lookup' definition that adopts their own mechanistic reading.
- Objection 8 disappears from the mechinterp reviewer's and generalist's lists.
- Stats reviewer: soundness 3 → 4, from the risk-separated met rate in the abstract, Table 3, and the caveats in the main text.

Parts A–D provide the evidence (each is worth about +1 per reviewer in their own payoff). Part E makes that evidence legible and keeps the novelty claim calibrated. Together the target is roughly 6/6/6/7, which is a clear accept. A strong accept needs C and D to come out MET, so that Table 1 rows 5–7 survive.

### risks
1. **The novelty argument depends on Parts C and D.** Table 1 rows 5–7 and two of the four regularities are conditional. If D(i), D(ii) and C fail, the argument rests on:
   - the route being a property of the prompt (value route unused under options-after);
   - G2's insufficiency;
   - behaviour and passages.

   Contribution then likely stays at about 3 rather than 3–4. The contingency table pre-writes the cuts so that no overclaim survives a failure.
2. **Density.** Three claims, four figures and three tables in 8 pages leave 0.36 page of slack. The cut order is fixed: Table 2 → Fig 2a; Fig 3c; the Fig 4 inset; A8. The checker enforces the page limit.
3. **Retro-labelling.** Assigning risk labels to A–I after their outcomes are known can look self-serving. Mitigations: the rule is mechanical and committed before stage 8; each label has a one-line justification; the paper reports A–I and J met rates separately and says the A–I labels are post hoc.
4. **The abstract's 'even in a neutral sentence' and 'seven families'** depend on JB3/A5 and on the B model roster. Their replacements are pre-written, and \nFamilies is computed by the scorer, not typed.
5. **Anonymous (2026) status at the deadline is unknown.** Case (b) needs the ICLR OpenReview forum to be public; check it on the day. Uploading the manuscript must be avoided, since it would signal authorship. E4 (DAS trained with the released code) remains a single third-person mention.
6. **Citation accuracy.**
   - Dai et al.'s venue is verified only through search results citing the ACL Anthology author page.
   - Prakash 2024's head names are not in its abstract, so they are not used.
   - Variengien & Winsor's ICLR title differs from the arXiv title, so add the note.
   - The tulchinskii2024wise abstract changed across arXiv versions; describe it only by what the current abstract says.
7. **'Lookup' terminology.** Adopting the sceptic's definition (the identity comes from the later token, and the key is the match target) could read as a retreat. It is stated as the definition from the first paragraph on, so ID_K is never over-read.
8. **Removing codes reduces in-text auditability.** This is offset by the Appendix A crosswalk, which maps every main-text sentence to its code, entry and outcome.

