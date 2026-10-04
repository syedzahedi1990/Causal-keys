# Paper versions

- **v1** (`paper2_v1.pdf`): the submission draft. Main results from GPU stages 1, 2, 3 and 3b. Source: commit `14f6f4e` plus the
  accuracy fixes in the commit "Paper v1: position against Prakash et al. ..." (lookback vocabulary mapped, role-swap claim
  narrowed, ICLR 2026 citation). v1 has no stage-4 content.
- **v2** (`paper2_v2.pdf`, built from `paper/` on this branch; `paper/main.pdf` is the same file): v1 plus GPU stage 4,
  preregistration F. The remap of the prior work was refit under the no-mention format with its authors' training
  code, with a same-code control, at Mistral-Small-24B. All gates and predictions F1-F4 were met. Post hoc, the
  fitting format shifts the key share in the two mixed formats (options-after, sentence-after). New: Sec. 4.1, the
  refit points in Fig. 2c, Table 5 (preregistration F), App. D and Table 7, and scope updates in the abstract,
  intro, limitations and conclusion. To fit 8 pages, the failed-predictions subsection is shorter (details in
  App. A).
