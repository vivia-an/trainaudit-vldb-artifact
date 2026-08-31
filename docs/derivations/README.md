# Derivation drafts

Working drafts that record **how a stated number was arrived at**, for claims the paper
presents as analytical rather than measured. They are the only written derivation for
several of them, so they are shipped here rather than left in the workspace.

Read them with three caveats:

1. **They are drafts, not paper text.** Section numbering, labels and prose differ from
   `paper/main.tex`; some carry internal notes at the top in Chinese.
2. **Some numbers predate the current Real-SE freeze.** `draft_S6_baselines.tex` quotes
   detection as 17/19, for instance; the current denominator is 17/18
   (`paper/numbers.tex`). Where a draft and the paper disagree, the paper and
   `numbers.tex` are authoritative.
3. **They say plainly which parts were not executed.** That matches the paper — §5.2
   states the class-coverage figures as an expressivity bound — and matches
   `../GAP_AUDIT.md`.

| File | Derives |
|---|---|
| `draft_S6_baselines.tex` | `tab:db-baselines`: why Manual SQL can express at most 3 of the 13 fault classes and Daikon-style mining at most 5, enumerated class by class, and why both inherit the −π_topo false-positive cost. **This is the derivation behind O3.** |
| `draft_RQ3_verified_mining_ablation.tex` | the four-arm guard ablation table (342 → 429 / 551 / 598) and its per-pattern breakdown |
| `draft_S63_overhead.tex` | the §5.5 overhead section, including the amortisation reasoning behind `fig:amortization` |
| `draft_S2_trace_relation_schema.tex` | the trace-relation schema of §2 — logical relations as views over `coredump(step, stage, data)` |
| `draft_S5_constraint_catalog.tex`, `draft_S5_verified_constraint_mining.tex` | the Pattern Catalog and the verified-mining pipeline sections |
| `draft_case_studies_violation_tables.tex` | the production case studies and their violation tables |
| `draft_related_work_sigmod.tex`, `draft_new_bib.bib` | related-work framing and the citations it adds |

Not shipped, deliberately: `EXPERIMENT_QUEUE.md`, `INTEGRATION_MAP.md` and
`sigmod_writing_constraints.md` from the same workspace directory. Those are about how to
write and sequence the paper, not about how a result was derived.
