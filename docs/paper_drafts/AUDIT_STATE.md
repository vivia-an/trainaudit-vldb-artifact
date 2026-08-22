
### 2026-08-23 — iteration 29: the diagnosis study's numbers are not in the artifact
Ran the shipped scorer, `diagnosis_s3_score.py`, which announced "Rater A not filled yet".
Following that up:

- `diagnosis_s3_rater_A.csv` — 17 rows, **every rating column empty**. So the two-rater design
  produced one rater, and there is no agreement measure for this study.
- `diagnosis_s3_rater_B.csv` — filled: **L1 yes on 17/17, L2 `N/A` on all 17**.
- `diagnosis_s3_input.json` — 17 cases, and **no field records a candidate-set size**;
  `rca_chain` is `[]` for every case. Its `meta.source` is `paper_table_baseline_3way.md`,
  which scores the superseded intermediate case set (O14).

`app:diagnosis` reports 9/17 (53%) at |S_rule|=1, median C1 = 6 with peak 12, 5 of 8 at C1 and
2 at C2, and draws 1→6→1 in `fig:diagnosis-mechanism`. **None of those cardinalities is
derivable from the shipped files.**

Rater B's 17/17 is worth noting: it is a *stronger* result than 53%, but it answers "does the
rule name identify the fault" rather than "is the candidate set of size 1". Substituting one
for the other would be wrong.

**O43**, check group 16. This is the same shape as O21 — an appendix quantification whose
source data is absent — and it is the second one found by running a shipped script rather than
reading it.
