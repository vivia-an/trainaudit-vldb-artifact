# Baseline measurement records

This directory contains the recorded clean-trace measurements for the database
baseline comparison.

| Baseline | Record | Evaluation protocol |
|---|---|---|
| Manual SQL | `manual_sql_baseline.json` | cross-rank evaluations on four clean databases |
| Daikon-style invariants | `daikon_style_baseline_loo.json` | leave-one-configuration-out |

The two records use the evaluation domain appropriate to each baseline. The
JSON files retain their exact denominators and counts so that rates can be
recomputed without relying on rounded values in the manuscript.
