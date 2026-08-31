# Frozen-catalog generalization records

This directory contains the recorded held-out evaluation for the frozen
constraint catalog and two free-form comparison arms. The temporal split is
defined by `freeze_bugs.json` and `heldout_bugs.json`; the arm-level records are
stored in `cov_A.jsonl`, `cov_B1.jsonl`, and `cov_B2.jsonl`.

The released aggregate is:

| Arm | Catalog size | Covered held-out cases |
|---|---:|---:|
| Frozen catalog | 35 | 207/249 (83.1%) |
| Free-form frozen | 15 | 119/249 (47.8%) |
| Free-form per-case proposal | per case | 138/249 (55.4%) |

The complete aggregate and catalog-size curve are in
`generalization_summary.csv`. Recompute them from the per-case records with:

```bash
python benchmark/eval/verify_catalog_generalization.py --check
```

The command also verifies the frozen/held-out split against the released corpus
manifest. It does not call an external model or rewrite any record.
