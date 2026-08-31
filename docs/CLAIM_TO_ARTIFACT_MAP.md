# Reproducible result map

This index maps the principal released results to their canonical records and
validation commands. Paths are relative to the repository root.

| Result | Canonical record | Validation command |
|---|---|---|
| 392-record upstream corpus | `benchmark/eval/silent_evidence_392.json`, `benchmark/eval/v2_full/category_392_v2.csv` | `python benchmark/eval/verify_corpus_construction.py --check` |
| taxonomy and 50-record annotation study | `benchmark/eval/irr_50_*`, `benchmark/eval/irr_50_annotations/` | `python benchmark/eval/verify_irr.py --check` and `python benchmark/eval/verify_taxonomy_table.py --check` |
| frozen 35-template catalog | `core/config/frozen_template_catalog.json`, `core/config/frozen_template_catalog.sha256` | `(cd core/config && sha256sum -c frozen_template_catalog.sha256)` |
| frozen-catalog held-out coverage | `benchmark/eval/catalog_generalization/` | `python benchmark/eval/verify_catalog_generalization.py --check` |
| Real-SE 17/18 detection outcomes | `benchmark/eval/real_sdc/real_se_detection.csv`, `benchmark/eval/real_sdc/real_sdc_manifest.json` | `python benchmark/eval/real_sdc/extract_detection_csv.py` |
| fixed-side 0/17 paired outcomes | `benchmark/eval/real_sdc/real_se_replay_outcomes.csv` | `python benchmark/eval/real_sdc/extract_replay_outcomes.py` |
| per-case appendix table | `benchmark/eval/real_sdc/real_se_detection.csv` | `python benchmark/eval/verify_appendix_detection_table.py --check` |
| matched Catalog/free-form ablation | `benchmark/eval/paper_v2/catalog_direct_ablation.csv`, `catalog_endpoint_pairing.csv` | `python benchmark/eval/paper_v2/verify_catalog_direct_ablation.py` |
| collector full-snapshot timing | `benchmark/injection/overhead_raw/`, `benchmark/injection/overhead_h20.csv` | `python benchmark/injection/parse_overhead_logs.py --check` |
| schema coverage | `benchmark/eval/paper_v2/schema_coverage_392.csv`, `benchmark/eval/extension_v3/trace_tier_392.json` | `python benchmark/eval/verify_tier_coverage_axis.py --check` |
| compiled SQL examples | `core/config/generated_sql.json` | `python core/validate_generated_sql.py --db <trace.db>` |

Run the stable offline set with:

```bash
bash scripts/check_release.sh
```

Large trace databases are checksummed release assets; see
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).
