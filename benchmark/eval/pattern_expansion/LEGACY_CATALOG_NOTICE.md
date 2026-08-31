# Legacy catalog artifact

This directory records the earlier P-numbered pattern-expansion experiment.
It is retained only for provenance and is not a semantic catalog consumed by
the paper, miner, DSL registry, or deployment code.

The canonical source is:

- `benchmark/eval/template_induction/frozen_template_catalog.json`
- SHA-256:
  `cfa30e182a4aa5c6637423dcbd95f15cb20c2b34a1da80000f5df6c6b8176734`
- implementation index: `trainaudit/trainaudit/catalog.py`

Do not use `pattern_catalog_v2.json` or its legacy identifiers in new
experiments. New annotations and grounded rules must use the canonical
`T01`--`T35` identifiers and names.
