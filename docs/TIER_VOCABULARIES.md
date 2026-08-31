# Tier and template vocabularies

TrainAudit uses several compact identifiers for different parts of the system.
They are intentionally separate and should not be added together unless a result
explicitly defines such an aggregation.

| Vocabulary | Defined in | Meaning |
|---|---|---|
| Schema tiers `S0`–`S6` | `paper/main.tex` | cumulative trace-field coverage used by the schema study |
| Integration tiers `T0`–`T4` | `core/trainaudit_pkg/trainaudit/tiers.py` | amount of framework knowledge available to a runtime rule |
| Catalog templates `T01`–`T35` | `core/config/frozen_template_catalog.json` | semantic relation templates produced by catalog induction |
| Predicate shapes | `core/trainaudit_pkg/trainaudit/dsl/predicate.py` | compilation form used by the DSL-to-SQL path |
| Predicate compositions | `benchmark/eval/v2_full/annotations_392_v2.json` | required combination of schema, topology, and precondition guards |
| Miner stages `S1`–`S5` | `core/agents/fsm_stages.py` | gap, evidence, synthesis, persistence, and reporting states |

## Common distinctions

The paper's schema tiers and the miner's FSM stages both use an `S` prefix but
measure unrelated objects. Likewise, `T0` and `T1` denote integration tiers,
whereas zero-padded identifiers such as `T01` and `T35` denote catalog templates.

The deployment inventory contains 32 distinct executable rule identifiers:
17 at integration tier T0 and 15 at T1. Thirteen of those rules also have a YAML
DSL representation. Consequently, the reported 45 deployment entries count
executable representations, while the unique semantic rule inventory contains
32 identifiers.

The frozen catalog contains 35 templates. Its eight recorded induction batches
grow the 27-template seed to exactly that final identifier set. Legacy `P`-numbered
pattern files under `benchmark/eval/pattern_expansion/` are retained only as source
records and are not part of the canonical T01–T35 catalog.
