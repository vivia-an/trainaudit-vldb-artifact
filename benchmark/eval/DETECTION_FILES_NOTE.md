# Which detection file backs the paper — read this first

Three generations of Real-SE detection data ship in this artifact. They score
**different case sets**, so their totals differ and only one matches the paper.

| File | Cases | Totals | Matches the paper? |
|---|---|---|---|
| `real_sdc/SMOKE_REPORT.md` → `real_sdc/real_se_detection.csv` | 17 (current set) | TrainAudit 17/17, TrainCheck 5/17 (+1 tool failure), Naïve 0/17, fixed-side FP 0/17 | **yes** — `numbers.tex` and appendix `tab:detection-results` are computed from this |
| `results.csv` | 17 (intermediate set) | TrainAudit 17/17, and one fixed-side alert — see below | no |
| `results_gpu.csv` | 14 (earlier H200 set) | 13/14 buggy detected, 12 clean fixed sides | no |

Overlap between the current set and `results.csv` is only **9 of 17** cases:
`results.csv` still carries `B2, B13, M-012, M-014, M-024, M-NEW-5, O-NEW-1, OC-NEW-3`,
while the current set adds `D-029, D-NEW-9, M-010, M-NEW-33, O-040, O-NEW-3, O-NEW-5,
OC-NEW-22`. `results_gpu.csv` shares only 6.

`real_se_detection.csv` is generated, not typed:

```bash
python3 real_sdc/extract_detection_csv.py
```

## The `OC-NEW-3` row, and why it is not a 1/17 false positive

`results.csv` lines 34–35 both carry `phase=buggy`, but the second row's message is
prefixed `[fixed]` and records `T1-sqrt-decay-front-loaded` firing with
`|slope|=0.5` on the fixed side. Read literally — the `phase` column is simply wrong on
that row — it is a fixed-side alert, i.e. a false positive, which is where the
long-standing "0/17 vs 1/17" discrepancy in `../../docs/experiment_registry.md` comes from.

It does not apply to the reported result: **`OC-NEW-3` is not in the current case set.**
It was dropped when Real-SE was re-frozen, and the current set's fixed sides are
0/17 (16 clean replays plus one case whose upstream fix rejects the faulty configuration
by assertion, footnoted in §5.2 of the paper).

Anyone re-deriving the numbers should use `real_se_detection.csv` and treat
`results.csv` / `results_gpu.csv` as historical. The same applies to the tables derived
from them: `paper_table_baseline_3way.md`, `paper_table_baseline_traincheck.md`
(which reports TrainCheck 10/17 on the intermediate set), `paper_table_gpu.md`.

## One inconsistency inside `SMOKE_REPORT.md` itself

Its "Why TrainCheck misses so many" section says "B1 / B3 missed despite TrainAudit's
`replica-cksum-equal` firing", but its own per-case table — and the paper — record
TrainCheck as **detecting** B1 (1/316). The table is the one the numbers come from; the
prose sentence should say B3 only.
