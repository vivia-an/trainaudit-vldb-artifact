# Baseline logs — what backs `tab:db-baselines`

The clean-trace false-positive row of `tab:db-baselines` (§5.2) is **measured**, not
derived. The runs are here; the filenames do not say which table they feed, so:

| Paper cell | Value | File | Recorded value |
|---|---|---|---|
| Manual SQL, clean FP/1M | 4.8×10⁵ | `manual_sql_baseline.json` | `clean_crossrank_fp_per_million` = 484,964 over 45,956 cross-rank evaluations on 4 clean databases |
| Daikon-style, clean FP/1M | 1.3×10⁵ | `daikon_style_baseline_loo.json` | `clean_fp_per_million` = 132,904 over 873,775 evaluations, leave-one-configuration-out |
| TrainAudit (verified), clean FP/1M | 25.8 | — | **not in the artifact**; see `../../docs/GAP_AUDIT.md` O3 |

Two denominators worth knowing before quoting these:

- Manual SQL's 4.8×10⁵ is the **cross-rank subset**. Across all 1,511,156 evaluations the
  same run gives 14,748/1M. The cross-rank figure is the right comparison — it is the
  subset where a topology guard would apply — but the two differ by 33×, so the basis
  should be stated wherever the number appears.
- Daikon-style's 1.3×10⁵ is the **leave-one-configuration-out** protocol, matching the
  paper's description. `daikon_style_baseline.json` is the simpler single-training-database
  variant and gives 446,173/1M; it is kept for provenance and is not the reported number.

The other claim in the same table — Manual SQL covering ≤3 of 13 classes and Daikon-style
≤5 — is an analytical expressivity bound, as §5.2 says. No harness scores it.

```bash
python3 ../../scripts/verify_paper_numbers.py     # recomputes both measured cells
```
