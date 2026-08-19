# Anonymous artifact (double-blind review)

This repository is the **anonymous code artifact** accompanying a under-review paper.
It implements the paper's verified mining pipeline (Pattern Catalog, FSM, Accept gate,
healthy-run, funnel reproduce) without revealing author identity.

## Withheld until camera-ready

- Author names and affiliations
- Canonical public repository URL (see `CITATION.cff`)
- Private contact emails / institutional paths

## Safe to cite in a blind review report

- System name **TrainAudit** (matches the anonymous manuscript)
- Frozen catalog hash, funnel counts, and `SDC_PAPER_ALIGN=1` paper path
- Offline checks: `python3 run_smoke.py`, `python3 scripts/reproduce_funnel_counts.py`

## Hosting note

Do **not** link this artifact to identifiable GitHub accounts or organization names in the
submission PDF until the review period ends. Prefer an anonymous artifact host or a
review-only private repo without profile linkage.
