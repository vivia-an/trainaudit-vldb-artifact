# 392-record corpus provenance audit

This audit records what the released artifacts support about construction of
the 392-record evidence corpus. It is intentionally narrower than the claims that
appeared in earlier paper drafts.

## Reconstructable construction

`manifest_v2.json` was assembled on 2026-05-10 by merging:

- an earlier 128-record issue/PR-curated pool;
- a 295-record pool with richer commit and reproduction metadata;
- one additional fully specified Megatron-LM record.

The first two pools share 32 incidents, leaving 96 records unique to the
128-record pool. The final arithmetic is therefore
`96 unique + 295 + 1 orphan = 392`. A second incomplete orphan outside both
source-pool counts was considered and excluded before assembly. The resulting
framework counts are Megatron-LM 110, DeepSpeed 123, OLMo 77, and OLMo-core
82.

## Source evidence retained in the manifest

| Retained provenance | Cases | Share |
|---|---:|---:|
| Has a fixed-commit identifier | 289 | 73.7% |
| No fixed commit, but has an issue/PR/release URL | 96 | 24.5% |
| Neither field is machine-resolvable in the merged manifest | 7 | 1.8% |

Of the 289 fixed-commit identifiers, 288 resolve in the released local Git
objects. Their commit dates range from 2021-08-06 through 2026-05-06. This is
the date span of retained, locally resolvable fixes; it is not evidence of a
predefined crawl window.

## What the artifacts do not establish

The repository does not currently contain a raw candidate ledger that records:

- the exact start and end dates of a uniform commit crawl;
- the SHA of each repository's default branch at the crawl boundary;
- the number of commits inspected before screening;
- per-candidate exclusion reasons before the final 392 records;
- a commit or issue URL for every retained record.

Consequently, the current 392-record corpus must not be described as an exhaustive
scan of every default-branch commit in a fixed interval. Establishing that
claim requires rebuilding the corpus from pinned repository snapshots and
rerunning the template-induction split.

## Why issue text is not sufficient for detector evaluation

Issue and PR discussions are useful discovery and explanatory evidence, but
they often omit the exact buggy revision, trigger configuration, executable
oracle, or fixed comparison. They may also combine multiple defects or report
only downstream symptoms. The Real-SE detector benchmark therefore requires a
buggy/fixed commit pair and a runnable driver even when an issue URL exists.
Issue-only records may support survey and catalog induction, but they do not
enter the detector-evaluation denominator.

## Real-SE selection

Real-SE is a coverage-constrained executable subset, not a random sample of the
392 records. A detector-coverable case must have:

1. auditable upstream provenance;
2. a semantic error that can continue without an immediate runtime failure;
3. an explicit buggy/fixed comparison;
4. a runnable method-level driver that triggers the violation;
5. archived command, environment, seed/GPU count, logs, and verdict;
6. no synthetic or pattern-only surrogate in place of the upstream bug.

The frozen manifest contains 17 such cases and one real observability-boundary
case, O-003. It covers all 13 taxonomy classes. O-022 is retained only as
survey/catalog evidence because it lacks a reproducible commit pair.
