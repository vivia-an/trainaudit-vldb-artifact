# IMPL_NOTES_S5 — Pattern Catalog Ablation (RQ-D5)

Executed 2026-07-17 on cluster `beijing`. Spec: `SPEC_S5_catalog_ablation.md`.

**Bottom line: SPEC §3 red line triggered.** yield_A/yield_B = 0.95 (A ≤ B).
The two arms produce an identical deployed rule set (Jaccard = 1.000). Results
and the recommended contribution-3 rewrite are in
`catalog_ablation/paper_table_catalog.md`. Step 4 (Real-SE detection) is
**blocked**, so only one of the two pre-registered headline metrics was obtained.

**Robustness check (arm A+).** A re-examination found the harness was discarding
4.6% of A-arm proposals for a reason that was the *harness's* fault, not the
catalog's (§6). Arm A+ repairs it and gives the catalog every benefit of the
doubt. The conclusion survives and strengthens: A+ yield = **0.0659** (0.91× of
B, i.e. worse than plain A), with **the same 11 deployed predicates**. Three
configurations — A, A+, B — deploy byte-identical rule sets.

---

## 1. Configuration (identical across arms except where noted)

| Setting | Value |
|---|---|
| L1 model | `deepseek-v4-flash` (pinned explicitly; the `deepseek-chat` alias resolves to it) |
| temperature | 1.0 |
| max_tokens | 32768 |
| replicates | 5 per arm |
| source chunks | 112 seed-file entries/replicate: megatron 30, deepspeed 31, olmo 24, olmo_core 27 |
| chunk truncation | first 8000 chars (same as `run_funnel.py:86`) |
| L2 / L3 / L4 | byte-identical across arms — unmodified repo code |
| L4 client | `PatternGuidedFilterLLM(deployed_rule_count=32)` — same simulator the published funnel used |
| **arm difference** | **`use_catalog` only** → selects the L1 system prompt |
| total L1 calls | 1120, **0 errors**, 2,545,982 prompt + 1,948,748 completion tokens |

Arms share a single `DeepSeekClient` instance, so model/temperature/max_tokens
are provably identical. The API key is read from `$DEEPSEEK_API_KEY`, never
written to disk.

### Deviations from SPEC §4.3 / §5 (and why)

1. **Source chunks.** SPEC §5 names `benchmark/eval/v2_full/v2_batch[1-4]_input.json`
   as the L1 input. Those files are the **bug-annotation corpus** (fields
   `bug_id` / `title` / `root_cause`), not source code. The real L1 input for the
   published 420 is `rebuttal_v1/A1_mining_funnel/seed_files.json` → files under
   `exp/frameworks/*`. We used the latter, so the arms stay comparable to the
   published funnel.
2. **L3 clean trace.** SPEC §4.3 names `benchmark/sweep/_runs/dense_190M`. It
   **does not exist** in this repo (never committed; not gitignored) and has no
   substitute. We used the per-framework clean traces `run_funnel.py:28-33`
   actually reads (`hunt_log/novel_hunt/{megatron_clean, deepspeed_bf16_only,
   olmo_core_baseline}`). **Both arms use the identical trace per framework**, so
   internal validity is unaffected; only comparability to the published
   skip-L3 numbers is reduced.
3. **Cost.** SPEC §9 estimated 40 L1 calls (assuming 4 chunks). The real corpus is
   112 files → 1120 calls. Still small; ran in ~20 min at 32-way concurrency.
4. **Step 4 not run.** See §5.

---

## 2. Code changes made to run this

### 2.1 `trainaudit/mining/layer1_hypothesis.py`

- Added `use_catalog: bool = True` to `propose_hypotheses()`; `False` selects
  `_SYSTEM_PROMPT_FREEFORM`.
- `_parse_hypothesis_response(text, *, use_catalog=True)`: the B arm forces
  `catalog_template_id=None` rather than SPEC §4.1's suggested
  `h.get("catalog_template_id")`. **This is stricter than the spec on purpose**:
  a free-form LLM that invents an id like `"T99"` would pass `.get()` and then
  die inside `Hypothesis.__post_init__` → `get_catalog_template()`, swallowed by
  `except Exception: continue` — SPEC §9's "most likely failure", silently
  zeroing arm B. Forcing `None` closes it.
- **Fixed a real parser bug** (see §3). Not an ablation knob — it is symmetric
  across arms and was required for *any* real LLM to work.

Regression: `trainaudit/tests/mining/` → **17 passed**.

### 2.2 Restored `trainaudit/trainaudit/verifier.py`

The whole `trainaudit` package was un-importable: `__init__.py:26` imports
`.verifier`, whose **source file was missing from disk and from `main`** (only
`__pycache__/verifier.cpython-311.pyc` survived). `run_funnel.py` — the script
that produced the paper's funnel — therefore did not run either.

Recovered from commit `6bfe98d` on the unmerged branch `ablation-d3-v2`
(2967 bytes, 2026-05-19). **This is a reconstruction, not the original**: the
pyc header shows the lost file was 2287 bytes dated 2026-05-05, so the two
versions differ. It is safe for this experiment — `verifier.py` only provides
`run_rules`/`summarize` (the T0/T1 rule executor) and is not on the mining
L1–L4 path; it was merely dragged in by the `__init__` import chain. **It would
matter for Step 4**, which is blocked anyway. Flagging it as a repo-integrity
issue independent of S5.

---

## 3. The parser bug that made "real LLM" impossible (root cause worth reading)

Both arms initially returned **zero** hypotheses despite the LLM emitting
perfectly-formed JSON. Two stacked causes:

1. **`max_tokens` starvation.** `deepseek-v4-flash` is a reasoning model: its
   `reasoning_content` is billed against `max_tokens` before any `content` is
   emitted. At the default 1024 the thinking pass consumed the entire budget and
   `content` came back **empty string**. Raised to 32768.
2. **`_parse_hypothesis_response` could not read bare JSON.** Its two probes were
   (a) a ```` ```json ```` fence, and (b) `text.rfind("{")` — which on nested
   bare JSON lands on the **last inner hypothesis object** and always fails to
   parse. A real LLM replies with bare JSON, so parsing always returned `[]`.

Cause 2 is confirmed by the simulator's own comment, `pattern_guided_llm.py:193-195`:

> *"Wrap in ```json fences so `layer1_hypothesis._parse_hypothesis_response`
> picks it up reliably (its fallback 'find last {' parser fails on nested JSON)."*

**The simulator worked around this parser defect by fence-wrapping its own
output, which is why the defect never surfaced in the published funnel.** Any
real LLM hits it immediately. Fix: try the whole payload and the outermost
`{...}` span before the legacy `rfind` probe. Symmetric across arms.

---

## 4. The B-arm prompt, in full (for reviewer strawman-judgement)

Per SPEC §4.2 the B prompt is the A prompt with **only** the template table and
the "Do not invent a new template id" instruction removed, and
`catalog_template_id` dropped from the required fields. Task statement, output
schema, effort budget ("2–4"), and source chunk are identical.

```
You are an LLM training framework auditor. Given a snippet of framework source
code, propose 2–4 invariants that, if violated at runtime, would indicate a
silent error in this code. Each Hypothesis must specify:
  - relation_type ∈ {cross_rank_equal, tensor_stat_bound, payload_field_compare,
    cross_step_monotonic, structural_presence, conditional_check}
  - entities: list of trace entities the invariant references (e.g. ['param', 'grad_norm'])
  - dimensions: ['rank', 'step'] etc.
  - rationale: one sentence on why this invariant matters

Return JSON: {"hypotheses": [...]}
```

The A-arm prompt is the unmodified repo `_SYSTEM_PROMPT`: same text, plus
`catalog_template_id: one id from the catalog below`, the instruction *"select
2–4 applicable entries from the frozen Pattern Catalog… Do not invent a new
template id or relation family"*, and the 35-line `T01…T35` table.

### The `catalog` vs `relation_type` boundary (SPEC §4.2 note)

`relation_type` is **retained** in arm B. It is the L2 dispatch key
(`_ENUMERATORS.get(hyp.relation_type)`); removing it would make arm B
unrunnable, which *would* be a strawman. **The catalog is the 35-entry template
table, not the 6 relation types.**

### Fairness evidence (arm B is not handicapped)

- B produces **more** usable hypotheses than A: 367 vs 330 median.
- B's proposals have a **100% legal** `relation_type` rate; A's is 95.4%.
- B's yield is **higher** (0.0724 vs 0.0688).
- B's L2 count is lower only because A emits more `entities` per hypothesis
  (2.13 vs 1.85 mean) and each entity can match a hookpoint — not because B was
  under-prompted.

---

## 5. Step 4 (Real-SE detection) — BLOCKED, not run

SPEC §3 pre-registered **two** headline metrics; only `yield` was obtained.

`real_sdc/run_same_harness.py` dispatches each of the 17 confirmed cases over
SSH to `eval-gpu-0` (megatron/deepspeed) or `beijing-olmo-gpu` (olmo/olmo-core)
and re-runs real training for 3 phases per case. From this cluster **both
hostnames fail to resolve**, and `benchmark/bugs/` contains **zero** `.duckdb`
traces, so there is no offline path either. `detection_comparison.csv` records
all 17 rows as `BLOCKED`.

This matters for interpretation: yield measures *efficiency*; detection measures
*capability*. We can only report the former. However, the survivor analysis
substitutes for it in the strongest possible way — **the two arms deploy the
identical 11 predicates**, so a detection run could not distinguish them either.
Any detection difference would have to come from rules that do not exist.

---

## 6. Anomaly ledger (SPEC §12.3)

| Anomaly | A | B |
|---|---:|---:|
| Raw items dropped: illegal/missing `relation_type` | **80** of 1,743 (over 5 reps; **4.6%**) | **0** of 1,841 |
| Hypotheses with empty `entities` (cannot enter `_parametric_enum`) | 0 | 0 |
| Hypotheses yielding zero L2 candidates | 65 median | 77 median |
| L1 API errors | 0 | 0 |

**The 80 A-arm drops are a harness defect, not an LLM failure.** An earlier pass
of this analysis called them "the model inventing illegal relation types" —
that was wrong, and it was the one place the catalog critique risked being
unfair. Checked across all 5 replicates: **80/80 carry a valid
`catalog_template_id`, and in 80/80 the "illegal" `relation_type` is exactly
that template's `relation_operator`** (`count_frequency_match` 15,
`reference_equivalence` 10, `value_scaling_consistency` 9, `boundedness` 8,
`ordering` 6, `monotonicity` 5, +12 more). The model used the catalog
correctly.

The cause: the A prompt renders entries as `T12: checkpoint-save-completeness
(count_frequency_match)` (`layer1_hypothesis.py:23-27`) then demands
`relation_type` from a **disjoint** six-value enum. The catalog's **20
`relation_operator` values** and L2's **6 `RelationType` dispatch keys** are
separate taxonomies with **no mapping between them anywhere in the codebase**:
`relation_operator` appears in exactly one place — the prompt string at
`layer1_hypothesis.py:25` — and is never consumed. Arm B never trips on this
only because its prompt shows no competing vocabulary.

**Arm A+ tests whether this gap is what suppressed the catalog** (see
`run_arm_a_plus.py`): it authors the missing operator→relation mapping and
replays the identical cached A-arm responses. It does not rescue the
contribution — yield **falls** to 0.0659 [0.0636–0.0702] vs B's 0.0724, because
the recovered proposals inflate L2 (5,365 → 5,887) without adding survivors, and
the deployed set is **the same 11 predicates**. The defect is real and worth
fixing on its own merits; it is not what makes the catalog inert.

### A-arm real LLM vs the published simulator funnel (SPEC §8)

| Stage | Published (simulator) | This run, arm A (real LLM, median) |
|---|---:|---:|
| L1 | 420 | 330 |
| L2 | 5,334 | 5,365 |
| L3 | 3,436 | 3,117 |
| L4 | 357 | 364 |

L2/L3/L4 land close to the published numbers while L1 is ~21% lower. This is
expected, not a bug: `PatternGuidedLLM` fires *every* regex-matched pattern spec
per file, whereas a real LLM obeys the "2–4" budget. The agreement downstream is
itself evidence for the mechanism in `paper_table_catalog.md` — **the funnel is
schema-driven, so it barely notices who wrote the hypotheses.**

**Both figures count per-seed-file sums, not distinct predicates**: arm A's L4 =
364 is 11 distinct predicates re-derived across 112 files (~11× in every stage).
`run_funnel.py` computes the published 357 the same way. The published funnel
should not be read as "357 distinct rules".

---

## 7. Files

| File | Contents |
|---|---|
| `catalog_ablation/deepseek_client.py` | LLM client; key from `$DEEPSEEK_API_KEY` |
| `catalog_ablation/run_l1.py` | Phase 1: 1120 L1 calls, resumable, raw responses cached |
| `catalog_ablation/l1_raw.jsonl` | every raw LLM response (1120 rows, auditable) |
| `catalog_ablation/run_ablation.py` | Phase 2: deterministic L2/L3/L4 replay |
| `catalog_ablation/run_arm_a_plus.py` | arm A+ robustness check: authors the missing operator→relation mapping, replays cached A responses |
| `catalog_ablation/arm_a_plus_results.json` | A+ per-rep funnel + survivors |
| `catalog_ablation/per_cell_results.json` | 40 cells (2 arms × 5 reps × 4 frameworks) |
| `catalog_ablation/make_outputs.py` | emits the SPEC §7 deliverables |
| `catalog_ablation/arm_{a,b}_results.json` | per-arm per-rep funnel + survivors |
| `catalog_ablation/catalog_ablation_summary.csv` | the §6 table's raw data |
| `catalog_ablation/detection_comparison.csv` | 17 rows, all `BLOCKED` |
| `catalog_ablation/paper_table_catalog.md` | paper table draft + caption + rewrite advice |

Reproduce: `DEEPSEEK_API_KEY=... python run_l1.py --reps 5 --workers 32 && python run_ablation.py && python make_outputs.py`
(`run_l1.py` resumes from `l1_raw.jsonl`; Phase 2 is deterministic.)

---

## 8. Acceptance against SPEC §13

> *"明确报告红线触发（A ≤ B），并给出 contribution 3 的改写建议"* — satisfied.
> *"两种结果都是成功。跑出『catalog 没用』不是实验失败，是我们自己先于 reviewer 2 知道了。"*

The caveat: with Step 4 blocked, this establishes that the catalog does not
improve **yield** and does not change the **deployed rule set**. It does not
independently measure detection — though identical rule sets leave nothing for a
detection run to find.
