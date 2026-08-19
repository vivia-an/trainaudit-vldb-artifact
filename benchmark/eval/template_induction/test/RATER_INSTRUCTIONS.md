# Locked-Test Rater Instructions (frozen before any rater saw a test case)

You are one of two INDEPENDENT raters (A or B) in a frozen-protocol validation of a
constraint-template catalog for silent training errors. The catalog is FROZEN: you
must not propose, merge, split, or edit templates. You have not seen and will not see
the other rater's output.

## Files you may read (ONLY these)
1. `template_coding_protocol.md` (§1 schema, §7 verdicts)
2. `frozen_template_catalog.json` (the frozen catalog)
3. `inputs/test_cases.json` (whitelisted case fields)

Do NOT read anything else in the repository. Ignore any template set you believe you
have seen elsewhere; only `frozen_template_catalog.json` counts.

## Per-case procedure
1. Extract privately (do not output the full extraction): semantic objects,
   expected relation, observable signal, topology scope, precondition, phase —
   from the case's whitelisted fields only.
2. Compare against the frozen catalog and output EXACTLY ONE verdict:
   - `MATCH` + `template_id`: core relation + object roles match one template; all
     residual differences are topology/phase/precondition guards. Guard values not
     listed in the template do NOT block a MATCH if they are the same guard *kind*;
     note them in `guard_note`.
   - `SINGLETON_NEW_RELATION`: relation is runtime-expressible but no frozen template
     covers it (a genuinely new relation).
   - `UNOBSERVABLE`: no runtime-measurable signal exists in a training run's trace.
   - `REFERENCE_DEPENDENT`: checkable only against a trusted reference execution
     (paired run / known-good baseline), not from the run itself.
   - `NON_GROUNDABLE`: relation matches a template but the topology or precondition
     guard cannot be reliably instantiated at runtime.
   - `INSUFFICIENT_EVIDENCE`: the case fields cannot support a determination.
3. For MATCH verdicts additionally output `groundable`: "yes" or "no: <reason>"
   (are schema + topology + precondition all instantiable at runtime?).
4. If torn between two templates, pick the better fit and put the runner-up in
   `alt_template_id`.

Decision order note: prefer MATCH over NON_GROUNDABLE over SINGLETON_NEW_RELATION;
use UNOBSERVABLE / REFERENCE_DEPENDENT only when the *relation itself* cannot be
witnessed by any in-run signal.

## Output format (JSONL, one object per case)
```json
{"case_id": "...", "verdict": "MATCH|SINGLETON_NEW_RELATION|UNOBSERVABLE|REFERENCE_DEPENDENT|NON_GROUNDABLE|INSUFFICIENT_EVIDENCE",
 "template_id": "T.. or null", "alt_template_id": "T.. or null",
 "groundable": "yes|no: reason|null",
 "relation_operator": "<your extracted canonical operator>",
 "guard_note": "<=1 sentence or null",
 "rationale": "<=1 sentence"}
```

## Final message
Report only: number of cases written, output path, verdict histogram. No per-case content.
