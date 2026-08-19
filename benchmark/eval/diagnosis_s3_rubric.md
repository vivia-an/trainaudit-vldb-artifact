# S3 Diagnosis Accuracy Rubric — Two-tier binary verdict

> Purpose: for each of the 17 TrainAudit DETECTED cases, decide whether the
> diagnostic signal (rule_id + rule_message at L1; multi-hop RCA chain at L2)
> is **sufficient for a framework engineer to localize the root cause without
> reading additional source code beyond what the rule already names**.
>
> Format is intentionally aligned with the IRR-50 rubric style used in
> `benchmark/eval/annotate_prompt.md` (binary verdict + short justification +
> confidence). The 13-class category taxonomy used there is **not** reused here
> because S3 measures diagnosis accuracy, not category prediction.

---

## SYSTEM

You are rating the diagnostic output of TrainAudit on real silent-error bugs.
For each case you see:

- `bug_id`, `framework`, `category`
- `rule_id` — the rule name that fired (this is the "L1 signal")
- `rule_message` — the runtime message attached to that rule fire
- `ground_truth_title` — one-line description of the bug from the upstream fix
- `ground_truth_summary` — 2-4 sentence description (commit message / PR body)
- `ground_truth_invariant` — the property the fix is meant to restore
- `rca_chain` — list of multi-hop drill-down steps (empty if not available)

Decide two binary verdicts and write a justification. Do not look at any other
paper file or annotation.

---

## OUTPUT format (per case)

```yaml
bug_id: <copy>
L1_correct: yes | no            # rule name -> fault class match
L2_correct: yes | no | N/A      # RCA chain -> leaf root cause; N/A if rca_chain == []
confidence: high | medium | low # rater's own certainty
justification: "<= 2 sentences, English ok, 30-300 chars"
```

CSV columns (in this order): `bug_id,L1_correct,L2_correct,confidence,justification`.

---

## L1 verdict — "rule name -> fault class"

**Question.** Does the `rule_id` (its name plus the one-line `rule_message`)
name the **same fault class** as the ground-truth invariant?

A rule name is treated as a fault signature. For L1=yes the rule must point at
the *same defective property* as the upstream fix, with at most one hop of
naming abstraction.

### L1 = yes if any of the following holds

1. **Exact match.** Rule names exactly the invariant the fix restores.
   Example: rule `T0-clip-grad-bounded` for B11 "grad clip uses max instead of
   min; ||grad|| > max_norm after clipping" — rule literally checks the
   invariant the fix restores.

2. **Property match.** Rule names the structural property whose violation
   *is* the bug, even if the rule does not name the buggy code path.
   Example: rule `T1-grad-replica-cksum-equal` for B2 "LinearWithFrozenWeight
   backward forgets to all-reduce grad_input" — rule reports that grad is not
   replicated across TP replicas, which is exactly what the missing all-reduce
   causes.

3. **Necessary-symptom match.** Rule names a symptom that is uniquely caused
   by the bug's fault class within this codebase (no other plausible
   explanation under the trigger conditions).
   Example: rule `T0-softmax-degenerate` for M-014 "topk=1 then softmax" — a
   degenerate one-hot router output under k=1 directly implies the
   topk-then-softmax mistake.

### L1 = no if

- Rule fires on an unrelated symptom (e.g., loss NaN rule fired on a sharding
  bug whose root cause never touches loss).
- Rule names a generic invariant (e.g., "loss_finite") that any of several
  fault classes could violate, and the rule_message does not narrow it down.
- Rule fires in the wrong subsystem (e.g., gradient rule for a dataloader
  bug).

**Tie-breakers.**

- *Latency.* If the rule fires several steps after the bug originates but
  still names the right property, count as **yes**; note the latency in the
  justification.
- *Multi-rule.* If multiple rules fired (not currently the case for our 17),
  take the **earliest fire that matches ground truth**; if none match, L1=no.
- *Naming style.* Underscores, prefixes (T0/T1), and tier numbers do not
  affect verdict — read the rule_id as a property name.

---

## L2 verdict — "RCA chain -> leaf root cause"

**Question.** Does the `rca_chain` (the multi-hop drill-down output) converge
to the **leaf root cause** named in `ground_truth_summary` (function, module,
or specific predicate)?

### L2 = yes if

- The chain names the same file/function/module that the fix patches, OR
- The chain narrows the failure to one specific predicate matching the
  invariant (e.g., "expert_bias buffer dtype == fp32 violated at TopKRouter
  forward").

### L2 = no if

- The chain points at a different subsystem than the fix.
- The chain stops at "some replica diverged" without naming which property
  or which module.

### L2 = N/A if

- `rca_chain` is empty (no multi-hop drill-down available for this case).
  This is the current state for all 17 records (see input.json `meta.rca_note`).
  When `rca_chain` is empty, **always** record `L2_correct: N/A`.

L2=N/A is excluded from the L2 denominator when aggregating.

---

## Confidence

| level | when to use |
|---|---|
| high | rule name and ground-truth invariant are an obvious 1:1 or property match; no ambiguity |
| medium | match requires one inferential step (rule names symptom; rule_message disambiguates) |
| low | rule name is generic, or you are guessing the connection |

---

## Justification

- One or two sentences, English (technical terms ok).
- Must say *why* yes or no, ideally by naming the bridge between rule_id and
  ground_truth_invariant.
- If verdict is yes via "necessary symptom" rather than exact match, say so.
- If you flagged latency or multi-rule, say so in 5-10 words.

Good examples:

- B11 yes: "Rule literally checks ||grad|| <= max_norm post clipping, which is
  exactly the invariant the fix restores."
- M-NEW-5 yes: "Rule reports the router lacks calculate_per_token_loss attr —
  the missing attr is the runtime manifestation of the loss-scaling fix."
- (hypothetical no) "Rule names a generic loss_finite check; rule_message
  does not disambiguate sharding vs dtype root cause."

Bad examples (do not write these):

- "Looks right." (no bridge)
- "Rule matches invariant." (no detail)

---

## Quality self-check

- [ ] All 17 rows present.
- [ ] `L1_correct` in {yes, no}.
- [ ] `L2_correct` in {yes, no, N/A}; N/A when and only when `rca_chain` is
  empty in input.json.
- [ ] `confidence` in {high, medium, low}.
- [ ] `justification` non-empty, 30-300 chars, names the bridge between rule
  and invariant.

---

## Known failure modes (cross-reference SPEC §8)

- *Rule fires but bug originates earlier*: still L1=yes if rule names the
  right property; note latency in justification.
- *Multi-rule fire*: take earliest matching rule.
- *RCA chain absent*: L2=N/A.
