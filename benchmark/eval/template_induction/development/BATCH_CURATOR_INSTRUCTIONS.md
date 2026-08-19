# Development-Batch Curator Instructions (frozen, identical for all batches)

You are the Curator processing ONE batch of the development stream in a frozen-protocol
grounded-theory experiment. Apply the frozen rules in
`template_coding_protocol.md` §§2–4 EXACTLY.

## Files you may read (ONLY these)
1. `template_coding_protocol.md`
2. `development/catalog_after_batch_<N-1>.json` (current catalog state)
3. `development/batch_inputs/batch_<N>_extractions.jsonl` (this batch's cases)

Do NOT read, grep, list, or explore anything else in the repository. Ignore any
template catalog you think you remember from elsewhere; only the catalog file counts.

## Per-case procedure (in batch order)
1. **MATCH first**: does the case's canonical relation + normalized semantic objects
   match an existing template, with all differences expressible as topology / phase /
   precondition guards (§2)? If yes → decision `MATCH(<template_id>)`. If the case
   needs a guard value not yet listed, extend that template's permitted guards
   (log as `guard_extension`, NOT a core edit).
2. If no match: normalize the operator against `operator_alias_map` (extend the map
   if the extractor coined a synonym).
3. Check the catalog's `singletons` list: if an earlier singleton has the same
   relation + objects and comes from an independent incident, PROMOTE: create a new
   template (next free sequential id, never reuse or renumber), supporting cases =
   the singleton's case + this case. Decision for this case: `NEW_TEMPLATE(<id>)`.
4. Otherwise decision `SINGLETON`: append to the catalog's singletons list.
5. Core merge/split edits to existing templates are allowed ONLY with an explicit
   §2/§3 clause citation, and must be logged under `core_edits`.

Same-incident duplicates (§4.1): if two cases are evidently the same incident
(same fix, same issue text), they count as ONE independent support.

## Outputs
1. `development/catalog_after_batch_<N>.json` — the FULL updated catalog, same
   structure as the input catalog (meta.stage = "dev_batch_<N>"; update
   positive_examples / support_count / singletons; template_ids append-only).
2. `development/batch_<N>_log.json`:
```json
{
 "batch": <N>,
 "decisions": [{"case_id": "...", "decision": "MATCH|NEW_TEMPLATE|SINGLETON",
                "template_id": "T.. or null", "guard_extension": "... or null",
                "note": "<=1 sentence"}],
 "n_matched": int, "n_new_singletons": int, "n_new_templates": int,
 "core_edits": [{"template_id": "...", "edit": "...", "rule_cited": "§..."}],
 "templates_total_after": int,
 "singletons_total_after": int
}
```
Every case in the batch gets exactly one decision. n_matched counts MATCH decisions
only.

## Final message
Report only: batch number, n_matched / n_new_singletons / n_new_templates,
templates_total_after, any core_edits (one line each). No per-case dumps.
