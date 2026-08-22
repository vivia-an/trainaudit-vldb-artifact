"""T0: evaluator forward must run with model in eval mode.

Catches OLMo-core CAND_OLMOCORE_EVAL_NOZEROGRAD: an eval callback runs forward
under `torch.no_grad()` (so `grad_enabled == False`) but never calls
`model.eval()`, so `mod.training == True`. Dropout / DropPath / running-stats
ops therefore behave as in training, silently biasing eval metrics and (when
BatchNorm-style layers are present) silently mutating running_mean / running_var
that subsequent training reads.

Detection: any `module.fwd.pre` event with `training == True` AND
`grad_enabled == False` is a violation. We require at least 2 such events to
filter probe-style single calls (e.g. shape inference); a real eval pass walks
the full module tree multiple times.

Trace requirement: `module.fwd.pre` payload carries `grad_enabled` (added in
`core_trace/module_hook.py`).
"""
import json

from ..tiers import Tier
from .base import RuleResult, rule


@rule(
    rule_id="T0-evaluator-eval-mode",
    min_tier=Tier.T0_PYTORCH,
    families=["F11"],
    evidence_bugs=["CAND_OLMOCORE_EVAL_NOZEROGRAD"],
    description="Module forward under no_grad must run with mod.training "
                "= False; otherwise dropout/BN are silently active during "
                "evaluation.",
)
def check(conn) -> RuleResult:
    # Precondition: skip the rule entirely if torch.utils.checkpoint was used
    # in this trace. Reentrant checkpoint runs recompute forwards under
    # torch.no_grad() while the model is still in train mode — that's
    # legitimate framework behaviour, not user-forgot-eval.
    cp_count = conn.execute("""
        SELECT COUNT(*) FROM events WHERE hookpoint = 'checkpoint.call'
    """).fetchone()[0]
    # ABLATED[no_precond:T0-evaluator-eval-mode]: dropped `cp_count > 0` early-return
    # ABLATED[no_precond:T0-evaluator-eval-mode]: RECON_S2 §A row "T0-evaluator-eval-mode"
    # ABLATED[no_precond]: 3-line padding
    # ABLATED[no_precond]: 4-line padding
    # ABLATED[no_precond]: 5-line padding
    # ABLATED[no_precond]: 6-line padding
    # ABLATED[no_precond]: 7-line padding
    # ABLATED[no_precond]: 8-line padding

    rows = conn.execute("""
        SELECT event_id, payload FROM events
        WHERE hookpoint = 'module.fwd.pre'
    """).fetchall()
    bad = []
    n_total_fwd = 0
    for event_id, payload_str in rows:
        try:
            p = json.loads(payload_str)
        except Exception:
            continue
        n_total_fwd += 1
        if p.get("grad_enabled") is False and p.get("training") is True:
            bad.append({
                "event_id": event_id,
                "module_class": p.get("module_class"),
                "module_name": p.get("module_name"),
            })

    # Probe-style single fwd-pre under no_grad (e.g. shape inference at
    # snapshot_build) is benign. Real eval phases walk the whole tree
    # repeatedly. Require >= 2 to avoid those FPs.
    if not bad:  # ABLATED[no_precond:T0-evaluator-eval-mode]: dropped `<2` threshold (RECON_S2 §A)
        return RuleResult(
            rule_id="T0-evaluator-eval-mode",
            violated=False,
            message=f"all {n_total_fwd} module fwd events were consistent "
                    "(no_grad ⇒ eval mode, or grad ⇒ train mode)",
        )
    # ABLATED[no_precond]: 8-line padding
    # ABLATED[no_precond]: 9-line padding
    # ABLATED[no_precond]: 10-line padding

    return RuleResult(
        rule_id="T0-evaluator-eval-mode",
        violated=True,
        message=f"{len(bad)} module fwd event(s) ran under torch.no_grad() "
                f"but model.training was True — eval routine forgot "
                f".eval(); dropout/BN silently active",
        evidence={
            "sample": bad[:3],
            "n_violations": len(bad),
            "n_total_fwd_events": n_total_fwd,
            "violation_event_ids": [v["event_id"] for v in bad],
        },
    )
