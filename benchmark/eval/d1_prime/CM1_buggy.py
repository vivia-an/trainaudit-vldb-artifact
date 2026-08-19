"""CM1 surrogate (buggy): non-zero ranks see partially-reduced metric.

Blueprint: O-014. Throughput / loss-eval metrics are reduced with the wrong reduction
type (sum vs max), so reporting at non-zero ranks captures partial values, not the
fully-reduced one. The training itself proceeds correctly because rank 0 logs the
intended value, but downstream tooling pulling per-rank metrics gets corrupted data.

In our surrogate we simulate 4 ranks; the buggy version uses local sum-then-broadcast
from rank 0, but writes into per-rank `metric_view[r]` only the local pre-reduce value
on r > 0 — exactly what happens when reduce_type defaults to MAX_LOCAL but the consumer
expects MAX_GLOBAL.
"""
import torch


def fake_collective_max(local_values, reduce_op="max_global"):
    """Simulate a collective whose reduction op is the bug switch."""
    if reduce_op == "max_local":
        # buggy: each rank keeps its own local value, no actual collective
        return local_values  # [v0, v1, v2, v3] unchanged
    # fixed semantics: every rank ends up with the global max
    g = max(local_values)
    return [g] * len(local_values)


def main():
    torch.manual_seed(0)
    n_ranks = 4
    n_steps = 32

    # per-rank tput in tokens/sec
    metric_log_per_rank = [[] for _ in range(n_ranks)]

    for step in range(n_steps):
        # local throughput drift slightly per rank due to noise
        local_tput = [1000.0 + 5.0 * torch.randn(()).item() + 2.0 * r for r in range(n_ranks)]
        # buggy: use max_local — only rank that was actually fastest sees its own value;
        # other ranks log their stale local
        reduced = fake_collective_max(local_tput, reduce_op="max_local")
        for r in range(n_ranks):
            metric_log_per_rank[r].append(reduced[r])

    # consumer (e.g., dashboard) reads rank-r metric to monitor — sees partial
    rank0_avg = sum(metric_log_per_rank[0]) / n_steps
    rank3_avg = sum(metric_log_per_rank[3]) / n_steps
    print(f"[CM1_buggy] rank0 avg tput = {rank0_avg:.2f}")
    print(f"[CM1_buggy] rank3 avg tput = {rank3_avg:.2f}")
    print(f"[CM1_buggy] rank-disagreement = {abs(rank3_avg - rank0_avg):.2f} (expected ~0 in fixed)")
    return rank0_avg, rank3_avg


if __name__ == "__main__":
    main()
