"""CM1 surrogate (fixed): metric collective uses max_global; all ranks see same value.
"""
import torch


def fake_collective_max(local_values, reduce_op="max_global"):
    if reduce_op == "max_local":
        return local_values
    g = max(local_values)
    return [g] * len(local_values)


def main():
    torch.manual_seed(0)
    n_ranks = 4
    n_steps = 32
    metric_log_per_rank = [[] for _ in range(n_ranks)]

    for step in range(n_steps):
        local_tput = [1000.0 + 5.0 * torch.randn(()).item() + 2.0 * r for r in range(n_ranks)]
        # fixed: use max_global — every rank ends with the same reduced value
        reduced = fake_collective_max(local_tput, reduce_op="max_global")
        for r in range(n_ranks):
            metric_log_per_rank[r].append(reduced[r])

    rank0_avg = sum(metric_log_per_rank[0]) / n_steps
    rank3_avg = sum(metric_log_per_rank[3]) / n_steps
    print(f"[CM1_fixed] rank0 avg tput = {rank0_avg:.2f}")
    print(f"[CM1_fixed] rank3 avg tput = {rank3_avg:.2f}")
    print(f"[CM1_fixed] rank-disagreement = {abs(rank3_avg - rank0_avg):.2f} (expected 0)")
    return rank0_avg, rank3_avg


if __name__ == "__main__":
    main()
