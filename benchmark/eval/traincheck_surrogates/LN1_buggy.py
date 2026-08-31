"""LN1 surrogate (buggy): aux_loss divided by micro_batch instead of token_count.

Triggers P16 Loss Component Normalization: each loss component's divisor must
match its declared reduction granularity (token-level → divided by num_valid_tokens,
sample-level → by micro-batch size). Buggy aux_loss is token-level but divided by
micro-batch (typically 4-32× off depending on seq_len).
"""
import torch
import torch.nn as nn


def compute_aux_loss(logits, mask, mode, micro_batch):
    """logits: (B, S, V), mask: (B, S), bool valid-token mask"""
    raw = (logits * logits).sum()
    if mode == "buggy":
        # buggy: divides by micro_batch instead of valid token count
        return raw / micro_batch
    else:
        # fixed: divides by num valid tokens
        return raw / mask.sum().clamp(min=1)


def main():
    import torch.nn as _nn; model = _nn.Linear(1, 1)  # for traincheck --models-to-track
    torch.manual_seed(0)
    micro_batch, seq_len, vocab = 4, 16, 32
    head = nn.Linear(8, vocab)

    losses = []
    for step in range(20):
        x = torch.randn(micro_batch, seq_len, 8)
        logits = head(x)
        mask = torch.ones(micro_batch, seq_len, dtype=torch.bool)  # all valid
        aux = compute_aux_loss(logits, mask, "buggy", micro_batch)
        losses.append(aux.item())

    print(f"[LN1_buggy] micro_batch={micro_batch}  seq_len={seq_len}  → expected divisor: {micro_batch*seq_len}")
    print(f"[LN1_buggy] actual divisor used: {micro_batch} (P16 violation)")
    print(f"[LN1_buggy] avg aux_loss = {sum(losses)/len(losses):.4f}")
    print(f"[LN1_buggy] (Expected magnitude when correct: {sum(losses)/len(losses)/seq_len:.4f}, off by {seq_len}x)")


if __name__ == "__main__":
    main()
