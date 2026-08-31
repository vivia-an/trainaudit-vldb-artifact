"""LN1 surrogate (fixed): aux_loss correctly divided by num_valid_tokens."""
import torch
import torch.nn as nn


def compute_aux_loss(logits, mask, mode, micro_batch):
    raw = (logits * logits).sum()
    if mode == "buggy":
        return raw / micro_batch
    else:
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
        mask = torch.ones(micro_batch, seq_len, dtype=torch.bool)
        aux = compute_aux_loss(logits, mask, "fixed", micro_batch)
        losses.append(aux.item())

    print(f"[LN1_fixed] micro_batch={micro_batch}  seq_len={seq_len}  → divisor used: {micro_batch*seq_len}")
    print(f"[LN1_fixed] avg aux_loss = {sum(losses)/len(losses):.4f}")


if __name__ == "__main__":
    main()
