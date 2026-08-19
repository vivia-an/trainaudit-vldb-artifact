"""O-NEW-9 surrogate (buggy): DataLoader silently emits out-of-vocab token ids
(>> vocab_size). Mirrors OLMo data-pipe regression where int32 cast wrapped
ids past the embedding range — embed lookup OOB silently undefined.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TruncatingTokenDataset(Dataset):
    def __init__(self, n=4, seq_len=16, vocab=50000):
        self.n = n
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.full((self.seq_len,), self.vocab + 100, dtype=torch.int64)


def main():
    torch.manual_seed(0)
    vocab = 50000
    model = nn.Embedding(vocab + 200, 8)  # avoid OOB crash; ids still > nominal vocab
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    dl = DataLoader(TruncatingTokenDataset(vocab=vocab), batch_size=2)
    for batch in dl:
        opt.zero_grad()
        h = model(batch)
        h.pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
