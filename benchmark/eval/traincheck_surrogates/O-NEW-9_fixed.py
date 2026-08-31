"""O-NEW-9 surrogate (fixed): DataLoader yields token ids in [0, vocab_size).

Reference run for TrainCheck.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    def __init__(self, n=4, seq_len=16, vocab=50000):
        self.n = n
        self.seq_len = seq_len
        self.vocab = vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab, (self.seq_len,), dtype=torch.int64)


def main():
    torch.manual_seed(0)
    vocab = 50000
    model = nn.Embedding(vocab, 8)
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    dl = DataLoader(TokenDataset(vocab=vocab), batch_size=2)
    for batch in dl:
        opt.zero_grad()
        h = model(batch)
        h.pow(2).sum().backward()
        opt.step()


if __name__ == "__main__":
    main()
