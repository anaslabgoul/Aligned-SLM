import torch
import torch.nn as nn
import numpy as np


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size, n_embed, mask: bool = False):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.head_size = head_size
        self.mask = mask

    def forward(self, x):
        _, seq_len, _ = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) / np.sqrt(self.head_size)

        if self.mask:
            tril = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            weights = weights.masked_fill(tril == 0, float("-inf"))

        weights = nn.functional.softmax(weights, dim=-1)
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embed, n_heads, mask: bool = False):
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, mask) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)
