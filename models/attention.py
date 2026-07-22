import torch
import torch.nn as nn
import numpy as np


class Head(nn.Module):
    """One head of self-attention.

    Supports incremental decoding: pass the (key, value) tensors returned by the
    previous step as `past_kv` and only the new token as `x`. Because causal
    masking means a token's key and value never depend on later tokens, the
    cached entries stay valid and only the new position has to be computed.
    """

    def __init__(self, head_size, n_embed, mask: bool = False):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.head_size = head_size
        self.mask = mask

    def forward(self, x, past_kv=None, use_cache: bool = False):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present = (k, v) if use_cache else None

        weights = q @ k.transpose(-2, -1) / np.sqrt(self.head_size)

        # With a cache the query block sits at the end of the key block, so the
        # mask compares absolute positions rather than assuming a square matrix.
        # A single query token may attend to everything cached before it, which
        # is why the q_len == 1 case needs no mask at all.
        query_len = q.size(1)
        if self.mask and query_len > 1:
            key_len = k.size(1)
            past_len = key_len - query_len
            query_pos = torch.arange(past_len, key_len, device=x.device)
            key_pos = torch.arange(key_len, device=x.device)
            disallowed = key_pos.unsqueeze(0) > query_pos.unsqueeze(1)
            weights = weights.masked_fill(disallowed, float("-inf"))

        weights = nn.functional.softmax(weights, dim=-1)
        return weights @ v, present


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embed, n_heads, mask: bool = False):
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed, mask) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x, past_kv=None, use_cache: bool = False):
        outputs = []
        presents = []
        for index, head in enumerate(self.heads):
            past = past_kv[index] if past_kv is not None else None
            out, present = head(x, past_kv=past, use_cache=use_cache)
            outputs.append(out)
            presents.append(present)

        out = self.proj(torch.cat(outputs, dim=-1))
        return out, (presents if use_cache else None)
