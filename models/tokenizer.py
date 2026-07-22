import re

import torch
import torch.nn as nn


text = "abcdefghijklmnopqrstuvwxyz0123456789*+-/^=()<>;:,.!? "
_SPECIAL_MARKER_PATTERN = re.compile(r"<bos>|<eos>", re.IGNORECASE)


class CharTokenizer(nn.Module):
    def __init__(self, vocab=text):
        super().__init__()
        self.special_tokens = ["<bos>", "<eos>", "<unk>"]
        unique_chars = list(dict.fromkeys(vocab))
        self.all_chars = self.special_tokens + unique_chars
        self.vocab_chars = set(unique_chars)

        self.stoi = {ch: i for i, ch in enumerate(self.all_chars)}
        self.itos = {i: ch for i, ch in enumerate(self.all_chars)}
        self.vocab_size = len(self.stoi)

        self.bos_token_id = self.stoi["<bos>"]
        self.eos_token_id = self.stoi["<eos>"]
        self.unk_token_id = self.stoi["<unk>"]

    def _strip_embedded_special_tokens(self, text: str) -> str:
        return _SPECIAL_MARKER_PATTERN.sub("", text)

    def encode(self, text, add_bos=True, add_eos=True):
        text = self._strip_embedded_special_tokens(text.lower())
        token_ids = [
            self.stoi[ch] if ch in self.vocab_chars else self.unk_token_id
            for ch in text
        ]
        if add_bos:
            token_ids = [self.bos_token_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_token_id]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        return "".join(self.itos.get(i, "<unk>") for i in x)


class CharEmbedding(nn.Module):
    def __init__(self, vocab_size=len(text), d_model=128, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, pos_offset: int = 0):
        # pos_offset is how many tokens are already in the KV cache. Without it,
        # every incrementally decoded token would be embedded as position 0.
        seq_len = x.size(1)
        position = torch.arange(
            pos_offset, pos_offset + seq_len, dtype=torch.long, device=x.device
        )

        tok_embed = self.token_embedding(x)
        pos_embed = self.position_embedding(position)
        embed = tok_embed + pos_embed
        embed = self.dropout(embed)
        return embed
