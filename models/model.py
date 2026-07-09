import torch
import torch.nn as nn
from tokenizer import CharTokenizer, CharEmbedding
from attention import MultiHeadAttention
from FFN import FFN


text = "abcdefghijklmnopqrstuvwxyz0123456789*+-/^=()<>;:,.!? "
vocab_size = len(text)
d_model = 128
n_heads = 4
n_layers = 32
max_seq_len = 1024
dropout = 0.1


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, mask=True)
        self.ffn = FFN(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class model(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.tokenizer = CharTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = CharEmbedding(self.vocab_size, d_model, max_seq_len, dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, self.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.fc(x)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=None):
        x = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = x.unsqueeze(0).to(self.device)
        for _ in range(max_new_tokens):
            logits = self.forward(x)
            if temperature == 0:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            else:
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits = torch.where(logits < v[:, [-1]], -float("inf"), logits)
                probs = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            if x.size(1) >= self.max_seq_len:
                break
        return self.tokenizer.decode(x[0].tolist())
