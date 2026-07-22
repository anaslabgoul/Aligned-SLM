import torch
import torch.nn as nn
from tokenizer import CharTokenizer, CharEmbedding
from attention import MultiHeadAttention
from FFN import FFN


text = "abcdefghijklmnopqrstuvwxyz0123456789*+-/^=()<>;:,.!? "
vocab_size = len(text)
d_model = 384
n_heads = 6
n_layers = 12
max_seq_len = 1024
dropout = 0.1


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, mask=True)
        self.ffn = FFN(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, past_kv=None, use_cache: bool = False):
        attn_out, present = self.attention(
            self.ln1(x), past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, present


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

    @staticmethod
    def cache_length(past_kvs) -> int:
        """Number of tokens already stored in a KV cache (0 if there is none)."""
        if not past_kvs or past_kvs[0] is None:
            return 0
        first_layer_first_head_keys = past_kvs[0][0][0]
        return first_layer_first_head_keys.size(1)

    def forward(self, x, past_kvs=None, use_cache: bool = False):
        """Returns logits, or (logits, kv_cache) when use_cache is True.

        Training calls this as model(x) and still gets a plain tensor back.
        """
        x = self.embedding(x, pos_offset=self.cache_length(past_kvs))

        presents = []
        for index, block in enumerate(self.blocks):
            past = past_kvs[index] if past_kvs is not None else None
            x, present = block(x, past_kv=past, use_cache=use_cache)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.fc(x)

        if use_cache:
            return logits, presents
        return logits

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """Generate a continuation of `prompt`.

        With use_cache=True each token is encoded once; without it the whole
        prefix is re-encoded every step. Both produce the same text — the flag
        exists so the two paths can be compared.
        """
        x = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        x = x.unsqueeze(0).to(self.device)

        generated = x
        step_input = x
        past_kvs = None

        for _ in range(max_new_tokens):
            if use_cache:
                logits, past_kvs = self.forward(
                    step_input, past_kvs=past_kvs, use_cache=True
                )
            else:
                logits = self.forward(generated)

            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(next_logits, top_k)
                    next_logits = torch.where(
                        next_logits < v[:, [-1]], -float("inf"), next_logits
                    )
                probs = nn.functional.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            # Only the new token needs encoding; the rest is in the cache.
            step_input = next_token

            if next_token.item() == self.tokenizer.eos_token_id:
                break
            if generated.size(1) >= self.max_seq_len:
                break

        return self.tokenizer.decode(generated[0].tolist())
