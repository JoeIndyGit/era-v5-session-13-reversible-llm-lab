from dataclasses import dataclass, asdict
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .reversible import ReversibleMidpointStack, ReversibleLeapfrogStack


@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    seq_len: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 22
    d_ff: int = 1024
    dropout: float = 0.0
    bias: bool = True
    reversible: bool = False
    reversible_variant: str = "midpoint"
    reversible_h: float = 0.25

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class TransformerDelta(nn.Module):
    """Return f_theta(x), i.e. the transformer update rather than x + f_theta(x)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_delta = self.attn(self.ln1(x))
        x_after_attn = x + attn_delta
        mlp_delta = self.fc2(F.gelu(self.fc1(self.ln2(x_after_attn)), approximate="tanh"))
        return attn_delta + mlp_delta


class BaselineStack(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.dropout != 0.0:
            raise ValueError("This experiment requires dropout=0 for deterministic reversible reconstruction.")

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        blocks = [TransformerDelta(cfg) for _ in range(cfg.n_layers)]

        if cfg.reversible:
            variant = cfg.reversible_variant.lower()
            stacks = {
                "midpoint": ReversibleMidpointStack,
                "leapfrog": ReversibleLeapfrogStack,
            }
            if variant not in stacks:
                raise ValueError(
                    f"Supported reversible variants are {sorted(stacks)}; received {cfg.reversible_variant!r}."
                )
            self.stack = stacks[variant](blocks, h=cfg.reversible_h)
        else:
            self.stack = BaselineStack(blocks)

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in blocks:
            torch.nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(block.fc2.weight, mean=0.0, std=residual_std)

        self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets: Optional[torch.Tensor] = None):
        _, t = idx.shape
        if t > self.cfg.seq_len:
            raise ValueError(f"sequence length {t} exceeds configured {self.cfg.seq_len}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.stack(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self):
        token_embedding = self.tok_emb.weight.numel()
        position_embedding = self.pos_emb.weight.numel()
        blocks = sum(p.numel() for p in self.stack.parameters())
        final_norm = sum(p.numel() for p in self.ln_f.parameters())
        return {
            "token_embedding": token_embedding,
            "position_embedding": position_embedding,
            "transformer_blocks": blocks,
            "final_norm": final_norm,
            "total": self.num_parameters(),
        }

    def architecture_summary(self):
        return {
            **self.cfg.to_dict(),
            "parameters": self.num_parameters(),
            "parameters_m": self.num_parameters() / 1e6,
            "parameter_breakdown": self.parameter_breakdown(),
        }
