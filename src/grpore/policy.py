"""The policy: a ~5k-parameter autoregressive GRU language model.

Small enough that a full GRPO run (rollouts + two optimization epochs)
finishes in seconds on a laptop CPU — which is the whole point: every
hyperparameter in the README ablation table was measured, not copied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import EOS, ID, MAX_STEPS, TOK, VOCAB


class Policy(nn.Module):
    def __init__(self, prompt_dim: int = 4, hidden: int = 48):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), hidden)
        self.gru = nn.GRUCell(hidden + prompt_dim, hidden)
        self.head = nn.Linear(hidden, len(VOCAB))
        self.prompt_dim = prompt_dim
        self.hidden = hidden

    # ---- rollout -----------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt_vec: torch.Tensor) -> tuple[list[int], list[float]]:
        """Sample one completion; returns (token_ids, per-token logprobs).

        Uses torch's global RNG — callers seed reproducibility via
        ``torch.manual_seed``.
        """
        h = torch.zeros(1, self.hidden)
        ids: list[int] = []
        lps: list[float] = []
        pv = prompt_vec.reshape(1, -1)
        x = torch.tensor([[0]], dtype=torch.long)  # BOS id 0 placeholder
        for _ in range(MAX_STEPS):
            inp = torch.cat([self.embed(x).squeeze(0), pv], dim=-1)
            h = self.gru(inp, h)
            logits = self.head(h)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample().item()
            ids.append(a)
            lps.append(dist.log_prob(torch.tensor(a)).item())
            if a == TOK[EOS]:
                break
            x = torch.tensor([[a]], dtype=torch.long)
        return ids, lps

    # ---- training-time scoring --------------------------------------
    def token_logprobs(self, prompt_vec: torch.Tensor, ids: list[int]) -> torch.Tensor:
        """Per-token logprob of a fixed sequence, differentiable."""
        h = torch.zeros(1, self.hidden)
        pv = prompt_vec.reshape(1, -1)
        x = torch.tensor([[0]], dtype=torch.long)
        out = []
        for a in ids:
            inp = torch.cat([self.embed(x).squeeze(0), pv], dim=-1)
            h = self.gru(inp, h)
            logp = F.log_softmax(self.head(h), dim=-1)
            out.append(logp[0, a])
            x = torch.tensor([[a]], dtype=torch.long)
        return torch.stack(out)


def prompt_vector(p) -> torch.Tensor:
    """Four features: scaled digits + scaled target. Deterministic."""
    n = p.numbers
    return torch.tensor([n[0] / 9.0, n[1] / 9.0, n[2] / 9.0, p.target / 99.0])


def tokens_to_str(ids: list[int]) -> str:
    return " ".join(ID[i] for i in ids)
