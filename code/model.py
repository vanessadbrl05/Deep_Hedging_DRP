"""Step 4 — the neural hedging policy.

A single feed-forward network F shared across time steps (time is an input),
applied recursively along each path:

    delta_k = F( t_k / T,  log(S_k/K) / (sigma_ref sqrt(T)),  delta_{k-1}  [, sqrt(v_k)/sigma_ref] )

Feeding the previous holding delta_{k-1} back in is what allows the network to
learn a no-trade region under transaction costs (it can decide "I'm close
enough, don't trade"). A closed-form delta has no notion of the current
position, which is exactly why it overtrades when trading is costly.
"""
import torch
from torch import nn


class HedgeNet(nn.Module):
    def __init__(self, K: float, T: float, sigma_ref: float = 0.2,
                 use_variance: bool = False, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.K, self.T, self.sigma_ref = K, T, sigma_ref
        self.use_variance = use_variance
        n_in = 3 + int(use_variance)
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, S: torch.Tensor, V: torch.Tensor | None = None) -> torch.Tensor:
        """S: (n, N+1) paths [, V: (n, N+1) variance paths]. Returns deltas (n, N)."""
        n, n_steps = S.shape[0], S.shape[1] - 1
        scale = self.sigma_ref * self.T**0.5
        log_m = torch.log(S / self.K) / scale
        prev = torch.zeros(n, 1)
        out = []
        for k in range(n_steps):
            feats = [torch.full((n, 1), k / n_steps), log_m[:, k:k + 1], prev]
            if self.use_variance:
                feats.append(V[:, k:k + 1].clamp(min=0).sqrt() / self.sigma_ref)
            prev = self.net(torch.cat(feats, dim=1))
            out.append(prev)
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def policy(self, k: int, n_steps: int, S: torch.Tensor, prev: torch.Tensor,
               vol: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate the policy on an arbitrary grid of states (for plotting)."""
        n = S.shape[0]
        feats = [torch.full((n, 1), k / n_steps),
                 (torch.log(S / self.K) / (self.sigma_ref * self.T**0.5)).view(-1, 1),
                 prev.view(-1, 1)]
        if self.use_variance:
            feats.append((vol / self.sigma_ref).view(-1, 1))
        return self.net(torch.cat(feats, dim=1)).squeeze(1)
