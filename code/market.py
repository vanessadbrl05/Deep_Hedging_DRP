"""Step 2 — market simulators.

All simulators return torch tensors of shape (n_paths, n_steps + 1) so they can
be fed straight into the network during training. Paths are simulated under the
REAL-WORLD measure (drift `mu`), because hedging performance is a real-world
question: we care about the PnL distribution we will actually experience. The
option premium, by contrast, is a risk-neutral price (see black_scholes.py).
"""
from dataclasses import dataclass

import torch


@dataclass
class GBMParams:
    S0: float = 100.0
    mu: float = 0.05        # real-world drift (annualised)
    sigma: float = 0.2      # volatility (annualised)
    r: float = 0.0          # risk-free rate
    T: float = 30 / 365     # maturity in years (30 days, as in Buehler et al.)
    n_steps: int = 30       # rebalance grid: daily


@dataclass
class HestonParams:
    S0: float = 100.0
    mu: float = 0.05
    r: float = 0.0
    T: float = 30 / 365
    n_steps: int = 30
    v0: float = 0.04        # initial variance (vol 20%)
    kappa: float = 1.5      # mean-reversion speed
    theta: float = 0.04     # long-run variance
    xi: float = 0.5         # vol of vol
    rho: float = -0.7       # spot/vol correlation (leverage effect)


def time_grid(T: float, n_steps: int) -> torch.Tensor:
    return torch.linspace(0.0, T, n_steps + 1)


def simulate_gbm(p: GBMParams, n_paths: int, generator: torch.Generator | None = None):
    """Exact GBM simulation: S_{k+1} = S_k exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)."""
    dt = p.T / p.n_steps
    Z = torch.randn(n_paths, p.n_steps, generator=generator)
    log_inc = (p.mu - 0.5 * p.sigma**2) * dt + p.sigma * dt**0.5 * Z
    log_S = torch.cat([torch.zeros(n_paths, 1), log_inc.cumsum(dim=1)], dim=1)
    return p.S0 * torch.exp(log_S)


def simulate_heston(p: HestonParams, n_paths: int, generator: torch.Generator | None = None):
    """Heston model with full-truncation Euler for the variance and log-Euler for spot.

    Full truncation: wherever v appears in drift/diffusion we use v+ = max(v, 0),
    but we let the stored v go negative. This is the scheme of Lord, Koekkoek and
    van Dijk (2010) and has the smallest bias among the simple Euler fixes.

    Returns (S, V), both of shape (n_paths, n_steps + 1).
    """
    dt = p.T / p.n_steps
    Z1 = torch.randn(n_paths, p.n_steps, generator=generator)
    Z3 = torch.randn(n_paths, p.n_steps, generator=generator)
    Z2 = p.rho * Z1 + (1 - p.rho**2) ** 0.5 * Z3   # correlated Brownian increment

    S = torch.empty(n_paths, p.n_steps + 1)
    V = torch.empty(n_paths, p.n_steps + 1)
    S[:, 0], V[:, 0] = p.S0, p.v0
    for k in range(p.n_steps):
        v_pos = V[:, k].clamp(min=0.0)
        sq = (v_pos * dt).sqrt()
        S[:, k + 1] = S[:, k] * torch.exp((p.mu - 0.5 * v_pos) * dt + sq * Z1[:, k])
        V[:, k + 1] = V[:, k] + p.kappa * (p.theta - v_pos) * dt + p.xi * sq * Z2[:, k]
    return S, V
