"""PnL engine shared by every hedging strategy.

Convention: we SOLD the option, received premium p0, and hedge by holding
delta_k shares over (t_k, t_{k+1}]. Terminal PnL of the seller:

    PnL = p0 + sum_k delta_k (S_{k+1} - S_k)                (hedge gains)
             - c * sum_{k=0}^{N} |delta_k - delta_{k-1}| S_k  (proportional costs)
             - payoff(S)

with delta_{-1} = 0 (start flat) and delta_N = 0 (unwind at maturity, i.e. the
option is cash-settled). Everything is expressed in discounted units
(S~_t = e^{-rt} S_t), which is exact for a self-financing strategy with a cash
account earning r. With r = 0 (our default) discounting is a no-op.

Because the delta hedge and the neural hedge both produce a holdings tensor of
shape (n_paths, n_steps), they are scored by the SAME function on the SAME
paths — which is what makes the comparison fair.
"""
import torch

from black_scholes import bs_call_delta


def call_payoff(S: torch.Tensor, K: float) -> torch.Tensor:
    return (S[:, -1] - K).clamp(min=0.0)


def pnl(S, deltas, payoff, premium, cost=0.0, r=0.0, T=1.0):
    """Terminal PnL per path.

    S       : (n, N+1) spot paths
    deltas  : (n, N)   holdings chosen at t_0 .. t_{N-1}
    payoff  : (n,)     option payoff at T (undiscounted)
    premium : float    price received at t_0
    cost    : float    proportional transaction cost (e.g. 0.001 = 10 bp)
    """
    n_steps = deltas.shape[1]
    t = torch.linspace(0.0, T, n_steps + 1)
    S_disc = S * torch.exp(-r * t)

    gains = (deltas * S_disc.diff(dim=1)).sum(dim=1)

    flat = torch.zeros(deltas.shape[0], 1, dtype=deltas.dtype)
    trades = torch.cat([deltas, flat], dim=1) - torch.cat([flat, deltas], dim=1)  # (n, N+1)
    costs = cost * (trades.abs() * S_disc).sum(dim=1)

    return premium + gains - costs - payoff * torch.exp(torch.tensor(-r * T))


def delta_hedge_holdings(S, K, sigma, r, T):
    """Classical discrete BS delta hedge. sigma may be a scalar or an (n, N+1)
    tensor of per-path, per-time vols (used for the Heston 'local vol' hedger)."""
    n_steps = S.shape[1] - 1
    t = torch.linspace(0.0, T, n_steps + 1)
    tau = (T - t[:-1]).expand(S.shape[0], -1)
    sig = sigma[:, :-1] if torch.is_tensor(sigma) and sigma.dim() == 2 else sigma
    return bs_call_delta(S[:, :-1], K, tau, sig, r)
