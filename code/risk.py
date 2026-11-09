"""Risk measures of terminal PnL (all differentiable, all 'lower is better').

Each returns rho(PnL): roughly 'how much cash must be added to make this
position acceptable'. For the translation-invariant ones (entropic, CVaR),
premium + rho(PnL - premium) is the seller's INDIFFERENCE PRICE.

- mean_std : -E[X] + lam * Std[X]     (Step 4 starting point; not monotone,
                                         penalises upside variance too)
- entropic : (1/lam) log E[exp(-lam X)] (exponential utility indifference)
- cvar     : mean of the worst (1-alpha) fraction of losses (expected shortfall)
"""
import math

import torch


def mean_std(pnl, lam=1.0):
    return -pnl.mean() + lam * pnl.std()


def entropic(pnl, lam=1.0):
    # logsumexp for numerical stability: log mean exp(-lam X)
    return (torch.logsumexp(-lam * pnl, dim=0) - math.log(pnl.numel())) / lam


def cvar(pnl, alpha=0.95):
    losses = -pnl
    k = max(1, int(round((1 - alpha) * losses.numel())))
    return losses.topk(k).values.mean()


def make_risk(name: str, param: float | None = None):
    """Factory: make_risk('cvar', 0.95), make_risk('entropic', 1.0), make_risk('mean_std', 1.0)."""
    if name == "mean_std":
        lam = 1.0 if param is None else param
        return lambda x: mean_std(x, lam)
    if name == "entropic":
        lam = 1.0 if param is None else param
        return lambda x: entropic(x, lam)
    if name == "cvar":
        alpha = 0.95 if param is None else param
        return lambda x: cvar(x, alpha)
    raise ValueError(name)


@torch.no_grad()
def summary(pnl: torch.Tensor) -> dict:
    """Evaluation metrics reported for every hedger."""
    q = torch.quantile(pnl, torch.tensor([0.01, 0.05]))
    return {
        "mean": pnl.mean().item(),
        "std": pnl.std().item(),
        "VaR95": -q[1].item(),
        "CVaR95": cvar(pnl, 0.95).item(),
        "CVaR99": cvar(pnl, 0.99).item(),
        "entropic(1)": entropic(pnl, 1.0).item(),
        "mean_std(1)": mean_std(pnl, 1.0).item(),
    }


def print_table(rows: dict[str, dict]):
    keys = list(next(iter(rows.values())).keys())
    w = max(len(n) for n in rows) + 2
    print(" " * w + "".join(f"{k:>13}" for k in keys))
    for name, m in rows.items():
        print(f"{name:<{w}}" + "".join(f"{m[k]:>13.4f}" for k in keys))
