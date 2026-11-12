"""Shared experiment setup: default market, fixed test set, result saving."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from black_scholes import bs_call_price
from hedging import call_payoff
from market import GBMParams, simulate_gbm

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

K = 100.0
N_TEST = 20_000
TEST_SEED = 2024

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def default_gbm(**kw) -> GBMParams:
    return GBMParams(**kw)


def bs_premium(p: GBMParams) -> float:
    return bs_call_price(p.S0, K, p.T, p.sigma, p.r).item()


def gbm_test_set(p: GBMParams):
    """The fixed evaluation paths. Same seed => every hedger sees the same paths."""
    return simulate_gbm(p, N_TEST, torch.Generator().manual_seed(TEST_SEED))


def gbm_sampler(p: GBMParams):
    return lambda n, g: (simulate_gbm(p, n, g), None)


def payoff(S):
    return call_payoff(S, K)


def save_json(obj, name):
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name)
    plt.close(fig)
    print(f"  saved results/{name}")
