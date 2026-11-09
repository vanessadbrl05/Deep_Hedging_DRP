"""Steps 2-3 — GBM simulator check + classical discrete delta-hedge baseline.
(Checkpoint for end of week 2.)

1. Sanity check the simulator: a Monte Carlo price under Q (mu = r) should match
   the Black-Scholes formula.
2. Delta-hedge the short call on 20,000 real-world paths, daily rebalancing.
3. Show WHY discrete rebalancing breaks the perfect hedge: the PnL std shrinks
   like 1/sqrt(N) as the grid gets finer, but never reaches 0 at finite N.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

from common import (K, N_TEST, bs_premium, default_gbm, gbm_test_set, payoff,
                    save_json, savefig)
from hedging import delta_hedge_holdings, pnl
from market import simulate_gbm
from risk import print_table, summary

p = default_gbm()
premium = bs_premium(p)
print(f"Black-Scholes price of ATM 30-day call: {premium:.4f}")

# --- 1. simulator sanity check under the risk-neutral measure -------------
q = default_gbm(mu=p.r)
S_q = simulate_gbm(q, 500_000, torch.Generator().manual_seed(1))
disc_pay = payoff(S_q) * np.exp(-p.r * p.T)
se = disc_pay.std().item() / np.sqrt(len(disc_pay))
print(f"Monte Carlo price under Q:              {disc_pay.mean().item():.4f} +/- {1.96 * se:.4f}")

# --- 2. delta-hedge baseline -----------------------------------------------
S = gbm_test_set(p)
deltas = delta_hedge_holdings(S, K, p.sigma, p.r, p.T)
pnl_delta = pnl(S, deltas, payoff(S), premium, cost=0.0, r=p.r, T=p.T)
stats = summary(pnl_delta)
print(f"\nDiscrete delta hedge, N={p.n_steps}, {N_TEST} paths, no costs:")
print_table({"delta": stats})

unhedged = pnl(S, torch.zeros_like(deltas), payoff(S), premium)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(unhedged.numpy(), bins=120, alpha=0.4, density=True, label="unhedged (short call)")
ax.hist(pnl_delta.numpy(), bins=120, alpha=0.8, density=True, label=f"delta hedge, N={p.n_steps}")
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="terminal PnL", ylabel="density", xlim=(-8, 4),
       title="Short ATM call: unhedged vs discrete delta hedge")
ax.legend()
savefig(fig, "step3_delta_pnl_hist.png")

# --- 3. hedging error vs rebalance frequency --------------------------------
grid = [5, 10, 20, 30, 60, 120, 240]
stds = []
for n in grid:
    pn = default_gbm(n_steps=n)
    Sn = simulate_gbm(pn, N_TEST, torch.Generator().manual_seed(7))
    dn = delta_hedge_holdings(Sn, K, pn.sigma, pn.r, pn.T)
    stds.append(pnl(Sn, dn, payoff(Sn), premium, r=pn.r, T=pn.T).std().item())
slope = np.polyfit(np.log(grid), np.log(stds), 1)[0]
print("\nPnL std vs N: " + ", ".join(f"N={n}: {s:.4f}" for n, s in zip(grid, stds)))
print(f"log-log slope = {slope:.3f}  (theory: -0.5)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(grid, stds, "o-", label="delta hedge PnL std")
ax.loglog(grid, stds[0] * (np.array(grid) / grid[0]) ** -0.5, "--", label=r"$\propto N^{-1/2}$")
ax.set(xlabel="number of rebalancing dates N", ylabel="std of terminal PnL",
       title=f"Discrete hedging error (fitted slope {slope:.2f})")
ax.legend()
savefig(fig, "step3_error_vs_rebalancing.png")

save_json({"premium": premium, "delta_frictionless": stats,
           "std_vs_N": dict(zip(map(str, grid), stds)), "slope": slope},
          "step3_baseline.json")
