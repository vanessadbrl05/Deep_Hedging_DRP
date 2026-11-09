"""Step 8 (stretch) — risk-measure ablation: how does the loss shape the strategy?

Train one network per risk measure at a fixed cost level, then score EVERY
network under EVERY metric on the same test paths. Each network should win on
the metric it was trained for — the off-diagonal entries show the trade-offs.

Usage: python step8_risk_ablation.py [--cost 0.002] [--iters 1500]
"""
import argparse

import matplotlib.pyplot as plt
import torch

from common import K, bs_premium, default_gbm, gbm_sampler, gbm_test_set, payoff, save_json, savefig
from hedging import delta_hedge_holdings, pnl
from model import HedgeNet
from risk import make_risk, print_table, summary
from train import train

ap = argparse.ArgumentParser()
ap.add_argument("--cost", type=float, default=0.002)
ap.add_argument("--iters", type=int, default=1500)
args = ap.parse_args()

p = default_gbm()
premium = bs_premium(p)
S = gbm_test_set(p)
measures = {"mean_std(1)": ("mean_std", 1.0), "entropic(1)": ("entropic", 1.0),
            "CVaR95": ("cvar", 0.95)}

pnls = {"delta": pnl(S, delta_hedge_holdings(S, K, p.sigma, p.r, p.T), payoff(S), premium, args.cost, p.r, p.T)}
for label, (name, param) in measures.items():
    print(f"\nTraining with {label} ...")
    m = HedgeNet(K, p.T, sigma_ref=p.sigma)
    train(m, gbm_sampler(p), payoff, premium, make_risk(name, param), cost=args.cost,
          r=p.r, T=p.T, n_iters=args.iters, log_every=500)
    with torch.no_grad():
        pnls[f"trained on {label}"] = pnl(S, m(S), payoff(S), premium, args.cost, p.r, p.T)

res = {n: summary(x) for n, x in pnls.items()}
print(f"\nRisk-measure ablation, cost = {args.cost}  (rows: hedger, columns: metric; lower risk is better)")
print_table(res)
save_json(res, "step8_risk_ablation.json")

fig, ax = plt.subplots(figsize=(8, 4.5))
for n, x in pnls.items():
    ax.hist(x.numpy(), bins=150, histtype="step", lw=1.4, density=True, label=n)
ax.set(xlabel="terminal PnL", ylabel="density", xlim=(-3, 2),
       title=f"Same market, different risk measures (cost {args.cost * 1e4:.0f} bp)")
ax.legend(fontsize=8)
savefig(fig, "step8_risk_ablation.png")
