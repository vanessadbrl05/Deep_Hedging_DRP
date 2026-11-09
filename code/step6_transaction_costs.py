"""Step 6 — proportional transaction costs. (Checkpoint for end of week 4.)

For each cost level c we retrain the deep hedger WITH costs in its PnL, and
compare against the delta hedge paying the same costs on the same paths.
Delta ignores costs entirely, so it overtrades; the network can learn to trade
less (a no-trade band around delta).

Usage: python step6_transaction_costs.py [--risk entropic] [--iters 1500]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from black_scholes import bs_call_delta
from common import (K, RESULTS, bs_premium, default_gbm, gbm_sampler, gbm_test_set,
                    payoff, save_json, savefig)
from hedging import delta_hedge_holdings, pnl
from model import HedgeNet
from risk import make_risk, print_table, summary
from train import train

ap = argparse.ArgumentParser()
ap.add_argument("--risk", default="entropic")
ap.add_argument("--risk-param", type=float, default=None)
ap.add_argument("--iters", type=int, default=1500)
ap.add_argument("--costs", type=float, nargs="+", default=[0.0, 0.0005, 0.001, 0.002, 0.005, 0.01])
ap.add_argument("--show-cost", type=float, default=0.002, help="cost level used for detail plots")
args = ap.parse_args()

p = default_gbm()
premium = bs_premium(p)
risk_fn = make_risk(args.risk, args.risk_param)
S = gbm_test_set(p)
d_bs = delta_hedge_holdings(S, K, p.sigma, p.r, p.T)


def turnover(d):
    flat = torch.zeros(d.shape[0], 1)
    return (torch.cat([d, flat], 1) - torch.cat([flat, d], 1)).abs().sum(1).mean().item()


rows, models = [], {}
for c in args.costs:
    print(f"\n=== cost = {c:.4f} ({c * 1e4:.0f} bp) ===")
    model = HedgeNet(K, p.T, sigma_ref=p.sigma)
    train(model, gbm_sampler(p), payoff, premium, risk_fn, cost=c, r=p.r, T=p.T,
          n_iters=args.iters, log_every=500)
    models[c] = model
    torch.save(model.state_dict(), RESULTS / f"step6_model_c{c:g}.pt")
    with torch.no_grad():
        d_net = model(S)
    pd = pnl(S, d_bs, payoff(S), premium, c, p.r, p.T)
    pn = pnl(S, d_net, payoff(S), premium, c, p.r, p.T)
    res = {"delta": summary(pd), "deep": summary(pn)}
    print_table(res)
    rows.append({"cost": c, **{f"{h}_{k}": v for h in res for k, v in res[h].items()},
                 "delta_risk": risk_fn(pd).item(), "deep_risk": risk_fn(pn).item(),
                 "delta_turnover": turnover(d_bs), "deep_turnover": turnover(d_net)})
save_json(rows, "step6_cost_sweep.json")

# --- figure 1: risk vs cost and the gap ------------------------------------
cs = np.array(args.costs) * 1e4
dr = np.array([r["delta_risk"] for r in rows])
nr = np.array([r["deep_risk"] for r in rows])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(cs, dr, "o-", label="delta hedge")
ax.plot(cs, nr, "s-", label="deep hedge")
ax.fill_between(cs, nr, dr, alpha=0.15, color="C2", label="gap")
ax.set(xlabel="proportional cost (bp)", ylabel=f"risk ({args.risk}) of PnL",
       title="Hedging risk vs transaction cost (lower is better)")
ax.legend()
ax = axes[1]
ax.plot(cs, [r["delta_turnover"] for r in rows], "o-", label="delta hedge")
ax.plot(cs, [r["deep_turnover"] for r in rows], "s-", label="deep hedge")
ax.set(xlabel="proportional cost (bp)", ylabel="mean total shares traded",
       title="Turnover: the network learns to trade less")
ax.legend()
savefig(fig, "step6_cost_sweep.png")

# --- figure 2: PnL histograms at one cost level -----------------------------
c = args.show_cost if args.show_cost in models else args.costs[len(args.costs) // 2]
model = models[c]
with torch.no_grad():
    d_net = model(S)
fig, ax = plt.subplots(figsize=(7, 4))
for name, d in [("delta", d_bs), ("deep hedge", d_net)]:
    x = pnl(S, d, payoff(S), premium, c, p.r, p.T).numpy()
    ax.hist(x, bins=120, alpha=0.55, density=True, label=f"{name} (mean {x.mean():.3f}, std {x.std():.3f})")
ax.set(xlabel="terminal PnL", ylabel="density", xlim=(-3, 2),
       title=f"PnL with {c * 1e4:.0f} bp proportional costs")
ax.legend()
savefig(fig, "step6_pnl_hist.png")

# --- figure 3: how much does each hedger trade from a given position? -------
# Delta always jumps all the way to N(d1) whatever it currently holds. The net
# sees its current holding and only moves part of the way: its trades are
# damped, which is how it saves costs.
k = p.n_steps // 2
spots = torch.linspace(88, 112, 300)
bs = bs_call_delta(spots, K, p.T * (1 - k / p.n_steps), p.sigma, p.r)
fig, ax = plt.subplots(figsize=(7, 4))
for prev, col in [(0.2, "C0"), (0.5, "C1"), (0.8, "C2")]:
    out = model.policy(k, p.n_steps, spots, prev=torch.full_like(spots, prev))
    ax.plot(spots, bs - prev, "--", color=col, lw=1)
    ax.plot(spots, out - prev, color=col, label=f"current holding = {prev}")
ax.axhline(0, color="k", lw=0.8)
ax.set(xlabel="spot", ylabel="trade size (new - current holding)",
       title=f"Trade size at day {k}, cost {c * 1e4:.0f} bp\n"
             "dashed: delta hedge, solid: deep hedge (damped toward zero)")
ax.legend(fontsize=8)
savefig(fig, "step6_trade_size.png")

# --- figure 4: one sample path ---------------------------------------------
i = 3
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.step(range(p.n_steps), d_bs[i], where="post", label="delta hedge")
ax.step(range(p.n_steps), d_net[i], where="post", label="deep hedge")
ax2 = ax.twinx()
ax2.plot(range(p.n_steps + 1), S[i], color="gray", alpha=0.4)
ax2.set_ylabel("spot", color="gray")
ax.set(xlabel="day", ylabel="holding", title=f"Holdings along one path (cost {c * 1e4:.0f} bp)")
ax.legend(loc="upper left")
savefig(fig, "step6_sample_path.png")
