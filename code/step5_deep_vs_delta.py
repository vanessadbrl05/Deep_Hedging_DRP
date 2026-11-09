"""Steps 4-5 — train the deep hedger in the frictionless GBM market and compare it
with the delta hedge on the same test paths. (Checkpoint for end of week 3.)

Success criterion from the booklet: the network should at least MATCH delta here.
In a frictionless BS world the delta hedge is (in the continuous limit) optimal,
so if the net doesn't recover something close to N(d1), it is broken.

Usage:  python step5_deep_vs_delta.py [--risk mean_std|entropic|cvar] [--iters 2000]
"""
import argparse

import matplotlib.pyplot as plt
import torch

from black_scholes import bs_call_delta
from common import (K, RESULTS, bs_premium, default_gbm, gbm_sampler, gbm_test_set,
                    payoff, save_json, savefig)
from hedging import delta_hedge_holdings, pnl
from model import HedgeNet
from risk import make_risk, print_table, summary
from train import train

ap = argparse.ArgumentParser()
ap.add_argument("--risk", default="mean_std")
ap.add_argument("--risk-param", type=float, default=None)
ap.add_argument("--iters", type=int, default=2000)
args = ap.parse_args()

p = default_gbm()
premium = bs_premium(p)

print(f"Training deep hedger (risk={args.risk}, frictionless)...")
model = HedgeNet(K, p.T, sigma_ref=p.sigma)
hist = train(model, gbm_sampler(p), payoff, premium, make_risk(args.risk, args.risk_param),
             cost=0.0, r=p.r, T=p.T, n_iters=args.iters)
torch.save(model.state_dict(), RESULTS / "step5_model.pt")

# --- evaluate on the shared test set --------------------------------------
S = gbm_test_set(p)
with torch.no_grad():
    d_net = model(S)
d_bs = delta_hedge_holdings(S, K, p.sigma, p.r, p.T)
res = {
    "delta": summary(pnl(S, d_bs, payoff(S), premium, r=p.r, T=p.T)),
    "deep":  summary(pnl(S, d_net, payoff(S), premium, r=p.r, T=p.T)),
}
print("\nFrictionless test set (20,000 paths):")
print_table(res)
print(f"mean |delta_net - delta_BS| over all steps: {(d_net - d_bs).abs().mean():.4f}")
save_json(res, "step5_frictionless.json")

# --- figure 1: PnL histograms ---------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for name, d in [("delta", d_bs), ("deep hedge", d_net)]:
    x = pnl(S, d, payoff(S), premium, r=p.r, T=p.T).numpy()
    ax.hist(x, bins=120, alpha=0.55, density=True, label=f"{name} (std {x.std():.3f})")
ax.set(xlabel="terminal PnL", ylabel="density", xlim=(-2.5, 2.5),
       title="Frictionless market: deep hedge vs delta hedge")
ax.legend()
savefig(fig, "step5_pnl_hist.png")

# --- figure 2: learned policy vs BS delta ----------------------------------
spots = torch.linspace(85, 115, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for k, c in [(0, "C0"), (15, "C1"), (27, "C2")]:
    tau = p.T * (1 - k / p.n_steps)
    bs = bs_call_delta(spots, K, tau, p.sigma, p.r)
    net = model.policy(k, p.n_steps, spots, prev=bs)  # holding = delta, so no 'trade' incentive
    ax.plot(spots, bs, "--", color=c, lw=1)
    ax.plot(spots, net, color=c, label=f"day {k}")
ax.set(xlabel="spot", ylabel="hedge ratio", title="Learned hedge (solid) vs Black-Scholes delta (dashed)")
ax.legend()
savefig(fig, "step5_policy_vs_delta.png")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(hist, lw=0.5)
ax.set(xlabel="iteration", ylabel="training risk", title="Training curve", ylim=(min(hist), sorted(hist)[int(0.98 * len(hist))]))
savefig(fig, "step5_training_curve.png")
