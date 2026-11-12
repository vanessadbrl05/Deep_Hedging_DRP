"""Step 7 — Heston stochastic volatility.

The market is now incomplete with only the stock as hedging instrument, so no
strategy is perfect. We compare on the same Heston test paths:

  * BS delta with constant vol sqrt(v0)      (misspecified classical hedger)
  * BS delta with current vol sqrt(v_t)      (vol-aware)
  * minimum-variance delta                   (classical SV hedger):
        Delta_MV = dC/dS + (rho * xi / S) * dC/dv
    which minimises Var(dC - Delta dS) under Heston dynamics; we approximate
    dC/dv with the BS vega at sigma = sqrt(v_t):  dC/dv = vega / (2 sqrt(v)).
  * deep hedge, variance HIDDEN (only t, spot, holding)
  * deep hedge, variance as an extra STATE

Premium = semi-analytic Heston price (checked against Monte Carlo below).

Usage: python step7_heston.py [--cost 0.0] [--risk entropic] [--iters 2000]
"""
import argparse
import math

import matplotlib.pyplot as plt
import torch

from black_scholes import bs_call_delta, heston_call_price, _d1_d2
from common import K, N_TEST, TEST_SEED, payoff, save_json, savefig
from hedging import delta_hedge_holdings, pnl
from market import HestonParams, simulate_heston
from model import HedgeNet
from risk import make_risk, print_table, summary
from train import train

ap = argparse.ArgumentParser()
ap.add_argument("--risk", default="entropic")
ap.add_argument("--risk-param", type=float, default=None)
ap.add_argument("--cost", type=float, default=0.0)
ap.add_argument("--iters", type=int, default=2000)
args = ap.parse_args()

h = HestonParams()
hp = dict(T=h.T, r=h.r, v0=h.v0, kappa=h.kappa, theta=h.theta, xi=h.xi, rho=h.rho)
premium = float(heston_call_price(h.S0, K, **hp))

# sanity check: Fourier price vs Monte Carlo under Q
hq = HestonParams(mu=h.r, n_steps=60)
Sq, _ = simulate_heston(hq, 400_000, torch.Generator().manual_seed(3))
mc = payoff(Sq) * math.exp(-h.r * h.T)
print(f"Heston price: Fourier {premium:.4f} | Monte Carlo {mc.mean():.4f} +/- {1.96 * mc.std() / len(mc)**0.5:.4f}")

S, V = simulate_heston(h, N_TEST, torch.Generator().manual_seed(TEST_SEED))
vol_t = V.clamp(min=1e-8).sqrt()
sig0 = h.v0**0.5


def min_variance_delta(S, vol):
    n = S.shape[1] - 1
    t = torch.linspace(0, h.T, n + 1)
    Sk, vk = S[:, :-1], vol[:, :-1]
    tau = (h.T - t[:-1]).expand_as(Sk)
    d1, _, tau = _d1_d2(Sk, K, tau, vk, h.r)
    vega = Sk * torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi) * tau.sqrt()
    dC_dv = vega / (2 * vk)
    return bs_call_delta(Sk, K, tau, vk, h.r) + h.rho * h.xi / Sk * dC_dv


hedgers = {
    "BS delta (const vol)": delta_hedge_holdings(S, K, sig0, h.r, h.T),
    "BS delta (vol_t)": delta_hedge_holdings(S, K, vol_t, h.r, h.T),
    "min-variance delta": min_variance_delta(S, vol_t),
}

risk_fn = make_risk(args.risk, args.risk_param)
sampler = lambda n, g: simulate_heston(h, n, g)
for use_v, name in [(False, "deep (vol hidden)"), (True, "deep (vol as state)")]:
    print(f"\nTraining {name} ...")
    m = HedgeNet(K, h.T, sigma_ref=sig0, use_variance=use_v)
    train(m, sampler, payoff, premium, risk_fn, cost=args.cost, r=h.r, T=h.T,
          n_iters=args.iters, log_every=500)
    with torch.no_grad():
        hedgers[name] = m(S, V)

res = {n: summary(pnl(S, d, payoff(S), premium, args.cost, h.r, h.T)) for n, d in hedgers.items()}
print(f"\nHeston test set, cost = {args.cost}:")
print_table(res)
save_json({"premium": premium, "cost": args.cost, "results": res}, f"step7_heston_c{args.cost:g}.json")

fig, ax = plt.subplots(figsize=(8, 4.5))
for n in ["BS delta (const vol)", "min-variance delta", "deep (vol hidden)", "deep (vol as state)"]:
    x = pnl(S, hedgers[n], payoff(S), premium, args.cost, h.r, h.T).numpy()
    ax.hist(x, bins=150, histtype="step", lw=1.4, density=True, label=f"{n} (std {x.std():.3f})")
ax.set(xlabel="terminal PnL", ylabel="density", xlim=(-3, 2.5),
       title=f"Heston market (rho={h.rho}, xi={h.xi}), cost {args.cost * 1e4:.0f} bp")
ax.legend(fontsize=8)
savefig(fig, f"step7_heston_hist_c{args.cost:g}.png")
