"""Training loop for the deep hedger.

Every iteration draws a FRESH batch of simulated paths (data is free in a
simulated market), runs the policy along them, computes terminal PnL with the
shared PnL engine, and takes an Adam step on the chosen risk measure.
"""
import time

import torch

from hedging import pnl


def train(model, sampler, payoff_fn, premium, risk_fn, cost=0.0, r=0.0, T=1.0,
          n_iters=2000, batch=4096, lr=3e-3, log_every=250, seed=0):
    """sampler(n, generator) -> (S, V_or_None);  payoff_fn(S) -> (n,) payoff."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 12345)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters, eta_min=lr / 20)
    history = []
    t0 = time.time()
    for it in range(1, n_iters + 1):
        S, V = sampler(batch, g)
        deltas = model(S, V)
        loss = risk_fn(pnl(S, deltas, payoff_fn(S), premium, cost, r, T))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if log_every and (it % log_every == 0 or it == 1):
            avg = sum(history[-log_every:]) / len(history[-log_every:])
            print(f"  iter {it:5d}  risk {avg:9.4f}  ({time.time() - t0:5.1f}s)", flush=True)
    return history
