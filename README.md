# Deep_Hedging_DRP
This is a project that aims to hedge options using a neural network policy insead of a closed form dela, and thus compare the two.

---

Following Buehler, Gonon, Teichmann & Wood (2019), *Deep Hedging*, a neural network learns a hedging
strategy for a short European call by directly minimising a risk measure of the terminal PnL.
It is compared with discrete Black–Scholes delta hedging on identical simulated paths: first in a
frictionless GBM market, then with proportional transaction costs, then under Heston stochastic
volatility.

## 1. Installation

Requires Python 3.10+ (tested with Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11, PyTorch 2.15).

```bash
git clone https://github.com/vanessadbrl05/Deep_Hedging_DRP.git
```

```bash
cd Deep_Hedging_DRP
```

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

No GPU is needed. Installing the CPU-only PyTorch build first avoids a multi-GB CUDA download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

(Plain `pip install -r requirements.txt` also works; it just downloads the larger CUDA build of PyTorch.)

## 2. Reproducing the experiments

Run every script from inside `code/`. Each one writes figures (`.png`), metrics (`.json`), and trained
weights (`.pt`) to `results/`.

```bash
cd code
```

| Command | What it does | Runtime* |
|---|---|---|
| `python step3_delta_baseline.py` | Checks the simulator against the BS price; delta-hedge baseline; hedging error vs rebalancing frequency | ~5 s |
| `python step5_deep_vs_delta.py` | Trains the deep hedger with no frictions and compares it with delta | ~3 min |
| `python step6_transaction_costs.py` | Retrains at 0, 5, 10, 20, 50, 100 bp proportional cost; plots the gap and turnover | ~13 min |
| `python step7_heston.py` | Heston market, no costs: classical SV hedgers vs deep (vol hidden / vol as state) | ~7 min |
| `python step7_heston.py --cost 0.002` | Same, with 20 bp costs | ~7 min |
| `python step8_risk_ablation.py` | Trains with mean–std, entropic, and CVaR at 20 bp; scores every net under every metric | ~8 min |

\*On 4 CPU cores. The whole pipeline takes about 40 minutes.

To run everything and keep the console logs:

```bash
for s in step3_delta_baseline step5_deep_vs_delta step6_transaction_costs step7_heston step8_risk_ablation; do python -u $s.py | tee ../results/${s%%_*}_log.txt; done; python -u step7_heston.py --cost 0.002 | tee ../results/step7_log_c0.002.txt
```

Useful flags (training scripts): `--risk {mean_std,entropic,cvar}`, `--risk-param` (λ for mean–std/entropic,
α for CVaR), `--iters`. Step 6 also takes `--costs ...` and `--show-cost`; Steps 7 and 8 take `--cost`.

Test paths and training use fixed seeds. Multithreaded CPU kernels are not bit-for-bit deterministic, so
retrained numbers can differ from those below in the third decimal place. The conclusions do not change.

## 3. Code map

| File | Role |
|---|---|
| `code/market.py` | GBM (exact) and Heston (full-truncation Euler) path simulators under the real-world measure |
| `code/black_scholes.py` | BS price, delta, and gamma (torch, vectorised); Heston call price via Fourier inversion |
| `code/hedging.py` | **The single PnL engine** shared by every hedger; BS delta-hedge holdings |
| `code/risk.py` | mean–std, entropic, and CVaR risk measures; evaluation summary table |
| `code/model.py` | `HedgeNet`: an MLP shared across time. Inputs: time, log-moneyness, current holding [, vol]. Output: next hedge ratio |
| `code/train.py` | Adam with cosine LR decay; a fresh batch of 4,096 simulated paths at every iteration |
| `code/common.py` | Default market, fixed 20,000-path test set, save helpers |

### Conventions

- We are the **seller** of the option: PnL = premium + hedge gains − transaction costs − payoff.
- The hedge starts flat and is unwound at maturity (cash settlement), so costs are paid on both ends.
- Costs are proportional: `c · |Δholding| · S`, so `c = 0.002` means 20 bp.
- Market: S0 = K = 100, σ = 20%, μ = 5% (real-world drift), r = 0, T = 30 days, daily rebalancing (N = 30).
- Heston: v0 = θ = 0.04, κ = 1.5, ξ = 0.5, ρ = −0.7.
- Risk measures are monetary ("cash needed to make the position acceptable"); **lower is better**.
- All hedgers are evaluated on **the same 20,000 test paths** (fixed seed), never seen in training.

## 4. Results

### Steps 2–3: simulator check and delta baseline
- Monte Carlo price under Q is 2.2915 ± 0.0096, against the Black–Scholes price of 2.2871.
- The daily delta hedge has PnL mean 0.001 and std 0.359. The hedge is fair but imperfect.
- The PnL std scales as N^−0.48 in the number of rebalancing dates (theory: N^−0.5). Discrete
  rebalancing is what breaks the perfect hedge.
- Figures: `step3_delta_pnl_hist.png`, `step3_error_vs_rebalancing.png`.

### Step 5: frictionless, deep vs delta
| | std | CVaR95 | entropic(1) | mean–std(1) |
|---|---|---|---|---|
| delta | 0.359 | 0.836 | 0.0670 | 0.3580 |
| deep (trained on mean–std) | 0.363 | 0.844 | 0.0666 | 0.3607 |

The network **matches delta** (within 0.75%) and recovers the N(d1) curve, including its sharpening
toward expiry, without ever being told the formula (`step5_policy_vs_delta.png`).

### Step 6: transaction costs (entropic risk, λ = 1)
| cost | delta | deep | deep improvement | turnover delta → deep |
|---|---|---|---|---|
| 0 bp | 0.067 | 0.070 | −3.8% | 2.74 → 2.74 |
| 5 bp | 0.209 | 0.196 | 6.1% | 2.74 → 2.41 |
| 10 bp | 0.353 | 0.312 | 11.5% | 2.74 → 2.22 |
| 20 bp | 0.646 | 0.522 | 19.1% | 2.74 → 1.97 |
| 50 bp | 1.560 | 1.076 | 31.0% | 2.74 → 1.69 |
| 100 bp | 3.199 | 1.911 | 40.3% | 2.74 → 1.52 |

Delta ignores costs, so its turnover never changes. The network uses its current holding to damp its
trades, moving only part of the way toward delta (`step6_trade_size.png`). The gap widens steadily
with cost (`step6_cost_sweep.png`).

### Step 7: Heston (entropic risk / CVaR99)
| hedger | no costs | 20 bp costs |
|---|---|---|
| BS delta, constant vol | 0.172 / 1.98 | 0.767 / 2.81 |
| BS delta, current vol √v_t | 0.157 / 1.69 | 0.727 / 2.41 |
| minimum-variance delta | 0.175 / 1.70 | 0.758 / 2.45 |
| deep, vol hidden | 0.165 / 1.76 | 0.616 / 2.29 |
| deep, vol as state | **0.155 / 1.66** | **0.608 / 2.24** |

Hedging only with the stock leaves vol risk unhedged, so every PnL std is ≈ 0.52, against 0.36 under GBM.
Giving the network the variance helps, most visibly in the tail. With costs, both deep hedgers beat
every classical hedger by about 16%.

### Step 8: risk-measure ablation (20 bp costs)
| trained on → | mean–std(1) | entropic(1) | CVaR95 | CVaR99 |
|---|---|---|---|---|
| delta | 0.961 | 0.646 | 1.568 | 2.012 |
| mean–std(1) | **0.870** | 0.535 | 1.416 | 1.843 |
| entropic(1) | 0.901 | **0.522** | 1.473 | 1.886 |
| CVaR95 | 0.924 | 0.563 | **1.294** | **1.562** |

Each network is best on the measure it was trained for. CVaR training buys the thinnest loss tail at
the cost of a worse mean–std score. The choice of loss *is* the choice of strategy.

## 5. Known limitations / next steps

- The stochastic-vol benchmark is a minimum-variance delta that uses the BS vega as a proxy for ∂C/∂v.
  An exact Heston delta and vega (from the Fourier price) would be a stronger baseline.
- Under Heston only the stock is traded. Adding a variance swap or an ATM option as a second instrument
  (as Buehler et al. do) would make the market approximately complete again.
- The network learns partial adjustment toward delta rather than a sharp no-trade band, which is
  probably a smoothness bias of the MLP. Feeding the BS delta as an input, or using a band-shaped
  output layer, are natural things to try next.
- Only one option and one maturity are covered. The network must be retrained when the contract changes.
