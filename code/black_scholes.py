"""Step 2 — analytical benchmarks.

Black-Scholes price / delta / gamma (torch, so they vectorise over whole path
tensors) and a semi-analytic Heston call price (numpy/scipy) used as the option
premium in the Heston experiment.
"""
import math

import numpy as np
import torch
from scipy.integrate import quad


def _d1_d2(S, K, tau, sigma, r):
    S = torch.as_tensor(S, dtype=torch.float32)
    tau = torch.as_tensor(tau, dtype=torch.float32).clamp(min=1e-10)
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    vol_sqrt = sigma * tau.sqrt()
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / vol_sqrt
    return d1, d1 - vol_sqrt, tau


def bs_call_price(S, K, tau, sigma, r=0.0):
    d1, d2, tau = _d1_d2(S, K, tau, sigma, r)
    N = torch.special.ndtr
    return S * N(d1) - K * torch.exp(-r * tau) * N(d2)


def bs_call_delta(S, K, tau, sigma, r=0.0):
    d1, _, _ = _d1_d2(S, K, tau, sigma, r)
    return torch.special.ndtr(d1)


def bs_call_gamma(S, K, tau, sigma, r=0.0):
    d1, _, tau = _d1_d2(S, K, tau, sigma, r)
    pdf = torch.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    return pdf / (torch.as_tensor(S) * sigma * tau.sqrt())


# ---------------------------------------------------------------------------
# Heston call price (Heston 1993, in the "little trap" form of Albrecher et al.
# 2007, which avoids branch-cut discontinuities in the complex logarithm).
# ---------------------------------------------------------------------------
def _heston_cf(u, S0, T, r, v0, kappa, theta, xi, rho):
    """Characteristic function of ln S_T under the risk-neutral measure."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (kappa * theta / xi**2) * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = ((b - d) / xi**2) * (1 - e) / (1 - g * e)
    return np.exp(iu * (np.log(S0) + r * T) + C + D * v0)


def heston_call_price(S0, K, T, r, v0, kappa, theta, xi, rho):
    args = (S0, T, r, v0, kappa, theta, xi, rho)
    lnK = np.log(K)
    fwd = S0 * np.exp(r * T)  # = cf(-i)

    def p1_integrand(u):
        return (np.exp(-1j * u * lnK) * _heston_cf(u - 1j, *args) / (1j * u * fwd)).real

    def p2_integrand(u):
        return (np.exp(-1j * u * lnK) * _heston_cf(u, *args) / (1j * u)).real

    P1 = 0.5 + quad(p1_integrand, 1e-8, 200, limit=500)[0] / np.pi
    P2 = 0.5 + quad(p2_integrand, 1e-8, 200, limit=500)[0] / np.pi
    return S0 * P1 - K * np.exp(-r * T) * P2
