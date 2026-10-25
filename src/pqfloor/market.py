"""Synthetic factor-model market and the cardinality-constrained mean-variance objective."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Market:
    """True monthly return moments of ``n`` assets."""

    mu: np.ndarray
    cov: np.ndarray

    def sample(self, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Estimate (mean, covariance) from ``T`` monthly Gaussian observations."""
        r = rng.multivariate_normal(self.mu, self.cov, size=T)
        return r.mean(0), np.cov(r, rowvar=False)


def make_market(n: int, rng: np.random.Generator, n_factors: int = 2) -> Market:
    """Random two-factor market: monthly means ~0.5 %, volatilities ~4-8 %."""
    B = rng.normal(0.0, 0.03, size=(n, n_factors)) + 0.03
    idio = rng.uniform(0.02, 0.05, size=n)
    cov = B @ B.T + np.diag(idio**2)
    mu = 0.005 + 0.004 * rng.standard_normal(n)
    return Market(mu=mu, cov=cov)


def bit_matrix(n: int) -> np.ndarray:
    """All 2**n bitstrings as a (2**n, n) 0/1 matrix; qubit 0 is the most significant bit."""
    idx = np.arange(2**n)[:, None]
    return ((idx >> (n - 1 - np.arange(n))) & 1).astype(float)


def objective(X: np.ndarray, mu: np.ndarray, cov: np.ndarray, k: int, q: float = 5.0) -> np.ndarray:
    """Mean-variance cost ``q w'Cw - mu'w`` with equal weights ``w = x / k`` (lower is better)."""
    W = X / k
    return q * np.einsum("ij,jk,ik->i", W, cov, W) - W @ mu
