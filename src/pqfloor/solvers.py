"""Classical baselines and the two QAOA variants (penalty and Dicke/XY-mixer)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------- classical baselines


def greedy(cost_fn, n: int, k: int) -> np.ndarray:
    """Forward selection: add the asset that lowers the cost most, k times."""
    x = np.zeros(n)
    for _ in range(k):
        best, bi = np.inf, -1
        for i in np.flatnonzero(x == 0):
            y = x.copy()
            y[i] = 1
            c = cost_fn(y[None])[0] if y.sum() == k else _partial(cost_fn, y, k)
            if c < best:
                best, bi = c, i
        x[bi] = 1
    return x


def _partial(cost_fn, y: np.ndarray, k: int) -> float:
    """Cost of a partial selection with weights 1/k (cash is the remainder)."""
    return float(cost_fn(y[None])[0])


def simulated_annealing(cost_fn, n: int, k: int, rng: np.random.Generator, steps: int = 400) -> np.ndarray:
    """Swap-move annealing on the cardinality-k set."""
    x = np.zeros(n)
    x[rng.choice(n, k, replace=False)] = 1
    c = cost_fn(x[None])[0]
    best, bc = x.copy(), c
    scale = abs(c) + 1e-9
    for t in range(steps):
        T = scale * 0.1 * (1 - t / steps) + 1e-12
        i = rng.choice(np.flatnonzero(x == 1))
        j = rng.choice(np.flatnonzero(x == 0))
        y = x.copy()
        y[i], y[j] = 0, 1
        cy = cost_fn(y[None])[0]
        if cy < c or rng.random() < np.exp(-(cy - c) / T):
            x, c = y, cy
            if c < bc:
                best, bc = x.copy(), c
    return best


def random_best_of(Xfeas: np.ndarray, cost_feas: np.ndarray, shots: int, rng: np.random.Generator) -> int:
    """Best of ``shots`` uniform feasible samples (equal-shot-budget baseline)."""
    idx = rng.integers(0, len(Xfeas), size=shots)
    return int(idx[np.argmin(cost_feas[idx])])


# ---------------------------------------------------------------- quantum simulators


class FullQAOA:
    """Penalty QAOA on the full 2**n space with the transverse-field mixer."""

    def __init__(self, n: int):
        self.n = n

    def _mix(self, psi: np.ndarray, beta: float) -> np.ndarray:
        c, s = np.cos(beta), -1j * np.sin(beta)
        U = np.array([[c, s], [s, c]])
        psi = psi.reshape((2,) * self.n)
        for ax in range(self.n):
            psi = np.moveaxis(np.tensordot(U, psi, axes=(1, ax)), 0, ax)
        return psi.reshape(-1)

    def state(self, cost: np.ndarray, params: np.ndarray) -> np.ndarray:
        psi = np.full(2**self.n, 2 ** (-self.n / 2), dtype=complex)
        for g, b in zip(params[0::2], params[1::2]):
            psi = self._mix(np.exp(-1j * g * cost) * psi, b)
        return psi


class XYQAOA:
    """QAOA on the Hamming-weight-k subspace with a ring XY mixer and Dicke initial state."""

    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        pop = np.array([bin(i).count("1") for i in range(2**n)])
        self.feas = np.flatnonzero(pop == k)
        pos = {int(s): i for i, s in enumerate(self.feas)}
        H = np.zeros((len(self.feas), len(self.feas)))
        for s, a in pos.items():
            for q in range(n):
                i, j = q, (q + 1) % n
                bi, bj = (s >> (n - 1 - i)) & 1, (s >> (n - 1 - j)) & 1
                if bi != bj:
                    H[a, pos[s ^ (1 << (n - 1 - i)) ^ (1 << (n - 1 - j))]] += 1.0
        self.lam, self.V = np.linalg.eigh(H)

    def state(self, cost_feas: np.ndarray, params: np.ndarray) -> np.ndarray:
        d = len(self.feas)
        psi = np.full(d, d ** -0.5, dtype=complex)
        for g, b in zip(params[0::2], params[1::2]):
            psi = np.exp(-1j * g * cost_feas) * psi
            psi = self.V @ (np.exp(-1j * b * self.lam) * (self.V.T @ psi))
        return psi


def train(energy_fn, p: int, rng: np.random.Generator, restarts: int = 3, scale: float = 1.0) -> np.ndarray:
    """Noiseless COBYLA training from random starts; ``scale`` normalises the cost angle."""
    best, bx = np.inf, None
    for _ in range(restarts):
        x0 = rng.uniform(0, 1, 2 * p) * np.tile([scale, 1.0], p)
        r = minimize(energy_fn, x0, method="COBYLA", options={"maxiter": 120, "rhobeg": 0.3})
        if r.fun < best:
            best, bx = r.fun, r.x
    return bx


# ---------------------------------------------------------------- gate-count and noise model


def cnot_full(n: int, p: int) -> int:
    """CNOTs per shot: one RZZ (2 CNOTs) per asset pair per layer; RX mixer is single-qubit."""
    return p * 2 * (n * (n - 1) // 2)


def cnot_xy(n: int, k: int, p: int, dicke_per_nk: float = 2.0) -> int:
    """CNOTs per shot: assumed ``dicke_per_nk * n * k`` for Dicke preparation, plus per layer
    the dense RZZ cost (2 CNOTs per pair) and a ring XY mixer (4 CNOTs per ring edge)."""
    return int(dicke_per_nk * n * k) + p * (2 * (n * (n - 1) // 2) + 4 * n)


def survival(p2q: float, n_cnot: int) -> float:
    """Probability that no CNOT fails under independent two-qubit depolarising error ``p2q``."""
    return (1.0 - p2q) ** n_cnot
