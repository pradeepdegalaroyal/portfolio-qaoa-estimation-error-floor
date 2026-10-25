"""One study cell: a market draw, an estimated instance, and every solver's portfolio."""
from __future__ import annotations

import numpy as np

from .market import Market, objective
from .solvers import (FullQAOA, XYQAOA, cnot_full, cnot_xy, greedy, random_best_of,
                      simulated_annealing, survival, train)


def _bits_to_int(x: np.ndarray) -> int:
    return int("".join(str(int(b)) for b in x), 2)


def run_cell(mkt: Market, n: int, k: int, T: int, p2q_list: list[float], shots: int, reps: int,
             seed: int, X: np.ndarray, xy: XYQAOA, full: FullQAOA) -> list[dict]:
    """Solve one estimated instance with all methods; score in-sample and out-of-sample.

    Metrics are normalised by the cost range over the feasible set: ``gap_in`` uses the
    estimated parameters, ``regret_oos`` uses the true parameters.
    """
    rng = np.random.default_rng(seed)
    mu_h, cov_h = mkt.sample(T, rng)
    feas = xy.feas
    Xf = X[feas]
    f_hat = objective(X, mu_h, cov_h, k)
    f_true = objective(X, mkt.mu, mkt.cov, k)
    fh, ft = f_hat[feas], f_true[feas]
    lo_h, rg_h = fh.min(), np.ptp(fh)
    lo_t, rg_t = ft.min(), np.ptp(ft)
    pos = {int(s): i for i, s in enumerate(feas)}

    def score(method: str, fi: int, p2q: float, ncx: int, rep: int) -> dict:
        return dict(seed=seed, T=T, method=method, p2q=p2q, rep=rep, cnots=ncx,
                    gap_in=(fh[fi] - lo_h) / rg_h, regret_oos=(ft[fi] - lo_t) / rg_t)

    rows: list[dict] = []
    cost_fn = lambda Y: objective(Y, mu_h, cov_h, k)  # noqa: E731
    rows.append(score("exact", int(np.argmin(fh)), 0.0, 0, 0))
    rows.append(score("greedy", pos[_bits_to_int(greedy(cost_fn, n, k))], 0.0, 0, 0))
    rows.append(score("anneal", pos[_bits_to_int(simulated_annealing(cost_fn, n, k, rng))], 0.0, 0, 0))

    # Costs are rescaled to unit range on the feasible set; the penalty weight is 1.5 per unit^2.
    wt = X.sum(1)
    c_pen = (f_hat - lo_h) / rg_h + 1.5 * (wt - k) ** 2
    c_xy = (fh - lo_h) / rg_h
    feas_mask = np.zeros(2**n, bool)
    feas_mask[feas] = True

    def energy(sim, c):
        def f(t):
            psi = sim.state(c, t)
            return float(np.real(np.vdot(psi, c * psi)))
        return f

    for p in (1, 2):
        P_pen = np.abs(full.state(c_pen, train(energy(full, c_pen), p, rng, scale=2.0))) ** 2
        P_xy = np.abs(xy.state(c_xy, train(energy(xy, c_xy), p, rng, scale=2.0))) ** 2
        n_pen, n_xy = cnot_full(n, p), cnot_xy(n, k, p)
        for p2q in p2q_list:
            lam_pen, lam_xy = 1 - survival(p2q, n_pen), 1 - survival(p2q, n_xy)
            Pn_pen = (1 - lam_pen) * P_pen + lam_pen / 2**n
            Pn_xy = (1 - lam_xy) * P_xy + lam_xy / len(feas)
            for rep in range(reps):
                s = rng.choice(2**n, size=shots, p=Pn_pen / Pn_pen.sum())
                s = s[feas_mask[s]]
                if len(s) == 0:  # no feasible sample: fall back to a random feasible portfolio
                    fi = int(rng.integers(len(feas)))
                else:
                    fi = pos[int(s[np.argmin(f_hat[s])])]
                rows.append(score(f"qaoa_penalty_p{p}", fi, p2q, n_pen, rep))
                s = rng.choice(len(feas), size=shots, p=Pn_xy / Pn_xy.sum())
                rows.append(score(f"qaoa_xy_p{p}", int(s[np.argmin(fh[s])]), p2q, n_xy, rep))
    for rep in range(reps):
        rows.append(score("random_shots", random_best_of(Xf, fh, shots, rng), 0.0, 0, rep))
    return rows
