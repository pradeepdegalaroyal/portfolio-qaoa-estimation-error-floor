import numpy as np

from pqfloor.market import bit_matrix, make_market, objective
from pqfloor.solvers import FullQAOA, XYQAOA, cnot_full, cnot_xy, survival


def test_objective_matches_manual():
    rng = np.random.default_rng(0)
    m = make_market(6, rng)
    X = bit_matrix(6)
    x = X[0b110100]
    w = x / 3
    assert np.isclose(objective(x[None], m.mu, m.cov, 3)[0], 5 * w @ m.cov @ w - w @ m.mu)


def test_xy_mixer_stays_in_feasible_subspace_and_normalised():
    xy = XYQAOA(6, 3)
    assert len(xy.feas) == 20
    c = np.random.default_rng(1).random(20)
    psi = xy.state(c, np.array([0.7, 0.4, 0.3, 0.9]))
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_full_qaoa_normalised():
    f = FullQAOA(5)
    c = np.random.default_rng(2).random(32)
    assert np.isclose(np.linalg.norm(f.state(c, np.array([0.5, 0.3]))), 1.0)


def test_noise_model_monotone_and_gate_counts():
    assert survival(0.0, 100) == 1.0
    assert survival(0.01, 100) < survival(0.001, 100)
    assert cnot_full(12, 2) == 2 * cnot_full(12, 1)
    assert cnot_xy(12, 5, 2) > cnot_xy(12, 5, 1)
