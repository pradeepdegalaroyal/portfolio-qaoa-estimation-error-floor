# Method

**Problem.** Choose k = 5 of n = 12 assets with equal weights w = x/k, minimising the mean-variance cost q·w'Σw − μ'w with q = 5. Returns follow a two-factor Gaussian model (monthly means about 0.5 %, volatilities 4-8 %). For each of 6 market seeds and each history length T in {24, 60, 120, 240, 600} months, the mean and covariance are estimated from T draws. Solvers see only the estimates; the true moments score the result.

**Metrics** (both normalised by the cost range over the feasible set): *in-sample gap* (estimated parameters, 0 = exact optimum) and *out-of-sample regret* (true parameters, 0 = true best portfolio). The headline quantity is the paired difference of out-of-sample regret to the exact in-sample optimum.

**Methods.**
- Classical baselines: exact enumeration (792 feasible sets), greedy forward selection, swap-move simulated annealing (400 steps), and uniform random feasible sampling with the same shot budget as the quantum methods (25 shots).
- Standard quantum baseline: penalty Quantum Approximate Optimization Algorithm (QAOA) on the full 4096-dimensional space, transverse-field mixer, penalty weight 1.5 on the cost scaled to unit range. Infeasible shots are discarded.
- Constrained variant: QAOA on the Hamming-weight-5 subspace with Dicke initial state and a ring XY mixer.
- Parameters are trained noiselessly (COBYLA, 3 restarts, 120 iterations) and evaluated under noise.

**Noise model.** Two-qubit depolarising error p per controlled-NOT (CNOT) gate; the output distribution is (1−λ)·ideal + λ·uniform with λ = 1 − (1−p)^G, where G is the CNOT count per shot (penalty: 2 per asset pair per layer; XY: an assumed 2·n·k for Dicke preparation plus the same cost layer and 4 per ring edge per layer). The uniform part is over all bitstrings for the penalty circuit and over feasible ones for the XY circuit. The Dicke gate count is an assumption, stated as such. The sweep is p in {0, 0.001, 0.003, 0.01}. A shot is taken from the noisy distribution; the best feasible shot by estimated cost is the answer. If no shot is feasible, a random feasible portfolio is returned.

**Ablation.** Depth p = 1 versus p = 2 for both circuits.

**Seeds and repetition.** 6 market seeds; 5 independent shot draws per cell, averaged within a seed. Statistics are mean ± standard deviation across seeds.
