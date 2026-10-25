# Research gap

## State of the art (from the supplied literature)

The brief labels every paper "other", so the roles below are my own reading: *recent* (2026-27), *landmark* (highly cited, accepted baseline), *review* (survey). Quantum Approximate Optimization Algorithm (QAOA) work targets the Noisy Intermediate-Scale Quantum (NISQ) regime. Blekos et al., 2024 (10.1016/j.physrep.2024.03.002) survey QAOA variants. Wołk et al., 2027 (10.1016/j.ins.2026.124187) note that standard QAOA is poorly suited to constrained problems. Restricting the mixer to the feasible subspace fixes this but enlarges the circuit and lowers noise robustness. Wang et al., 2027 (10.15302/frontphys.2027.023301) ask whether variational algorithms stay viable as hardware moves towards fault tolerance. Alzboun and Abed-alguni, 2027 (10.1016/j.ins.2026.124009) ask why variational circuits often fail to beat classical methods.

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Blekos et al. (10.1016/j.physrep.2024.03.002) | 2024 | review | Surveys QAOA and variants | Error in the problem inputs themselves |
| Wołk et al. (10.1016/j.ins.2026.124187) | 2027 | recent | Fewer-gate constrained hypercube mixer | Whether a better optimum matters for the end use |
| Wang et al. (10.15302/frontphys.2027.023301) | 2027 | review | Variational algorithms towards fault tolerance | Application-level cost of sub-optimality |
| Alzboun and Abed-alguni (10.1016/j.ins.2026.124009) | 2027 | recent | Explains the quantum gap in reinforcement learning | Optimisation tasks with noisy inputs |
| Peral-García et al. (10.1016/j.cosrev.2024.100619) | 2024 | review | Systematic review of quantum machine learning | Quantitative noise-versus-data-error trade-off |
| Huynh et al. (10.1016/j.cosrev.2026.101072) | 2027 | review | Surveys quantum-inspired machine learning | Subspace simulation of QAOA for finance |
| Yang et al. (10.1109/COMST.2023.3254481) | 2023 | review | Survey of issues in quantum computing | Application-level error budgets |
| Caleffi et al. (10.1016/j.comnet.2024.110672) | 2024 | review | Distributed quantum computing survey | Single-device QAOA noise budgets |
| Zobrist et al. (10.1038/s41586-024-08449-y) | 2025 | landmark | Error correction below threshold | Near-term variational workloads |
| Sivak et al. (10.1038/s41586-023-05782-6) | 2023 | landmark | Real-time error correction beyond break-even | Variational optimisation |
| Evered et al. (10.1038/s41586-023-06481-y) | 2023 | landmark | High-fidelity parallel entangling gates | Application-level impact of gate error |
| Kouskiya and Oskay (10.1016/j.cma.2026.119443) | 2027 | adjacent | Variational algorithm for finite-element mechanics | Input-uncertainty floor |

## Gap

Every QAOA study scores a solution against the optimum of the *given* instance. In finance the instance is itself an estimate: the mean and covariance of asset returns come from a finite history, so the exact optimum of the estimated problem is a noisy proxy for the best portfolio. The literature above never asks how much of QAOA's optimisation error is invisible once this input error is counted, nor how that interacts with the extra gates of constrained mixers.

**Hypothesis (falsifiable).** For a cardinality-constrained mean-variance selection (12 assets, choose 5), the out-of-sample penalty of a noisy QAOA portfolio relative to the exact in-sample optimum shrinks as the return history gets shorter, and for histories of 24 months it is no larger than 0.02 of the cost range, for both a penalty QAOA and a Dicke-state XY-mixer QAOA, up to a two-qubit gate error of 0.003.

**Success metric.** Paired difference in out-of-sample regret (true-parameter cost, normalised by the feasible range) between a method and the exact in-sample optimum, mean ± standard deviation over market seeds. The hypothesis is refuted if the difference at 24 months exceeds 0.02 at that noise level, or does not fall as history shortens.

**Scale.** Statevector simulation, 12 qubits (4096 amplitudes, or 792 in the subspace), six random markets. Noise is a global depolarising approximation driven by two-qubit gate counts, not a device model.

**Novelty in one sentence.** This study measures QAOA solution quality against the out-of-sample value of the portfolio rather than the in-sample optimum, and compares penalty and constrained-mixer circuits at equal shots under a gate-count noise model.
