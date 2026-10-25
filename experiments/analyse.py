"""Aggregate raw rows into tables and make the three figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VIR = plt.get_cmap("viridis")
QX = ["qaoa_penalty_p1", "qaoa_penalty_p2", "qaoa_xy_p1", "qaoa_xy_p2"]
CLASSICAL = ["exact", "greedy", "anneal", "random_shots"]


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over shot repetitions; add paired regret difference to the exact in-sample optimum."""
    g = df.groupby(["seed", "T", "method", "p2q", "cnots"], as_index=False)[["gap_in", "regret_oos"]].mean()
    ex = g[g.method == "exact"].set_index(["seed", "T"])["regret_oos"]
    g["delta_oos"] = g.regret_oos.values - ex.reindex(list(zip(g.seed, g["T"]))).values
    return g


def _agg(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    s = d.groupby(keys)[["gap_in", "regret_oos", "delta_oos"]].agg(["mean", "std"])
    s.columns = ["_".join(c) for c in s.columns]
    return s.reset_index()


def analyse(df: pd.DataFrame, out: Path, figdir: Path, meta: dict) -> None:
    """Write summary CSVs to ``out`` and figures to ``figdir``."""
    figdir.mkdir(exist_ok=True)
    g = per_seed(df)
    ps = sorted(meta["p2q"])
    ref_p = ps[len(ps) // 2]
    base = g[g.method.isin(CLASSICAL)]
    qd = g[g.method.isin(QX)]
    tab = pd.concat([base[base.p2q == 0.0], qd[qd.p2q == ref_p]])
    _agg(tab, ["T", "method"]).to_csv(out / "summary_ref_noise.csv", index=False)
    _agg(qd, ["T", "method", "p2q"]).to_csv(out / "summary_noise_sweep.csv", index=False)
    Ts = meta["T"]
    sh = qd[qd.p2q == ref_p].groupby(["T", "method"])[["delta_oos", "regret_oos"]].mean()
    sh["optimisation_share_of_regret"] = sh.delta_oos / sh.regret_oos
    sh.reset_index().to_csv(out / "optimisation_share.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m in zip(axs, ["qaoa_penalty_p2", "qaoa_xy_p2"]):
        d = qd[qd.method == m].groupby(["p2q", "T"]).delta_oos.mean().unstack()
        im = ax.imshow(d.values, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(d.columns)), d.columns)
        ax.set_yticks(range(len(d.index)), d.index)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                ax.text(j, i, f"{d.values[i, j]:.3f}", ha="center", va="center",
                        color="w" if d.values[i, j] < np.nanmean(d.values) else "k", fontsize=8)
        ax.set_title(m.replace("_", " "))
        ax.set_xlabel("history length T (monthly observations)")
        fig.colorbar(im, ax=ax, label="extra out-of-sample regret vs exact")
    axs[0].set_ylabel("two-qubit gate error rate")
    fig.suptitle("Optimisation error that survives estimation error (mean over seeds)")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_heatmap_extra_regret.png", dpi=200)
    plt.close(fig)

    T0 = Ts[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ms = ["exact", "greedy", "random_shots", "qaoa_penalty_p2", "qaoa_xy_p2"]
    for i, m in enumerate(ms):
        d = g[(g.method == m) & (g["T"] == T0) & (g.p2q == (ref_p if m.startswith("qaoa") else 0.0))]
        v = np.sort(d.regret_oos.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=VIR(i / (len(ms) - 1)),
                label=m.replace("_", " "), lw=2)
    ax.set_xlabel("out-of-sample regret (fraction of feasible cost range)")
    ax.set_ylabel("empirical CDF over market seeds")
    ax.set_title(f"Out-of-sample regret, T = {T0} months, gate error {ref_p}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_ecdf_oos_regret.png", dpi=200)
    plt.close(fig)

    T1 = Ts[len(Ts) // 2]
    d = pd.concat([base[(base["T"] == T1) & (base.p2q == 0.0)], qd[(qd["T"] == T1) & (qd.p2q == ref_p)]])
    a = d.groupby("method")[["cnots", "gap_in", "regret_oos"]].mean()
    a.to_csv(out / "pareto_points.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(a.cnots, a.gap_in, c=a.regret_oos, cmap="viridis", s=90, edgecolor="k")
    for m, r in a.iterrows():
        ax.annotate(m.replace("_", " "), (r.cnots, r.gap_in), fontsize=7, xytext=(4, 4), textcoords="offset points")
    front, best = [], np.inf
    for _, r in a.sort_values("cnots").iterrows():
        if r.gap_in < best:
            front.append((r.cnots, r.gap_in))
            best = r.gap_in
    ax.step(*zip(*front), where="post", color="k", ls="--", label="Pareto front")
    fig.colorbar(sc, label="out-of-sample regret (fraction of range)")
    ax.set_xlabel("two-qubit gates per shot (0 = classical)")
    ax.set_ylabel("in-sample optimality gap (fraction of range)")
    ax.set_title(f"Cost vs quality, T = {T1}, gate error {ref_p}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig3_pareto_gates_vs_gap.png", dpi=200)
    plt.close(fig)
