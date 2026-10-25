"""Reproduce every result: ``python experiments/run_all.py`` (``--quick`` for a smoke run)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from pqfloor.market import bit_matrix, make_market  # noqa: E402
from pqfloor.solvers import FullQAOA, XYQAOA  # noqa: E402
from pqfloor.study import run_cell  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        n, k, seeds, Ts, noise, shots, reps = 8, 3, 2, [24, 120], [0.0, 0.003], 15, 3
    else:
        n, k, seeds, Ts, noise, shots, reps = 12, 5, 6, [24, 60, 120, 240, 600], [0.0, 0.001, 0.003, 0.01], 25, 5
    out = ROOT / ("results_quick" if a.quick else "results")
    out.mkdir(exist_ok=True)
    X, xy, full = bit_matrix(n), XYQAOA(n, k), FullQAOA(n)
    t0, rows = time.time(), []
    for s in range(seeds):
        mkt = make_market(n, np.random.default_rng(1000 + s))
        for T in Ts:
            rows += run_cell(mkt, n, k, T, noise, shots, reps, seed=10 * s + T, X=X, xy=xy, full=full)
        print(f"seed {s + 1}/{seeds} done, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw.csv", index=False)
    meta = dict(n=n, k=k, seeds=seeds, T=Ts, p2q=noise, shots=shots, reps=reps,
                runtime_s=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    from analyse import analyse
    analyse(df, out, ROOT / ("figures_quick" if a.quick else "figures"), meta)


if __name__ == "__main__":
    main()
