"""Create compact reviewer-facing summaries from benchmark result files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def interquartile_range(values: pd.Series) -> float:
    """Return the 75th-percentile minus the 25th-percentile range."""
    return float(values.quantile(0.75) - values.quantile(0.25))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    return parser.parse_args()


def summarize_runs(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["topology", "algorithm"], as_index=False)
        .agg(
            runs=("seed", "count"),
            successes=("success", "sum"),
            success_rate=("success", "mean"),
            median_combined_error=("combined_error", "median"),
            iqr_combined_error=("combined_error", interquartile_range),
            best_combined_error=("combined_error", "min"),
            median_flow_norm=("flow_norm", "median"),
            median_conc_norm=("conc_norm", "median"),
            median_flow_max_abs_error=("flow_max_abs_error", "median"),
            median_conc_max_abs_error=("conc_max_abs_error", "median"),
            median_n_eval=("n_eval", "median"),
            iqr_n_eval=("n_eval", interquartile_range),
            median_runtime_s=("runtime_s", "median"),
            iqr_runtime_s=("runtime_s", interquartile_range),
            median_normal_exit_count=("normal_exit_count", "median"),
            median_budget_exhausted_count=("budget_exhausted_count", "median"),
        )
        .sort_values(["topology", "median_combined_error", "algorithm"])
    )


def summarize_restarts(restarts: pd.DataFrame) -> pd.DataFrame:
    if restarts.empty:
        return pd.DataFrame()
    return (
        restarts.groupby(["topology", "exit_status"], as_index=False)
        .agg(
            restarts=("start_index", "count"),
            median_n_eval=("n_eval", "median"),
            median_combined_error=("combined_error", "median"),
        )
        .sort_values(["topology", "exit_status"])
    )


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.results_dir / "raw_runs.csv")
    if raw.empty:
        raise ValueError("raw_runs.csv contains no benchmark records")
    summary = summarize_runs(raw)
    summary.to_csv(args.results_dir / "analysis_by_topology_algorithm.csv", index=False)

    winners = raw.loc[raw.groupby("topology")["combined_error"].idxmin()].sort_values("topology")
    winners.to_csv(args.results_dir / "analysis_winners_by_topology.csv", index=False)

    restart_path = args.results_dir / "raw_slsqp_restarts.csv"
    if restart_path.exists():
        restart_summary = summarize_restarts(pd.read_csv(restart_path))
        restart_summary.to_csv(args.results_dir / "analysis_slsqp_restarts.csv", index=False)

    print("\nTopology/algorithm summary")
    print(summary.to_string(index=False))
    print("\nBest run per topology")
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
