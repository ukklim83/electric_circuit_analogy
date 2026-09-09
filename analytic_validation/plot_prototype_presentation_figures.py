"""Create presentation figures from prototype curvature-analysis outputs.

Run from the analytic_validation directory after
prototype_joint_convexity_analysis.py and prototype_directional_confirmation.py
have generated their output files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).with_name("outputs") / "prototype_joint_convexity"
FIGURE_DIR = ROOT / "presentation_figures"
NEGATIVE_TOLERANCE = 1.0e-3


def _load_hessian_rows() -> list[dict[str, float]]:
    with (ROOT / "prototype_hessian_witnesses.csv").open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _load_precision_data() -> tuple[dict, list[dict[str, float]]]:
    confirmation = json.loads((ROOT / "prototype_directional_confirmation.json").read_text(encoding="utf-8"))
    with (ROOT / "prototype_jensen_checks.csv").open(newline="", encoding="utf-8") as handle:
        double_precision = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return confirmation, double_precision


def _save(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=400, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def make_hessian_screening_figure(rows: list[dict[str, float]]) -> None:
    # This full-width source figure is intended to be reduced by approximately
    # 50% on insertion into a two-column manuscript.  Use >=24 pt at source so
    # the final rendered text is >=12 pt.
    font_family = "Times New Roman"
    title_size, axis_size, tick_size, note_size, legend_size = 26, 24, 24, 24, 24
    objective = np.array([row["objective"] for row in rows])
    lambda_min = np.array([row["lambda_min"] for row in rows])
    samples = np.array([int(row["sample"]) for row in rows])
    stable = np.array([bool(int(row.get("stable_negative_curvature", 1))) for row in rows])
    negative = lambda_min < -NEGATIVE_TOLERANCE
    negative = negative & stable
    nonnegative = ~negative & stable
    fraction = negative.mean()

    fig, (ax_scatter, ax_distribution) = plt.subplots(
        1, 2, figsize=(13.2, 6.3), gridspec_kw={"width_ratios": [1.60, 1.0], "wspace": 0.42}
    )

    stable_lambda = lambda_min[stable]
    ax_scatter.axhspan(stable_lambda.min() * 1.08, 0.0, color="#fbe4e4", zorder=0, label="negative-curvature region")
    ax_scatter.axhline(0.0, color="black", linewidth=1.1)
    if nonnegative.any():
        ax_scatter.scatter(np.log10(objective[nonnegative]), lambda_min[nonnegative], s=42, color="#7a7a7a", alpha=0.85, label="non-negative samples")
    ax_scatter.scatter(np.log10(objective[negative]), lambda_min[negative], s=48, color="#d95f02", alpha=0.9, label=r"$\lambda_{\min}<-10^{-3}$")
    ax_scatter.set_xlabel(r"$\log_{10} J_{\mathrm{proto}}(\mathbf{z})$", fontsize=axis_size, fontname=font_family)
    ax_scatter.set_ylabel(r"$\lambda_{\min}[\nabla^2 J_{\mathrm{proto}}(\mathbf{z})]$", fontsize=axis_size, fontname=font_family)
    ax_scatter.grid(axis="x", alpha=0.18)
    replicate_count = len({int(row.get("replicate", 1)) for row in rows})

    jitter = np.random.default_rng(20260809).normal(0.0, 0.045, size=stable_lambda.size)
    colors = np.where(negative[stable], "#d95f02", "#7a7a7a")
    ax_distribution.axhspan(stable_lambda.min() * 1.08, 0.0, color="#fbe4e4", zorder=0)
    ax_distribution.axhline(0.0, color="black", linewidth=1.1)
    ax_distribution.boxplot(stable_lambda, positions=[0], widths=0.34, vert=True, showfliers=False, patch_artist=True, boxprops={"facecolor": "#d9e6f2", "edgecolor": "#3a3a3a"}, medianprops={"color": "#1f4e79", "linewidth": 1.7})
    ax_distribution.scatter(jitter, stable_lambda, c=colors, s=36, alpha=0.9, zorder=3)
    ax_distribution.set_xticks([0])
    ax_distribution.set_xticklabels(["interior Sobol samples"], fontsize=tick_size, fontname=font_family)
    ax_distribution.set_ylabel(r"$\lambda_{\min}[\nabla^2 J_{\mathrm{proto}}]$", fontsize=axis_size, fontname=font_family)
    ax_distribution.grid(axis="y", alpha=0.18)

    for axis in (ax_scatter, ax_distribution):
        axis.tick_params(axis="both", labelsize=tick_size, width=1.1, length=5)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontname(font_family)

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.96)
    _save(fig, "figure_1_hessian_screening")


def make_replicate_figure(rows: list[dict[str, float]]) -> None:
    if "replicate" not in rows[0]:
        return
    replicate_ids = sorted({int(row["replicate"]) for row in rows})
    if len(replicate_ids) < 2:
        return
    grouped = [[row for row in rows if int(row["replicate"]) == replicate] for replicate in replicate_ids]
    all_lambda = np.array([row["lambda_min"] for row in rows if int(row.get("stable_negative_curvature", 1))])
    y_lower = min(all_lambda.min() * 1.08, -0.1)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), sharex=True, sharey=True)
    fig.suptitle("Independent scrambled-Sobol Hessian screenings", fontsize=16, fontweight="bold", y=1.01)
    for axis, replicate, replicate_rows in zip(axes.ravel(), replicate_ids, grouped):
        replicate_rows = [row for row in replicate_rows if int(row.get("stable_negative_curvature", 1))]
        lambdas = np.array([row["lambda_min"] for row in replicate_rows])
        objective = np.array([row["objective"] for row in replicate_rows])
        selected = np.array([int(row.get("is_selected_witness", 0)) for row in replicate_rows], dtype=bool)
        witness = int(np.flatnonzero(selected)[0]) if selected.any() else int(np.argmin(lambdas))
        negative = lambdas < -NEGATIVE_TOLERANCE
        axis.axhspan(y_lower, 0.0, color="#fbe4e4", zorder=0)
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.scatter(np.log10(objective[negative]), lambdas[negative], s=18, color="#d95f02", alpha=0.78)
        if (~negative).any():
            axis.scatter(np.log10(objective[~negative]), lambdas[~negative], s=18, color="#7a7a7a", alpha=0.78)
        axis.scatter(np.log10(objective[witness]), lambdas[witness], s=150, marker="*", color="#b2182b", edgecolor="black", linewidth=0.55, zorder=3)
        axis.set_title(f"replicate {replicate}: stable {len(replicate_rows)}/256; min = {lambdas.min():.3f}", loc="left", fontsize=11, fontweight="bold")
        axis.set_xlabel(r"$\log_{10}J_{\mathrm{proto}}(\mathbf{z})$")
        axis.grid(axis="x", alpha=0.15)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\lambda_{\min}[\nabla^2 J_{\mathrm{proto}}]$")
    fig.tight_layout()
    _save(fig, "figure_3_independent_sobol_replicates")


def _stable_spectra(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return objective values and full ordered spectra from stable points."""
    stable_rows = [row for row in rows if int(row.get("stable_negative_curvature", 1))]
    if not stable_rows or "eigenvalue_1" not in stable_rows[0]:
        raise ValueError("Full ordered spectra are absent. Re-run prototype_joint_convexity_analysis.py.")
    objective = np.array([row["objective"] for row in stable_rows])
    spectra = np.array([[row[f"eigenvalue_{rank}"] for rank in range(1, 9)] for row in stable_rows])
    return objective, spectra


def make_rank_distribution_figure(rows: list[dict[str, float]]) -> None:
    """Show the magnitude and sign of all eight ordered curvature directions."""
    objective, spectra = _stable_spectra(rows)
    fig, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(13.2, 6.5), gridspec_kw={"width_ratios": [1.65, 1.0]})
    fig.suptitle("Ordered Hessian-eigenvalue spectra across stable interior samples", fontsize=16, fontweight="bold", y=1.02)
    rng = np.random.default_rng(20260809)
    negative_fraction = np.mean(spectra < -NEGATIVE_TOLERANCE, axis=0)
    for ax, ranks, title in ((ax_low, range(0, 6), "(a) Lower six ordered eigenvalues"), (ax_high, range(6, 8), "(b) Upper two ordered eigenvalues")):
        positions = np.arange(1, len(ranks) + 1)
        selected = spectra[:, list(ranks)]
        lower = min(float(np.min(selected)), -0.1) * 1.08
        ax.axhspan(lower, 0.0, color="#fbe4e4", zorder=0, label="negative-curvature directions")
        ax.axhline(0.0, color="black", linewidth=1.1)
        ax.boxplot(
            [spectra[:, rank] for rank in ranks], positions=positions, widths=0.52, showfliers=False,
            patch_artist=True, boxprops={"facecolor": "#d9e6f2", "edgecolor": "#3a3a3a"},
            medianprops={"color": "#1f4e79", "linewidth": 1.8}, whiskerprops={"color": "#3a3a3a"}, capprops={"color": "#3a3a3a"},
        )
        for local_rank, global_rank in enumerate(ranks):
            jitter = rng.normal(positions[local_rank], 0.055, size=spectra.shape[0])
            ax.scatter(jitter, spectra[:, global_rank], color="#d95f02", s=10, alpha=0.18, linewidths=0, rasterized=True)
            ax.text(positions[local_rank], 0.025, f"{negative_fraction[global_rank]:.1%}", ha="center", va="bottom", fontsize=9, color="#7a3b00")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("ordered eigenvalue rank")
        ax.set_xticks(positions)
        ax.set_xticklabels([rf"$\lambda_{rank + 1}$" for rank in ranks])
        ax.grid(axis="y", alpha=0.18)
    ax_low.set_ylabel(r"eigenvalue of $\nabla^2J_{\mathrm{proto}}(\mathbf{z})$")
    ax_high.set_ylabel(r"eigenvalue of $\nabla^2J_{\mathrm{proto}}(\mathbf{z})$")
    ax_low.legend(frameon=False, loc="center left", bbox_to_anchor=(0.01, 0.42))
    ax_low.text(0.02, 0.975, "Percent labels: fraction with eigenvalue < −10⁻³", transform=ax_low.transAxes, fontsize=8.7, va="top")
    fig.tight_layout()
    _save(fig, "figure_4_ordered_eigenvalue_rank_distributions")


def make_negative_count_figure(rows: list[dict[str, float]]) -> None:
    """Summarize how many independent negative-curvature directions occur per point."""
    objective, spectra = _stable_spectra(rows)
    counts = np.count_nonzero(spectra < -NEGATIVE_TOLERANCE, axis=1)
    bins = np.arange(-0.5, 9.5, 1.0)
    histogram = np.bincount(counts, minlength=9)
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.set_title("Number of negative-curvature directions per stable interior sample", fontsize=15, fontweight="bold", pad=12)
    bars = ax.bar(np.arange(9), histogram, color="#d95f02", edgecolor="#8a3c00", width=0.74)
    ax.set_xlabel(r"$n_-(\mathbf{z})=\#\{i:\lambda_i(\mathbf{z})<-10^{-3}\}$")
    ax.set_ylabel("number of stable Sobol samples")
    ax.set_xticks(np.arange(9))
    ax.set_ylim(0.0, max(histogram) * 1.18)
    for bar, value in zip(bars, histogram):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + max(histogram) * 0.025, f"{value}\n({value / len(counts):.1%})", ha="center", va="bottom", fontsize=10)
    ax.text(0.98, 0.95, f"stable N = {len(counts)}\nmedian $n_-$ = {np.median(counts):.0f}\nmean $n_-$ = {np.mean(counts):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    _save(fig, "figure_5_negative_eigenvalue_count_histogram")


def make_objective_spectrum_fan_figure(rows: list[dict[str, float]]) -> None:
    """Show objective-conditioned median spectra and interquartile bands."""
    objective, spectra = _stable_spectra(rows)
    log_objective = np.log10(objective)
    # Equal-count bins stabilize quantiles without suggesting a physical x-grid.
    edges = np.quantile(log_objective, np.linspace(0.0, 1.0, 13))
    edges = np.unique(edges)
    centers: list[float] = []
    medians: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (log_objective >= lower) & (log_objective <= upper if upper == edges[-1] else log_objective < upper)
        if np.count_nonzero(mask) < 8:
            continue
        centers.append(float(np.median(log_objective[mask])))
        medians.append(np.median(spectra[mask], axis=0))
        lows.append(np.quantile(spectra[mask], 0.25, axis=0))
        highs.append(np.quantile(spectra[mask], 0.75, axis=0))
    centers_array = np.array(centers)
    median_array = np.vstack(medians)
    low_array = np.vstack(lows)
    high_array = np.vstack(highs)
    selected_ranks = (0, 1, 2, 7)
    colors = ("#b2182b", "#ef8a62", "#2166ac", "#542788")
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.set_title("Objective-conditioned ordered Hessian spectrum", fontsize=15, fontweight="bold", pad=12)
    y_lower = min(float(np.min(low_array[:, selected_ranks])), -0.1) * 1.08
    ax.axhspan(y_lower, 0.0, color="#fbe4e4", zorder=0)
    ax.axhline(0.0, color="black", linewidth=1.1)
    for rank, color in zip(selected_ranks, colors):
        label = rf"$\lambda_{rank + 1}$ median ± IQR"
        ax.fill_between(centers_array, low_array[:, rank], high_array[:, rank], color=color, alpha=0.16)
        ax.plot(centers_array, median_array[:, rank], "o-", color=color, linewidth=2.1, markersize=4.5, label=label)
    ax.set_xlabel(r"$\log_{10}J_{\mathrm{proto}}(\mathbf{z})$; equal-count objective bins")
    ax.set_ylabel(r"ordered Hessian eigenvalue")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="lower left")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    _save(fig, "figure_6_objective_conditioned_spectrum_fan")


def make_precision_figure(confirmation: dict, double_precision: list[dict[str, float]]) -> None:
    high_precision = confirmation["symmetric_difference_and_jensen_checks"]
    h_high = np.array([float(row["step"]) for row in high_precision])
    curvature_high = np.array([float(row["directional_second_difference"]) for row in high_precision])
    gap_high = np.array([float(row["jensen_gap"]) for row in high_precision])
    h_double = np.array([row["step"] for row in double_precision])
    curvature_double = np.array([row["directional_second_difference"] for row in double_precision])
    gap_double = np.array([row["jensen_gap_center_minus_endpoint_average"] for row in double_precision])

    fig, (ax_curvature, ax_gap) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    fig.suptitle("Arbitrary-precision verification of the negative-curvature witness", fontsize=16, fontweight="bold", y=1.02)

    for axis in (ax_curvature, ax_gap):
        axis.set_xscale("log")
        axis.set_xticks(h_high)
        axis.set_xticklabels([f"{h:g}" for h in h_high])
        axis.grid(which="both", alpha=0.18)
        axis.set_xlabel(r"perturbation scale $h$ along $\mathbf{v}_{\min}$")

    ax_curvature.axhline(0.0, color="black", linewidth=1.1)
    ax_curvature.plot(h_high, curvature_high, "o-", color="#1f4e79", linewidth=2.4, label="80-digit matrix solve", zorder=2)
    ax_curvature.plot(h_double, curvature_double, "s", color="#7a7a7a", markerfacecolor="white", markeredgewidth=1.7, markersize=8, label="float64 matrix solve", zorder=3)
    ax_curvature.set_ylabel(r"$[g(h)-2g(0)+g(-h)]/h^2$")
    ax_curvature.set_title("(a) Directional curvature", loc="left", fontweight="bold")
    ax_curvature.legend(frameon=False, fontsize=9, loc="best")
    ax_curvature.text(0.03, 0.05, r"all estimates $<0$; square markers overlap the 80-digit curve", transform=ax_curvature.transAxes, fontsize=9)

    ax_gap.plot(h_high, gap_high, "o-", color="#1f4e79", linewidth=2.4, label="80-digit matrix solve", zorder=2)
    ax_gap.plot(h_double, gap_double, "s", color="#7a7a7a", markerfacecolor="white", markeredgewidth=1.7, markersize=8, label="float64 matrix solve", zorder=3)
    ax_gap.set_yscale("log")
    ax_gap.set_ylabel(r"$g(0)-[g(-h)+g(h)]/2$")
    ax_gap.set_title("(b) Direct Jensen-inequality violation", loc="left", fontweight="bold")
    ax_gap.text(0.03, 0.05, r"all gaps $>0$; double-precision squares overlap high-precision points", transform=ax_gap.transAxes, fontsize=9)

    _save(fig, "figure_2_precision_robustness")


def main() -> None:
    global ROOT, FIGURE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT, help="Analysis output directory to visualize.")
    args = parser.parse_args()
    ROOT = args.input_dir
    FIGURE_DIR = ROOT / "presentation_figures"
    rows = _load_hessian_rows()
    confirmation, double_precision = _load_precision_data()
    make_hessian_screening_figure(rows)
    make_replicate_figure(rows)
    make_rank_distribution_figure(rows)
    make_negative_count_figure(rows)
    make_objective_spectrum_fan_figure(rows)
    make_precision_figure(confirmation, double_precision)
    print(f"Wrote presentation figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
