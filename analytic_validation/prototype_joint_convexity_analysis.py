"""Numerical curvature witnesses for the exact eight-resistance prototype.

Run from ``electric_circuit_analogy/analytic_validation``:

    python prototype_joint_convexity_analysis.py --samples 512
    python prototype_joint_convexity_analysis.py --samples 256 --replicates 4

The script samples a normalized resistance box, estimates full Hessians by
central differences, verifies the strongest negative-curvature candidate by a
directional second difference and a Jensen inequality check, and writes only
new files below ``analytic_validation/outputs/prototype_joint_convexity``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from prototype_reduced_objective import build_prototype_objective, objective_definition_latex


def finite_difference_hessian(evaluate, z: np.ndarray, step: float) -> np.ndarray:
    """Central-difference Hessian of a scalar function on normalized variables."""
    z = np.asarray(z, dtype=float)
    n = z.size
    if np.any(z - step < 0.0) or np.any(z + step > 1.0):
        raise ValueError("Hessian point must be at least one finite-difference step from each bound.")
    hessian = np.empty((n, n), dtype=float)
    center = evaluate(z)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = step
        hessian[i, i] = (evaluate(z + e_i) - 2.0 * center + evaluate(z - e_i)) / step**2
        for j in range(i + 1, n):
            e_j = np.zeros(n)
            e_j[j] = step
            value = (
                evaluate(z + e_i + e_j)
                - evaluate(z + e_i - e_j)
                - evaluate(z - e_i + e_j)
                + evaluate(z - e_i - e_j)
            ) / (4.0 * step**2)
            hessian[i, j] = hessian[j, i] = value
    return 0.5 * (hessian + hessian.T)


def directional_second_difference(evaluate, z: np.ndarray, direction: np.ndarray, step: float) -> float:
    """Evaluate curvature along a unit direction using a symmetric difference."""
    return (evaluate(z + step * direction) - 2.0 * evaluate(z) + evaluate(z - step * direction)) / step**2


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128, help="Sobol samples in the interior normalized box.")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--replicates", type=int, default=1, help="Independent scrambled Sobol designs to screen.")
    parser.add_argument("--step", type=float, default=3.0e-3)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("outputs") / "prototype_joint_convexity")
    args = parser.parse_args()
    if not 0.0 < args.step < 0.25:
        raise ValueError("--step must lie between 0 and 0.25.")
    if not _is_power_of_two(args.samples):
        raise ValueError("--samples must be a power of two so each Sobol design retains its balance property.")
    if args.replicates < 1:
        raise ValueError("--replicates must be at least one.")

    prototype = build_prototype_objective()
    evaluate = prototype.evaluate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prototype_objective_definition.tex").write_text(objective_definition_latex(), encoding="utf-8")
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "objective": "exact matrix-defined eight-resistance prototype reduced objective",
                "coordinates": "normalized resistance vector z in [0, 1]^8",
                "samples_per_design": args.samples,
                "replicates": args.replicates,
                "total_samples": args.samples * args.replicates,
                "sobol_seeds": [args.seed + replicate for replicate in range(args.replicates)],
                "finite_difference_step": args.step,
                "lower_resistance_bound": prototype.lower_bounds.tolist(),
                "upper_resistance_bound": prototype.upper_bounds.tolist(),
                "interpretation": "Negative curvature is a counterexample to joint convexity for this prototype; it is not a multimodality claim or a benchmark-wide result.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict] = []
    all_points: list[np.ndarray] = []
    eigenvectors: list[np.ndarray] = []
    power = int(np.log2(args.samples))
    for replicate in range(args.replicates):
        sobol_seed = args.seed + replicate
        sampler = qmc.Sobol(d=8, scramble=True, seed=sobol_seed)
        raw = sampler.random_base2(power)
        # Keep every stencil in the normalized box.
        points = args.step + (1.0 - 2.0 * args.step) * raw
        for index, z in enumerate(points):
            hessian = finite_difference_hessian(evaluate, z, args.step)
            eigenvalues, vectors = np.linalg.eigh(hessian)
            direction = vectors[:, 0]
            # Eigenvectors are sign-indeterminate.  Canonicalizing the sign makes
            # the stored witness direction reproducible across later checks.
            direction *= np.sign(direction[np.argmax(np.abs(direction))])
            row = {
                "replicate": replicate + 1,
                "sobol_seed": sobol_seed,
                "sample": index,
                "global_sample": len(rows),
                "objective": evaluate(z),
                "lambda_min": eigenvalues[0],
                "lambda_max": eigenvalues[-1],
            }
            # Ordered eigenvalues describe local curvature along eight
            # orthogonal coupled-resistance directions.  Preserve the entire
            # spectrum for later rank and negative-direction diagnostics.
            row.update({f"eigenvalue_{i + 1}": eigenvalues[i] for i in range(8)})
            row.update({f"z_{i}": z[i] for i in range(8)})
            row.update({f"v_min_{i}": direction[i] for i in range(8)})
            rows.append(row)
            all_points.append(z)
            eigenvectors.append(direction)

    # A Hessian eigenvalue obtained at one finite-difference scale can be
    # step-sensitive near sharp or poorly conditioned response regions.  Keep
    # an inexpensive line-direction check at h/2 and h, then use the strongest
    # stable negative candidate for the reported witness.
    stable_indices: list[int] = []
    for index, (row, z, direction) in enumerate(zip(rows, all_points, eigenvectors)):
        direction = direction / np.linalg.norm(direction)
        half_step = args.step / 2.0
        curvature_half = directional_second_difference(evaluate, z, direction, half_step)
        curvature_full = directional_second_difference(evaluate, z, direction, args.step)
        relative_difference = abs(curvature_half - curvature_full) / max(1.0, abs(curvature_full))
        is_stable = curvature_half < -1.0e-3 and curvature_full < -1.0e-3 and relative_difference <= 0.25
        row.update(
            {
                "directional_curvature_half_step": curvature_half,
                "directional_curvature_step": curvature_full,
                "directional_curvature_relative_difference": relative_difference,
                "stable_negative_curvature": int(is_stable),
                "is_selected_witness": 0,
            }
        )
        if is_stable:
            stable_indices.append(index)
    if not stable_indices:
        raise RuntimeError("No step-stable negative-curvature witness was found.")
    witness_index = min(stable_indices, key=lambda index: rows[index]["lambda_min"])
    rows[witness_index]["is_selected_witness"] = 1
    write_csv(args.output_dir / "prototype_hessian_witnesses.csv", rows)

    summaries = []
    for replicate in range(1, args.replicates + 1):
        replicate_rows = [row for row in rows if row["replicate"] == replicate]
        lambda_values = np.array([row["lambda_min"] for row in replicate_rows])
        summaries.append(
            {
                "replicate": replicate,
                "sobol_seed": args.seed + replicate - 1,
                "samples": len(replicate_rows),
                "negative_count": int(np.count_nonzero(lambda_values < -1.0e-3)),
                "negative_fraction": float(np.mean(lambda_values < -1.0e-3)),
                "stable_negative_count": int(sum(row["stable_negative_curvature"] for row in replicate_rows)),
                "stable_negative_fraction": float(np.mean([row["stable_negative_curvature"] for row in replicate_rows])),
                "lambda_min_minimum": float(lambda_values.min()),
                "lambda_min_median": float(np.median(lambda_values)),
            }
        )
    write_csv(args.output_dir / "prototype_hessian_replicate_summary.csv", summaries)
    witness = all_points[witness_index]
    direction = eigenvectors[witness_index]
    direction /= np.linalg.norm(direction)

    # Use several steps; a stable negative sign is stronger evidence than one
    # finite-difference scale alone.
    verification_steps = np.array([args.step / 2.0, args.step, args.step * 2.0])
    verification = []
    for step in verification_steps:
        if np.any(witness - step * direction < 0.0) or np.any(witness + step * direction > 1.0):
            continue
        minus = witness - step * direction
        plus = witness + step * direction
        center_value = evaluate(witness)
        minus_value = evaluate(minus)
        plus_value = evaluate(plus)
        verification.append(
            {
                "step": step,
                "directional_second_difference": directional_second_difference(evaluate, witness, direction, step),
                "J_minus": minus_value,
                "J_center": center_value,
                "J_plus": plus_value,
                "jensen_gap_center_minus_endpoint_average": center_value - 0.5 * (minus_value + plus_value),
            }
        )
    write_csv(args.output_dir / "prototype_jensen_checks.csv", verification)

    line_t = np.linspace(-2.0 * args.step, 2.0 * args.step, 161)
    line_j = np.array([evaluate(witness + t * direction) for t in line_t])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(line_t, line_j, color="#1f4e79", linewidth=2.5)
    ax.scatter([0.0], [evaluate(witness)], color="#c00000", zorder=3, label="Hessian witness")
    ax.set_xlabel(r"$t$ along minimum-curvature eigenvector")
    ax.set_ylabel(r"$J_{\mathrm{proto}}(\mathbf{z}_0+t\mathbf{v})$")
    ax.set_title("Eight-resistance prototype: directional curvature witness")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_S5_prototype_joint_convexity.pdf", dpi=300)
    plt.close(fig)

    print(f"Wrote prototype analysis to: {args.output_dir}")
    print(f"Witness sample: {witness_index}")
    print(f"Minimum Hessian eigenvalue: {rows[witness_index]['lambda_min']:.6e}")
    if verification:
        print("Directional second differences:", [f"{row['directional_second_difference']:.6e}" for row in verification])
        print("Jensen gaps:", [f"{row['jensen_gap_center_minus_endpoint_average']:.6e}" for row in verification])


if __name__ == "__main__":
    main()
