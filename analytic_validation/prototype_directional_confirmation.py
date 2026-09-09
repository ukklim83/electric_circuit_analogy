"""Arbitrary-precision directional confirmation of a prototype curvature witness.

This script does not expand the full symbolic eight-dimensional Hessian.  It
fixes the witness point and minimum-curvature direction saved by
``prototype_joint_convexity_analysis.py``, re-solves the same flow and
concentration matrices with mpmath arbitrary precision, and evaluates
g(t)=J(z0+t*v).  It reports both ``mp.diff(g, 0, 2)`` and independently
computed symmetric second differences.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from prototype_reduced_objective import LOWER_BOUND, Q_DESIRED, UPPER_BOUND, _incidence_numpy, _junction_constraints


def _matrix_from_numpy(array: np.ndarray) -> mp.matrix:
    rows, cols = array.shape
    matrix = mp.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = mp.mpf(str(array[i, j]))
    return matrix


def _diag(values: list[mp.mpf]) -> mp.matrix:
    matrix = mp.zeros(len(values), len(values))
    for i, value in enumerate(values):
        matrix[i, i] = value
    return matrix


def _slice(matrix: mp.matrix, rows: range, cols: range) -> mp.matrix:
    return mp.matrix([[matrix[i, j] for j in cols] for i in rows])


def _vstack(parts: list[mp.matrix]) -> mp.matrix:
    rows: list[list[mp.mpf]] = []
    for part in parts:
        rows.extend([[part[i, j] for j in range(part.cols)] for i in range(part.rows)])
    return mp.matrix(rows)


def _concat_vectors(parts: list[mp.matrix]) -> mp.matrix:
    return mp.matrix([part[i] for part in parts for i in range(part.rows)])


def _build_high_precision_evaluator():
    """Return J(z) evaluated by the same matrix equations at arbitrary precision."""
    A = _matrix_from_numpy(_incidence_numpy())
    E_o_np, E_m_np, E_i_np = _junction_constraints(_incidence_numpy())
    E_o, E_m, E_i = _matrix_from_numpy(E_o_np), _matrix_from_numpy(E_m_np), _matrix_from_numpy(E_i_np)
    q_desired = mp.mpf(1) / mp.mpf(60_000_000)
    lower, upper = mp.mpf(str(LOWER_BOUND)), mp.mpf(str(UPPER_BOUND))

    def evaluate(z: list[mp.mpf]) -> mp.mpf:
        if len(z) != 8:
            raise ValueError("Expected eight normalized variables.")
        r = [lower + value * (upper - lower) for value in z]
        lhs = mp.zeros(18, 18)
        for i in range(8):
            lhs[i, i] = r[i]
        for i in range(8):
            for j in range(8):
                lhs[i, 8 + j] = A[i, j]
                lhs[8 + j, i] = A[i, j]
        lhs[8, 16] = lhs[16, 8] = mp.mpf(1)
        lhs[9, 17] = lhs[17, 9] = mp.mpf(1)
        rhs = mp.matrix([0] * 14 + [-q_desired, -q_desired] + [0, 0])
        q = mp.lu_solve(lhs, rhs)[:8]
        q_out, q_mid, q_in = q[:2], q[2:6], q[6:8]
        Q_out, Q_mid, Q_in = _diag(q_out), _diag(q_mid), _diag(q_in)
        Q_target, F_in = _diag([q_desired, q_desired]), _diag([-q_desired, -q_desired])

        A_oo, A_om, A_oi = _slice(A, range(0, 2), range(0, 2)), _slice(A, range(0, 2), range(2, 6)), _slice(A, range(0, 2), range(6, 8))
        A_mo, A_mm, A_mi = _slice(A, range(2, 6), range(0, 2)), _slice(A, range(2, 6), range(2, 6)), _slice(A, range(2, 6), range(6, 8))
        A_io, A_im, A_ii = _slice(A, range(6, 8), range(0, 2)), _slice(A, range(6, 8), range(2, 6)), _slice(A, range(6, 8), range(6, 8))

        def hstack(left: mp.matrix, right: mp.matrix) -> mp.matrix:
            return mp.matrix([[left[i, j] if j < left.cols else right[i, j - left.cols] for j in range(left.cols + right.cols)] for i in range(left.rows)])

        C = _vstack(
            [
                hstack(A_oo.T * Q_out - Q_target, A_mo.T * Q_mid),
                hstack(A_om.T * Q_out, A_mm.T * Q_mid),
                hstack(A_oi.T * Q_out, A_mi.T * Q_mid),
                hstack(E_o, E_m),
            ]
        )
        normal = C.T * C

        def solve_component(inlet: list[int]) -> mp.matrix:
            inlet_vector = mp.matrix(inlet)
            rhs_conc = _concat_vectors(
                [
                    -A_io.T * Q_in * inlet_vector,
                    -A_im.T * Q_in * inlet_vector,
                    (F_in - A_ii.T * Q_in) * inlet_vector,
                    -E_i * inlet_vector,
                ]
            )
            return mp.lu_solve(normal, C.T * rhs_conc)

        x1, x2 = solve_component([1, 0]), solve_component([0, 1])
        x_out = [x1[0], x1[1], x2[0], x2[1]]
        x_target = [mp.mpf("0.7"), mp.mpf("0.3"), mp.mpf("0.3"), mp.mpf("0.7")]
        flow_term = sum((q_out[i] - q_desired) ** 2 for i in range(2)) / (2 * q_desired) ** 2
        concentration_term = sum((x_out[i] - x_target[i]) ** 2 for i in range(4))
        return flow_term + concentration_term

    return evaluate


def _read_witness(path: Path) -> tuple[int, list[mp.mpf], list[mp.mpf], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "v_min_0" not in rows[0]:
        raise ValueError("Witness CSV lacks v_min columns. Re-run prototype_joint_convexity_analysis.py first.")
    # Prefer the witness selected by the full-screening script after its
    # two-step directional-stability check.  Fall back to the raw most-negative
    # Hessian candidate for backwards-compatible older CSV files.
    selected = [row for row in rows if row.get("is_selected_witness") == "1"]
    if len(selected) > 1:
        raise ValueError("Witness CSV contains more than one selected witness.")
    witness = selected[0] if selected else min(rows, key=lambda row: float(row["lambda_min"]))
    return (
        int(witness["sample"]),
        [mp.mpf(witness[f"z_{i}"]) for i in range(8)],
        [mp.mpf(witness[f"v_min_{i}"]) for i in range(8)],
        float(witness["lambda_min"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).with_name("outputs") / "prototype_joint_convexity"
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--steps", nargs="+", default=["0.0015", "0.003", "0.006"])
    parser.add_argument("--mp-diff", action="store_true", help="Also attempt mpmath's adaptive numerical derivative; this can be slow.")
    args = parser.parse_args()
    mp.mp.dps = args.dps

    sample, z0, direction, double_lambda = _read_witness(args.input_dir / "prototype_hessian_witnesses.csv")
    norm = mp.sqrt(sum(value**2 for value in direction))
    direction = [value / norm for value in direction]
    evaluate = _build_high_precision_evaluator()

    def g(t: mp.mpf) -> mp.mpf:
        return evaluate([z + t * v for z, v in zip(z0, direction)])

    center = g(mp.mpf("0"))
    checks = []
    for raw_step in args.steps:
        step = mp.mpf(raw_step)
        minus, plus = g(-step), g(step)
        checks.append(
            {
                "step": str(step),
                "directional_second_difference": str((plus - 2 * center + minus) / step**2),
                "J_minus": str(minus),
                "J_center": str(center),
                "J_plus": str(plus),
                "jensen_gap": str(center - (minus + plus) / 2),
            }
        )
    result = {
        "method": "arbitrary-precision line-restriction confirmation",
        "precision_decimal_digits": args.dps,
        "witness_sample": sample,
        "double_precision_full_hessian_lambda_min": double_lambda,
        "mpmath_second_derivative_g_dd_0": None,
        "z0": [str(value) for value in z0],
        "unit_direction": [str(value) for value in direction],
        "symmetric_difference_and_jensen_checks": checks,
        "scope": "Confirms a negative-curvature direction for the prototype. It is not a full symbolic Hessian or a multimodality proof.",
    }
    if args.mp_diff:
        # This adaptive derivative is optional because it may repeatedly call
        # the high-precision matrix solver at very small steps.
        result["mpmath_second_derivative_g_dd_0"] = str(mp.diff(g, mp.mpf("0"), 2))
    output = args.input_dir / "prototype_directional_confirmation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote: {output}")
    if result["mpmath_second_derivative_g_dd_0"] is not None:
        print(f"mp.diff g''(0): {result['mpmath_second_derivative_g_dd_0']}")
    for row in checks:
        print(f"h={row['step']}: finite difference={row['directional_second_difference']}, Jensen gap={row['jensen_gap']}")


if __name__ == "__main__":
    main()
