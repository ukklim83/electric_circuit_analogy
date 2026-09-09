"""Multi-start SLSQP variant of :mod:`electric_analogy`.

This backend receives the unified ``electric_analogy`` solver module explicitly
and provides a scalar, bounded multi-start SLSQP implementation without a
circular or implicit import.

The first restart is the legacy initial design.  The remaining restarts are
Latin-hypercube points in normalized design space.  The solution with the
lowest weighted combined residual is written to ``new_length_mat.csv``.
"""

from __future__ import annotations

import contextlib as _contextlib
import io as _io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds as _ScipyBounds
from scipy.optimize import minimize as _scipy_minimize
from scipy.stats import qmc

class EvaluationBudgetExceeded(RuntimeError):
    """Raised before a new circuit evaluation would exceed a restart FES cap."""


def _target_concentrations(
    conc_csv: str, address: str, solver_module
) -> np.ndarray:
    """Return target fractions in the same outlet-major ordering as the solver."""
    concentration_df = solver_module.conc_mat_produce(conc_csv, address)
    fraction_vectors = [
        concentration_df[f"fraction{index + 1}"].to_numpy(dtype=float)
        for index in range(len(concentration_df.columns))
    ]
    return np.vstack(fraction_vectors).T.ravel()


def _make_starts(initial_z: np.ndarray, n_starts: int, seed: int | None) -> np.ndarray:
    """Construct one legacy start plus independent Latin-hypercube starts."""
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
    if n_starts == 1:
        return initial_z.reshape(1, -1)
    sampler = qmc.LatinHypercube(d=initial_z.size, seed=seed)
    return np.vstack((initial_z, sampler.random(n_starts - 1)))


def execute_length_change(
    whatToSolve,
    changing_edges,
    inc_csv,
    length_csv,
    conc_csv,
    args,
    ini_list,
    inlet_node_idx,
    inlet_edge_idx,
    outlet_node_idx,
    outlet_edge_idx,
    conc_wght,
    flow_wght,
    address,
    popSize=200,
    nGen=200,
    *,
    n_starts: int = 20,
    maxiter: int = 1000,
    seed: int | None = 1,
    lower_scale: float = 0.8,
    upper_scale: float = 1.2,
    lower_delta_mm: float | None = None,
    upper_delta_mm: float | None = None,
    ftol: float = 1e-12,
    finite_diff_rel_step: float = 1e-6,
    progress_every: int = 5,
    output_suffix: str = "slsqp",
    solver_module=None,
    max_evaluations_per_start: int | list[int] | tuple[int, ...] | None = None,
):
    """Optimize channel lengths using independently restarted SLSQP.

    Parameters through ``address`` retain the original function signature.
    ``popSize`` and ``nGen`` are accepted only so legacy callers continue to
    run; they have no SLSQP meaning.  Use ``n_starts`` and ``maxiter`` instead.

    Lengths are normalized to ``z in [0, 1]`` before optimization.  Bounds are
    still expressed relative to the legacy design, using ``lower_scale`` and
    ``upper_scale`` (0.8x--1.2x by default).  The scalar objective is

    ``hypot(flow_wght * ||flow residual||_2,
            conc_wght * ||concentration residual||_2)``.

    Non-negative predicted flow and concentration are retained as SLSQP
    inequality constraints, matching the intent of the original NSGA-II
    feasibility conditions.

    When ``max_evaluations_per_start`` is supplied, every unique circuit solve
    (including finite-difference points) is counted.  The cap is enforced
    before the solver call; ``maxiter`` remains only a safety limit.
    """
    if not 0 < lower_scale < upper_scale:
        raise ValueError("Require 0 < lower_scale < upper_scale")
    if maxiter < 1:
        raise ValueError("maxiter must be at least 1")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")

    if solver_module is None:
        raise ValueError(
            "solver_module must be provided by the unified electric_analogy dispatcher"
        )
    solver = solver_module
    suffix = f"_{output_suffix}" if output_suffix else ""
    started = solver.time.time()
    edge_indices = np.asarray(changing_edges, dtype=int)
    if edge_indices.ndim != 1 or edge_indices.size == 0:
        raise ValueError("changing_edges must contain at least one edge index")

    length_df = pd.read_csv(Path(address) / length_csv, index_col="edge")
    initial_length = length_df["length"].to_numpy(dtype=float)
    if np.any(edge_indices < 0) or np.any(edge_indices >= initial_length.size):
        raise IndexError("changing_edges contains an index outside length_csv")

    initial_variables = initial_length[edge_indices]
    lower = (
        initial_variables * lower_scale
        if lower_delta_mm is None
        else initial_variables + float(lower_delta_mm)
    )
    upper = (
        initial_variables * upper_scale
        if upper_delta_mm is None
        else initial_variables + float(upper_delta_mm)
    )
    span = upper - lower
    if np.any(lower <= 0):
        raise ValueError("Every lower length bound must be positive")
    if np.any(span <= 0):
        raise ValueError("Every upper length bound must exceed its lower bound")
    initial_z = (initial_length[edge_indices] - lower) / span

    (
        _is_length_unknown,
        is_source_unknown,
        is_concentration_unknown,
    ) = solver.variable_setup(whatToSolve)
    eta, width, height, n = args
    inc_mat = solver.inc_mat_produce(inc_csv, address)
    inlet_list = [list(item) for item in ini_list if item[3] == "i"]
    outlet_list = [list(item) for item in ini_list if item[3] == "o"]
    if not outlet_list:
        raise ValueError("At least one outlet is required for length optimization")

    _, _, target_flow, target_flow_scaled = solver.calculate_currents(
        whatToSolve, inc_csv, initial_length, ini_list, args, address
    )
    target_flow_scaled = np.asarray(target_flow_scaled, dtype=float)
    target_concentration = _target_concentrations(conc_csv, address, solver)
    threshold = 1e-5

    # SLSQP calls the objective and constraints separately at the same point.
    # Cache the last full circuit solve so each candidate is solved once.
    cache: dict[str, Any] = {"z": None, "result": None, "evaluations": 0}
    restart_budget: dict[str, Any] = {
        "limit": None,
        "evaluations_at_start": 0,
        "best_z": None,
        "best_result": None,
    }

    def evaluate(z: np.ndarray) -> dict[str, Any]:
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        if cache["z"] is not None and np.array_equal(z, cache["z"]):
            return cache["result"]
        limit = restart_budget["limit"]
        used = cache["evaluations"] - restart_budget["evaluations_at_start"]
        if limit is not None and used >= limit:
            raise EvaluationBudgetExceeded(
                f"restart evaluation budget exhausted: {used} >= {limit}"
            )

        candidate_length = initial_length.copy()
        candidate_length[edge_indices] = lower + z * span
        block_matrix = solver.construct_block_mat(
            candidate_length, eta, (width, height, n), inc_mat
        )
        rhs = solver.construct_RHS(inc_mat, ini_list)
        grounded_matrix, grounded_rhs, ground_index = solver.ground_block_mat(
            block_matrix, inc_mat, rhs, outlet_list
        )
        # Some legacy concentration routines print internal matrix dimensions.
        # Keep terminal output focused on restart and iteration-level progress.
        with _contextlib.redirect_stdout(_io.StringIO()):
            currents, _, solved_rhs = solver.Kirchhoff_solver(
                block_matrix,
                grounded_matrix,
                rhs,
                grounded_rhs,
                inc_mat,
                is_source_unknown,
                ini_list,
                outlet_list,
                inlet_list,
                ground_index,
            )
        # Outlet edge currents are signed opposite to the positive inlet-flow
        # convention.  Compare physical flow fractions, not their orientation.
        predicted_flow = np.abs(
            np.asarray(currents[: len(outlet_list)], dtype=float)
        )
        inlet_current, _, _, _ = solver.calculate_currents(
            whatToSolve, inc_csv, candidate_length, ini_list, args, address
        )
        predicted_flow_scaled = predicted_flow / abs(float(inlet_current))

        if is_concentration_unknown and len(inlet_list) > 1:
            with _contextlib.redirect_stdout(_io.StringIO()):
                concentration_vectors = solver.conc_calculation(
                    solved_rhs,
                    currents,
                    inc_mat,
                    inlet_list,
                    outlet_list,
                    inlet_node_idx,
                    inlet_edge_idx,
                )
            predicted_concentration = np.vstack(concentration_vectors).T.ravel()
        else:
            predicted_concentration = np.empty(0, dtype=float)

        if predicted_concentration.size != target_concentration.size:
            raise ValueError(
                "Predicted and target concentration dimensions differ: "
                f"{predicted_concentration.size} != {target_concentration.size}"
            )

        flow_residual = predicted_flow_scaled - target_flow_scaled
        concentration_residual = predicted_concentration - target_concentration
        flow_norm = float(flow_wght * np.linalg.norm(flow_residual))
        concentration_norm = float(conc_wght * np.linalg.norm(concentration_residual))
        result = {
            "length": candidate_length,
            "flow": predicted_flow_scaled,
            "concentration": predicted_concentration,
            "flow_residual": flow_residual,
            "concentration_residual": concentration_residual,
            "flow_norm": flow_norm,
            "concentration_norm": concentration_norm,
            "combined_error": float(np.hypot(flow_norm, concentration_norm)),
        }
        cache.update(z=z.copy(), result=result, evaluations=cache["evaluations"] + 1)
        if (
            restart_budget["best_result"] is None
            or result["combined_error"] < restart_budget["best_result"]["combined_error"]
        ):
            restart_budget["best_z"] = z.copy()
            restart_budget["best_result"] = result
        return result

    def scalar_objective(z: np.ndarray) -> float:
        return float(evaluate(z)["combined_error"])

    def feasibility(z: np.ndarray) -> np.ndarray:
        result = evaluate(z)
        return np.concatenate(
            (
                result["flow"] + threshold,
                result["concentration"] + threshold,
            )
        )

    starts = _make_starts(initial_z, n_starts, seed)
    if max_evaluations_per_start is None:
        restart_limits: list[int | None] = [None] * len(starts)
    elif isinstance(max_evaluations_per_start, int):
        if max_evaluations_per_start < 1:
            raise ValueError("max_evaluations_per_start must be at least 1")
        restart_limits = [max_evaluations_per_start] * len(starts)
    else:
        restart_limits = [int(limit) for limit in max_evaluations_per_start]
        if len(restart_limits) != len(starts) or any(limit < 1 for limit in restart_limits):
            raise ValueError("Provide one positive FES cap for every restart")
    restart_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_restart: int | None = None

    print(
        "Multi-start SLSQP: "
        f"{n_starts} starts, maxiter={maxiter}, normalized bounds=[0, 1], "
        f"restart FES caps={restart_limits}."
    )
    if popSize != 200 or nGen != 200:
        print(
            "Note: legacy popSize/nGen arguments are ignored by SLSQP "
            f"(received popSize={popSize}, nGen={nGen})."
        )

    for restart_index, z0 in enumerate(starts):
        iteration = 0
        evaluations_before = int(cache["evaluations"])
        restart_budget.update(
            limit=restart_limits[restart_index],
            evaluations_at_start=evaluations_before,
            best_z=None,
            best_result=None,
        )

        def report_iteration(xk: np.ndarray, *_args: object) -> None:
            nonlocal iteration
            iteration += 1
            if iteration == 1 or iteration % progress_every == 0:
                result = evaluate(xk)
                print(
                    f"SLSQP restart {restart_index + 1}/{len(starts)}, "
                    f"iteration {iteration}: objective={result['combined_error']:.6e}, "
                    f"flow_max={np.max(np.abs(result['flow_residual'])):.6e}, "
                    f"conc_max={np.max(np.abs(result['concentration_residual'])):.6e}"
                )

        scipy_result = None
        exit_status = "solver_stopped"
        message = ""
        try:
            scipy_result = _scipy_minimize(
                scalar_objective,
                z0,
                method="SLSQP",
                jac="2-point",
                bounds=_ScipyBounds(np.zeros_like(z0), np.ones_like(z0)),
                constraints={"type": "ineq", "fun": feasibility},
                callback=report_iteration,
                options={
                    "maxiter": maxiter,
                    "ftol": ftol,
                    "finite_diff_rel_step": finite_diff_rel_step,
                    "disp": False,
                },
            )
            candidate = np.asarray(scipy_result.x, dtype=float)
            exit_status = "converged" if scipy_result.success else "solver_stopped"
            message = str(scipy_result.message)
        except EvaluationBudgetExceeded as exc:
            candidate = restart_budget["best_z"]
            exit_status = "budget_exhausted"
            message = str(exc)

        if candidate is None:
            candidate = np.asarray(z0, dtype=float)
        try:
            result = evaluate(candidate)
        except EvaluationBudgetExceeded:
            candidate = restart_budget["best_z"]
            result = restart_budget["best_result"]
        if result is None:
            raise RuntimeError("No candidate was evaluated during the SLSQP restart")
        restart_rows.append(
            {
                "restart_index": restart_index,
                "start_type": "initial" if restart_index == 0 else "latin_hypercube",
                "success": bool(scipy_result is not None and scipy_result.success),
                "exit_status": exit_status,
                "message": message,
                "iterations": int(getattr(scipy_result, "nit", iteration)),
                "solver_nfev": int(getattr(scipy_result, "nfev", 0)),
                "max_circuit_evaluations": restart_limits[restart_index],
                "unique_circuit_evaluations": int(cache["evaluations"]) - evaluations_before,
                "combined_error": result["combined_error"],
                "flow_norm": result["flow_norm"],
                "concentration_norm": result["concentration_norm"],
                "flow_max_abs_error": float(np.max(np.abs(result["flow_residual"]))),
                "concentration_max_abs_error": float(
                    np.max(np.abs(result["concentration_residual"]))
                ),
            }
        )
        print(
            f"SLSQP restart {restart_index + 1}/{len(starts)} finished: "
            f"status={exit_status}, evaluations={int(cache['evaluations']) - evaluations_before}, "
            f"objective={result['combined_error']:.6e}."
        )
        if best_result is None or result["combined_error"] < best_result["combined_error"]:
            best_result = result
            best_restart = restart_index

    assert best_result is not None and best_restart is not None
    length_df["length"] = best_result["length"]
    output_dir = Path(address)
    length_df.to_csv(output_dir / f"new_length_mat{suffix}.csv")
    restart_df = pd.DataFrame(restart_rows)
    restart_df["selected_as_best"] = restart_df.index == best_restart
    restart_df.to_csv(output_dir / f"slsqp_restart_summary{suffix}.csv", index=False)

    elapsed = solver.time.time() - started
    print(
        f"Selected restart {best_restart + 1}/{len(starts)}; "
        f"objective={best_result['combined_error']:.6e}; time={elapsed:.2f} s."
    )
    if "optimization" in whatToSolve:
        return elapsed, length_df
    return length_df
