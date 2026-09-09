"""Particle-swarm backend for the unified :mod:`electric_analogy` API.

The module receives the circuit solver explicitly, uses pymoo's scalar PSO in
normalized channel-length coordinates, and does not import ``electric_analogy``.
"""

from __future__ import annotations

import contextlib as _contextlib
import io as _io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.pso import PSO as _PymooPSO
from pymoo.core.callback import Callback as _PymooCallback
from pymoo.core.problem import Problem as _PymooProblem
from pymoo.core.termination import Termination as _PymooTermination
from pymoo.optimize import minimize as _pymoo_minimize

def _target_concentrations(
    conc_csv: str, address: str, solver_module
) -> np.ndarray:
    """Return targets in the outlet-major ordering used by ``conc_calculation``."""
    concentration_df = solver_module.conc_mat_produce(conc_csv, address)
    fractions = [
        concentration_df[f"fraction{index + 1}"].to_numpy(dtype=float)
        for index in range(len(concentration_df.columns))
    ]
    return np.vstack(fractions).T.ravel()


class _RelativeStagnationTermination(_PymooTermination):
    """Terminate after the best scalar error stagnates or the generation cap."""

    def __init__(self, max_generations: int, patience: int, relative_tolerance: float):
        super().__init__()
        self.max_generations = max_generations
        self.patience = patience
        self.relative_tolerance = relative_tolerance
        self.best_seen = np.inf
        self.stagnant_generations = 0

    def _update(self, algorithm) -> float:
        current = float(np.min(algorithm.pop.get("F")))
        if current < self.best_seen * (1.0 - self.relative_tolerance):
            self.best_seen = current
            self.stagnant_generations = 0
        else:
            self.stagnant_generations += 1
        if algorithm.n_gen >= self.max_generations:
            return 1.0
        return 1.0 if self.stagnant_generations >= self.patience else 0.0


class _PSOProgressCallback(_PymooCallback):
    """Print compact generation-level progress without extra circuit solves."""

    def __init__(self, evaluator, every_generations: int, total_generations: int, termination):
        super().__init__()
        self.evaluator = evaluator
        self.every_generations = max(1, every_generations)
        self.total_generations = total_generations
        self.termination = termination
        self.last_generation = 0

    def notify(self, algorithm) -> None:
        generation = int(algorithm.n_gen)
        if generation == self.last_generation:
            return
        if generation != 1 and generation % self.every_generations != 0:
            return
        self.last_generation = generation
        print(
            f"PSO generation {generation}/{self.total_generations}: "
            f"evaluations={self.evaluator.evaluations}, "
            f"best_objective={self.evaluator.best_error:.6e}, "
            f"stagnant_generations={self.termination.stagnant_generations}",
            flush=True,
        )


class _CircuitEvaluator:
    """Evaluate one normalized length vector through the legacy circuit solver."""

    def __init__(
        self,
        what_to_solve,
        changing_edges,
        inc_csv,
        length_csv,
        conc_csv,
        args,
        ini_list,
        inlet_node_idx,
        inlet_edge_idx,
        address,
        flow_weight,
        concentration_weight,
        lower_scale,
        upper_scale,
        lower_delta_mm,
        upper_delta_mm,
        lower_bounds,
        upper_bounds,
        solver_module,
    ):
        self.solver = solver_module
        self.what_to_solve = what_to_solve
        self.inc_csv = inc_csv
        self.address = address
        self.args = args
        self.edge_indices = np.asarray(changing_edges, dtype=int)
        self.initial_df = pd.read_csv(Path(address) / length_csv, index_col="edge")
        self.initial_length = self.initial_df["length"].to_numpy(dtype=float)
        if self.edge_indices.ndim != 1 or self.edge_indices.size == 0:
            raise ValueError("changing_edges must contain at least one edge index")
        if np.any(self.edge_indices < 0) or np.any(self.edge_indices >= self.initial_length.size):
            raise IndexError("changing_edges contains an index outside length_csv")
        initial_variables = self.initial_length[self.edge_indices]
        if (lower_bounds is None) != (upper_bounds is None):
            raise ValueError("lower_bounds and upper_bounds must be supplied together")
        if lower_bounds is not None:
            self.lower = np.asarray(lower_bounds, dtype=float)
            self.upper = np.asarray(upper_bounds, dtype=float)
            if (
                self.lower.shape != initial_variables.shape
                or self.upper.shape != initial_variables.shape
            ):
                raise ValueError("Explicit bounds must match the number of changing edges")
        else:
            self.lower = (
                initial_variables * lower_scale
                if lower_delta_mm is None
                else initial_variables + float(lower_delta_mm)
            )
            self.upper = (
                initial_variables * upper_scale
                if upper_delta_mm is None
                else initial_variables + float(upper_delta_mm)
            )
        if np.any(~np.isfinite(self.lower)) or np.any(~np.isfinite(self.upper)):
            raise ValueError("Every length bound must be finite")
        self.span = self.upper - self.lower
        if np.any(self.lower <= 0):
            raise ValueError("Every lower length bound must be positive")
        if np.any(self.span <= 0):
            raise ValueError("Every upper length bound must exceed its lower bound")
        self.initial_z = (self.initial_length[self.edge_indices] - self.lower) / self.span
        self.eta, self.width, self.height, self.n = args
        self.inc_mat = self.solver.inc_mat_produce(inc_csv, address)
        self.ini_list = ini_list
        self.inlet_list = [list(item) for item in ini_list if item[3] == "i"]
        self.outlet_list = [list(item) for item in ini_list if item[3] == "o"]
        if not self.outlet_list:
            raise ValueError("At least one outlet is required for length optimization")
        _, self.is_source_unknown, self.is_concentration_unknown = self.solver.variable_setup(
            what_to_solve
        )
        _, _, _, self.target_flow = self.solver.calculate_currents(
            what_to_solve, inc_csv, self.initial_length, ini_list, args, address
        )
        self.target_flow = np.asarray(self.target_flow, dtype=float)
        self.target_concentration = _target_concentrations(conc_csv, address, self.solver)
        self.inlet_node_idx = inlet_node_idx
        self.inlet_edge_idx = inlet_edge_idx
        self.flow_weight = flow_weight
        self.concentration_weight = concentration_weight
        self.evaluations = 0
        self.best_error = np.inf
        self.best_z: np.ndarray | None = None

    def evaluate(self, z: np.ndarray) -> dict[str, Any]:
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        candidate_length = self.initial_length.copy()
        candidate_length[self.edge_indices] = self.lower + z * self.span
        matrix = self.solver.construct_block_mat(
            candidate_length, self.eta, (self.width, self.height, self.n), self.inc_mat
        )
        rhs = self.solver.construct_RHS(self.inc_mat, self.ini_list)
        grounded_matrix, grounded_rhs, ground_index = self.solver.ground_block_mat(
            matrix, self.inc_mat, rhs, self.outlet_list
        )
        with _contextlib.redirect_stdout(_io.StringIO()):
            currents, _, solved_rhs = self.solver.Kirchhoff_solver(
                matrix, grounded_matrix, rhs, grounded_rhs, self.inc_mat,
                self.is_source_unknown, self.ini_list, self.outlet_list,
                self.inlet_list, ground_index,
            )
        inlet_current, _, _, _ = self.solver.calculate_currents(
            self.what_to_solve,
            self.inc_csv,
            candidate_length,
            self.ini_list,
            self.args,
            self.address,
        )
        if inlet_current == 0:
            raise ZeroDivisionError("Inlet flow is zero; normalized outlet flow is undefined")
        # Outlet edge currents use the opposite sign convention to positive
        # inlet flow.  Optimize physical flow fractions independent of edge orientation.
        predicted_flow = np.abs(
            np.asarray(currents[: len(self.outlet_list)], dtype=float)
        ) / abs(float(inlet_current))
        if self.is_concentration_unknown and len(self.inlet_list) > 1:
            with _contextlib.redirect_stdout(_io.StringIO()):
                concentration_vectors = self.solver.conc_calculation(
                    solved_rhs, currents, self.inc_mat, self.inlet_list,
                    self.outlet_list, self.inlet_node_idx, self.inlet_edge_idx,
                )
            predicted_concentration = np.vstack(concentration_vectors).T.ravel()
        else:
            predicted_concentration = np.empty(0, dtype=float)
        if predicted_concentration.size != self.target_concentration.size:
            raise ValueError("Predicted and target concentration dimensions differ")
        flow_residual = predicted_flow - self.target_flow
        concentration_residual = predicted_concentration - self.target_concentration
        flow_norm = float(self.flow_weight * np.linalg.norm(flow_residual))
        concentration_norm = float(self.concentration_weight * np.linalg.norm(concentration_residual))
        result = {
            "length": candidate_length,
            "flow": predicted_flow,
            "concentration": predicted_concentration,
            "flow_residual": flow_residual,
            "concentration_residual": concentration_residual,
            "flow_norm": flow_norm,
            "concentration_norm": concentration_norm,
            "combined_error": float(np.hypot(flow_norm, concentration_norm)),
        }
        self.evaluations += 1
        if result["combined_error"] < self.best_error:
            self.best_error = result["combined_error"]
            self.best_z = z.copy()
        return result


class _PSOProblem(_PymooProblem):
    def __init__(
        self,
        evaluator: _CircuitEvaluator,
        enforce_nonnegative: bool,
        feasibility_threshold: float,
    ):
        super().__init__(
            n_var=evaluator.edge_indices.size,
            n_obj=1,
            n_ieq_constr=2 if enforce_nonnegative else 0,
            xl=0.0,
            xu=1.0,
        )
        self.evaluator = evaluator
        self.enforce_nonnegative = enforce_nonnegative
        self.feasibility_threshold = feasibility_threshold

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        candidates = np.asarray(x, dtype=float)
        results = [self.evaluator.evaluate(candidate) for candidate in candidates]
        out["F"] = np.asarray([[result["combined_error"]] for result in results])
        if self.enforce_nonnegative:
            out["G"] = np.asarray(
                [
                    [
                        -np.min(result["flow"]) - self.feasibility_threshold,
                        -np.min(result["concentration"]) - self.feasibility_threshold,
                    ]
                    for result in results
                ]
            )


def execute_length_change(
    whatToSolve, changing_edges, inc_csv, length_csv, conc_csv, args, ini_list,
    inlet_node_idx, inlet_edge_idx, outlet_node_idx, outlet_edge_idx, conc_wght,
    flow_wght, address, popSize=50, nGen=None, *, max_fes_per_dim: int = 10_000,
    seed: int | None = 1, stagnation_generations: int = 300,
    relative_improvement_tol: float = 1e-6, progress_every_generations: int = 50,
    lower_scale: float = 0.8, upper_scale: float = 1.2,
    lower_delta_mm: float | None = None, upper_delta_mm: float | None = None,
    lower_bounds=None, upper_bounds=None,
    enforce_nonnegative: bool = False, feasibility_threshold: float = 1e-5,
    output_suffix: str = "pso", solver_module=None,
):
    """Optimize lengths using a bounded, scalar particle-swarm search.

    ``popSize`` is the swarm size.  If ``nGen`` is omitted, the generation cap
    is computed from ``max_fes_per_dim * number_of_design_variables``.  This
    follows the benchmark protocol while retaining the legacy 0.8x--1.2x
    length bounds by default.
    """
    if popSize < 2:
        raise ValueError("popSize must be at least 2 for PSO")
    if not 0 < lower_scale < upper_scale:
        raise ValueError("Require 0 < lower_scale < upper_scale")
    if solver_module is None:
        raise ValueError(
            "solver_module must be provided by the unified electric_analogy dispatcher"
        )
    solver = solver_module
    suffix = f"_{output_suffix}" if output_suffix else ""
    evaluator = _CircuitEvaluator(
        whatToSolve, changing_edges, inc_csv, length_csv, conc_csv, args, ini_list,
        inlet_node_idx, inlet_edge_idx, address, flow_wght, conc_wght,
        lower_scale, upper_scale, lower_delta_mm, upper_delta_mm,
        lower_bounds, upper_bounds, solver,
    )
    if nGen is None:
        nGen = max(1, math.ceil(max_fes_per_dim * evaluator.edge_indices.size / popSize))
    nGen = int(nGen)
    if nGen < 1:
        raise ValueError("nGen must be at least 1")
    termination = _RelativeStagnationTermination(
        nGen, stagnation_generations, relative_improvement_tol
    )
    callback = _PSOProgressCallback(
        evaluator, progress_every_generations, nGen, termination
    )
    print(
        f"PSO: population={popSize}, max_generations={nGen}, "
        f"max_fes={popSize * nGen}, normalized bounds=[0, 1]."
    )
    started = solver.time.time()
    result = _pymoo_minimize(
        _PSOProblem(evaluator, enforce_nonnegative, feasibility_threshold),
        _PymooPSO(pop_size=popSize), termination,
        seed=seed, verbose=False, callback=callback,
    )
    best_z = np.asarray(result.X if result.X is not None else evaluator.best_z, dtype=float)
    best = evaluator.evaluate(best_z)
    length_df = evaluator.initial_df.copy()
    length_df["length"] = best["length"]
    output_dir = Path(address)
    length_df.to_csv(output_dir / f"new_length_mat{suffix}.csv")
    pd.DataFrame([{
        "seed": seed, "population": popSize, "max_generations": nGen,
        "evaluations": evaluator.evaluations, "combined_error": best["combined_error"],
        "flow_norm": best["flow_norm"], "concentration_norm": best["concentration_norm"],
        "flow_max_abs_error": float(np.max(np.abs(best["flow_residual"]))),
        "concentration_max_abs_error": float(np.max(np.abs(best["concentration_residual"]))),
        "minimum_flow": float(np.min(best["flow"])),
        "minimum_concentration": float(np.min(best["concentration"])),
        "nonnegative_constraints_enabled": enforce_nonnegative,
        "stagnant_generations": termination.stagnant_generations,
    }]).to_csv(output_dir / f"pso_optimization_summary{suffix}.csv", index=False)
    print(
        f"PSO finished: objective={best['combined_error']:.6e}, "
        f"evaluations={evaluator.evaluations}, time={solver.time.time() - started:.2f} s.",
        flush=True,
    )
    if "optimization" in whatToSolve:
        return solver.time.time() - started, length_df
    return length_df
