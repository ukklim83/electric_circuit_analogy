"""Run a reviewer-facing optimizer benchmark on the shared circuit solver.

Protocol
--------
* All optimizers work in normalized channel-length coordinates ``z in [0, 1]^D``.
* Each topology receives ``max_fes_per_dim * D`` objective evaluations.
* NSGA-II preserves the manuscript population size of 600.
* PSO uses a 50-particle swarm with an early stagnation termination.
* SLSQP performs every planned restart independently: the original design plus
  Latin-hypercube starts.  Each restart has its own evaluation allowance.

The output directory contains both one row per outer run (``raw_runs.csv``)
and one row per SLSQP restart (``raw_slsqp_restarts.csv``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import qmc

try:
    from .objective_wrapper import BenchmarkObjective, ObjectiveResult, make_objective
except ImportError:  # pragma: no cover - allows running this file as a script.
    from objective_wrapper import BenchmarkObjective, ObjectiveResult, make_objective


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results_reviewer_multistart_v2"


@dataclass
class OptimizerSettings:
    max_fes_per_dim: int = 10_000
    nsga2_pop_size: int = 600
    pso_pop_size: int = 50
    slsqp_starts: int = 20
    slsqp_maxiter: int = 1_000
    lower_scale: float = 1.0 / 3.0
    upper_scale: float = 3.0
    flow_abs_tolerance: float = 1e-3
    conc_abs_tolerance: float = 1e-3
    pso_stagnation_generations: int = 300
    pso_rel_improvement_tol: float = 1e-6
    nsga2_progress_every_generations: int = 10
    pso_progress_every_generations: int = 100
    slsqp_progress_every_iterations: int = 10


@dataclass
class RunRecord:
    topology: str
    algorithm: str
    seed: int
    max_fes: int
    n_eval: int
    runtime_s: float
    flow_norm: float
    conc_norm: float
    combined_error: float
    flow_max_abs_error: float
    conc_max_abs_error: float
    success: bool
    message: str
    length_file: str
    planned_starts: int = 1
    attempted_starts: int = 1
    normal_exit_count: int = 0
    budget_exhausted_count: int = 0
    best_restart_index: int | None = None


@dataclass
class RestartRecord:
    topology: str
    outer_seed: int
    start_index: int
    start_type: str
    max_fes: int
    n_eval: int
    runtime_s: float
    exit_status: str
    scipy_success: bool
    scipy_message: str
    flow_norm: float
    conc_norm: float
    combined_error: float
    flow_max_abs_error: float
    conc_max_abs_error: float
    selected_as_best: bool = False


class EvaluationBudgetExceeded(RuntimeError):
    """Raised before a solver would exceed its independent evaluation budget."""


class ProgressReporter:
    """Emit compact, append-only progress events without evaluating the objective."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")

    @staticmethod
    def _display_value(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def emit(self, event: str, **fields: object) -> None:
        """Print one human-readable event and persist the same event as JSONL."""
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        payload = {
            "timestamp": timestamp,
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        display = " | ".join(
            f"{key}={self._display_value(value)}" for key, value in fields.items()
        )
        print(f"[{timestamp}] {event}" + (f" | {display}" if display else ""), flush=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fh.flush()


class PymooProgressCallback(Callback):
    """Report generation-level progress using existing optimizer state only."""

    def __init__(
        self,
        reporter: ProgressReporter,
        counted: "CountedObjective",
        topology: str,
        algorithm: str,
        seed: int,
        total_generations: int,
        evaluation_limit: int,
        every_generations: int,
        termination: "RelativeStagnationTermination | None" = None,
    ) -> None:
        super().__init__()
        self.reporter = reporter
        self.counted = counted
        self.topology = topology
        self.algorithm = algorithm
        self.seed = seed
        self.total_generations = total_generations
        self.evaluation_limit = evaluation_limit
        self.every_generations = max(1, every_generations)
        self.termination = termination
        self.last_reported_generation = 0

    def notify(self, algorithm) -> None:
        generation = int(algorithm.n_gen)
        if generation == self.last_reported_generation:
            return
        if (
            generation != 1
            and generation % self.every_generations != 0
            and generation < self.total_generations
        ):
            return
        self.last_reported_generation = generation
        fields: dict[str, object] = {
            "topology": self.topology,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "generation": generation,
            "total_generations": self.total_generations,
            "n_eval": self.counted.n_eval,
            "max_fes": self.evaluation_limit,
            "progress_pct": 100.0 * self.counted.n_eval / self.evaluation_limit,
            "best_combined_error": self.counted.best_error,
        }
        if self.termination is not None:
            fields["stagnant_generations"] = self.termination.stagnant_generations
        self.reporter.emit("GENERATION", **fields)


class CountedObjective:
    """Count normalized objective evaluations and retain the best sampled point."""

    def __init__(self, objective: BenchmarkObjective, max_eval: int | None = None) -> None:
        self.objective = objective
        self.n_eval = 0
        self.max_eval = max_eval
        self.best_z: np.ndarray | None = None
        self.best_error = np.inf

    def evaluate(self, candidates: np.ndarray) -> ObjectiveResult:
        z = np.asarray(candidates, dtype=float)
        count = 1 if z.ndim == 1 else z.shape[0]
        if self.max_eval is not None and self.n_eval + count > self.max_eval:
            raise EvaluationBudgetExceeded(
                f"Evaluation budget would be exceeded: {self.n_eval + count} > {self.max_eval}"
            )
        self.n_eval += count
        result = self.objective.evaluate_normalized(z)
        rows = z.reshape(count, -1)
        best_idx = int(np.argmin(result.combined_error))
        best_error = float(result.combined_error[best_idx])
        if np.isfinite(best_error) and best_error < self.best_error:
            self.best_error = best_error
            self.best_z = rows[best_idx].copy()
        return result

    def scalar(self, candidate: np.ndarray) -> float:
        return float(self.evaluate(candidate).combined_error[0])

    def multi(self, candidates: np.ndarray) -> np.ndarray:
        result = self.evaluate(candidates)
        return np.vstack((result.flow_norm, result.conc_norm)).T


class PymooMultiObjectiveProblem(Problem):
    def __init__(self, counted: CountedObjective) -> None:
        super().__init__(
            n_var=counted.objective.n_var,
            n_obj=2,
            n_constr=0,
            xl=0.0,
            xu=1.0,
        )
        self.counted = counted

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        out["F"] = self.counted.multi(x)


class PymooScalarProblem(Problem):
    def __init__(self, counted: CountedObjective) -> None:
        super().__init__(
            n_var=counted.objective.n_var,
            n_obj=1,
            n_constr=0,
            xl=0.0,
            xu=1.0,
        )
        self.counted = counted

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        out["F"] = self.counted.evaluate(x).combined_error.reshape(-1, 1)


class RelativeStagnationTermination(Termination):
    """Stop PSO when its scalar best has not improved for a fixed window."""

    def __init__(
        self, max_generations: int, patience: int, relative_tolerance: float
    ) -> None:
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


def max_fes(objective: BenchmarkObjective, settings: OptimizerSettings) -> int:
    return settings.max_fes_per_dim * objective.n_var


def final_result(objective: BenchmarkObjective, z: np.ndarray) -> ObjectiveResult:
    return objective.evaluate_normalized(np.asarray(z, dtype=float))


def is_success(result: ObjectiveResult, settings: OptimizerSettings) -> bool:
    return bool(
        result.flow_max_abs_error[0] <= settings.flow_abs_tolerance
        and result.conc_max_abs_error[0] <= settings.conc_abs_tolerance
    )


def make_record(
    objective: BenchmarkObjective,
    algorithm: str,
    seed: int,
    evaluation_limit: int,
    n_eval: int,
    runtime_s: float,
    result: ObjectiveResult,
    message: str,
    settings: OptimizerSettings,
    **extra,
) -> RunRecord:
    return RunRecord(
        topology=objective.config.name,
        algorithm=algorithm,
        seed=seed,
        max_fes=evaluation_limit,
        n_eval=n_eval,
        runtime_s=runtime_s,
        flow_norm=float(result.flow_norm[0]),
        conc_norm=float(result.conc_norm[0]),
        combined_error=float(result.combined_error[0]),
        flow_max_abs_error=float(result.flow_max_abs_error[0]),
        conc_max_abs_error=float(result.conc_max_abs_error[0]),
        success=is_success(result, settings),
        message=message,
        length_file="",
        **extra,
    )


def run_nsga2(
    objective: BenchmarkObjective,
    seed: int,
    settings: OptimizerSettings,
    reporter: ProgressReporter,
) -> tuple[np.ndarray, RunRecord, list[RestartRecord]]:
    evaluation_limit = max_fes(objective, settings)
    n_gen = max(1, evaluation_limit // settings.nsga2_pop_size)
    counted = CountedObjective(objective, max_eval=n_gen * settings.nsga2_pop_size)
    problem = PymooMultiObjectiveProblem(counted)
    callback = PymooProgressCallback(
        reporter,
        counted,
        objective.config.name,
        "NSGA-II",
        seed,
        n_gen,
        evaluation_limit,
        settings.nsga2_progress_every_generations,
    )

    started = time.perf_counter()
    result = pymoo_minimize(
        problem,
        NSGA2(pop_size=settings.nsga2_pop_size),
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
        callback=callback,
    )
    runtime = time.perf_counter() - started

    zs = np.asarray(result.X, dtype=float)
    if zs.ndim == 1:
        zs = zs.reshape(1, -1)
    final = objective.evaluate_normalized(zs)
    best_z = zs[int(np.argmin(final.combined_error))]
    best = final_result(objective, best_z)
    record = make_record(
        objective, "NSGA-II", seed, evaluation_limit, counted.n_eval, runtime,
        best, f"completed {n_gen} generations", settings,
    )
    return best_z, record, []


def make_slsqp_starts(objective: BenchmarkObjective, seed: int, n_starts: int) -> np.ndarray:
    if n_starts < 1:
        raise ValueError("slsqp_starts must be at least one")
    if n_starts == 1:
        return objective.initial_normalized.reshape(1, -1)
    sampler = qmc.LatinHypercube(d=objective.n_var, seed=seed)
    return np.vstack((objective.initial_normalized, sampler.random(n_starts - 1)))


def run_slsqp(
    objective: BenchmarkObjective,
    seed: int,
    settings: OptimizerSettings,
    reporter: ProgressReporter,
) -> tuple[np.ndarray, RunRecord, list[RestartRecord]]:
    evaluation_limit = max_fes(objective, settings)
    per_start_limit = evaluation_limit // settings.slsqp_starts
    starts = make_slsqp_starts(objective, seed, settings.slsqp_starts)
    restarts: list[RestartRecord] = []
    best_z: np.ndarray | None = None
    best_error = np.inf
    best_restart_index: int | None = None
    started_outer = time.perf_counter()

    for idx, z0 in enumerate(starts):
        counted = CountedObjective(objective, max_eval=per_start_limit)
        started = time.perf_counter()
        iteration = 0
        start_type = "initial" if idx == 0 else "latin_hypercube"
        reporter.emit(
            "SLSQP_START",
            topology=objective.config.name,
            algorithm="SLSQP",
            seed=seed,
            start_index=idx + 1,
            total_starts=len(starts),
            start_type=start_type,
            max_fes=per_start_limit,
        )

        def report_iteration(_xk: np.ndarray, *_args: object) -> None:
            nonlocal iteration
            iteration += 1
            if (
                iteration != 1
                and iteration % settings.slsqp_progress_every_iterations != 0
            ):
                return
            reporter.emit(
                "SLSQP_ITERATION",
                topology=objective.config.name,
                algorithm="SLSQP",
                seed=seed,
                start_index=idx + 1,
                total_starts=len(starts),
                iteration=iteration,
                n_eval=counted.n_eval,
                max_fes=per_start_limit,
                progress_pct=100.0 * counted.n_eval / per_start_limit,
                best_combined_error=counted.best_error,
            )

        scipy_result = None
        exit_status = "numerical_failure"
        message = ""
        try:
            scipy_result = scipy_minimize(
                counted.scalar,
                z0,
                method="SLSQP",
                jac="2-point",
                bounds=objective.normalized_bounds,
                callback=report_iteration,
                options={
                    "maxiter": settings.slsqp_maxiter,
                    "ftol": 1e-12,
                    "finite_diff_rel_step": 1e-6,
                    "disp": False,
                },
            )
            candidate = np.asarray(scipy_result.x, dtype=float)
            exit_status = "converged" if scipy_result.success else "solver_stopped"
            message = str(scipy_result.message)
        except EvaluationBudgetExceeded as exc:
            candidate = counted.best_z
            exit_status = "budget_exhausted"
            message = str(exc)
        except Exception as exc:  # pragma: no cover - retained as run evidence.
            candidate = counted.best_z
            message = f"{type(exc).__name__}: {exc}"

        if candidate is None:
            candidate = np.asarray(z0, dtype=float)
        result = final_result(objective, candidate)
        runtime = time.perf_counter() - started
        restart = RestartRecord(
            topology=objective.config.name,
            outer_seed=seed,
            start_index=idx,
            start_type="initial" if idx == 0 else "latin_hypercube",
            max_fes=per_start_limit,
            n_eval=counted.n_eval,
            runtime_s=runtime,
            exit_status=exit_status,
            scipy_success=bool(scipy_result is not None and scipy_result.success),
            scipy_message=message,
            flow_norm=float(result.flow_norm[0]),
            conc_norm=float(result.conc_norm[0]),
            combined_error=float(result.combined_error[0]),
            flow_max_abs_error=float(result.flow_max_abs_error[0]),
            conc_max_abs_error=float(result.conc_max_abs_error[0]),
        )
        restarts.append(restart)
        reporter.emit(
            "SLSQP_END",
            topology=objective.config.name,
            algorithm="SLSQP",
            seed=seed,
            start_index=idx + 1,
            total_starts=len(starts),
            start_type=start_type,
            iterations=iteration,
            n_eval=counted.n_eval,
            max_fes=per_start_limit,
            exit_status=exit_status,
            best_combined_error=restart.combined_error,
            runtime_s=runtime,
        )
        if np.isfinite(restart.combined_error) and restart.combined_error < best_error:
            best_error = restart.combined_error
            best_z = candidate.copy()
            best_restart_index = idx

    if best_z is None:  # Defensive fallback preserving the original design.
        best_z = objective.initial_normalized.copy()
        best_restart_index = 0
    restarts[best_restart_index].selected_as_best = True
    best = final_result(objective, best_z)
    total_n_eval = sum(item.n_eval for item in restarts)
    normal_exits = sum(item.exit_status == "converged" for item in restarts)
    budget_exhausted = sum(item.exit_status == "budget_exhausted" for item in restarts)
    record = make_record(
        objective,
        "SLSQP",
        seed,
        evaluation_limit,
        total_n_eval,
        time.perf_counter() - started_outer,
        best,
        f"best of {len(restarts)} independent restarts",
        settings,
        planned_starts=settings.slsqp_starts,
        attempted_starts=len(restarts),
        normal_exit_count=normal_exits,
        budget_exhausted_count=budget_exhausted,
        best_restart_index=best_restart_index,
    )
    return best_z, record, restarts


def run_pso(
    objective: BenchmarkObjective,
    seed: int,
    settings: OptimizerSettings,
    reporter: ProgressReporter,
) -> tuple[np.ndarray, RunRecord, list[RestartRecord]]:
    evaluation_limit = max_fes(objective, settings)
    n_gen = max(1, evaluation_limit // settings.pso_pop_size)
    counted = CountedObjective(objective, max_eval=n_gen * settings.pso_pop_size)
    problem = PymooScalarProblem(counted)
    termination = RelativeStagnationTermination(
        n_gen, settings.pso_stagnation_generations, settings.pso_rel_improvement_tol
    )
    callback = PymooProgressCallback(
        reporter,
        counted,
        objective.config.name,
        "PSO",
        seed,
        n_gen,
        evaluation_limit,
        settings.pso_progress_every_generations,
        termination,
    )

    started = time.perf_counter()
    result = pymoo_minimize(
        problem,
        PSO(pop_size=settings.pso_pop_size),
        termination,
        seed=seed,
        verbose=False,
        callback=callback,
    )
    runtime = time.perf_counter() - started
    best_z = np.asarray(result.X, dtype=float)
    best = final_result(objective, best_z)
    record = make_record(
        objective, "PSO", seed, evaluation_limit, counted.n_eval, runtime,
        best, "converged or stopped after relative stagnation", settings,
    )
    return best_z, record, []


def save_length_vector(
    objective: BenchmarkObjective, z: np.ndarray, record: RunRecord, results_dir: Path
) -> RunRecord:
    length_dir = results_dir / "optimized_lengths"
    length_dir.mkdir(parents=True, exist_ok=True)
    algorithm = record.algorithm.lower().replace("-", "").replace(" ", "_")
    path = length_dir / f"{record.topology}_{algorithm}_seed{record.seed}.csv"
    length_df = objective.length_df.copy()
    length_df["length"] = objective.decode(z)
    length_df.to_csv(path)
    record.length_file = str(path)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", nargs="+", default=["a", "b", "c", "d"])
    parser.add_argument(
        "--algorithms", nargs="+", default=["nsga2", "slsqp", "pso"],
        choices=["nsga2", "slsqp", "pso"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 31)))
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-fes-per-dim", type=int, default=10_000)
    parser.add_argument("--nsga2-pop-size", type=int, default=600)
    parser.add_argument("--pso-pop-size", type=int, default=50)
    parser.add_argument("--slsqp-starts", type=int, default=20)
    parser.add_argument("--slsqp-maxiter", type=int, default=1_000)
    parser.add_argument("--flow-abs-tolerance", type=float, default=1e-3)
    parser.add_argument("--conc-abs-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--nsga2-progress-every-generations",
        type=int,
        default=10,
        help="Emit an NSGA-II progress event every N generations.",
    )
    parser.add_argument(
        "--pso-progress-every-generations",
        type=int,
        default=100,
        help="Emit a PSO progress event every N generations.",
    )
    parser.add_argument(
        "--slsqp-progress-every-iterations",
        type=int,
        default=10,
        help="Emit an SLSQP progress event every N accepted iterations.",
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=None,
        help="Append structured JSONL progress events here (default: results-dir/progress_events.jsonl).",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a small wiring-check protocol.")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> OptimizerSettings:
    settings = OptimizerSettings(
        max_fes_per_dim=args.max_fes_per_dim,
        nsga2_pop_size=args.nsga2_pop_size,
        pso_pop_size=args.pso_pop_size,
        slsqp_starts=args.slsqp_starts,
        slsqp_maxiter=args.slsqp_maxiter,
        flow_abs_tolerance=args.flow_abs_tolerance,
        conc_abs_tolerance=args.conc_abs_tolerance,
        nsga2_progress_every_generations=max(1, args.nsga2_progress_every_generations),
        pso_progress_every_generations=max(1, args.pso_progress_every_generations),
        slsqp_progress_every_iterations=max(1, args.slsqp_progress_every_iterations),
    )
    if args.smoke:
        settings.max_fes_per_dim = 10
        settings.nsga2_pop_size = 4
        settings.pso_pop_size = 5
        settings.slsqp_starts = 2
        settings.slsqp_maxiter = 5
        settings.pso_stagnation_generations = 3
    return settings


def write_outputs(
    records: list[dict], restart_records: list[dict], results_dir: Path,
    settings: OptimizerSettings,
) -> None:
    runs = pd.DataFrame(records)
    restarts = pd.DataFrame(restart_records)
    runs.to_csv(results_dir / "raw_runs.csv", index=False)
    restarts.to_csv(results_dir / "raw_slsqp_restarts.csv", index=False)
    if not runs.empty:
        summary = (
            runs.groupby(["topology", "algorithm"], as_index=False)
            .agg(
                runs=("seed", "count"),
                success_rate=("success", "mean"),
                median_combined_error=("combined_error", "median"),
                best_combined_error=("combined_error", "min"),
                median_flow_norm=("flow_norm", "median"),
                median_conc_norm=("conc_norm", "median"),
                median_n_eval=("n_eval", "median"),
                median_runtime_s=("runtime_s", "median"),
                median_normal_exit_count=("normal_exit_count", "median"),
                median_budget_exhausted_count=("budget_exhausted_count", "median"),
            )
        )
        summary.to_csv(results_dir / "summary_by_topology_algorithm.csv", index=False)
    with (results_dir / "settings.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(settings), fh, indent=2)


def main() -> None:
    args = parse_args()
    settings = settings_from_args(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    progress_log = args.progress_log or args.results_dir / "progress_events.jsonl"
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter(progress_log)
    runners: dict[str, Callable[..., tuple[np.ndarray, RunRecord, list[RestartRecord]]]] = {
        "nsga2": run_nsga2,
        "slsqp": run_slsqp,
        "pso": run_pso,
    }
    records: list[dict] = []
    restart_records: list[dict] = []

    for topology in args.topologies:
        objective = make_objective(
            topology,
            lower_scale=settings.lower_scale,
            upper_scale=settings.upper_scale,
        )
        for seed in args.seeds:
            for algorithm in args.algorithms:
                algorithm_label = {"nsga2": "NSGA-II", "slsqp": "SLSQP", "pso": "PSO"}[algorithm]
                evaluation_limit = max_fes(objective, settings)
                reporter.emit(
                    "RUN_START",
                    topology=topology,
                    algorithm=algorithm_label,
                    seed=seed,
                    max_fes=evaluation_limit,
                    progress_log=str(progress_log),
                )
                try:
                    z, record, restarts = runners[algorithm](objective, seed, settings, reporter)
                except Exception as exc:
                    reporter.emit(
                        "RUN_ERROR",
                        topology=topology,
                        algorithm=algorithm_label,
                        seed=seed,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise
                records.append(asdict(save_length_vector(objective, z, record, args.results_dir)))
                restart_records.extend(asdict(item) for item in restarts)
                write_outputs(records, restart_records, args.results_dir, settings)
                reporter.emit(
                    "RUN_END",
                    topology=topology,
                    algorithm=algorithm_label,
                    seed=seed,
                    n_eval=record.n_eval,
                    max_fes=record.max_fes,
                    best_combined_error=record.combined_error,
                    flow_max_abs_error=record.flow_max_abs_error,
                    conc_max_abs_error=record.conc_max_abs_error,
                    success=record.success,
                    runtime_s=record.runtime_s,
                    checkpoint="raw_runs.csv",
                )


if __name__ == "__main__":
    main()
