"""Shared objective wrapper for optimizer benchmarks.

This module adapts the existing topology-specific ``electric_analogy.py`` files
without editing them. It prepares the same targets and bounds used by
``execute_length_change()`` and exposes a compact evaluator for NSGA-II, SLSQP,
and PSO benchmark drivers.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import importlib.util
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

try:
    from .benchmark_configs import TopologyConfig, get_topology_config
except ImportError:  # pragma: no cover - allows running this file as a script.
    from benchmark_configs import TopologyConfig, get_topology_config


WHAT_TO_SOLVE = ["length", "source", "concentration"]


@dataclass(frozen=True)
class ObjectiveResult:
    flow_diff: np.ndarray
    conc_diff: np.ndarray
    flow: np.ndarray
    conc: np.ndarray
    flow_norm: np.ndarray
    conc_norm: np.ndarray
    combined_error: np.ndarray
    flow_max_abs_error: np.ndarray
    conc_max_abs_error: np.ndarray


class BenchmarkObjective:
    """Evaluate optimizer candidates for one topology."""

    def __init__(
        self,
        config: TopologyConfig,
        *,
        flow_weight: float = 0.5,
        conc_weight: float = 0.5,
        lower_scale: float = 0.5,
        upper_scale: float = 2.0,
    ) -> None:
        self.config = config
        self.flow_weight = flow_weight
        self.conc_weight = conc_weight
        self.lower_scale = lower_scale
        self.upper_scale = upper_scale
        self.module = load_topology_module(config)
        patch_constant_geometry_functions(self.module)

        self.length_df = pd.read_csv(config.address / config.length_csv, index_col="edge")
        self.initial_length = self.length_df["length"].to_numpy(dtype=float)
        self.lower_bounds = lower_scale * self.initial_length
        self.upper_bounds = upper_scale * self.initial_length
        self.inc_mat = self.module.inc_mat_produce(config.inc_csv, str(config.address))

        self.target_conc = self._load_target_concentration()
        (
            self.inlet_current,
            self.outlet_current,
            self.target_flow_unscaled,
            self.target_flow,
        ) = self.module.calculate_currents(
            WHAT_TO_SOLVE,
            config.inc_csv,
            self.initial_length,
            _as_mutable_ini_list(config.ini_list),
            config.args,
            str(config.address),
        )

    @property
    def n_var(self) -> int:
        return len(self.config.changing_edges)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return list(zip(self.lower_bounds, self.upper_bounds, strict=True))

    @property
    def normalized_bounds(self) -> list[tuple[float, float]]:
        """Unit-cube bounds used by every reviewer-facing optimizer."""
        return [(0.0, 1.0)] * self.n_var

    @property
    def initial_normalized(self) -> np.ndarray:
        return self.encode(self.initial_length)

    def encode(self, lengths: np.ndarray) -> np.ndarray:
        """Map physical channel lengths into the unit hypercube."""
        lengths = np.asarray(lengths, dtype=float)
        return (lengths - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def decode(self, normalized: np.ndarray) -> np.ndarray:
        """Map unit-hypercube coordinates back into physical channel lengths."""
        normalized = np.asarray(normalized, dtype=float)
        return self.lower_bounds + normalized * (self.upper_bounds - self.lower_bounds)

    def evaluate_lengths(self, candidates: np.ndarray) -> ObjectiveResult:
        """Evaluate one or more candidate length vectors.

        Parameters
        ----------
        candidates:
            Shape ``(n_var,)`` for one candidate or ``(n_candidates, n_var)`` for
            vectorized evaluation. For the current a/b/c/d topologies,
            ``changing_edges`` spans every edge, so each candidate is a complete
            length vector.
        """
        x = np.asarray(candidates, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.n_var:
            raise ValueError(
                f"{self.config.name}: expected {self.n_var} variables, got {x.shape[1]}"
            )

        if tuple(self.config.changing_edges) != tuple(range(len(self.initial_length))):
            raise NotImplementedError(
                "The benchmark wrapper currently expects changing_edges to cover "
                "all channel lengths. Extend candidates into full length vectors "
                "before calling the existing Pareto evaluator for partial-edge cases."
            )

        flow, conc = self._evaluate_direct(x)

        flow = np.asarray(flow, dtype=float)
        conc = np.asarray(conc, dtype=float)
        flow_diff = np.abs(self.target_flow - flow)
        conc_diff = np.abs(self.target_conc - conc)

        flow_norm = np.linalg.norm(flow_diff, axis=1) * self.flow_weight
        conc_norm = np.linalg.norm(conc_diff, axis=1) * self.conc_weight
        combined_error = np.sqrt(flow_norm**2 + conc_norm**2)
        flow_max_abs_error = np.max(flow_diff, axis=1)
        conc_max_abs_error = np.max(conc_diff, axis=1)

        return ObjectiveResult(
            flow_diff=flow_diff,
            conc_diff=conc_diff,
            flow=flow,
            conc=conc,
            flow_norm=flow_norm,
            conc_norm=conc_norm,
            combined_error=combined_error,
            flow_max_abs_error=flow_max_abs_error,
            conc_max_abs_error=conc_max_abs_error,
        )

    def evaluate_normalized(self, candidates: np.ndarray) -> ObjectiveResult:
        """Evaluate one or more unit-hypercube candidates."""
        z = np.asarray(candidates, dtype=float)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        if np.any(z < 0.0) or np.any(z > 1.0):
            raise ValueError("Normalized optimizer coordinates must lie in [0, 1].")
        return self.evaluate_lengths(self.decode(z))

    def evaluate(self, candidates: np.ndarray) -> ObjectiveResult:
        """Backward-compatible alias for physical-length evaluation."""
        return self.evaluate_lengths(candidates)

    def scalar_objective(self, candidate: np.ndarray) -> float:
        """Return scalar combined error for local/swarm optimizers."""
        return float(self.evaluate(candidate).combined_error[0])

    def scalar_normalized(self, candidate: np.ndarray) -> float:
        """Return the common scalar target-matching objective in z-coordinates."""
        return float(self.evaluate_normalized(candidate).combined_error[0])

    def multi_objective(self, candidates: np.ndarray) -> np.ndarray:
        """Return two-column objective values for multi-objective optimizers."""
        result = self.evaluate(candidates)
        return np.vstack((result.flow_norm, result.conc_norm)).T

    def multi_objective_normalized(self, candidates: np.ndarray) -> np.ndarray:
        """Return two objective columns for unit-hypercube candidates."""
        result = self.evaluate_normalized(candidates)
        return np.vstack((result.flow_norm, result.conc_norm)).T

    def _load_target_concentration(self) -> np.ndarray:
        conc_df = pd.read_csv(self.config.address / self.config.conc_csv)
        target_conc_cols = []
        for col in conc_df.columns:
            if col.startswith("fraction"):
                target_conc_cols.append(conc_df[[col]].to_numpy())
        if not target_conc_cols:
            raise ValueError(f"{self.config.name}: no fraction columns found")
        return np.vstack(tuple(target_conc_cols)).T.flatten()

    def _evaluate_direct(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate candidates through the original solver functions in memory."""
        flows = []
        concs = []
        for candidate in candidates:
            i_vec, conc_vec = self._solve_candidate(candidate)
            flows.append(self._scale_outlet_flow(np.asarray(i_vec, dtype=float)))
            concs.append(self._flatten_concentrations(conc_vec))

        return np.vstack(flows), np.vstack(concs)

    def _solve_candidate(self, candidate: np.ndarray) -> tuple[np.ndarray, list[np.ndarray] | None]:
        eta, w, h, n = self.config.args
        _, is_source_unknown, is_conc_unknown = self.module.variable_setup(WHAT_TO_SOLVE)
        ini_list = _as_mutable_ini_list(self.config.ini_list)
        ini_list_outlet = [item for item in ini_list if item[3] == "o"]
        ini_list_inlet = [item for item in ini_list if item[3] == "i"]

        arg = (w, h, n)
        a_mat = self.module.construct_block_mat(candidate, eta, arg, self.inc_mat)
        b_f = self.module.construct_RHS(self.inc_mat, ini_list)
        a_ground, b_f_ground, gnd_idx = self.module.ground_block_mat(
            a_mat,
            self.inc_mat,
            b_f,
            ini_list_outlet,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            i_vec, _, b_f = self.module.Kirchhoff_solver(
                a_mat,
                a_ground,
                b_f,
                b_f_ground,
                self.inc_mat,
                is_source_unknown,
                ini_list,
                ini_list_outlet,
                ini_list_inlet,
                gnd_idx,
            )

        conc_vec = None
        if is_conc_unknown and len(ini_list_inlet) > 1:
            conc_vec = self.module.conc_calculation(
                b_f,
                i_vec,
                self.inc_mat,
                ini_list_inlet,
                ini_list_outlet,
                self.config.inlet_node_idx,
                self.config.inlet_edge_idx,
            )

        return i_vec, conc_vec

    def _scale_outlet_flow(self, i_vec: np.ndarray) -> np.ndarray:
        outlet_count = len(self.target_flow)
        outlet_flow = i_vec[:outlet_count]
        if self.inlet_current == 0:
            raise ZeroDivisionError(f"{self.config.name}: inlet current is zero")
        return outlet_flow / (-1.0 * self.inlet_current)

    def _flatten_concentrations(self, conc_vec: list[np.ndarray] | None) -> np.ndarray:
        if conc_vec is None:
            return np.array([], dtype=float)
        outlet_count = len(self.target_flow)
        return np.concatenate(
            tuple(np.asarray(item, dtype=float)[:outlet_count] for item in conc_vec)
        )


def load_topology_module(config: TopologyConfig) -> ModuleType:
    """Load one topology-specific electric_analogy.py with local resistance.py."""
    module_path = config.address / "electric_analogy.py"
    module_name = f"_benchmark_electric_analogy_{config.name}"

    # The original files import ``resistance`` as a local top-level module.
    # Remove prior topology modules so each load resolves against config.address.
    sys.modules.pop("resistance", None)
    sys.modules.pop(module_name, None)

    original_path = copy.copy(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(config.address))
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path = original_path
        sys.dont_write_bytecode = original_dont_write_bytecode


def patch_constant_geometry_functions(module: ModuleType) -> None:
    """Cache expensive resistance geometry factors within one loaded topology."""
    if hasattr(module, "rectangular"):
        module.rectangular = functools.lru_cache(maxsize=None)(module.rectangular)
    if hasattr(module, "circular"):
        module.circular = functools.lru_cache(maxsize=None)(module.circular)


def _as_mutable_ini_list(ini_list: tuple[tuple[object, ...], ...]) -> list[list[object]]:
    return [list(item) for item in ini_list]


def make_objective(
    name: str,
    *,
    flow_weight: float = 0.5,
    conc_weight: float = 0.5,
    lower_scale: float = 0.5,
    upper_scale: float = 2.0,
) -> BenchmarkObjective:
    """Create a benchmark objective by topology name."""
    return BenchmarkObjective(
        get_topology_config(name),
        flow_weight=flow_weight,
        conc_weight=conc_weight,
        lower_scale=lower_scale,
        upper_scale=upper_scale,
    )


def main() -> None:
    """Smoke-test all configured topologies at their initial lengths."""
    for name in ("a", "b", "c", "d"):
        objective = make_objective(name)
        result = objective.evaluate(objective.initial_length)
        print(
            name,
            f"n_var={objective.n_var}",
            f"flow_norm={result.flow_norm[0]:.6g}",
            f"conc_norm={result.conc_norm[0]:.6g}",
            f"combined={result.combined_error[0]:.6g}",
        )


if __name__ == "__main__":
    main()
