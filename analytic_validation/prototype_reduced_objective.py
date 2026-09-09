"""Eight-resistance symbolic prototype and its numerical reduced objective.

The legacy analytic derivation expands individual residuals with SymPy.  Full
expansion of the coupled concentration normal equations is prohibitively large
for repeated Hessian scans.  This module therefore preserves the *exact*
symbolic matrix formulation (with state variables eliminated as matrix
inverses) and evaluates the same equations numerically for each resistance
vector.  No surrogate model is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import sympy as sp


Q_DESIRED = 1.0e-6 / 60.0
LOWER_BOUND = 5.0e6
UPPER_BOUND = 1.0e11


@dataclass(frozen=True)
class PrototypeObjective:
    """Exact matrix definition plus a callable reduced scalar objective."""

    resistance_symbols: tuple[sp.Symbol, ...]
    normalized_symbols: tuple[sp.Symbol, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    flow_system: tuple[sp.Matrix, sp.Matrix]
    concentration_system: tuple[sp.Matrix, sp.Matrix, sp.Matrix]
    q_out_symbolic: sp.Matrix
    x_out_symbolic: sp.Matrix
    objective_definition: sp.Expr
    evaluate: Callable[[np.ndarray], float]

    def resistance_from_normalized(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.shape != (8,):
            raise ValueError(f"Expected an eight-variable normalized vector, got {z.shape}.")
        return self.lower_bounds + z * (self.upper_bounds - self.lower_bounds)


def _incidence_numpy() -> np.ndarray:
    return np.array(
        [
            [1, 0, -1, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0],
            [0, 0, 1, 0, 0, -1, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 1, 0, -1, 0, 0],
            [0, 0, 0, 0, 1, 0, -1, 0],
            [0, 0, 0, 0, 0, 1, 0, -1],
        ],
        dtype=float,
    )


def _junction_constraints(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the legacy split-node continuity rows for this topology."""
    rows: list[np.ndarray] = []
    for node in range(A.shape[1]):
        column = A[:, node]
        if np.count_nonzero(column == -1) > 1:
            incoming = (column == 1).astype(float)
            for outgoing_edge in np.flatnonzero(column == -1):
                row = incoming.copy()
                row[outgoing_edge] = -1.0
                rows.append(row)
    if not rows:
        return np.empty((0, 2)), np.empty((0, 4)), np.empty((0, 2))
    E = np.vstack(rows)
    E_om, E_i = np.hsplit(E, [6])
    E_o, E_m = np.hsplit(E_om, [2])
    return E_o, E_m, E_i


def _numeric_reduced_objective(z: np.ndarray) -> float:
    """Evaluate the exact prototype equations after numerical state elimination."""
    z = np.asarray(z, dtype=float)
    if z.shape != (8,):
        raise ValueError(f"Expected an eight-variable normalized vector, got {z.shape}.")
    if np.any(z < 0.0) or np.any(z > 1.0):
        raise ValueError("Normalized prototype variables must lie in [0, 1].")
    r = LOWER_BOUND + z * (UPPER_BOUND - LOWER_BOUND)
    A = _incidence_numpy()

    # Exact numerical counterpart of the augmented symbolic flow system.
    base = np.block([[np.diag(r), A], [A.T, np.zeros((8, 8))]])
    constraints = np.zeros((16, 2))
    constraints[8, 0] = 1.0
    constraints[9, 1] = 1.0
    lhs = np.block([[base, constraints], [constraints.T, np.zeros((2, 2))]])
    rhs = np.concatenate([np.zeros(8), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Q_DESIRED, -Q_DESIRED], np.zeros(2)])
    q = np.linalg.solve(lhs, rhs)[:8]

    q_out, q_mid, q_in = q[:2], q[2:6], q[6:8]
    Q_out, Q_mid, Q_in = np.diag(q_out), np.diag(q_mid), np.diag(q_in)
    Q_target = np.diag([Q_DESIRED, Q_DESIRED])
    F_in = np.diag([-Q_DESIRED, -Q_DESIRED])

    A_oo, A_om, A_oi = A[:2, :2], A[:2, 2:6], A[:2, 6:8]
    A_mo, A_mm, A_mi = A[2:6, :2], A[2:6, 2:6], A[2:6, 6:8]
    A_io, A_im, A_ii = A[6:8, :2], A[6:8, 2:6], A[6:8, 6:8]
    E_o, E_m, E_i = _junction_constraints(A)
    lhs_conc = np.vstack(
        [
            np.hstack([A_oo.T @ Q_out - Q_target, A_mo.T @ Q_mid]),
            np.hstack([A_om.T @ Q_out, A_mm.T @ Q_mid]),
            np.hstack([A_oi.T @ Q_out, A_mi.T @ Q_mid]),
            np.hstack([E_o, E_m]),
        ]
    )
    normal = lhs_conc.T @ lhs_conc

    def solve_component(inlet: np.ndarray) -> np.ndarray:
        rhs_conc = np.concatenate(
            [
                -A_io.T @ Q_in @ inlet,
                -A_im.T @ Q_in @ inlet,
                (F_in - A_ii.T @ Q_in) @ inlet,
                -E_i @ inlet,
            ]
        )
        return np.linalg.solve(normal, lhs_conc.T @ rhs_conc)

    x1 = solve_component(np.array([1.0, 0.0]))
    x2 = solve_component(np.array([0.0, 1.0]))
    x_out = np.array([x1[0], x1[1], x2[0], x2[1]])
    x_target = np.array([0.7, 0.3, 0.3, 0.7])

    flow_term = np.sum((q_out - Q_DESIRED) ** 2) / (2.0 * Q_DESIRED) ** 2
    concentration_term = np.sum((x_out - x_target) ** 2)
    objective = float(flow_term + concentration_term)
    if not np.isfinite(objective):
        raise FloatingPointError("Prototype objective is non-finite at the requested point.")
    return objective


@lru_cache(maxsize=1)
def build_prototype_objective() -> PrototypeObjective:
    """Build the symbolic matrix definition and numerical scalar evaluator."""
    names = ("r_o_1", "r_o_2", "r_m_1", "r_m_2", "r_m_3", "r_m_4", "r_i_1", "r_i_2")
    r = sp.symbols(" ".join(names), positive=True)
    z = sp.symbols("z_0:8", real=True)
    A = sp.Matrix(_incidence_numpy().astype(int))
    R = sp.diag(*r)
    base = R.row_join(A).col_join(A.T.row_join(sp.zeros(8, 8)))
    constraints = sp.zeros(16, 2)
    constraints[8, 0] = 1
    constraints[9, 1] = 1
    flow_lhs = base.row_join(constraints).col_join(constraints.T.row_join(sp.zeros(2, 2)))
    flow_rhs = sp.Matrix([0] * 8 + [0] * 6 + [-sp.Float(Q_DESIRED)] * 2 + [0] * 2)
    # Matrix symbols retain the exact state-elimination structure without
    # forcing SymPy to expand the very large rational concentration formula.
    # The callable evaluator below solves these exact matrices numerically.
    q_hat = sp.MatrixSymbol("q_hat(r)", 2, 1)
    x_hat = sp.MatrixSymbol("x_hat(r)", 4, 1)
    q_out_symbolic = sp.Matrix([q_hat[0, 0], q_hat[1, 0]])
    x_out_symbolic = sp.Matrix([x_hat[i, 0] for i in range(4)])
    C_symbolic = sp.MatrixSymbol("C(r)", 12, 6)
    b_1_symbolic = sp.MatrixSymbol("b_1(r)", 12, 1)
    b_2_symbolic = sp.MatrixSymbol("b_2(r)", 12, 1)
    q_target = sp.Matrix([sp.Float(Q_DESIRED), sp.Float(Q_DESIRED)])
    x_target = sp.Matrix([sp.Rational(7, 10), sp.Rational(3, 10), sp.Rational(3, 10), sp.Rational(7, 10)])
    # This compact expression is the exact reduced-objective definition after
    # the flow and concentration state solves.  It intentionally avoids an
    # expanded rational expression that is too large for reproducible scans.
    q_residual = q_out_symbolic - q_target
    x_residual = x_out_symbolic - x_target
    objective_definition = q_residual.dot(q_residual) / (2 * sp.Float(Q_DESIRED)) ** 2 + x_residual.dot(x_residual)
    return PrototypeObjective(
        resistance_symbols=r,
        normalized_symbols=z,
        lower_bounds=np.full(8, LOWER_BOUND),
        upper_bounds=np.full(8, UPPER_BOUND),
        flow_system=(flow_lhs, flow_rhs),
        concentration_system=(C_symbolic, b_1_symbolic, b_2_symbolic),
        q_out_symbolic=q_out_symbolic,
        x_out_symbolic=x_out_symbolic,
        objective_definition=objective_definition,
        evaluate=_numeric_reduced_objective,
    )


def objective_definition_latex() -> str:
    """Return manuscript-ready LaTeX for the exact matrix-defined objective."""
    return r"""\[
\mathbf y(\mathbf r)=\mathbf K(\mathbf r)^{-1}\mathbf b,
\qquad
\mathbf x_k(\mathbf r)=
\left[\mathbf C(\mathbf r)^\mathsf{T}\mathbf C(\mathbf r)\right]^{-1}
\mathbf C(\mathbf r)^\mathsf{T}\mathbf b_k(\mathbf r),
\]
\[
J_{\mathrm{proto}}(\mathbf z)=
\left[\frac{\|\hat{\mathbf q}_o(\mathbf z)-\mathbf q_o^*\|_2}
{\sum_i q_i^*}\right]^2+
\|\hat{\mathbf x}_o(\mathbf z)-\mathbf x_o^*\|_2^2,
\qquad
\mathbf r(\mathbf z)=\mathbf r_{\mathrm{lb}}+
\mathbf z\odot(\mathbf r_{\mathrm{ub}}-\mathbf r_{\mathrm{lb}}).
\]
Here \(\mathbf K\) is the augmented symbolic flow matrix and
\(\mathbf C\) is the concentration matrix constructed from its exact flow
solution.  The numerical evaluator solves these same matrices at each
\(\mathbf z\); it is not a fitted surrogate model."""
