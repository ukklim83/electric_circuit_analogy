"""Resolve reproducible edge-selection and length-bound optimization profiles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

EDGE_MODES = ("reviewed", "all")
BOUND_MODES = ("additive", "relative")
DEFAULT_LOWER_SCALE = 0.5
DEFAULT_UPPER_SCALE = 2.0


@dataclass(frozen=True)
class OptimizationProfile:
    """Concrete optimizer inputs resolved from user-facing mode names."""

    edge_mode: str
    bound_mode: str
    edge_indices: np.ndarray
    initial_lengths_mm: np.ndarray
    lower_bounds_mm: np.ndarray
    upper_bounds_mm: np.ndarray
    lower_delta_mm: float
    upper_delta_mm: float
    lower_scale: float
    upper_scale: float

    @property
    def edge_ids(self) -> list[str]:
        return [f"E{index + 1:02d}" for index in self.edge_indices]

    @property
    def cad_release_eligible(self) -> bool:
        """Only the CAD-reviewed production profile is release eligible."""
        return self.edge_mode == "reviewed" and self.bound_mode == "additive"

    def as_dict(self) -> dict[str, object]:
        bounds = [
            {
                "edge_index": int(index),
                "edge_id": edge_id,
                "initial_length_mm": float(initial),
                "lower_bound_mm": float(lower),
                "upper_bound_mm": float(upper),
            }
            for index, edge_id, initial, lower, upper in zip(
                self.edge_indices,
                self.edge_ids,
                self.initial_lengths_mm,
                self.lower_bounds_mm,
                self.upper_bounds_mm,
            )
        ]
        return {
            "edge_mode": self.edge_mode,
            "bound_mode": self.bound_mode,
            "changing_edge_indices": [int(value) for value in self.edge_indices],
            "changing_edge_ids": self.edge_ids,
            "requested_bounds": {
                "additive_delta_mm": {
                    "lower": self.lower_delta_mm,
                    "upper": self.upper_delta_mm,
                },
                "relative_scale": {
                    "lower": self.lower_scale,
                    "upper": self.upper_scale,
                },
            },
            "resolved_edge_bounds": bounds,
            "cad_release_eligible": self.cad_release_eligible,
            "cad_release_note": (
                "Eligible for the existing CAD controller workflow."
                if self.cad_release_eligible
                else "Exploratory optimizer profile; CAD controller qualification is required."
            ),
        }


def resolve_optimization_profile(
    initial_lengths_mm: Sequence[float],
    reviewed_edges: Sequence[int],
    *,
    edge_mode: str = "reviewed",
    bound_mode: str = "additive",
    lower_delta_mm: float = -5.5,
    upper_delta_mm: float = 2.0,
    lower_scale: float = DEFAULT_LOWER_SCALE,
    upper_scale: float = DEFAULT_UPPER_SCALE,
) -> OptimizationProfile:
    """Resolve edge indices and exact lower/upper arrays, failing closed."""
    edge_mode = str(edge_mode).strip().lower()
    bound_mode = str(bound_mode).strip().lower()
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode must be one of: {', '.join(EDGE_MODES)}")
    if bound_mode not in BOUND_MODES:
        raise ValueError(f"bound_mode must be one of: {', '.join(BOUND_MODES)}")

    initial = np.asarray(initial_lengths_mm, dtype=float)
    if initial.ndim != 1 or initial.size == 0:
        raise ValueError("initial_lengths_mm must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(initial)) or np.any(initial <= 0):
        raise ValueError("Every initial channel length must be finite and positive")

    reviewed = np.asarray(reviewed_edges, dtype=int)
    if reviewed.ndim != 1 or reviewed.size == 0:
        raise ValueError("reviewed_edges must contain at least one edge index")
    if len(np.unique(reviewed)) != reviewed.size:
        raise ValueError("reviewed_edges contains duplicates")
    if np.any(reviewed < 0) or np.any(reviewed >= initial.size):
        raise IndexError("reviewed_edges contains an index outside the length matrix")

    edges = reviewed.copy() if edge_mode == "reviewed" else np.arange(initial.size)
    selected = initial[edges]
    if not lower_delta_mm < upper_delta_mm:
        raise ValueError("lower_delta_mm must be smaller than upper_delta_mm")
    if not 0 < lower_scale < upper_scale:
        raise ValueError("Require 0 < lower_scale < upper_scale")

    if bound_mode == "additive":
        lower = selected + float(lower_delta_mm)
        upper = selected + float(upper_delta_mm)
    else:
        lower = selected * float(lower_scale)
        upper = selected * float(upper_scale)

    unsafe = np.flatnonzero(lower <= 0)
    if unsafe.size:
        details = ", ".join(
            f"E{int(edges[position]) + 1:02d} "
            f"(initial={selected[position]:.6g} mm, lower={lower[position]:.6g} mm)"
            for position in unsafe
        )
        raise ValueError(
            f"{edge_mode}+{bound_mode} produces nonpositive lower bounds: {details}. "
            "Use --bound-mode relative or select a CAD-qualified edge set."
        )
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Every resolved length bound must be finite")
    if np.any(upper <= lower):
        raise ValueError("Every upper length bound must exceed its lower bound")
    if np.any(selected < lower) or np.any(selected > upper):
        raise ValueError("The initial design must lie inside every resolved bound")

    return OptimizationProfile(
        edge_mode=edge_mode,
        bound_mode=bound_mode,
        edge_indices=edges,
        initial_lengths_mm=selected.copy(),
        lower_bounds_mm=lower,
        upper_bounds_mm=upper,
        lower_delta_mm=float(lower_delta_mm),
        upper_delta_mm=float(upper_delta_mm),
        lower_scale=float(lower_scale),
        upper_scale=float(upper_scale),
    )
