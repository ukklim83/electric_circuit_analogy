"""Canonical topology configurations for system1 through system4."""
from __future__ import annotations
from dataclasses import dataclass
from math import nan
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOWER_LENGTH_DELTA_MM = -5.5
DEFAULT_UPPER_LENGTH_DELTA_MM = 2.0
DEFAULT_LOWER_LENGTH_SCALE = 0.5
DEFAULT_UPPER_LENGTH_SCALE = 2.0

@dataclass(frozen=True)
class SystemConfig:
    name: str
    args: tuple[float, float, float, int]
    outlet_edge_idx: tuple[int, ...]
    outlet_node_idx: tuple[int, ...]
    inlet_edge_idx: tuple[int, ...]
    inlet_node_idx: tuple[int, ...]
    changing_edges: tuple[int, ...]
    ini_list: tuple[tuple[Any, ...], ...]
    lower_length_delta_mm: float = DEFAULT_LOWER_LENGTH_DELTA_MM
    upper_length_delta_mm: float = DEFAULT_UPPER_LENGTH_DELTA_MM
    lower_length_scale: float = DEFAULT_LOWER_LENGTH_SCALE
    upper_length_scale: float = DEFAULT_UPPER_LENGTH_SCALE
    inc_csv: str = "incidence_mat.csv"
    length_csv: str = "length_mat.csv"
    conc_csv: str = "concentration_mat.csv"
    plot_type: tuple[str, ...] = ("compare", "concentration", "difference")

    def __post_init__(self) -> None:
        if not self.changing_edges:
            raise ValueError(f"{self.name}: changing_edges must not be empty")
        if len(set(self.changing_edges)) != len(self.changing_edges):
            raise ValueError(f"{self.name}: changing_edges contains duplicates")
        if any(index < 0 for index in self.changing_edges):
            raise ValueError(f"{self.name}: changing_edges contains a negative index")
        if self.lower_length_delta_mm >= self.upper_length_delta_mm:
            raise ValueError(f"{self.name}: lower length delta must be smaller than upper")
        if not 0 < self.lower_length_scale < self.upper_length_scale:
            raise ValueError(f"{self.name}: require 0 < lower length scale < upper")

    @property
    def input_dir(self) -> Path:
        return REPOSITORY_ROOT / self.name

    @property
    def reviewed_changing_edges(self) -> tuple[int, ...]:
        """CAD-reviewed subset; ``changing_edges`` remains a compatibility alias."""
        return self.changing_edges

    def mutable_ini_list(self) -> list[list[Any]]:
        return [list(item) for item in self.ini_list]

SYSTEM_CONFIGS: dict[str, SystemConfig] = {
    "system1": SystemConfig(
        "system1", (1.002e-5, 100e-6, 100e-6, 20001),
        (0, 1, 2, 3), (0, 1, 2, 3), (9, 10), (9, 10), (0, 1, 2, 3, 4, 6),
        (("f", 0, nan, "o", "m^3/s"), ("f", 1, nan, "o", "m^3/s"),
         ("f", 2, nan, "o", "m^3/s"), ("f", 3, nan, "o", "m^3/s"),
         ("f", 9, 2.89e-9 / 60, "i", "m^3/s"),
         ("f", 10, 1.11e-9 / 60, "i", "m^3/s")),
    ),
    "system2": SystemConfig(
        "system2", (1.002e-3, 500e-6, 500e-6, 20001),
        (0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (15, 16), (14, 15), (0, 1, 2, 3, 4, 7, 10, 12, 14),
        (("f", 0, nan, "o", "m^3/s"), ("f", 1, nan, "o", "m^3/s"),
         ("f", 2, nan, "o", "m^3/s"), ("f", 3, nan, "o", "m^3/s"),
         ("f", 4, nan, "o", "m^3/s"),
         ("f", 14, 10e-6 / 60, "i", "m^3/s"),
         ("f", 15, 10e-6 / 60, "i", "m^3/s")),
    ),
    "system3": SystemConfig(
        "system3", (1.002e-3, 500e-6, 500e-6, 20001),
        (0, 1, 2, 3, 4), (0, 1, 2, 3, 4), (24, 25), (20, 21),
        (0, 1, 2, 3, 4, 11, 12, 13, 14, 19, 20, 21),
        (("f", 0, nan, "o", "m^3/s"), ("f", 1, nan, "o", "m^3/s"),
         ("f", 2, nan, "o", "m^3/s"), ("f", 3, nan, "o", "m^3/s"),
         ("f", 4, nan, "o", "m^3/s"),
         ("f", 20, 10e-6 / 60, "i", "m^3/s"),
         ("f", 21, 10e-6 / 60, "i", "m^3/s")),
        lower_length_delta_mm=-5.5,
        upper_length_delta_mm=2.0,
    ),
    "system4": SystemConfig(
        "system4", (1.002e-3, 500e-6, 500e-6, 20001),
        (0, 1, 2, 3), (0, 1, 2, 3), (29, 30, 31), (23, 24, 25), (0, 1, 2, 3, 10, 11, 12, 13, 14, 21, 22, 23, 24),
        (("f", 0, nan, "o", "m^3/s"), ("f", 1, nan, "o", "m^3/s"),
         ("f", 2, nan, "o", "m^3/s"), ("f", 3, nan, "o", "m^3/s"),
         ("f", 23, 6.6e-6 / 60, "i", "m^3/s"),
         ("f", 24, 6.6e-6 / 60, "i", "m^3/s"),
         ("f", 25, 6.6e-6 / 60, "i", "m^3/s")),
    ),
}

def get_system_config(name: str) -> SystemConfig:
    key = str(name).strip().lower()
    try:
        return SYSTEM_CONFIGS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown system {name!r}; choose one of: {', '.join(SYSTEM_CONFIGS)}") from exc
