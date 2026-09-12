"""Topology configurations for the reviewer-facing optimizer benchmark.

The four benchmark labels use vendored legacy systems so all geometry inputs,
boundary conditions, and target files remain reproducible inside this checkout:
``a=system4``, ``b=system3``, ``c=system2``, and ``d=system1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_INPUT_DIR = Path(__file__).resolve().parent / "inputs"


@dataclass(frozen=True)
class TopologyConfig:
    name: str
    address: Path
    inc_csv: str
    length_csv: str
    conc_csv: str
    args: tuple[float, float, float, int]
    ini_list: tuple[tuple[Any, ...], ...]
    inlet_node_idx: tuple[int, ...]
    inlet_edge_idx: tuple[int, ...]
    outlet_node_idx: tuple[int, ...]
    outlet_edge_idx: tuple[int, ...]
    changing_edges: tuple[int, ...]


TOPOLOGY_CONFIGS: dict[str, TopologyConfig] = {
    "a": TopologyConfig(
        name="a",
        address=BENCHMARK_INPUT_DIR / "system4",
        inc_csv="incidence_mat.csv",
        length_csv="length_mat.csv",
        conc_csv="concentration_mat.csv",
        args=(1.002e-3, 500e-6, 500e-6, 20001),
        outlet_edge_idx=(0, 1, 2, 3),
        outlet_node_idx=(0, 1, 2, 3),
        inlet_edge_idx=(29, 30, 31),
        inlet_node_idx=(23, 24, 25),
        changing_edges=tuple(range(32)),
        ini_list=(
            ("f", 0, float("nan"), "o", "m^3/s"),
            ("f", 1, float("nan"), "o", "m^3/s"),
            ("f", 2, float("nan"), "o", "m^3/s"),
            ("f", 3, float("nan"), "o", "m^3/s"),
            ("f", 23, 6.6e-6 / 60, "i", "m^3/s"),
            ("f", 24, 6.6e-6 / 60, "i", "m^3/s"),
            ("f", 25, 6.6e-6 / 60, "i", "m^3/s"),
        ),
    ),
    "b": TopologyConfig(
        name="b",
        address=BENCHMARK_INPUT_DIR / "system3",
        inc_csv="incidence_mat.csv",
        length_csv="length_mat.csv",
        conc_csv="concentration_mat.csv",
        args=(1.002e-3, 500e-6, 500e-6, 20001),
        outlet_edge_idx=(0, 1, 2, 3, 4),
        outlet_node_idx=(0, 1, 2, 3, 4),
        inlet_edge_idx=(15, 16),
        inlet_node_idx=(14, 15),
        changing_edges=tuple(range(17)),
        ini_list=(
            ("f", 0, float("nan"), "o", "m^3/s"),
            ("f", 1, float("nan"), "o", "m^3/s"),
            ("f", 2, float("nan"), "o", "m^3/s"),
            ("f", 3, float("nan"), "o", "m^3/s"),
            ("f", 4, float("nan"), "o", "m^3/s"),
            ("f", 14, 10e-6 / 60, "i", "m^3/s"),
            ("f", 15, 10e-6 / 60, "i", "m^3/s"),
        ),
    ),
    "c": TopologyConfig(
        name="c",
        address=BENCHMARK_INPUT_DIR / "system2",
        inc_csv="incidence_mat.csv",
        length_csv="length_mat.csv",
        conc_csv="concentration_mat.csv",
        args=(1.002e-3, 500e-6, 500e-6, 20001),
        outlet_edge_idx=(0, 1, 2, 3, 4),
        outlet_node_idx=(0, 1, 2, 3, 4),
        inlet_edge_idx=(24, 25),
        inlet_node_idx=(20, 21),
        changing_edges=tuple(range(26)),
        ini_list=(
            ("f", 0, float("nan"), "o", "m^3/s"),
            ("f", 1, float("nan"), "o", "m^3/s"),
            ("f", 2, float("nan"), "o", "m^3/s"),
            ("f", 3, float("nan"), "o", "m^3/s"),
            ("f", 4, float("nan"), "o", "m^3/s"),
            ("f", 20, 10e-6 / 60, "i", "m^3/s"),
            ("f", 21, 10e-6 / 60, "i", "m^3/s"),
        ),
    ),
    "d": TopologyConfig(
        name="d",
        address=BENCHMARK_INPUT_DIR / "system1",
        inc_csv="incidence_mat.csv",
        length_csv="length_mat.csv",
        conc_csv="concentration_mat.csv",
        args=(1.002e-5, 100e-6, 100e-6, 20001),
        outlet_edge_idx=(0, 1, 2, 3),
        outlet_node_idx=(0, 1, 2, 3),
        inlet_edge_idx=(9, 10),
        inlet_node_idx=(9, 10),
        changing_edges=tuple(range(11)),
        ini_list=(
            ("f", 0, float("nan"), "o", "m^3/s"),
            ("f", 1, float("nan"), "o", "m^3/s"),
            ("f", 2, float("nan"), "o", "m^3/s"),
            ("f", 3, float("nan"), "o", "m^3/s"),
            ("f", 9, 2.89e-9 / 60, "i", "m^3/s"),
            ("f", 10, 1.11e-9 / 60, "i", "m^3/s"),
        ),
    ),
}


def get_topology_config(name: str) -> TopologyConfig:
    """Return one topology configuration by short name."""
    return TOPOLOGY_CONFIGS[name]
