"""Shared command-line driver for every supported circuit topology."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import Sequence

try:
    from .electric_analogy import execute_functions, execute_length_change, plotting_outlet
    from .system_configs import REPOSITORY_ROOT, SYSTEM_CONFIGS, get_system_config
except ImportError:  # Direct execution from codes/.
    from electric_analogy import execute_functions, execute_length_change, plotting_outlet
    from system_configs import REPOSITORY_ROOT, SYSTEM_CONFIGS, get_system_config

OPTIMIZERS = ("nsga2", "pso", "slsqp")
INPUT_FILES = ("incidence_mat.csv", "length_mat.csv", "concentration_mat.csv")

def _prepare_run_directory(config, output_dir: Path) -> Path:
    """Copy immutable topology inputs into a writable result directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in INPUT_FILES:
        source = config.input_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing {config.name} input: {source}")
        destination = output_dir / filename
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    return output_dir

def run(
    system: str = "system1",
    optimizer: str = "nsga2",
    *,
    seed: int = 1,
    pop_size: int | None = None,
    n_gen: int | None = None,
    n_starts: int = 20,
    maxiter: int = 1000,
    output_dir: str | Path | None = None,
    optimize: bool = True,
):
    """Run one topology with the canonical solver and selected optimizer."""
    config = get_system_config(system)
    optimizer = str(optimizer).strip().lower()
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of: {', '.join(OPTIMIZERS)}")
    run_dir = (
        REPOSITORY_ROOT / "results" / config.name / optimizer
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    input_dir = config.input_dir.resolve()
    resolved_run_dir = run_dir.resolve()
    if resolved_run_dir == input_dir or input_dir in resolved_run_dir.parents:
        raise ValueError(
            "The output directory must not be the system input directory or one "
            "of its subdirectories. Use results/<system>/<optimizer> or another "
            "separate directory."
        )
    run_dir = _prepare_run_directory(config, run_dir)
    what_to_solve = ["source", "concentration"]
    length_csv = config.length_csv
    nametag = "total"
    ini_list = config.mutable_ini_list()

    if optimize:
        what_to_solve.insert(0, "length")
        default_pop = {"nsga2": 600, "pso": 50, "slsqp": 200}[optimizer]
        selected_pop = default_pop if pop_size is None else pop_size
        selected_gen = 600 if optimizer == "nsga2" and n_gen is None else n_gen
        options: dict[str, object] = {}
        if optimizer == "pso":
            options.update(seed=seed, max_fes_per_dim=10_000)
        elif optimizer == "slsqp":
            options.update(seed=seed, n_starts=n_starts, maxiter=maxiter)
        execute_length_change(
            what_to_solve, list(config.changing_edges), config.inc_csv,
            length_csv, config.conc_csv, config.args, ini_list,
            list(config.inlet_node_idx), list(config.inlet_edge_idx),
            list(config.outlet_node_idx), list(config.outlet_edge_idx),
            0.5, 0.5, str(run_dir), popSize=selected_pop,
            nGen=selected_gen, optimizer=optimizer, **options,
        )
        length_csv = "new_length_mat.csv"
        nametag = "revised"

    i_vec, conc_vec = execute_functions(
        what_to_solve, config.inc_csv, length_csv, config.args, ini_list,
        list(config.inlet_node_idx), list(config.inlet_edge_idx), nametag,
        str(run_dir),
    )
    plotting_outlet(
        i_vec, list(config.outlet_edge_idx), config.plot_type, str(run_dir),
        what_to_solve, ini_list, config.args, conc_vec, nametag,
        config.inc_csv, length_csv, config.conc_csv,
    )
    print(f"Results written to: {run_dir}")
    return i_vec, conc_vec, run_dir

def build_parser(default_system: str = "system1") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve or optimize an electric-analogy circuit.")
    parser.add_argument("--system", choices=tuple(SYSTEM_CONFIGS), default=default_system)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="nsga2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--n-gen", type=int)
    parser.add_argument("--n-starts", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--solve-only", action="store_true", help="Skip length optimization.")
    return parser

def main(default_system: str = "system1", argv: Sequence[str] | None = None):
    args = build_parser(default_system).parse_args(argv)
    return run(
        system=args.system, optimizer=args.optimizer, seed=args.seed,
        pop_size=args.pop_size, n_gen=args.n_gen, n_starts=args.n_starts,
        maxiter=args.maxiter, output_dir=args.output_dir,
        optimize=not args.solve_only,
    )

if __name__ == "__main__":
    main()
