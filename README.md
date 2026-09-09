# Electric Circuit Analogy

**English** | [한국어](README_ko.md)

Electric Circuit Analogy is a Python program for analyzing microfluidic channel
networks as equivalent electrical circuits and optimizing channel lengths to
match target outlet flow rates and concentrations. It supports standard circuit
analysis together with NSGA-II, PSO, and multi-start SLSQP optimization.

## Features

- Incidence-matrix-based channel network analysis
- Per-channel flow-rate and pressure calculations using Kirchhoff's laws
- Outlet concentration calculations for multicomponent fluids
- Channel-length optimization with NSGA-II, PSO, or multi-start SLSQP
- Four example topologies and input datasets: system1 through system4
- Separate result directories for each system and optimizer

## Requirements

- Python 3.10 or later
- NumPy 1.24 or later and earlier than 3
- pandas 2 or later and earlier than 4
- SciPy 1.10 or later and earlier than 2
- Matplotlib 3.7 or later and earlier than 4
- pymoo 0.6 or later and earlier than 0.7

## Installation

From the repository root, install the dependencies into the Python environment
you intend to use. A system Python installation, a conda environment, or any
other environment manager may be used.

```bash
python -m pip install -r requirements.txt
```

If your system uses `python3` as the Python command, run
`python3 -m pip install -r requirements.txt` instead. If you use conda or another
environment manager, select the intended environment before running the command.

Verify the installation and CLI with:

```bash
python electric_analogy_programming.py --help
```

## Quick start

For a first run, analyze the original system1 design without optimization:

```bash
python electric_analogy_programming.py --system system1 --solve-only
```

By default, the results are written to `results/system1/nsga2/`. With
`--solve-only`, `--optimizer` is not used for optimization but still supplies
the optimizer name in the result path.

Select an optimizer to optimize channel lengths:

```bash
# NSGA-II
python electric_analogy_programming.py --system system1 --optimizer nsga2

# Particle Swarm Optimization
python electric_analogy_programming.py --system system2 --optimizer pso

# Multi-start SLSQP
python electric_analogy_programming.py --system system3 --optimizer slsqp
```

Running the launcher without arguments performs NSGA-II length optimization on
system1. The default settings can require substantial computation. For a quick
functional test, use smaller populations and iteration counts:

```bash
python electric_analogy_programming.py --system system1 --optimizer nsga2 --pop-size 20 --n-gen 10
python electric_analogy_programming.py --system system1 --optimizer pso --pop-size 10 --n-gen 10
python electric_analogy_programming.py --system system1 --optimizer slsqp --n-starts 2 --maxiter 20
```

## Command-line options

| Option | Description | Default |
|---|---|---|
| `--system` | System to run: `system1`, `system2`, `system3`, or `system4` | `system1` |
| `--optimizer` | Optimization method: `nsga2`, `pso`, or `slsqp` | `nsga2` |
| `--seed` | Random seed for PSO/SLSQP | `1` |
| `--pop-size` | NSGA-II population size or PSO swarm size | NSGA-II `600`; PSO `50` |
| `--n-gen` | Maximum NSGA-II/PSO generations | NSGA-II `600`; PSO derives it from its evaluation budget |
| `--n-starts` | Number of independent SLSQP restarts | `20` |
| `--maxiter` | Maximum iterations in each SLSQP run | `1000` |
| `--output-dir` | User-defined result directory | `results/<system>/<optimizer>/` |
| `--solve-only` | Analyze the input design without length optimization | Disabled |

For reproducible PSO/SLSQP runs, record the `--seed` together with all other
optimizer options.

## Input data

Each system directory must contain these three input CSV files:

```text
system1/
├── incidence_mat.csv
├── length_mat.csv
└── concentration_mat.csv
```

- `incidence_mat.csv`: incidence matrix describing connections between edges
  and nodes
- `length_mat.csv`: channel length for each edge
- `concentration_mat.csv`: target component concentrations at each outlet

Fluid properties, channel cross sections, inlet/outlet indices, inlet flow
rates, and the list of optimizable edges are defined for each system in
`codes/system_configs.py`. The edge and node order in the CSV files must match
the indices in that configuration.

All systems use additive optimization bounds of `baseline - 5.5 mm` and
`baseline + 2.0 mm` for their configured changing edges. A run fails closed if
the selected changing-edge set would produce a nonpositive lower length bound.
The reviewed changing-edge sets are:

| System | Topology | Optimizable logical edges |
|---|---|---|
| `system1` | system1 | E01-E05, E07 |
| `system2` | `b` | E01-E05, E08, E11, E13, E15 |
| `system3` | `c` | E01-E05, E12-E15, E20-E22 |
| `system4` | `3_4` | E01-E04, E11-E15, E22-E25 |

All edges not listed for a system remain fixed.

Treat the system directories as original input storage. The program copies the
input CSV files to the result directory and works on those copies. It rejects an
`--output-dir` that points to a system input directory or one of its
subdirectories.

See [`conc_matrix_explanation.md`](conc_matrix_explanation.md) for more details
about the concentration matrix.

## Results

The default result layout is:

```text
results/
└── system1/
    ├── nsga2/
    ├── pso/
    └── slsqp/
```

At the end of a run, the program prints the actual result directory as
`Results written to: ...`. Depending on the selected mode, generated files may
include:

- `new_length_mat.csv`: optimized channel lengths
- `current_vec_total.csv`, `voltage_vec_total.csv`: original-design flow rates
  and pressures
- `current_vec_revised.csv`, `voltage_vec_revised.csv`: optimized-design flow
  rates and pressures
- `outlet_*.png`: outlet flow-rate and concentration plots
- `pareto_front.png`, `pareto_optimal_*.csv`: NSGA-II results
- `pso_optimization_summary.csv`: PSO run summary
- `slsqp_restart_summary.csv`: results for each SLSQP restart

Use `--output-dir` to select a custom result location:

```bash
python electric_analogy_programming.py --system system4 --optimizer pso --output-dir my_results/system4_pso
```

## Running from Python

You can call the shared driver from another Python program instead of using the
CLI:

```python
from codes.electric_analogy_programming import run

flow, concentration, result_dir = run(
    system="system1",
    optimizer="pso",
    seed=42,
    pop_size=20,
    n_gen=50,
)
```

Pass `optimize=False` to analyze a design without optimization:

```python
flow, concentration, result_dir = run(
    system="system1",
    optimizer="nsga2",
    optimize=False,
)
```

Import `codes.electric_analogy` when direct access to the lower-level circuit
analysis API is required. The root-level `electric_analogy.py` is a compatibility
facade for existing `import electric_analogy` users.

## Project structure

```text
electric_circuit_analogy/
├── electric_analogy.py                 # Root compatibility facade
├── electric_analogy_programming.py     # Main CLI launcher
├── requirements.txt
├── codes/
│   ├── electric_analogy.py             # Shared solver and NSGA-II
│   ├── electric_analogy_pso.py         # PSO backend
│   ├── electric_analogy_slsqp.py       # Multi-start SLSQP backend
│   ├── electric_analogy_programming.py # Shared CLI/driver
│   ├── system_configs.py               # Configuration for system1-system4
│   └── resistance.py                   # Channel resistance formulas
├── system1/                            # Input data and compatibility launcher
├── system2/
├── system3/
├── system4/
└── results/                            # Generated at runtime; ignored by Git
```

## System-specific launchers

Launchers inside the system directories are also available. Running commands
from the repository root is recommended.

```bash
python system1/electric_analogy_programming.py --optimizer pso
python system4/electric_analogy_programming.py --optimizer slsqp
```

Each launcher selects its corresponding system by default and delegates all
calculations to the same shared driver and solver.

## Troubleshooting

- If a `ModuleNotFoundError` occurs, confirm that the dependencies are installed
  in the Python environment currently running the program, then run
  `python -m pip install -r requirements.txt` again.
- If an input-file error occurs, confirm that all three CSV files exist in the
  selected system directory.
- If an output-path `ValueError` occurs, select a separate directory outside
  `system1` through `system4`.
- If optimization takes too long, reduce `--pop-size`, `--n-gen`, `--n-starts`,
  or `--maxiter` for test runs.
- Run `python electric_analogy_programming.py --help` to view the exact options
  supported by the current version.
