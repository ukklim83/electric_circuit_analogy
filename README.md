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
- SymPy 1.12 or later and earlier than 2
- mpmath 1.3 or later and earlier than 1.4
- openpyxl 3.1 or later and earlier than 4

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

By default, the results are written to
`results/system1/nsga2/reviewed_additive/`. With `--solve-only`, `--optimizer`
is not used for optimization but still supplies the optimizer name in the
result path.

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
| `--edge-mode` | `reviewed`: configured CAD-reviewed subset; `all`: every logical edge | `reviewed` |
| `--bound-mode` | `additive`: initial −5.5/+2.0 mm; `relative`: initial 0.5x/2.0x | `additive` |
| `--output-dir` | User-defined result directory | `results/<system>/<optimizer>/<edge_mode>_<bound_mode>/` |
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

The edge set and bounds are independently selectable from one codebase:

| Mode | Meaning | CAD release status |
|---|---|---|
| `reviewed + additive` | Reviewed edges, `initial - 5.5 mm` to `initial + 2.0 mm` | Production default; eligible for the existing CAD workflow |
| `reviewed + relative` | Reviewed edges, `initial × 0.5` to `initial × 2.0` | Exploratory; requires new CAD calibration |
| `all + relative` | Every edge, `initial × 0.5` to `initial × 2.0` | Exploratory; unreviewed edges may lack CAD controllers |
| `all + additive` | Every edge, `initial - 5.5 mm` to `initial + 2.0 mm` | Fails closed when any lower bound is nonpositive |

The additive mode fails closed rather than silently clipping invalid bounds.
For example, short fixed edges included by `--edge-mode all` can make
`initial - 5.5 mm <= 0`; the error lists every unsafe logical edge. The
two bound profiles and the reviewed changing-edge sets are defined in
`codes/system_configs.py`:

| System | Topology | Optimizable logical edges |
|---|---|---|
| `system1` | system1 | E01-E05, E07 |
| `system2` | `b` | E01-E05, E08, E11, E13, E15 |
| `system3` | `c` | E01-E05, E12-E15, E20-E22 |
| `system4` | `3_4` | E01-E04, E11-E15, E22-E25 |

All edges not listed for a system remain fixed.

Examples:

```bash
# Production-default profile
python electric_analogy_programming.py --system system3 --optimizer nsga2 \
  --edge-mode reviewed --bound-mode additive

# Whole-network exploratory optimization
python electric_analogy_programming.py --system system3 --optimizer pso \
  --edge-mode all --bound-mode relative
```

Every optimization writes `optimization_run_config.json`, including the
selected modes, changing-edge indices and E-names, exact initial/lower/upper
lengths, requested deltas/scales, and `cad_release_eligible`. This file is the
authoritative record of the profile actually sent to the optimizer.

Treat the system directories as original input storage. The program copies the
input CSV files to the result directory and works on those copies. It rejects an
`--output-dir` that points to a system input directory or one of its
subdirectories.

See [`conc_matrix_explanation.md`](conc_matrix_explanation.md) for more details
about the concentration matrix.

The compact symbolic and curvature-analysis workflow is documented in
[`analytic_validation/README.md`](analytic_validation/README.md). Its scope is
the eight-resistance prototype; it is not a replacement for system-specific
solver regression or CAD validation.

## Published optimizer benchmark

The machine-readable 30-seed outcomes underlying Supplementary Table S11 are
provided in
[`benchmark/published_results/digital_discovery_v1`](benchmark/published_results/digital_discovery_v1).
They cover NSGA-II, PSO, and multi-start SLSQP for each of the four network
topologies. [`benchmark/README.md`](benchmark/README.md) documents the data
fields, aggregation command, and the correspondence between the benchmark's
legacy topology identifiers and Supplementary Fig. 11/Table S11.

## Results

The default result layout is:

```text
results/
└── system1/
    ├── nsga2/
    │   ├── reviewed_additive/
    │   └── all_relative/
    ├── pso/
    └── slsqp/
```

At the end of a run, the program prints the actual result directory as
`Results written to: ...`. Depending on the selected mode, generated files may
include:

- `new_length_mat.csv`: optimized channel lengths
- `optimization_run_config.json`: resolved edge/bound profile and CAD eligibility
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
├── analytic_validation/                # Symbolic prototype and curvature checks
├── benchmark/                           # 30-seed optimizer benchmark and data
├── codes/
│   ├── electric_analogy.py             # Shared solver and NSGA-II
│   ├── electric_analogy_pso.py         # PSO backend
│   ├── electric_analogy_slsqp.py       # Multi-start SLSQP backend
│   ├── electric_analogy_programming.py # Shared CLI/driver
│   ├── optimization_profiles.py        # Shared edge/bound mode resolver
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
