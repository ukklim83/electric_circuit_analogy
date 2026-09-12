# Optimizer benchmark data

This directory contains the curated machine-readable outcomes and the supporting
scripts for the 30-seed comparison of NSGA-II, PSO, and multi-start SLSQP across
four microfluidic network topologies.

## Published dataset

The files used for the reported comparison are in
[`published_results/digital_discovery_v1`](published_results/digital_discovery_v1):

- `optimizer_benchmark_raw_runs.csv` contains one row for every topology,
  optimizer, and seed (4 topologies x 3 optimizers x 30 seeds = 360 rows).
- `analysis_by_topology_algorithm.csv` contains the median and interquartile-range
  statistics derived from the raw runs.
- `summary_by_topology_algorithm.csv` is a compact secondary summary.
- `settings.json` records the benchmark settings used for the archived runs.

The `length_file` column in the raw CSV preserves the provenance path recorded
during the original execution. Those run-specific optimized-length files are not
required to reproduce the aggregate statistics and are not included here.

## Correspondence to Supplementary Table S11

Supplementary Table S11 reports the median and interquartile range over the same
30 seeds for final combined error (`combined_error`), runtime (`runtime_s`), and
function evaluations (`n_eval`). A run was counted as successful when both
`flow_max_abs_error` and `conc_max_abs_error` were below 1e-3.

The benchmark retained its original internal topology identifiers. Their mapping
to the manuscript labels is:

| Supplementary Fig. 11 / Table S11 label | Network name | Internal `topology` value |
|---|---|---|
| (a) | Oh-reference | `d` |
| (b) | Modified Lee | `b` |
| (c) | Pyramidal | `c` |
| (d) | Variable inlet/outlet | `a` |

Thus, the rows shown in Supplementary Table S11 can be traced directly to
`optimizer_benchmark_raw_runs.csv` after applying this label mapping. The exact
aggregates used by the table are provided in
`analysis_by_topology_algorithm.csv`.

## Recreate the aggregate statistics

From the repository root, run:

```bash
python benchmark/analyze_benchmark_results.py benchmark/published_results/digital_discovery_v1
```

This regenerates `analysis_by_topology_algorithm.csv` from the 360 raw outcomes.
The analysis requires pandas, which is included in the main
`requirements.txt`.

## Rerun the benchmark

The archived benchmark runner and its legacy input snapshots are included to
preserve the numerical protocol used to generate the published outcomes. The
full 30-seed comparison is computationally expensive. To inspect its CLI:

```bash
python benchmark/run_optimizer_benchmark.py --help
```

The authoritative settings for the published run are in
`published_results/digital_discovery_v1/settings.json`. The `inputs` directory
is intentionally isolated from the main examples because it preserves the
legacy solver/input snapshots used by this benchmark.
