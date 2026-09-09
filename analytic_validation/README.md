# Analytic validation assets

**English** | [한국어](README_ko.md)

This directory contains a compact, reproducible subset of the former
`analytic/` working directory. Generated plots, expanded symbolic text files,
optimizer outputs, notebooks with embedded output, duplicate solver copies,
and debug-only scripts are intentionally excluded.

## Scope

These assets study the mathematical structure of an eight-resistance,
two-inlet/two-outlet prototype. They provide symbolic equations and numerical
counterexamples to joint convexity of the prototype objective.

They do **not** by themselves validate:

- the numerical accuracy of every `system1` through `system4` topology;
- NSGA-II, PSO, or SLSQP convergence to a global optimum;
- the `reviewed`/`all` edge modes or `additive`/`relative` length bounds; or
- CAD geometry, manufacturability, clearance, or controller calibration.

Those claims require separate solver regression and CAD acceptance tests.

## Included files

| File | Role | Status |
|---|---|---|
| `prototype_reduced_objective.py` | Exact matrix-defined eight-resistance objective with numerical state elimination | Current core |
| `prototype_joint_convexity_analysis.py` | Sobol sampling, finite-difference Hessians, eigenvalue screening, and Jensen checks | Current core |
| `prototype_directional_confirmation.py` | Arbitrary-precision confirmation of the selected negative-curvature direction | Current core |
| `plot_prototype_presentation_figures.py` | Optional figures from the generated validation reports | Optional reporting |
| `analytic_validation_script.py` | Original large symbolic derivation and pairwise parameter plots | Legacy reference |
| `CV_NCV_map.xlsx` | Manually classified convex/non-convex map consumed by the legacy script | Legacy input |

`analytic_validation_script.py` is retained for traceability. It is much slower,
depends on a local LaTeX installation for text rendering, and includes manual
classification. Prefer the three `prototype_*` scripts for reproducible
curvature evidence.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

The analytic workflow additionally uses SymPy, mpmath, and openpyxl; they are
listed in the repository requirements.

## Recommended workflow

Run these commands from this directory:

```bash
python prototype_joint_convexity_analysis.py --samples 128 --replicates 1
python prototype_directional_confirmation.py
python plot_prototype_presentation_figures.py
```

For the more extensive replicated screening used in the source study:

```bash
python prototype_joint_convexity_analysis.py --samples 256 --replicates 4 \
  --output-dir outputs/prototype_joint_convexity_n256_b4
python prototype_directional_confirmation.py \
  --input-dir outputs/prototype_joint_convexity_n256_b4
python plot_prototype_presentation_figures.py \
  --input-dir outputs/prototype_joint_convexity_n256_b4
```

All generated files are written under `outputs/`, which is excluded from Git.

## Interpretation

The validation sequence is:

1. Build the exact prototype flow and concentration matrix equations.
2. Eliminate state variables numerically without introducing a surrogate.
3. Estimate the full Hessian over interior Sobol samples.
4. Find a step-stable negative eigenvalue and its direction.
5. Confirm the same direction with arbitrary-precision symmetric differences.
6. Check a positive Jensen gap along the witness line.

A stable negative directional curvature is sufficient to disprove joint
convexity for this prototype. It is not proof of multiple local optima and
should not be generalized automatically to every topology.

## Legacy workflow

The original symbolic script can be run with:

```bash
python analytic_validation_script.py
```

It writes generated equations and plots to `outputs/legacy_symbolic/` and reads
the retained `CV_NCV_map.xlsx`. Expect high symbolic-computation cost and very
large figures.
