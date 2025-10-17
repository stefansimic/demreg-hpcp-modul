# DEMREG Optimization Report

## Context and Problem Statement

- We recover **DEM(T)** from multi-channel counts using a temperature response matrix and **GSVD-regularized** inversion.
- Outputs per pixel: `DEM(T)`, uncertainty `edem`, log-temperature grid `elogt`, regularized counts `dn_reg`, and **χ²** misfit.
- The optimization focuses on the core pipeline implemented in the four modules:
  - `demmap_pos.py` (per-pixel solver / parallelization)
  - `dem_inv_gsvd.py` (GSVD + Tikhonov regularization core)
  - `dem_reg_map.py` (μ-grid search / discrepancy principle)

## Profiling & Analysis of the Baseline

> TODO STEFAN

**Method.** We profiled the baseline using representative inputs and recorded function-level costs (cProfile/flame), memory behavior, and parallel utilization.

**Environment (baseline run):**

- CPU: {CPU_MODEL}, threads: 1
- Python/NumPy: 3.11.14 / 2.3.3
- Command:
  ```bash
  python bench_demreg_sxs.py --baseline_dir Baseline --improved_dir CPU_Vectorization --width 256 --height 256 --threads 1 --repeats 3 --outdir {OUTDIR}
  ```

**Key findings:**

- The **per-pixel inversion** (GSVD + μ-selection) dominated wall time.
- **Python loops** and **temporary allocations** on the hot path caused overhead.
- **Process pool + BLAS threads** risked **oversubscription**, reducing effective throughput.
- μ-grid length and search bounds affected time predictably; overly long grids wasted work.

## Implemented Strategies

We used the following optimizationg strategies.

- CPU Vectorization using numpy
- Utilizing GPU rescoursec with cupy
- Parallel Computing with Dask

### CPU Vectorization

> TODO SHANE

### GPU with cupy

> TODO GIDEON

### Parallel Computing with Dask

> TODO STEFAN

## Benchmark Design

> TODO SHANE

## Results & Comparisons

> TODO GIDEON

## Reflections

> TODO ALL
