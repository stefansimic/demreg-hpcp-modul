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

A second major optimization effort focused on accelerating the DEMREG solver
using GPU computing. The initial approach was straightforward: we ported the
vectorized CPU implementation to run on the GPU by replacing NumPy with CuPy.
In particular, the GSVD computation and the discrepancy principle μ-grid search
were executed on the GPU without changing the underlying algorithm.

This first attempt successfully offloaded the heavy linear algebra operations
to the GPU, but performance gains were disappointing. The main reason was
architectural: the solver still processed each pixel sequentially and performed
one GSVD per pixel. Even though each SVD was faster on the GPU, the high number
of small kernel launches, memory transfers, and repeated allocations led to
significant overhead. In practice, the GPU implementation was slower than
the optimized CPU vectorized version for typical image sizes.

To address this, we explored a second strategy: re-designing the algorithm to
better match the GPU’s strengths. Instead of calling GSVD individually for
every pixel, the idea was to implement a fully vectorized GPU kernel that
processes batches of pixels simultaneously. In theory, this approach would have
significantly reduced kernel launch overhead and improved parallel occupancy.
However, we were unable to get this version to work within the project
timeframe, so it remained a concept rather than a working implementation.

We also experimented with an alternative solver based on a simplified L²
discrepancy formulation (`demmap_pos_gpu_l2`). This method avoids the GSVD per
pixel and instead computes the regularization parameter and solution using
purely vectorized linear algebra across all pixels. The result is an algorithm
that runs orders of magnitude faster on the GPU because it performs fewer small
decompositions and benefits from massive parallelism. However, this comes at a
cost: the simplified formulation produces slightly less accurate DEM
reconstructions and higher χ² residuals compared to the GSVD-based reference
implementation.

In summary:

- ✅ Approach 1 – Direct CuPy Port: Minimal code changes and mathematically
  identical to the CPU version. It offloads the GSVD and μ-search to the GPU, but
  still processes each pixel individually, leading to significant overhead and in
  many cases slower performance than the CPU vectorized version.

- ⚠️ Approach 2 – Batched GPU Solver (Concept Only): A redesigned solver
  intended to process many pixels in parallel in a fully vectorized GPU kernel.
  This approach would have drastically reduced per-pixel overhead and improved
  parallel efficiency, but we were unable to get a stable implementation working
  within the project timeframe.

- ✅ Approach 3 – Simplified L² Discrepancy Solver: A new algorithm that avoids
  per-pixel GSVD entirely and solves the regularization problem using vectorized
  linear algebra. This runs orders of magnitude faster on the GPU by leveraging
  large-scale parallelism, but at the cost of slightly lower accuracy and higher
  χ² residuals.

> 💡 **Key Insight:** This trade-off highlights a key lesson in GPU
> acceleration: simply offloading existing CPU code rarely achieves good
> performance. To fully leverage GPU capabilities, algorithmic redesign is
> often required.

Both the simple and L² solution can be found in the `./pyhon/gpu` directory.

### Parallel Computing with Dask

> TODO STEFAN

## Benchmark Design

> TODO SHANE

## Results & Comparisons

> TODO GIDEON

## Reflections

> TODO ALL
