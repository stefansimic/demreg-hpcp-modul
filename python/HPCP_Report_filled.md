# DEMREG Optimization Report
_Date: 2025-10-16_

> This report documents the optimization of the DEM reconstruction pipeline, covering: (1) baseline profiling and analysis; (2) implemented strategies and rationale; (3) benchmark results and comparisons; and (4) reflections on what worked, what did not, and why.  
---

## 1. Executive Summary
- **Objective:** Speed up DEM reconstruction while preserving scientific quality (χ² stability, DEM consistency).
- **Approach:** Profile the baseline, form hypotheses, implement targeted improvements (vectorization, thread/process tuning, numerical guards), and evaluate using a **same-run, side-by-side** harness.
- **Outcome (headline):** Time reduced by **57.43%** on 256×256 map while maintaining comparable χ² (mean/median).  
- **Key enablers:** Removing Python-level loops on hot paths, avoiding BLAS oversubscription, careful μ-grid selection, and small numerical stability tweaks.

---

## 2. Context and Problem Statement
- We recover **DEM(T)** from multi-channel counts using a temperature response matrix and **GSVD-regularized** inversion.
- Outputs per pixel: `DEM(T)`, uncertainty `edem`, log-temperature grid `elogt`, regularized counts `dn_reg`, and **χ²** misfit.
- The optimization focuses on the core pipeline implemented in the four modules:
  - `dn2dem_pos.py` (orchestration / binning / mapping)
  - `demmap_pos.py` (per-pixel solver / parallelization)
  - `dem_inv_gsvd.py` (GSVD + Tikhonov regularization core)
  - `dem_reg_map.py` (μ-grid search / discrepancy principle)

---

## 3. Profiling & Analysis of the Baseline
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

---

## 4. Implemented Strategies & Rationale
1. **Vectorization & memory locality** — Replace Python loops with NumPy ops; keep arrays contiguous; reuse buffers.  
2. **Thread/process coordination** — Limit BLAS threads (e.g., OMP/OPENBLAS/MKL = 1) when using processes; tune chunk sizes.  
3. **μ-grid + discrepancy** — Use log-spaced μ with sensible bounds; select via discrepancy principle; early exit when met.  
4. **Numerical guards & stability** — Clip/guard invalid values; finite checks; stable masking (boolean masks).  
5. **I/O & progress niceties (optional)** — Make progress bars optional; keep out of critical kernels.

---

## 5. Benchmark Design
**Harness:** **Same-run, side-by-side**; both implementations see identical inputs and environment.  
**Artifacts:** `aggregate_sxs_summary.json`, `aggregate_sxs_arrays.npz` (or single-run equivalents).  
**Dataset:** 256×256 (65,536 px)  
**Metrics:** Wall time (s), **% reduction vs baseline**, χ² mean/median, DEM overlays.

---

## 6. Results & Comparisons
**Timing (wall clock, mean ± std):**
- Baseline: **15.307 ± 0.263 s**  
- Improved: **6.516 ± 0.129 s**  
- **% time reduction vs baseline:** **57.43%** (std 0.44)

**χ² stability (per-run means/medians):**
- Baseline mean/median: **49.8 / 49.8**
- Improved mean/median: **45.3 / 45.3**

**Qualitative checks:**
- DEM median curves over logT are consistent between implementations.
- Residual distributions are comparable; no systematic bias observed.

---

## 7. Reflections
**What worked well**
- Vectorization and memory-layout fixes removed Python overhead on hot paths.
- Limiting BLAS threads when using processes prevented oversubscription.
- A shorter, log-spaced μ-grid preserved accuracy while reducing compute.

**What did not (or needed care)**
- Too many workers and long μ-grids increased contention with little benefit.
- Floating-point reduction order made tiny diffs; single-thread BLAS reduced drift.
- Some baseline masking logic needed boolean masks instead of `np.where` tuples.

**Why**
- The workload is allocation- and bandwidth-sensitive. Reducing per-pixel Python work and coordinating threads lets optimized BLAS do most of the heavy lifting without fighting for resources.

---

## 8. Reproducibility
**Example command:**
```bash
python bench_demreg_sxs.py   --baseline_dir Baseline   --improved_dir CPU_Vectorization   --width 256 --height 256   --threads 1   --repeats 3   --outdir {OUTDIR}   --seed 132 --seed_fixed
```

**Environment:**
- Python: 3.11.14, NumPy: 2.3.3
- BLAS threads (recommended): `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

**Artifacts to archive:**
- `{OUTDIR}/aggregate_sxs_summary.json`
- `{OUTDIR}/aggregate_sxs_arrays.npz`
- `{OUTDIR}/aggregate_timings.csv`

---

## 9. Appendix
| Key | Value |
|---|---|
| Labels | baseline, improved |
| Pixels / WxH | 256×256 (65,536 px) |
| Number of runs | 3 |

