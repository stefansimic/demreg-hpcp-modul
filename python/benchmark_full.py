import numpy as np
import cupy as cp
import time
import csv
import os
from dotenv import load_dotenv
from sys import path as sys_path
from itertools import product

# -----------------------------
# 🔧 Benchmark Settings
# -----------------------------
PIXELS_LIST = [200, 1000, 2000, 5000]
NT_LIST = [200, 400]
NF = 6
REPEATS = 5
OUTPUT_CSV = "./results/benchmark_results.csv"

# -----------------------------
# 📁 Import solvers
# -----------------------------
load_dotenv()
demreg_path = os.getenv("DEMREG_PATH")
if demreg_path:
    sys_path.append(demreg_path)
    sys_path.append(demreg_path + "/Baseline")
    sys_path.append(demreg_path + "/vectorized")
    sys_path.append(demreg_path + "/gpu")
    sys_path.append(demreg_path + "/dask_opt")
else:
    raise RuntimeError("DEMREG_PATH is not set. Please define it in your environment.")

from demmap_pos import demmap_pos
from demmap_pos_vectorized import demmap_pos_vectorized
from demmap_pos_dask import demmap_pos_dask
from demmap_pos_gpu import demmap_pos_gpu
from demmap_pos_gpu_l2 import demmap_pos_gpu_l2

# Register solvers for iteration
ALGORITHMS = {
    "cpu": demmap_pos,
    "vectorized": demmap_pos_vectorized,
    "dask": demmap_pos_dask,
    "gpu": demmap_pos_gpu,
    "gpu-l2": demmap_pos_gpu_l2,
}


# -----------------------------
# 📊 Data Generation
# -----------------------------
def generate_test_data(na, nf, nt):
    logt = np.linspace(5.7, 7.1, nt)
    dlogt = np.full(nt, 0.05)

    d1, m1, s1 = 4e22, 6.5, 0.15
    root2pi = np.sqrt(2 * np.pi)
    dem_model = (d1 / (root2pi * s1)) * np.exp(-(logt - m1) ** 2 / (2 * s1**2))

    rmatrix = np.random.rand(nt, nf) * 1e-24
    tc_full = dem_model[:, None] * rmatrix * (10**logt[:, None]) * np.log(10**dlogt[:, None])
    dn_base = np.sum(tc_full, axis=0)

    dd = np.zeros((na, nf))
    ed = np.zeros((na, nf))
    np.random.seed(42)
    for i in range(na):
        variation = 0.8 + 0.4 * np.random.rand()
        dd[i, :] = dn_base * variation
        ed[i, :] = 0.1 * dd[i, :]
        dd[i, :] += np.random.randn(nf) * ed[i, :]
    return dd, ed, rmatrix, logt, dlogt


# -----------------------------
# 🧪 Benchmark Execution
# -----------------------------
def run_solver(solver, dd, ed, rmatrix, logt, dlogt):
    glc = np.ones(ed.shape[1])
    dem_norm0 = np.ones((dd.shape[0], logt.shape[0]))
    return solver(dd, ed, rmatrix, logt, dlogt, glc, dem_norm0=dem_norm0)


def benchmark_algorithm(name, solver, dd, ed, rmatrix, logt, dlogt, repeats=5):
    times = []
    results = None

    # Warmup run (optional but improves consistency)
    _ = run_solver(solver, dd, ed, rmatrix, logt, dlogt)

    for _ in range(repeats):
        start = time.perf_counter()
        results = run_solver(solver, dd, ed, rmatrix, logt, dlogt)
        end = time.perf_counter()
        times.append(end - start)

    median_time = np.median(times)
    std_time = np.std(times)
    return median_time, std_time, results

# -----------------------------
# 🚀 Main Benchmark Loop
# -----------------------------
def main():
    results = []
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    for pixels, nt in product(PIXELS_LIST, NT_LIST):
        print(f"\n📊 Benchmarking for pixels={pixels}, nt={nt}")

        # Generate data once per parameter combo
        dd, ed, rmatrix, logt, dlogt = generate_test_data(pixels, NF, nt)

        # Always run CPU baseline for comparison
        cpu_time, cpu_std, cpu_result = benchmark_algorithm(
            "cpu", ALGORITHMS["cpu"], dd, ed, rmatrix, logt, dlogt, REPEATS
        )

        for algo_name, solver in ALGORITHMS.items():
            print(f"▶️ Running {algo_name.upper()}...")
            median_time, std_time, result = benchmark_algorithm(
                algo_name, solver, dd, ed, rmatrix, logt, dlogt, REPEATS
            )
            _, _, _, chisq, _ = result
            chisq_median = np.median(chisq)

            results.append(
                [
                    algo_name,
                    pixels,
                    NF,
                    nt,
                    REPEATS,
                    median_time,
                    std_time,
                    chisq_median,
                ]
            )

    # Save results
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "pixels",
                "nf",
                "nt",
                "repeats",
                "median_time",
                "std_time",
                "median_chisq",
            ]
        )
        writer.writerows(results)

    print(f"\n✅ Benchmarking complete. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
