import numpy as np
import time
import os
from dotenv import load_dotenv
from sys import path as sys_path

# --- Generate Realistic Test Data ---
def generate_test_data(na, nf, nt):
    logt = np.linspace(5.7, 7.1, nt)
    dlogt = np.full(nt, 0.05)

    d1, m1, s1 = 4e22, 6.5, 0.15
    root2pi = np.sqrt(2 * np.pi)
    dem_model = (d1 / (root2pi * s1)) * np.exp(-(logt - m1)**2 / (2 * s1**2))

    rmatrix = np.random.rand(nt, nf) * 1e-24

    tc_full = dem_model[:, None] * rmatrix * (10**logt[:, None]) * np.log(10**dlogt[:, None])
    dn_base = np.sum(tc_full, axis=0)  # (nf,)

    dd = np.zeros((na, nf))
    ed = np.zeros((na, nf))
    np.random.seed(42)
    for i in range(na):
        variation = 0.8 + 0.4 * np.random.rand()
        dd[i, :] = dn_base * variation
        ed[i, :] = 0.1 * dd[i, :]
        dd[i, :] += np.random.randn(nf) * ed[i, :]  # Messrauschen

    return dd, ed, rmatrix, logt, dlogt


# --- Benchmark helper function ---
def benchmark(func, name, *args, repeats=3, **kwargs):
    times = []
    result = None
    for i in range(repeats):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
        print(f"   {name} run {i+1}/{repeats}: {times[-1]:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"✅ {name} avg runtime: {avg_time:.3f}s ± {std_time:.3f}s\n")
    return result, avg_time


def describe_chisq(name, chisq):
    median = np.median(chisq)
    p25, p75 = np.percentile(chisq, [25, 75])
    p95 = np.percentile(chisq, 95)
    print(f"   📉 {name} reduced χ²:")
    print(f"      median = {median:.3f}")
    print(f"      IQR    = [{p25:.3f}, {p75:.3f}]")
    print(f"      95%ile = {p95:.3f}")


# =====================================================================
# Main entry point
# =====================================================================
def main():
    # --- Setup paths ---
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

    # --- Try GPU version ---
    try:
        import cupy as cp
        from demmap_pos_gpu import demmap_pos_gpu
        from demmap_pos_gpu_l2 import demmap_pos_gpu_l2
        HAS_GPU = True
    except ImportError:
        HAS_GPU = False

    # --- Benchmark Parameters ---
    na = 200
    nf = 6
    nt = 200
    repeats = 1

    dd, ed, rmatrix, logt, dlogt = generate_test_data(na, nf, nt)
    glc = np.ones(nf)
    dem_norm0 = np.ones((na, nt))

    print(f"🚀 Benchmark Setup:")
    print(f"   Pixels:         {na}")
    print(f"   Channels (nf):  {nf}")
    print(f"   Temp bins (nt): {nt}\n")

    # -------------------------------------------------------------------
    # Run Baseline Benchmark
    # -------------------------------------------------------------------
    print("🧠 Running baseline...")
    (baseline_result, baseline_edem, baseline_elogt, baseline_chisq, baseline_dn_reg), baseline_time = benchmark(
        demmap_pos,
        "Baseline",
        dd, ed, rmatrix, logt, dlogt, glc,
        dem_norm0=dem_norm0,
        repeats=repeats
    )

    # -------------------------------------------------------------------
    # Run Vectorized Benchmark
    # -------------------------------------------------------------------
    print("🖥️  Running vectorized solver...")
    (vectorized_result, vectorized_edem, vectorized_elogt, vectorized_chisq, vectorized_dn_reg), vectorized_time = benchmark(
        demmap_pos_vectorized,
        "Vectorized",
        dd, ed, rmatrix, logt, dlogt, glc,
        dem_norm0=dem_norm0,
        repeats=repeats
    )

    # -------------------------------------------------------------------
    # Run Dask Benchmark
    # -------------------------------------------------------------------
    print("🖥️  Running dask solver...")
    (dask_result, dask_edem, dask_elogt, dask_chisq, dask_dn_reg), dask_time = benchmark(
        demmap_pos_dask,
        "Dask",
        dd, ed, rmatrix, logt, dlogt, glc,
        dem_norm0=dem_norm0,
        repeats=repeats
    )

    # -------------------------------------------------------------------
    # Run GPU Benchmark (CuPy)
    # -------------------------------------------------------------------
    if HAS_GPU:
        print("🧪 Running GPU (CuPy) solver...")
        (gpu_result, gpu_edem, gpu_elogt, gpu_chisq, gpu_dn_reg), gpu_time = benchmark(
            demmap_pos_gpu,
            "GPU (CuPy)",
            dd, ed, rmatrix, logt, dlogt, glc,
            dem_norm0=dem_norm0,
            repeats=repeats
        )

        print("🧪 Running GPU L2 (CuPy) solver...")
        (gpu_l2_result, gpu_l2_edem, gpu_l2_elogt, gpu_l2_chisq, gpu_l2_dn_reg), gpu_l2_time = benchmark(
            demmap_pos_gpu_l2,
            "GPU L2 (CuPy)",
            dd, ed, rmatrix, logt, dlogt, glc,
            dem_norm0=dem_norm0,
            repeats=repeats
        )
    else:
        print("⚠️ CuPy / GPU implementation not available — skipping GPU benchmark.")
        gpu_result = gpu_chisq = None
        gpu_time = np.nan

        gpu_l2_result = gpu_chisq = None
        gpu_l2_time = np.nan

    # -------------------------------------------------------------------
    # 📊 CHI²-Analyse (nur interne Solver-Werte)
    # -------------------------------------------------------------------
    print("\n📊 Data-space quality analysis (χ²)")
    describe_chisq("Baseline", baseline_chisq)
    describe_chisq("Vectorized", vectorized_chisq)
    describe_chisq("Dask", dask_chisq)
    if HAS_GPU:
        describe_chisq("GPU", gpu_chisq)
        describe_chisq("GPU l2", gpu_l2_chisq)

    # -------------------------------------------------------------------
    # ⚡ Performance
    # -------------------------------------------------------------------
    print("\n⚡ Performance:")
    speedup_vec = baseline_time / vectorized_time
    print(f"   ⚡ Vectorized speedup: {speedup_vec:.2f}× faster than baseline")

    if HAS_GPU and not np.isnan(gpu_time):
        speedup_gpu = baseline_time / gpu_time
        speedup_gpu_l2 = baseline_time / gpu_l2_time

        print(f"   ⚡ GPU speedup: {speedup_gpu:.2f}× faster than baseline")
        print(f"   ⚡ GPU L2 speedup: {speedup_gpu_l2:.2f}× faster than baseline")
        print()
        print(f"   ⚡ GPU vs Vectorized: {vectorized_time / gpu_time:.2f}× faster than vectorized")
        print(f"   ⚡ GPU L2 vs Vectorized: {vectorized_time / gpu_l2_time:.2f}× faster than vectorized")
        print()
        print(f"   ⚡ GPU L2 vs GPU: {gpu_time / gpu_l2_time:.2f}× faster than GPU")


# =====================================================================
# Safe entry point for multiprocessing / Dask
# =====================================================================
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
