import numpy as np
import time
import os
from dotenv import load_dotenv
from sys import path as sys_path

# --- Setup paths ---
load_dotenv()
demreg_path = os.getenv("DEMREG_PATH")
if demreg_path:
    sys_path.append(demreg_path)
else:
    raise RuntimeError("DEMREG_PATH is not set. Please define it in your environment.")

from demmap_pos import demmap_pos
from demmap_pos_vectorized import demmap_pos_vectorized

# --- Benchmark Parameters ---
na = 200
nf = 6
nt = 200
repeats = 3

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

dd, ed, rmatrix, logt, dlogt = generate_test_data(na, nf, nt)
glc = np.ones(nf)
dem_norm0 = np.ones((na, nt))

print(f"🚀 Benchmark Setup:")
print(f"   Pixels:         {na}")
print(f"   Channels (nf):  {nf}")
print(f"   Temp bins (nt): {nt}\n")

# -------------------------------------------------------------------
# Benchmark helper function
# -------------------------------------------------------------------
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
# 📊 CHI²-Analyse (nur interne Solver-Werte)
# -------------------------------------------------------------------
print("\n📊 Data-space quality analysis (χ²)")

def describe_chisq(name, chisq):
    median = np.median(chisq)
    p25, p75 = np.percentile(chisq, [25, 75])
    p95 = np.percentile(chisq, 95)
    print(f"   📉 {name} reduced χ²:")
    print(f"      median = {median:.3f}")
    print(f"      IQR    = [{p25:.3f}, {p75:.3f}]")
    print(f"      95%ile = {p95:.3f}")

describe_chisq("Baseline", baseline_chisq)
describe_chisq("Vectorized", vectorized_chisq)

chisq_diff = vectorized_chisq - baseline_chisq
chisq_diff_median = np.median(chisq_diff)
chisq_diff_p95 = np.percentile(np.abs(chisq_diff), 95)
frac_close = np.mean(np.abs(chisq_diff) < 0.2)

print(f"\n   🔍 Δχ² (Vectorized - Baseline):")
print(f"      median Δχ² = {chisq_diff_median:.3f}")
print(f"      95%ile |Δχ²| = {chisq_diff_p95:.3f}")
print(f"      ✅ {frac_close*100:.1f}% der Pixel liegen innerhalb ±0.2 χ²-Differenz\n")

# -------------------------------------------------------------------
# 📉 DEM-space comparison
# -------------------------------------------------------------------
print("📉 DEM-space comparison:")

rel_diff = np.linalg.norm(baseline_result - vectorized_result) / np.linalg.norm(baseline_result)
cos_sim = np.median(
    np.sum(baseline_result * vectorized_result, axis=1)
    / (np.linalg.norm(baseline_result, axis=1) * np.linalg.norm(vectorized_result, axis=1) + 1e-30)
)

print(f"   🔍 Relative L2 difference: {rel_diff:.3e}")
print(f"   🤝 Median cosine similarity: {cos_sim:.3f}")

# -------------------------------------------------------------------
# ⚡ Performance
# -------------------------------------------------------------------
print("\n⚡ Performance:")
speedup = baseline_time / vectorized_time
print(f"   ⚡ speedup: {speedup:.2f}× faster than baseline\n")

# Sanity check
if np.median(vectorized_chisq) > 2.0 or np.median(vectorized_chisq) < 0.5:
    print("⚠️  Warning: Vectorized χ² deviates strongly from expected ~1. Check regularization/normalization!")
