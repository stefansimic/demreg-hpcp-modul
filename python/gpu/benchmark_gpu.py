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
from demmap_pos_gpu import demmap_pos_gpu

# --- Benchmark Parameters ---
na = 2000     # Anzahl DEMs (Pixel)
nf = 6        # AIA-Kanäle
nt = 200      # Temperatur-Bins
repeats = 3   # Mehrfach messen für stabilere Zeiten

# --- Generate Random Test Data ---
dd = np.random.random((na, nf)) * 1e3
ed = np.random.random((na, nf)) * 20 + 5
rmatrix = np.random.random((nt, nf)) * 1e-23
logt = np.linspace(5.5, 7.5, nt)
dlogt = np.gradient(logt)
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
# Run CPU Benchmark
# -------------------------------------------------------------------
print("🧠 Running CPU baseline...")
(cpu_result, *_), cpu_time = benchmark(
    demmap_pos,
    "CPU",
    dd, ed, rmatrix, logt, dlogt, glc,
    dem_norm0=dem_norm0,
    repeats=repeats
)


# -------------------------------------------------------------------
# Run GPU Benchmark
# -------------------------------------------------------------------
print("🖥️  Running GPU solver...")
(gpu_result, *_), gpu_time = benchmark(
    demmap_pos_gpu,
    "GPU",
    dd, ed, rmatrix, logt, dlogt, glc,
    dem_norm0=dem_norm0,
    repeats=repeats
)


# -------------------------------------------------------------------
# Result Comparison
# -------------------------------------------------------------------
print("📊 Result comparison:")

rel_diff = np.linalg.norm(cpu_result - gpu_result) / np.linalg.norm(cpu_result)
abs_diff = np.abs(cpu_result - gpu_result)
max_diff = abs_diff.max()
mean_diff = abs_diff.mean()

print(f"   🔍 Relative L2 difference:  {rel_diff:.3e}")
print(f"   📉 Mean absolute difference: {mean_diff:.3e}")
print(f"   📈 Max absolute difference:  {max_diff:.3e}\n")

speedup = cpu_time / gpu_time
print(f"⚡ GPU speedup: {speedup:.2f}× faster than CPU\n")

# Optional: sanity check
if rel_diff > 1e-2:
    print("⚠️  Warning: Relative difference > 1e-2 – check solver parameters!")
