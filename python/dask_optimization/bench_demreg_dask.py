#!/usr/bin/env python3
"""
bench_demreg.py — clean rewrite

Purpose
-------
Baseline benchmark for the DEMREG pipeline with **guaranteed artifact outputs**:
- Always saves arrays to:   <outdir>/outputs_last.npz
- Always saves preview to:  <outdir>/outputs_last_preview.json (human-readable)
- Timing summaries to:      <outdir>/bench_results.json
- Per-run CSV to:           <outdir>/bench_runs.csv

It exercises the solver through dn2dem_pos() end‑to‑end using either
synthetic AIA-like responses or an optional real AIA .sav file.

Usage
-----
# 1) 1D strip (1024 pixels)
python bench_demreg.py --mode synthetic --pixels 1024 --repeats 5 --warmup 1 --threads 1 --outdir bench_out

# 2) 2D image (256x256)
python bench_demreg.py --mode synthetic --width 256 --height 256 --repeats 3 --threads 1 --outdir bench_out

# 3) Save a golden baseline NPZ explicitly
python bench_demreg.py --mode synthetic --pixels 2048 --save-baseline bench_out/golden_baseline.npz

# 4) Compare current outputs to a golden baseline
python bench_demreg.py --mode synthetic --pixels 2048 --compare bench_out/golden_baseline.npz
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
# DEMREG entry point
from dn2dem_pos_dask import dn2dem_pos_dask

try:
    import scipy.io as sio  # only needed if --mode aia
except Exception:
    sio = None

# --------------------------- helpers ---------------------------

def pin_threads(threads: int):
    # Make timings reproducible and avoid BLAS oversubscription
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(threads)


def capture_env():
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
    }


def synth_response_bank(nf: int = 6):
    # Smooth positive response curves around six bands (logT in [5.7, 7.1])
    tresp_logt = np.linspace(5.7, 7.1, 200)
    centers = np.array([5.9, 6.1, 6.3, 6.5, 6.8, 7.0][:nf])
    widths = np.full(nf, 0.10)
    trmatrix = np.stack([
        np.exp(-0.5 * ((tresp_logt - c) / w) ** 2) for c, w in zip(centers, widths)
    ], axis=1) * 1e-27
    return trmatrix, tresp_logt


def load_aia_response_from_sav(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if sio is None:
        raise RuntimeError("scipy is required to load .sav files. Please `conda install scipy`. ")
    dat = sio.readsav(str(path))
    tresp_logt = np.array(dat['logt']).astype(float)
    nf = len(dat['tr'][:])
    nt = len(tresp_logt)
    trmatrix = np.zeros((nt, nf), dtype=float)
    for i in range(nf):
        trmatrix[:, i] = dat['tr'][i]
    return trmatrix, tresp_logt


def gaussian_dem(tresp_logt):
    d1, m1, s1 = 4e22, 6.5, 0.15
    root2pi = np.sqrt(2 * np.pi)
    return (d1 / (root2pi * s1)) * np.exp(-(tresp_logt - m1) ** 2 / (2 * s1 ** 2))


def make_synthetic_counts(trmatrix: np.ndarray, tresp_logt: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    nt, nf = trmatrix.shape
    dem_mod = gaussian_dem(tresp_logt)

    tresp_dlogt = np.full(nt, 0.05)
    tc_full = dem_mod[:, None] * trmatrix * (10 ** tresp_logt)[:, None] * np.log(10 ** tresp_dlogt)[:, None]
    dn_in = tc_full.sum(axis=0)

    gains = np.array([18.3, 17.6, 17.7, 18.3, 18.3, 17.6][:nf])
    dn2ph = gains * np.array([94, 131, 171, 193, 211, 335][:nf]) / 3397.0
    rdnse = np.array([1.14, 1.18, 1.15, 1.20, 1.20, 1.18][:nf])
    exp_time = 2.9
    shotnoise = np.sqrt(dn2ph * (dn_in * exp_time)) / dn2ph / exp_time
    edn_in = np.sqrt(rdnse ** 2 + shotnoise ** 2)

    dn_in = dn_in * (1.0 + 1e-6 * rng.standard_normal(size=nf))
    return dn_in.astype(float), edn_in.astype(float)


def default_temperature_grid():
    return np.logspace(5.7, 7.1, 35)  # 35 edges → 34 bins


def build_problem(mode: str, pixels: int, width: int, height: int, aia_resp: str | None, seed: int):
    temps = default_temperature_grid()
    if mode == "aia":
        if not aia_resp:
            raise ValueError("--aia-resp is required when mode=aia")
        trmatrix, tresp_logt = load_aia_response_from_sav(Path(aia_resp))
    else:
        trmatrix, tresp_logt = synth_response_bank(6)

    dn1, edn1 = make_synthetic_counts(trmatrix, tresp_logt, seed)

    if width > 0 and height > 0:
        nf = dn1.shape[0]
        dn = np.broadcast_to(dn1, (width, height, nf)).copy()
        edn = np.broadcast_to(edn1, (width, height, nf)).copy()
        shape_tag = {"kind": "image", "width": int(width), "height": int(height)}
        npx = width * height
    else:
        nf = dn1.shape[0]
        dn = np.broadcast_to(dn1, (pixels, nf)).copy()
        edn = np.broadcast_to(edn1, (pixels, nf)).copy()
        shape_tag = {"kind": "row", "pixels": int(pixels)}
        npx = pixels

    return {"dn": dn, "edn": edn, "tresp": trmatrix, "tresp_logt": tresp_logt, "temps": temps, "shape": shape_tag, "npx": int(npx)}


def run_once(payload):
    t0 = time.perf_counter()
    dem, edem, elogt, chisq, dn_reg = dn2dem_pos_dask(payload["dn"], payload["edn"], payload["tresp"], payload["tresp_logt"], payload["temps"])
    t1 = time.perf_counter()
    return (dem, edem, elogt, chisq, dn_reg, t1 - t0)


def save_outputs_npz(path: Path, dem, edem, elogt, chisq, dn_reg, meta: dict):
    np.savez_compressed(path, dem=np.asarray(dem), edem=np.asarray(edem), elogt=np.asarray(elogt), chisq=np.asarray(chisq), dn_reg=np.asarray(dn_reg), meta=json.dumps(meta))


def summarize_preview(dem, edem, elogt, chisq, dn_reg):
    def stats(x):
        x = np.asarray(x)
        return {"shape": list(x.shape), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "p50": float(np.median(x)), "p99": float(np.quantile(x, 0.99))}
    return {"dem": stats(dem), "edem": stats(edem), "elogt": stats(elogt), "chisq": stats(chisq), "dn_reg": stats(dn_reg)}


def compare_against(baseline_npz: Path, current, rtol: float, atol: float, chi2_tol: float):
    d = np.load(baseline_npz, allow_pickle=True)
    b_dem, b_edem, b_elogt, b_chi2, b_dn = d["dem"], d["edem"], d["elogt"], d["chisq"], d["dn_reg"]
    dem, edem, elogt, chisq, dn_reg = current
    def ok(a,b): return np.allclose(a, b, rtol=rtol, atol=atol)
    report = {
        "dem": ok(dem, b_dem),
        "edem": ok(edem, b_edem),
        "elogt": ok(elogt, b_elogt),
        "dn_reg": ok(dn_reg, b_dn),
    }
    med_ok = abs(float(np.median(chisq)) - float(np.median(b_chi2))) <= chi2_tol
    overall = all(report.values()) and med_ok
    return overall, report, {"median_ok": med_ok, "median_diff": float(abs(np.median(chisq) - np.median(b_chi2)))}

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="DEMREG baseline benchmark with NPZ + preview outputs")
    ap.add_argument("--mode", choices=["synthetic","aia"], default="synthetic")
    ap.add_argument("--aia-resp", type=str, default=None)
    ap.add_argument("--pixels", type=int, default=1024)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="bench_out")
    # optional regression
    ap.add_argument("--save-baseline", type=str, default=None)
    ap.add_argument("--compare", type=str, default=None)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-7)
    ap.add_argument("--chi2-tol", type=float, default=1e-3)

    args = ap.parse_args()

    pin_threads(args.threads)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    payload = build_problem(args.mode, args.pixels, args.width, args.height, args.aia_resp, args.seed)

    # Warmups (not recorded)
    for _ in range(max(0, args.warmup)):
        _ = run_once(payload)

    # Timed repeats
    results = []
    last = None
    for i in range(max(1, args.repeats)):
        dem, edem, elogt, chisq, dn_reg, elapsed = run_once(payload)
        last = (dem, edem, elogt, chisq, dn_reg)
        results.append({"elapsed_s": elapsed, "chisq_mean": float(np.mean(chisq)), "chisq_median": float(np.median(chisq)), "dem_shape": list(np.shape(dem)), "chisq_shape": list(np.shape(chisq))})
        print(f"Run {i+1}/{args.repeats}: {elapsed:.4f}s, chi²≈{results[-1]['chisq_mean']:.3f}")

    # Aggregate timings
    times = np.array([r["elapsed_s"] for r in results], float)
    elapsed_mean = float(times.mean()); elapsed_std = float(times.std(ddof=1)) if len(times)>1 else 0.0
    npx = payload["npx"]
    throughput = float(npx / elapsed_mean) if elapsed_mean>0 else float("nan")

    env = capture_env()

    # ALWAYS save NPZ + preview from last run
    outputs_npz = outdir / "outputs_last.npz"
    save_outputs_npz(outputs_npz, *last, meta={"env": env, "shape": payload["shape"], "mode": args.mode})
    preview_json = outdir / "outputs_last_preview.json"
    with open(preview_json, "w") as f:
        json.dump({"bench_info": {"env": env, "shape": payload["shape"], "mode": args.mode}, "outputs_summary": summarize_preview(*last)}, f, indent=2)
    print(f"Saved outputs: {outputs_npz} Saved preview: {preview_json}")

    # Optional: save golden & compare
    if args.save_baseline:
        save_outputs_npz(Path(args.save_baseline), *last, meta={"env": env, "shape": payload["shape"], "mode": args.mode})
        print(f"Saved golden baseline: {args.save_baseline}")
    cmp_info = None
    if args.compare:
        ok, rep, extra = compare_against(Path(args.compare), last, args.rtol, args.atol, args.chi2_tol)
        cmp_info = {"ok": ok, "report": rep, **extra, "baseline": args.compare}
        print(f"Comparison overall OK: {ok}")

    # Write timing JSON + CSV
    bench_summary = {"elapsed_mean_s": elapsed_mean, "elapsed_std_s": elapsed_std, "throughput_px_per_s": throughput, "px_count": int(npx), "results_each": results, "bench_info": {"env": env, "shape": payload["shape"], "mode": args.mode}, "comparison": cmp_info}

    json_path = outdir / "bench_results.json"
    with open(json_path, "w") as f: json.dump(bench_summary, f, indent=2)

    csv_path = outdir / "bench_runs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run","elapsed_s","chisq_mean","chisq_median","dem_shape","chisq_shape"])
        for i, r in enumerate(results, 1):
            w.writerow([i, r["elapsed_s"], r["chisq_mean"], r["chisq_median"], r["dem_shape"], r["chisq_shape"]])

    print(f"Wrote {json_path} and {csv_path}")

if __name__ == "__main__":
    main()
