#!/usr/bin/env python3
"""
Side-by-side DEMREG benchmark (single combined artifact)

Updates the original bench script to:
- Import the pipeline twice: once from Baseline/, once from CPU_Vectorization/ (or any two dirs)
- Run both on the same input payload in a single process
- Avoid module cache collisions between identical filenames
- Write one JSON + one NPZ for the report notebook to read

Outputs (in --outdir):
  - sxs_summary.json   (timings, chisq summaries, shapes, env)
  - sxs_arrays.npz     (DEM/edem/elogt/chisq/dn_reg for both A and B)
"""
import argparse
import contextlib
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

# --------------------------- import isolation ---------------------------
DEMREG_MODULES = ["dn2dem_pos", "demmap_pos", "dem_inv_gsvd", "dem_reg_map"]

@contextlib.contextmanager
def impl_import_scope(impl_dir: Path):
    """Temporarily import DEMREG modules from impl_dir, avoiding cache collisions."""
    impl_dir = impl_dir.resolve()
    sys.path.insert(0, str(impl_dir))
    # drop any cached modules with these names so we load fresh from impl_dir
    stash = {m: sys.modules.pop(m, None) for m in DEMREG_MODULES}
    try:
        yield
    finally:
        # clean out any newly imported modules to keep global state pristine
        for m in DEMREG_MODULES:
            sys.modules.pop(m, None)
            if stash[m] is not None:
                sys.modules[m] = stash[m]
        if sys.path and sys.path[0] == str(impl_dir):
            sys.path.pop(0)

def run_impl(impl_dir: Path, payload: Dict[str, np.ndarray], label: str) -> Dict:
    with impl_import_scope(impl_dir):
        from dn2dem_pos import dn2dem_pos  # imports its companions from impl_dir
        t0 = time.perf_counter()
        dem, edem, elogt, chisq, dn_reg = dn2dem_pos(
            payload["dn"], payload["edn"], payload["tresp"], payload["tresp_logt"], payload["temps"]
        )
        elapsed = time.perf_counter() - t0

    chisq = np.asarray(chisq).ravel()
    return {
        "label": label,
        "elapsed_s": float(elapsed),
        "dem": np.asarray(dem), "edem": np.asarray(edem),
        "elogt": np.asarray(elogt), "chisq": chisq, "dn_reg": np.asarray(dn_reg),
        "chisq_summary": {
            "mean": float(chisq.mean()),
            "median": float(np.median(chisq)),
            "p95": float(np.percentile(chisq, 95)),
            "min": float(chisq.min()), "max": float(chisq.max()), "count": int(chisq.size),
        },
        "shapes": {
            "dem": list(np.shape(dem)),
            "edem": list(np.shape(edem)),
            "elogt": list(np.shape(elogt)),
            "dn_reg": list(np.shape(dn_reg)),
        },
    }

# --------------------------- reproducible env ---------------------------
def pin_threads(threads: int):
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

# --------------------------- synthetic problem ---------------------------
def synth_response_bank(nf: int = 6):
    tresp_logt = np.linspace(5.7, 7.1, 200)
    centers = np.array([5.9, 6.1, 6.3, 6.5, 6.8, 7.0][:nf])
    widths = np.full(nf, 0.10)
    trmatrix = np.stack(
        [np.exp(-0.5 * ((tresp_logt - c) / w) ** 2) for c, w in zip(centers, widths)],
        axis=1
    ) * 1e-27
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

def build_problem(pixels: int, width: int, height: int, seed: int):
    temps = default_temperature_grid()
    trmatrix, tresp_logt = synth_response_bank(6)
    dn1, edn1 = make_synthetic_counts(trmatrix, tresp_logt, seed)

    if width > 0 and height > 0:
        nf = dn1.shape[0]
        dn = np.broadcast_to(dn1, (width, height, nf)).copy()
        edn = np.broadcast_to(edn1, (width, height, nf)).copy()
        npx = width * height
    else:
        nf = dn1.shape[0]
        dn = np.broadcast_to(dn1, (pixels, nf)).copy()
        edn = np.broadcast_to(edn1, (pixels, nf)).copy()
        npx = pixels

    return {
        "dn": dn, "edn": edn, "tresp": trmatrix, "tresp_logt": tresp_logt, "temps": temps,
        "npx": int(npx),
    }

# --------------------------- main ---------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Side-by-side DEMREG benchmark (single artifact, with repeats)")
    ap.add_argument("--baseline_dir", type=str, default="Baseline")
    ap.add_argument("--improved_dir", type=str, default="CPU_Vectorization")
    ap.add_argument("--pixels", type=int, default=1024, help="Used only if width/height are zero")
    ap.add_argument("--width", type=int, default=0, help="2-D width (if >0, pixels is ignored)")
    ap.add_argument("--height", type=int, default=0, help="2-D height (if >0, pixels is ignored)")
    ap.add_argument("--repeats", type=int, default=1, help="How many repeated runs")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_fixed", action="store_true", help="Use the same seed for all repeats")
    ap.add_argument("--outdir", type=str, default="out_data")
    args = ap.parse_args()

    # Pin BLAS/OMP threads for fairness/repro
    pin_threads(args.threads)
    env = capture_env()

    base_dir = Path(args.baseline_dir).resolve()
    imp_dir  = Path(args.improved_dir).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Baseline dir not found: {base_dir}")
    if not imp_dir.exists():
        raise FileNotFoundError(f"Improved dir not found: {imp_dir}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    repeats = max(1, args.repeats)

    # Accumulators for aggregate
    elapsed_A, elapsed_B = [], []
    chisq_mean_A, chisq_mean_B = [], []
    chisq_med_A,  chisq_med_B  = [], []
    chisq_p95_A,  chisq_p95_B  = [], []
    seeds = []


    for r in range(1, repeats + 1):
        seed_r = args.seed if args.seed_fixed else (args.seed + (r - 1))
        seeds.append(seed_r)
        payload = build_problem(args.pixels, args.width, args.height, seed=seed_r)

        resA = run_impl(base_dir, payload, "baseline")
        resB = run_impl(imp_dir,  payload, "improved")

        # Collect for aggregate
        elapsed_A.append(resA["elapsed_s"]);              elapsed_B.append(resB["elapsed_s"])
        chisq_mean_A.append(resA["chisq_summary"]["mean"]);  chisq_mean_B.append(resB["chisq_summary"]["mean"])
        chisq_med_A.append(resA["chisq_summary"]["median"]); chisq_med_B.append(resB["chisq_summary"]["median"])
        chisq_p95_A.append(resA["chisq_summary"]["p95"]);    chisq_p95_B.append(resB["chisq_summary"]["p95"])

        # Per-run outputs (if repeats==1, write directly to outdir; else to outdir/run{r})
        outdir_run = outdir if repeats == 1 else (outdir / f"run{r}")
        outdir_run.mkdir(parents=True, exist_ok=True)

        # Per-run JSON
        summary = {
            "labels": [resA["label"], resB["label"]],
            "elapsed_s": {resA["label"]: resA["elapsed_s"], resB["label"]: resB["elapsed_s"]},
            "speedup_baseline_over_improved": float(resA["elapsed_s"] / resB["elapsed_s"]) if resB["elapsed_s"] > 0 else float("nan"),
            "chisq_summary": {resA["label"]: resA["chisq_summary"], resB["label"]: resB["chisq_summary"]},
            "shapes": {resA["label"]: resA["shapes"], resB["label"]: resB["shapes"]},
            "env": env,
            "payload_info": {"pixels": args.pixels, "width": args.width, "height": args.height, "seed": seed_r},
            "run_index": r,
            "repeats": repeats,
        }
        (outdir_run / "sxs_summary.json").write_text(json.dumps(summary, indent=2))

        # Per-run arrays
        np.savez_compressed(outdir_run / "sxs_arrays.npz",
            dem_A = resA["dem"],   edem_A = resA["edem"],   elogt_A = resA["elogt"],
            chisq_A = resA["chisq"], dn_reg_A = resA["dn_reg"],
            dem_B = resB["dem"],   edem_B = resB["edem"],   elogt_B = resB["elogt"],
            chisq_B = resB["chisq"], dn_reg_B = resB["dn_reg"],
            labelA = resA["label"], labelB = resB["label"],
        )

    # Aggregate summary at root
    agg = {
        "repeats": repeats,
        "payload_info": {"pixels": args.pixels, "width": args.width, "height": args.height, "seed_base": args.seed},
        "env": env,
        "elapsed_s": {
            "baseline": {"values": elapsed_A, "mean": float(np.mean(elapsed_A)), "std": float(np.std(elapsed_A, ddof=1)) if len(elapsed_A) > 1 else 0.0},
            "improved": {"values": elapsed_B, "mean": float(np.mean(elapsed_B)), "std": float(np.std(elapsed_B, ddof=1)) if len(elapsed_B) > 1 else 0.0},
            "speedup_baseline_over_improved": float(np.mean(np.array(elapsed_A) / np.array(elapsed_B))),
        },
        "chisq_summary": {
            "baseline": {
                "mean_mean": float(np.mean(chisq_mean_A)),
                "mean_median": float(np.mean(chisq_med_A)),
                "mean_p95": float(np.mean(chisq_p95_A)),
            },
            "improved": {
                "mean_mean": float(np.mean(chisq_mean_B)),
                "mean_median": float(np.mean(chisq_med_B)),
                "mean_p95": float(np.mean(chisq_p95_B)),
            },
        },
    }
    (outdir / "aggregate_sxs_summary.json").write_text(json.dumps(agg, indent=2))

    with open(outdir / "aggregate_timings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run", "seed",
            "t_baseline_s", "t_improved_s", "speedup (in %)",
            "chisq_mean_baseline", "chisq_median_baseline",
            "chisq_mean_improved", "chisq_median_improved",
        ])
        for i, (seed_i, ta, tb, mA, mdA, mB, mdB) in enumerate(
            zip(seeds, elapsed_A, elapsed_B, chisq_mean_A, chisq_med_A, chisq_mean_B, chisq_med_B),
            start=1
        ):
            speedup_pct = 100.0 * ((ta / tb))
            w.writerow([
                i, seed_i,
                f"{ta:.6f}", f"{tb:.6f}", f"{speedup_pct:.2f}",
                f"{mA:.6f}", f"{mdA:.6f}",
                f"{mB:.6f}", f"{mdB:.6f}",
            ])



    print(f"[OK] Wrote aggregate to {outdir/'aggregate_sxs_summary.json'}")
    if repeats > 1:
        print(f"Per-run artifacts under: {outdir}/run1..run{repeats}")

if __name__ == "__main__":
    main()
