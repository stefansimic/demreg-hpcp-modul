"""
Minimal benchmark script in dask_opt that mirrors the original bench_demreg.py
but imports the dask-enabled modules (same call signatures).
"""

import numpy as np
import time

from demmap_pos_dask import demmap_pos
from dn2dem_pos_dask import dn2dem_pos

def main():
    print("\nRunning minimal benchmark (dask-enabled modules)")
    na, nf, nt = 500, 6, 200
    dd = np.random.random((na, nf)) * 1e3
    ed = np.random.random((na, nf)) * 20 + 5
    rmatrix = np.random.random((nt, nf)) * 1e-23
    logt = np.linspace(5.5, 7.5, nt)
    dlogt = np.gradient(logt)
    glc = np.ones(nf)
    dem_norm0 = np.ones((na, nt))

    print("Benchmark demmap_pos")
    t0 = time.time()
    dem, edem, elogt, chisq, dn_reg = demmap_pos(dd, ed, rmatrix, logt, dlogt, glc, dem_norm0=dem_norm0)
    t1 = time.time()
    print(f"  demmap_pos runtime: {t1-t0:.2f} s")

    print("Benchmark dn2dem_pos")
    t0 = time.time()
    dem, edem, elogt, chisq, dn_reg = dn2dem_pos(dd, ed, rmatrix, logt, dlogt, glc, dem_norm0=dem_norm0)
    t1 = time.time()
    print(f"  dn2dem_pos runtime: {t1-t0:.2f} s")

if __name__ == '__main__':
    main()
