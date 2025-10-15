"""
Dask-enabled replacement for dn2dem_pos.py with the same public interface.
The function name and signature are identical to the original, so this module
can be swapped in without changing call sites.
"""

import numpy as np
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import os

from demmap_pos_dask import demmap_pos  # use the dask-enabled demmap_pos internally


def dn2dem_pos(dd, ed, rmatrix, logt, dlogt, glc,
               reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
               dem_norm0=None, nmu=42, warn=False,
               l_emd=False, rscl=False):
    """
    Keep the same interface as original dn2dem_pos.

    Returns:
        dem, edem, elogt, chisq, dn_reg
    """

    dd = np.asarray(dd)
    ed = np.asarray(ed)
    na = dd.shape[0]
    nt = rmatrix.shape[0]

    if dem_norm0 is None:
        dem_norm0 = np.ones((na, nt))

    num_workers = max(1, (os.cpu_count() or 4))

    tasks = []
    for i in range(na):
        # call demmap_pos for a single-row slice; demmap_pos returns arrays for that slice
        task = delayed(demmap_pos)(
            dd[i:i+1, :], ed[i:i+1, :], rmatrix, logt, dlogt, glc,
            reg_tweak, max_iter, rgt_fact, dem_norm0[i:i+1, :],
            nmu, warn, l_emd, rscl
        )
        tasks.append(task)

    with ProgressBar():
        results = compute(*tasks, scheduler="threads", num_workers=num_workers)

    dem = np.vstack([r[0] for r in results])
    edem = np.vstack([r[1] for r in results])
    elogt = np.vstack([r[2] for r in results])
    chisq = np.array([r[3] for r in results])
    dn_reg = np.vstack([r[4] for r in results])

    return dem, edem, elogt, chisq, dn_reg
