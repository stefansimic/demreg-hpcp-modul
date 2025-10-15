"""
Dask-enabled dn2dem_pos with exactly the same interface as the original.

Implements a parallel version that distributes the per-pixel DEM inversions
over available CPU threads using Dask. The algorithm internally calls the
dask_opt.demmap_pos.demmap_pos() function (which itself is Dask-optimized).
"""

import numpy as np
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import os

# Import the Dask-enabled demmap_pos (same interface)
from demmap_pos_dask import demmap_pos


def dn2dem_pos(dn_in, edn_in, tresp, tresp_logt, temps,
               reg_tweak=1.0, max_iter=10, gloci=0,
               rgt_fact=1.5, dem_norm0=None, nmu=40,
               warn=False, emd_int=False, emd_ret=False,
               l_emd=False, non_pos=False, rscl=False):
    """
    Dask-parallel dn2dem_pos with the same interface as original.
    Accepts dn_in either as (n_pixels, nf) or as (width, height, nf).
    Returns dem shaped like input (i.e. image -> image, row -> row).
    """
    import numpy as np
    from dask import delayed, compute
    from dask.diagnostics import ProgressBar
    import os
    from demmap_pos_dask import demmap_pos  # internal dask-enabled routine

    # ensure arrays
    dn = np.asarray(dn_in)
    edn = np.asarray(edn_in)
    tresp = np.asarray(tresp)
    tresp_logt = np.asarray(tresp_logt)
    temps = np.asarray(temps)

    # Detect image mode (width, height, nf) vs flat mode (npx, nf)
    input_was_image = False
    if dn.ndim == 3:
        input_was_image = True
        width, height, nf = dn.shape
        npx = width * height
        # flatten to (npx, nf)
        dn_flat = dn.reshape((npx, nf))
        edn_flat = edn.reshape((npx, nf))
    elif dn.ndim == 2:
        dn_flat = dn
        edn_flat = edn
        npx, nf = dn_flat.shape
    else:
        raise ValueError("dn_in must be 2D (n_pixels,nf) or 3D (w,h,nf)")

    nt = tresp.shape[0]

    if dem_norm0 is None:
        dem_norm0 = np.ones((npx, nt))
    else:
        dem_norm0 = np.asarray(dem_norm0)
        # if provided as image, flatten similarly
        if dem_norm0.ndim == 3:
            dem_norm0 = dem_norm0.reshape((npx, nt))

    # number of workers: keep simple, let system decide
    num_workers = max(1, (os.cpu_count() or 4))

    # Build delayed tasks: call demmap_pos on 1-row slices (shape (1,nf))
    tasks = []
    dlogt = np.gradient(tresp_logt)  # must match tresp length (nt)
    glc = np.ones(tresp.shape[1])    # kept for interface compatibility (placeholder)
    batch_size = 512
    for start in range(0, npx, batch_size):
        stop = min(start + batch_size, npx)
        task = delayed(demmap_pos)(
            dn_flat[start:stop, :],
            edn_flat[start:stop, :],
            tresp,
            tresp_logt,
            dlogt,
            glc,
            reg_tweak,
            max_iter,
            rgt_fact,
            dem_norm0[start:stop, :],
            nmu,
            warn,
            l_emd,
            rscl
        )
        tasks.append(task)


    # compute in parallel
    with ProgressBar():
        results = compute(*tasks, scheduler="threads", num_workers=num_workers)

    # stack results: each r is (dem_row, edem_row, elogt_row, chisq_scalar/array?, dn_reg_row)
    dem_rows = np.vstack([r[0] for r in results])    # (npx, nt)
    edem_rows = np.vstack([r[1] for r in results])   # (npx, nt)
    elogt_rows = np.vstack([r[2] for r in results])  # (npx, nt) or (npx, ?) depending on demmap
    chisq_arr = np.array([r[3] for r in results])    # (npx,)
    dn_reg_rows = np.vstack([r[4] for r in results]) # (npx, nf)

    # reshape back to image if needed
    if input_was_image:
        dem = dem_rows.reshape((width, height, dem_rows.shape[1]))
        edem = edem_rows.reshape((width, height, edem_rows.shape[1]))
        elogt = elogt_rows.reshape((width, height, elogt_rows.shape[1])) if elogt_rows.ndim == 2 else elogt_rows
        dn_reg = dn_reg_rows.reshape((width, height, dn_reg_rows.shape[1]))
        chisq = chisq_arr.reshape((width, height)) if chisq_arr.size == width*height else chisq_arr
    else:
        dem = dem_rows
        edem = edem_rows
        elogt = elogt_rows
        dn_reg = dn_reg_rows
        chisq = chisq_arr

    return dem, edem, elogt, chisq, dn_reg



if __name__ == "__main__":
    # Simple self-test
    na, nf, nt = 8, 6, 50
    dn_in = np.random.random((na, nf)) * 1e3
    edn_in = np.random.random((na, nf)) * 20 + 5
    tresp = np.random.random((nt, nf)) * 1e-23
    tresp_logt = np.linspace(5.5, 7.5, nt)
    temps = np.logspace(5.5, 7.5, nt)

    dem, edem, elogt, chisq, dn_reg = dn2dem_pos(dn_in, edn_in, tresp, tresp_logt, temps)
    print("dn2dem_pos (Dask) sanity: mean chi² =", np.mean(chisq))
