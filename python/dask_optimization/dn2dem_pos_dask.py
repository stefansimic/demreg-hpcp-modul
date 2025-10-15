"""
Dask-optimierte Version von dn2dem_pos.py
-----------------------------------------
Ziel:
  Parallelisierte DEM-Inversion (DN → DEM)
  basierend auf Dask und den dask_opt-Modulen:
    - demmap_pos_dask
    - dem_inv_gsvd_dask
    - dem_reg_map_dask
"""

import numpy as np
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar

from dask_optimization.demmap_pos_dask import demmap_pos_dask
from dask_optimization.dem_inv_gsvd_dask import dem_inv_gsvd_dask
from dask_optimization.dem_reg_map_dask import dem_reg_map_dask


def _dn2dem_single(dd, ed, rmatrix, logt, dlogt, glc,
                   reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
                   dem_norm0_row=None, nmu=42, warn=False, l_emd=False, rscl=False):
    """
    Einzelne DEM-Inversion für einen Pixel oder Datensatz.
    (Das ist im Prinzip eine Wrapper-Funktion um die drei Kernroutinen.)
    """
    dem, edem, elogt, chisq, dn_reg = demmap_pos_dask(
        dd[None, :], ed[None, :], rmatrix, logt, dlogt, glc,
        reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
        dem_norm0=dem_norm0_row[None, :] if dem_norm0_row is not None else None,
        nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl
    )

    return dem[0], edem[0], elogt[0], chisq[0], dn_reg[0]


def dn2dem_pos_dask(dd, ed, rmatrix, logt, dlogt, glc,
                    reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
                    dem_norm0=None, nmu=42, warn=False,
                    l_emd=False, rscl=False, chunksize=32):
    """
    Hauptfunktion: Dask-Parallelisierung über Pixel.
    """

    dd = np.asarray(dd)
    ed = np.asarray(ed)
    na, nf = dd.shape
    nt = rmatrix.shape[0]

    if dem_norm0 is None:
        dem_norm0 = np.ones((na, nt))

    # Task-Erstellung
    tasks = []
    for i in range(na):
        task = delayed(_dn2dem_single)(
            dd[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
            reg_tweak, max_iter, rgt_fact,
            dem_norm0[i, :], nmu, warn, l_emd, rscl
        )
        tasks.append(task)

    # Ausführung
    with ProgressBar():
        results = compute(*tasks, scheduler="threads", num_workers=8)

    # Ergebnisse zusammenfassen
    dem = np.stack([r[0] for r in results])
    edem = np.stack([r[1] for r in results])
    elogt = np.stack([r[2] for r in results])
    chisq = np.array([r[3] for r in results])
    dn_reg = np.stack([r[4] for r in results])

    return dem, edem, elogt, chisq, dn_reg


if __name__ == "__main__":
    # Demo-Testlauf
    print("Running Dask version of dn2dem_pos ...")

    na, nf, nt = 64, 6, 200
    dd = np.random.random((na, nf)) * 1e3
    ed = np.random.random((na, nf)) * 20 + 5
    rmatrix = np.random.random((nt, nf)) * 1e-23
    logt = np.linspace(5.5, 7.5, nt)
    dlogt = np.gradient(logt)
    glc = np.ones(nf)

    dem, edem, elogt, chisq, dn_reg = dn2dem_pos_dask(dd, ed, rmatrix, logt, dlogt, glc)
    print(f"✅ Done. Mean χ² = {np.mean(chisq):.3f}")
