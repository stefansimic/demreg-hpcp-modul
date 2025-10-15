import numpy as np
import dask
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from numpy import diag

from dem_inv_gsvd_dask import dem_inv_gsvd_dask
from dem_reg_map_dask import dem_reg_map_dask

def _dem_single_pixel(dnin, ednin, rmatrix, logt, dlogt, glc,
                      reg_tweak, max_iter, rgt_fact,
                      dem_norm0_row, nmu, warn, l_emd, rscl):
    """
    Hilfsfunktion: führt die DEM-Berechnung für einen einzelnen Pixel aus.
    Wird später per Dask parallelisiert.
    """
    nf = rmatrix.shape[1]
    nt = rmatrix.shape[0]

    dem_reg_lwght = np.array(dem_norm0_row, dtype=float)
    dem_reg_lwght[~np.isfinite(dem_reg_lwght)] = np.nanmin(dem_reg_lwght[np.isfinite(dem_reg_lwght)])
    dem_reg_lwght[dem_reg_lwght <= 0] = np.min(dem_reg_lwght[dem_reg_lwght > 0])

    if l_emd:
        L = np.diag(1 / abs(dem_reg_lwght))
    else:
        L = np.diag(np.sqrt(dlogt) / np.sqrt(abs(dem_reg_lwght)))

    sva, svb, U, V, W = dem_inv_gsvd_dask(rmatrix.T, L)

    ndem = 1
    piter = 0
    rgt = reg_tweak
    dem_reg_out = np.zeros(nt)

    while (ndem > 0) and (piter < max_iter):
        lamb = dem_reg_map_dask(sva, svb, U, W, dnin, ednin, rgt, nmu)
        sa2 = sva[:nf] ** 2
        sb2 = svb[:nf] ** 2
        denom = sa2 + sb2 * lamb
        filt_vec = np.divide(sva[:nf], denom, where=denom != 0)

        U6 = U[:nf, :nf]
        Z = filt_vec[:, None] * U6
        if nt > nf:
            Kcore = np.vstack([Z, np.zeros((nt - nf, nf))])
        else:
            Kcore = Z[:nt, :]
        kdag = W @ Kcore

        dem_reg_out = (kdag @ dnin).squeeze()
        ndem = int(np.sum(dem_reg_out < 0))
        rgt = rgt_fact * rgt
        piter += 1

    if (warn and (piter == max_iter)):
        print("Warning - max iterations reached in positivity regularisation")

    dn_reg = (rmatrix.T @ dem_reg_out).squeeze()
    residuals = (dnin - dn_reg) / ednin
    chisq = np.sum(residuals ** 2) / nf

    delxi2 = kdag @ kdag.T
    edem = np.sqrt(np.clip(np.diag(delxi2), 0.0, np.inf))

    # Temperatur-Auflösungsabschätzung
    elogt = np.zeros(nt)

    if rscl:
        mnrat = np.mean(dnin / dn_reg)
        dem_reg_out *= mnrat
        edem *= mnrat
        dn_reg = (rmatrix.T @ dem_reg_out).squeeze()
        chisq = np.sum(((dnin - dn_reg) / ednin) ** 2) / nf

    return dem_reg_out, edem, elogt, chisq, dn_reg


def demmap_pos_dask(dd, ed, rmatrix, logt, dlogt, glc,
                    reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
                    dem_norm0=None, nmu=42, warn=False,
                    l_emd=False, rscl=False, chunksize=100):
    """
    Dask-basierte Version von demmap_pos.
    Führt die Berechnungen für mehrere Pixel parallel aus.
    """

    dd = np.asarray(dd)
    ed = np.asarray(ed)
    rmatrix = np.asarray(rmatrix)
    logt = np.asarray(logt)
    dlogt = np.asarray(dlogt)

    na = dd.shape[0]
    nf = dd.shape[1]
    nt = rmatrix.shape[0]

    if dem_norm0 is None:
        dem_norm0 = np.ones((na, nt))

    delayed_results = []
    for i in range(na):
        dnin = dd[i, :]
        ednin = ed[i, :]
        norm0_row = dem_norm0[i, :]

        task = delayed(_dem_single_pixel)(
            dnin, ednin, rmatrix, logt, dlogt, glc,
            reg_tweak, max_iter, rgt_fact,
            norm0_row, nmu, warn, l_emd, rscl
        )
        delayed_results.append(task)

    with ProgressBar():
        computed = compute(*delayed_results, scheduler="threads", num_workers=8)

    # Ergebnisse zusammenführen
    dem = np.stack([r[0] for r in computed])
    edem = np.stack([r[1] for r in computed])
    elogt = np.stack([r[2] for r in computed])
    chisq = np.array([r[3] for r in computed])
    dn_reg = np.stack([r[4] for r in computed])

    return dem, edem, elogt, chisq, dn_reg


if __name__ == "__main__":
    # Beispieltestlauf
    na, nf, nt = 100, 6, 200
    dd = np.random.random((na, nf)) * 1e3
    ed = np.random.random((na, nf)) * 20 + 5
    rmatrix = np.random.random((nt, nf)) * 1e-23
    logt = np.linspace(5.5, 7.5, nt)
    dlogt = np.gradient(logt)
    glc = np.ones(nf)

    print("Running Dask version of DEMMAP_POS...")
    dem, edem, elogt, chisq, dn_reg = demmap_pos_dask(dd, ed, rmatrix, logt, dlogt, glc)
    print(f"✅ Done. Mean χ² = {np.mean(chisq):.3f}")
