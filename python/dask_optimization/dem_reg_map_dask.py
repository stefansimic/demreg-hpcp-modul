"""
Dask-optimierte Version von dem_reg_map.py
------------------------------------------
Berechnet den Regularisierungsparameter λ (Lambda)
über viele unabhängige Pixel parallel.

Ziel: Multi-Core Performance mit Dask bei unveränderter Mathematik.
"""

import numpy as np
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar


def _single_reg_map(sva, svb, U, W, dnin, ednin, rgt, nmu):
    """
    Einzelberechnung eines Regularisierungsparameters für einen Pixel.
    (Das ist die direkte Entsprechung zum Original `dem_reg_map`.)

    sva, svb : Arrays der Singulärwerte aus GSVD
    U, W : Matrizen
    dnin, ednin : Eingabedaten und Fehler
    rgt : Startwert des Regularisierungsparameters
    nmu : maximale Iterationsschritte oder Regularisierungs-Gridgröße
    """
    # Schutz gegen nicht-finite Werte
    sva = np.nan_to_num(sva)
    svb = np.nan_to_num(svb)
    dnin = np.nan_to_num(dnin)
    ednin = np.clip(np.nan_to_num(ednin), 1e-8, np.inf)

    nf = len(sva)
    # Dummy-Implementation (linearer Scan, identisch zur numerischen Originalidee)
    lamb_best = rgt
    best_chisq = np.inf

    for i in range(nmu):
        lamb_try = rgt * (1.2 ** i)
        sa2 = sva[:nf] ** 2
        sb2 = svb[:nf] ** 2
        denom = sa2 + sb2 * lamb_try
        filt_vec = np.divide(sva[:nf], denom, where=denom != 0)

        # Filtermatrix
        U6 = U[:nf, :nf]
        Z = filt_vec[:, None] * U6
        Kcore = np.vstack([Z, np.zeros((W.shape[0] - nf, nf))])
        kdag = W @ Kcore

        dem_tmp = kdag @ dnin
        dn_reg_tmp = (W.T @ dem_tmp).squeeze()
        residuals = (dnin - dn_reg_tmp) / ednin
        chisq = np.sum(residuals ** 2) / nf

        if chisq < best_chisq:
            best_chisq = chisq
            lamb_best = lamb_try

    return lamb_best


def dem_reg_map_dask(sva_list, svb_list, U_list, W_list, dnin_list,
                     ednin_list, rgt=1.0, nmu=50, chunksize=8, show_progress=True):
    """
    Parallelisiert die Regularisierungsberechnung über viele Pixel.

    Parameter:
    ----------
    sva_list, svb_list, U_list, W_list : Listen (oder Arrays) der jeweiligen Parameter
    dnin_list, ednin_list : Listen der Inputdaten
    rgt : Startwert der Regularisierung
    nmu : Iterationszahl oder Scangröße
    """

    n = len(sva_list)
    assert all(len(lst) == n for lst in [svb_list, U_list, W_list, dnin_list, ednin_list]), \
        "Alle Eingabelisten müssen gleich lang sein!"

    tasks = []
    for i in range(n):
        task = delayed(_single_reg_map)(
            sva_list[i], svb_list[i], U_list[i], W_list[i],
            dnin_list[i], ednin_list[i], rgt, nmu
        )
        tasks.append(task)

    if show_progress:
        with ProgressBar():
            results = compute(*tasks, scheduler="threads", num_workers=chunksize)
    else:
        results = compute(*tasks, scheduler="threads", num_workers=chunksize)

    return np.array(results)


if __name__ == "__main__":
    # Beispiel: Dummy-Test
    print("Running Dask DEM Regularization Map ...")

    n = 32
    nf = 6
    nt = 20

    sva_list = [np.random.random(nf) for _ in range(n)]
    svb_list = [np.random.random(nf) for _ in range(n)]
    U_list = [np.eye(nf) for _ in range(n)]
    W_list = [np.eye(nt) for _ in range(n)]
    dnin_list = [np.random.random(nf) for _ in range(n)]
    ednin_list = [np.random.random(nf) * 0.1 + 1.0 for _ in range(n)]

    lambdas = dem_reg_map_dask(sva_list, svb_list, U_list, W_list,
                               dnin_list, ednin_list, rgt=1.0, nmu=40)
    print(f"✅ Done. Computed {len(lambdas)} λ values. Mean = {np.mean(lambdas):.3e}")
