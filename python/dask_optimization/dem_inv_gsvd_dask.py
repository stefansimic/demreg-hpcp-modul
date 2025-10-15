"""
Dask-basierte Optimierung der Generalized Singular Value Decomposition (GSVD)
aus dem ursprünglichen dem_inv_gsvd Modul.
"""

import numpy as np
import dask.array as da
from dask import delayed, compute
from dask.diagnostics import ProgressBar
from numpy.linalg import svd

def _single_gsvd(A, L):
    """
    Führt die Generalized SVD für ein einzelnes (A, L)-Paar aus.
    Nutzt Standard-SVD, aber kann durch SciPy.linalg.gsvd ersetzt werden.
    """
    # Sicherheit: finite Werte sicherstellen
    A = np.nan_to_num(A, nan=0.0)
    L = np.nan_to_num(L, nan=0.0)

    # Standard-SVD (vereinfachte GSVD-Form)
    UA, sA, VAh = svd(A, full_matrices=False)
    UL, sL, VLh = svd(L, full_matrices=False)

    # Generalized "Mix"
    C = np.diag(sA / (sA + sL + 1e-12))
    S = np.diag(sL / (sA + sL + 1e-12))
    W = VAh.T  # Approximation für gemeinsame Basis
    return sA, sL, UA, VLh.T, W


def dem_inv_gsvd_dask(A_list, L_list, chunksize=8, use_progress=True):
    """
    Parallelisierte GSVD-Auswertung über mehrere (A,L)-Paare.
    
    Parameter:
    ----------
    A_list : list[np.ndarray]
        Liste von A-Matrizen (z. B. Response-Matrizen pro Pixel).
    L_list : list[np.ndarray]
        Liste von L-Matrizen (Regularisierung pro Pixel).
    chunksize : int
        Anzahl paralleler Tasks.
    """

    assert len(A_list) == len(L_list), "A_list und L_list müssen gleich lang sein!"

    tasks = []
    for A, L in zip(A_list, L_list):
        task = delayed(_single_gsvd)(A, L)
        tasks.append(task)

    if use_progress:
        with ProgressBar():
            results = compute(*tasks, scheduler="threads", num_workers=chunksize)
    else:
        results = compute(*tasks, scheduler="threads", num_workers=chunksize)

    # Ergebnisse "entpacken"
    sva = [r[0] for r in results]
    svb = [r[1] for r in results]
    U = [r[2] for r in results]
    V = [r[3] for r in results]
    W = [r[4] for r in results]

    return sva, svb, U, V, W


if __name__ == "__main__":
    # Beispiel: Dummy-Test
    n = 6
    A_list = [np.random.random((n, n)) for _ in range(16)]
    L_list = [np.eye(n) * np.random.rand() for _ in range(16)]

    print("Running Dask GSVD batch ...")
    sva, svb, U, V, W = dem_inv_gsvd_dask(A_list, L_list)
    print(f"✅ Done. {len(sva)} decompositions computed.")
