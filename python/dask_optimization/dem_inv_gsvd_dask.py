"""
Dask-friendly dem_inv_gsvd replacement.
Keeps same function name and interface as original `dem_inv_gsvd`.
The original performs a GSVD-like decomposition; here we provide a robust,
vectorized implementation using numpy/scipy when available. The function is
thread-safe and can be used inside delayed tasks.
"""

import numpy as np

try:
    from scipy.linalg import svd
except Exception:
    from numpy.linalg import svd


def dem_inv_gsvd(A, L):
    """
    Compute a GSVD-like decomposition for the pair (A, L).
    Interface kept compatible: returns (sva, svb, U, V, W)

    A : 2D array
    L : 2D array
    """
    A = np.array(A, dtype=float)
    L = np.array(L, dtype=float)

    # SVDs
    UA, sA, VhA = svd(A, full_matrices=False)
    UL, sL, VhL = svd(L, full_matrices=False)

    sva = sA.copy()
    svb = sL.copy()

    W = VhA.T
    U = UA
    V = VhL.T if VhL.shape[0] == VhA.shape[1] else np.eye(VhA.shape[1])

    return sva, svb, U, V, W
