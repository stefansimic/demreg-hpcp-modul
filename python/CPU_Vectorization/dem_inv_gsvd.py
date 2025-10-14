import numpy as np
from numpy.linalg import svd, pinv, solve

def dem_inv_gsvd(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # A @ inv(B) without forming inv(B)
    AB1 = solve(B.T, A.T).T  # (m, n)
    m, n = AB1.shape
    N = max(m, n)

    # Pad to square once to keep downstream shapes stable
    C = np.zeros((N, N), dtype=np.float64)
    C[:m, :n] = AB1

    # Economy SVD on the square pad (still returns NxN here)
    u, s, vh = svd(C, full_matrices=False)

    beta = 1.0 / np.sqrt(1.0 + s * s)
    alpha = s * beta

    # inv(diag(beta)) @ vh == row-scale vh by 1/beta
    vh_scaled = (vh.T / beta).T
    w2 = pinv(vh_scaled @ B)

    return alpha, beta, u.T[:, :m], vh.T, w2
