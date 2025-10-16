import cupy as cp

def dem_inv_gsvd_gpu(A, B):
    # Ensure float64 for numerical parity with NumPy path
    A = cp.asarray(A, dtype=cp.float64)
    B = cp.asarray(B, dtype=cp.float64)

    # A @ inv(B) without forming inv(B): solve(B.T, A.T).T
    AB1 = cp.linalg.solve(B.T, A.T).T  # (m, n)
    m, n = AB1.shape
    N = max(m, n)

    # Pad once to square to keep downstream shapes stable
    C = cp.zeros((N, N), dtype=cp.float64)
    C[:m, :n] = AB1

    # Economy SVD on square pad (returns NxN here)
    u, s, vh = cp.linalg.svd(C, full_matrices=False)

    beta = 1.0 / cp.sqrt(1.0 + s * s)
    alpha = s * beta

    # inv(diag(beta)) @ vh == row-scale vh by 1/beta
    vh_scaled = (vh.T / beta).T
    w2 = cp.linalg.pinv(vh_scaled @ B)

    return alpha, beta, u.T[:, :m], vh.T, w2
