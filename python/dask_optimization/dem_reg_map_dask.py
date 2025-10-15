"""
Dask-friendly dem_reg_map replacement.
Same function name and compatible interface as the original dem_reg_map.
This implementation is written to be robust and vectorizable for per-pixel calls.
"""

import numpy as np

def dem_reg_map(sva, svb, U, W, dnin, ednin, rgt=1.0, nmu=42):
    """
    Compute regularization parameter lambda for one pixel.
    Interface identical to original: returns a scalar lambda.
    """
    sva = np.array(sva, dtype=float)
    svb = np.array(svb, dtype=float)
    dnin = np.array(dnin, dtype=float)
    ednin = np.array(ednin, dtype=float)
    ednin = np.clip(ednin, 1e-8, np.inf)

    nf = min(len(sva), W.shape[1]) if hasattr(W, "shape") else len(sva)

    best_lamb = rgt
    best_chisq = np.inf

    for i in range(nmu):
        lamb_try = rgt * (1.2 ** i)
        sa2 = sva[:nf] ** 2
        sb2 = svb[:nf] ** 2
        denom = sa2 + sb2 * lamb_try
        filt_vec = np.divide(sva[:nf], denom, where=denom != 0)

        U6 = U[:nf, :nf]
        Z = filt_vec[:, None] * U6
        if W.shape[0] > nf:
            Kcore = np.vstack([Z, np.zeros((W.shape[0] - nf, nf))])
        else:
            Kcore = Z[:W.shape[0], :]

        kdag = W @ Kcore
        dem_tmp = kdag @ dnin
        dn_reg_tmp = (W.T @ dem_tmp).squeeze()
        residuals = (dnin - dn_reg_tmp) / ednin
        chisq = np.sum(residuals ** 2) / max(1, len(dnin))

        if chisq < best_chisq:
            best_chisq = chisq
            best_lamb = lamb_try

    return best_lamb
