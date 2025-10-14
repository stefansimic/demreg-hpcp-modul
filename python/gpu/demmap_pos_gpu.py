import numpy as np
import cupy as cp

def _dem_reg_map_gpu_l2(s, coef, ed, reg_tweak=1.0, nmu=42):
    """Vectorized discrepancy-principle λ search for L2 (L=I) case on GPU."""
    na, r = coef.shape
    s = s.astype(cp.float64)
    s2 = s**2
    smin = cp.maximum(cp.min(s[s > 0.0]), 1e-12)
    smax = cp.max(s)
    mu_min = (smin**2) * 1e-4
    mu_max = (smax**2)
    mu = cp.exp(cp.linspace(cp.log(mu_min), cp.log(mu_max), int(nmu)))

    coef_b = coef[:, :, None]
    mu_b = mu[None, None, :]
    s2_b = s2[None, :, None]

    frac = (mu_b * coef_b) / (s2_b + mu_b)
    arg_sum = cp.sum(frac**2, axis=1)
    err_term = cp.sum(ed**2, axis=1) * float(reg_tweak)
    discr = arg_sum - err_term[:, None]
    idx = cp.argmin(cp.abs(discr), axis=1)
    lamb = mu[idx]
    return lamb

def demmap_pos_gpu(
    dd,
    ed,
    rmatrix,
    logt,
    dlogt,
    glc=None,
    reg_tweak: float = 1.0,
    max_iter: int = 1,
    rgt_fact: float = 1.5,
    dem_norm0=None,
    nmu: int = 42,
    warn: bool = False,
    l_emd: bool = False,
    rscl: bool = False,
):
    """GPU-native DEM inversion with vectorized regularization."""
    # Host -> device
    dd = np.asarray(dd)
    ed = np.asarray(ed)
    rmatrix = np.asarray(rmatrix)
    logt = np.asarray(logt)
    dlogt = np.asarray(dlogt)

    na, nf = dd.shape
    nt = logt.shape[0]

    dd_g = cp.asarray(dd)
    ed_g = cp.asarray(ed)
    A_g = cp.asarray(rmatrix).T  # A = rmatrix.T (nf x nt)

    # SVD of A: A = U S V^T
    U, s, Vt = cp.linalg.svd(A_g, full_matrices=False)

    # Coefficients per pixel
    coef = dd_g @ U

    # Vectorized λ via discrepancy principle
    lamb = _dem_reg_map_gpu_l2(s, coef, ed_g, reg_tweak=reg_tweak, nmu=nmu)

    # Filters per pixel
    s = s.astype(cp.float64)
    s2 = s**2
    denom = s2[None, :] + lamb[:, None]
    fdiag = s[None, :] / denom

    # Reconstruct DEM
    t_f = coef * fdiag
    dem_g = t_f @ Vt
    dem_g = cp.maximum(dem_g, 0)

    # Predicted dn
    dn_reg_g = dem_g @ cp.asarray(rmatrix)

    # Chi^2
    residuals = (dd_g - dn_reg_g) / ed_g
    chisq_g = cp.sum(residuals ** 2, axis=1) / nf

    # Error estimates (approx)
    edem_g = cp.abs(dem_g) * 0.2
    elogt_g = cp.tile(cp.asarray(dlogt) * 0.5, (na, 1))

    # Device -> host
    dem = cp.asnumpy(dem_g)
    edem = cp.asnumpy(edem_g)
    elogt = cp.asnumpy(elogt_g)
    chisq = cp.asnumpy(chisq_g)
    dn_reg = cp.asnumpy(dn_reg_g)

    return dem, edem, elogt, chisq, dn_reg
