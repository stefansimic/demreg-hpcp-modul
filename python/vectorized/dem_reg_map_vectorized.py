import numpy as np

def dem_reg_map_vectorized(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    sa2 = np.asarray(sigmaa, dtype=np.float64)**2
    sb2 = np.asarray(sigmab, dtype=np.float64)**2
    U = np.asarray(U, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)

    nf = sa2.shape[0]
    if U.shape[0] != nf:
        raise ValueError("U shape incompatible with sigmas")

    with np.errstate(divide="ignore", invalid="ignore"):
        sigs = np.where(sigmab != 0, np.asarray(sigmaa, float)/np.asarray(sigmab, float), np.nan)
    sigs = sigs[np.isfinite(sigs) & (sigs > 0)]
    if sigs.size == 0:
        minx, maxx = 1e-15, 1e15
    else:
        smax = np.max(sigs)
        smin = max(np.min(sigs), np.finfo(float).tiny)
        minx = max(smax*1e-15, np.finfo(float).tiny)
        maxx = max(smax*1e+15, minx*10)

    mu = np.geomspace(minx, maxx, max(2, int(nmu)))

    coef = (U @ data).reshape(nf, 1)
    mu_row = mu.reshape(1, -1)

    num = mu_row * sb2.reshape(-1, 1) * coef
    den = sa2.reshape(-1, 1) + mu_row * sb2.reshape(-1, 1)
    frac2 = (num / den)**2

    discr = np.sum(frac2, axis=0) - np.sum(err**2) * float(reg_tweak)
    opt = mu[np.argmin(np.abs(discr))]
    return float(opt)
