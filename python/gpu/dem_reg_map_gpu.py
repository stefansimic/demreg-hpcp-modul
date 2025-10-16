import cupy as cp

def dem_reg_map_gpu(sigmaa, sigmab, U, W, data, err, reg_tweak, nmu=500):
    sa2 = cp.asarray(sigmaa, dtype=cp.float64) ** 2
    sb2 = cp.asarray(sigmab, dtype=cp.float64) ** 2
    U = cp.asarray(U, dtype=cp.float64)
    data = cp.asarray(data, dtype=cp.float64)
    err = cp.asarray(err, dtype=cp.float64)

    nf = sa2.shape[0]
    if U.shape[0] != nf:
        raise ValueError("U shape incompatible with sigmas")

    sigs = cp.where(sigmab != 0, cp.asarray(sigmaa, float) / cp.asarray(sigmab, float), cp.nan)
    sigs = sigs[cp.isfinite(sigs) & (sigs > 0)]

    if sigs.size == 0:
        minx, maxx = 1e-15, 1e15
    else:
        smax = sigs.max()
        smin = cp.maximum(sigs.min(), cp.finfo(float).tiny)
        minx = cp.maximum(smax * 1e-15, cp.finfo(float).tiny)
        maxx = cp.maximum(smax * 1e+15, minx * 10)

    mu = cp.logspace(cp.log10(minx), cp.log10(maxx), max(2, int(nmu)))

    coef = (U @ data).reshape(nf, 1)
    mu_row = mu.reshape(1, -1)

    num = mu_row * sb2.reshape(-1, 1) * coef
    den = sa2.reshape(-1, 1) + mu_row * sb2.reshape(-1, 1)

    den_safe = cp.where(den == 0, 1.0, den)
    frac2 = (num / den_safe) ** 2

    discr = cp.sum(frac2, axis=0) - cp.sum(err ** 2) * float(reg_tweak)
    opt_idx = cp.argmin(cp.abs(discr))
    opt = mu[opt_idx]

    return float(opt.item())
