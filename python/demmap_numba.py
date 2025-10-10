
import numpy as np
from numpy import diag
from dem_inv_gsvd import dem_inv_gsvd
from dem_reg_map import dem_reg_map
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from numba import njit, prange

# -----------------------------
# Numba-accelerated primitives
# -----------------------------

@njit(cache=True, fastmath=True)
def _compute_rmatrixin(rmatrix, ednin):
    nt, nf = rmatrix.shape[0], rmatrix.shape[1]
    rmatrixin = np.empty((nt, nf))
    for k in range(nf):
        # guard division-by-zero (should not happen if ednin>0)
        denom = ednin[k] if ednin[k] != 0.0 else 1.0
        for t in range(nt):
            rmatrixin[t, k] = rmatrix[t, k] / denom
    return rmatrixin

@njit(cache=True, fastmath=True)
def _compute_dn_scaled(dnin, ednin):
    nf = dnin.shape[0]
    dn = np.empty(nf)
    edn = np.empty(nf)
    for k in range(nf):
        denom = ednin[k] if ednin[k] != 0.0 else 1.0
        dn[k] = dnin[k] / denom
        edn[k] = ednin[k] / denom
    return dn, edn

@njit(cache=True, fastmath=True)
def _compute_emloci(dnin, rmatrix, gdglc):
    nt = rmatrix.shape[0]
    m = gdglc.shape[0]
    emloci = np.empty((nt, m))
    for j in range(m):
        ch = gdglc[j]
        # dnin[ch] / rmatrix[:, ch]
        for t in range(nt):
            emloci[t, j] = dnin[ch] / rmatrix[t, ch]
    return emloci

@njit(cache=True, fastmath=True)
def _rowwise_min_ignore_zeros(A):
    """Row-wise min that ignores zeros; if a row is all zero -> 0."""
    n, m = A.shape
    out = np.zeros(n)
    for i in range(n):
        first = True
        mn = 0.0
        for j in range(m):
            v = A[i, j]
            if v != 0.0:
                if first:
                    mn = v
                    first = False
                elif v < mn:
                    mn = v
        out[i] = 0.0 if first else mn
    return out

@njit(cache=True, fastmath=True)
def _moving_avg_valid(x, window):
    # valid (= no padding): length = len(x) - window + 1
    n = x.shape[0]
    w = window
    out_len = n - w + 1
    out = np.empty(out_len)
    invw = 1.0 / w
    run = 0.0
    # initial window
    for i in range(w):
        run += x[i]
    out[0] = run * invw
    # slide
    for i in range(w, n):
        run += x[i] - x[i-w]
        out[i - w + 1] = run * invw
    return out

@njit(cache=True, fastmath=True)
def _clamp_min_inplace(x, minval):
    for i in range(x.shape[0]):
        if x[i] < minval:
            x[i] = minval


# -------------------------------------------
# Pixel solver with Numba-accelerated helpers
# -------------------------------------------

def dem_pix_numba(dnin, ednin, rmatrix, logt, dlogt, glc,
                  reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
                  dem_norm0=None, nmu=42, warn=True, l_emd=False, rscl=False):
    """
    Drop-in replacement for dem_pix() that accelerates the pure-NumPy inner
    loops with Numba where possible. The GSVD and regularization mapping are
    left as-is (Python/NumPy), to preserve numerical behavior.
    """
    nf = rmatrix.shape[1]
    nt = logt.shape[0]

    if dem_norm0 is None:
        dem_norm0 = np.ones(nt, dtype=np.float64)

    dem = np.zeros(nt, dtype=np.float64)
    edem = np.zeros(nt, dtype=np.float64)
    elogt = np.zeros(nt, dtype=np.float64)
    chisq = 0.0
    dn_reg = np.zeros(nf, dtype=np.float64)

    # Scaled inputs
    rmatrixin = _compute_rmatrixin(rmatrix, ednin)
    dn, edn = _compute_dn_scaled(dnin, ednin)

    # checks
    if (np.isnan(dn).sum() == 0) and (np.isinf(dn).sum() == 0) and (np.all(dn > 0)):
        ndem = 1
        piter = 0
        rgt = reg_tweak

        # Build initial weighting dem_reg_lwght
        if (np.prod(dem_norm0) == 1.0) or (dem_norm0[0] == 0):
            # no user-supplied weighting
            if np.sum(glc) > 0:
                # EM loci approach
                gdglc = np.nonzero(glc > 0)[0].astype(np.int64)
                emloci = _compute_emloci(dnin, rmatrix, gdglc)
                dem_model = _rowwise_min_ignore_zeros(emloci)
                dem_reg_lwght = dem_model.copy()
            else:
                # self-norm approach: one reg pass with L = diag(1/sqrt(dlogt))
                L = np.diag(1.0/np.sqrt(dlogt[:]))
                sva, svb, U, V, W = dem_inv_gsvd(rmatrixin.T, L)

                # map reg once
                lamb = dem_reg_map(sva, svb, U, W, dn, edn, rgt, nmu)

                nf = rmatrix.shape[1]

                if U.shape[0] == nf:
                    # Klassischer Fall: U ist (nf x nf), W ggf. (nt x nt)
                    U_nf = U[:nf, :nf]
                    W_nf = W[:, :nf] if W.shape[1] >= nf else W
                    filt = np.zeros((nf, nf))
                    diags = min(len(sva), len(svb), nf)
                    for kk in range(diags):
                        filt[kk, kk] = sva[kk] / (sva[kk]**2 + svb[kk]**2 * lamb)

                    kdag = W_nf @ (filt.T @ U_nf)

                else:
                    # U ist NICHT (nf x nf), typischerweise (nt x nt)
                    # -> benutze die ersten nf Spalten von U
                    U_nf = U[:, :nf]                      # (nt x nf)
                    # Filter als (nt x nt) mit nur den ersten diags Diagonaleinträgen
                    filt = np.zeros((U.shape[0], U.shape[0]))
                    diags = min(len(sva), len(svb), nf, U.shape[0])
                    for kk in range(diags):
                        filt[kk, kk] = sva[kk] / (sva[kk]**2 + svb[kk]**2 * lamb)

                    # (filt.T @ U_nf): (nt x nt) @ (nt x nf) -> (nt x nf)
                    # W: (nt x nt), also Ergebnis (nt x nf)
                    kdag = W @ (filt.T @ U_nf)

                dr0 = (kdag @ dn).squeeze()

                fcofmax = 1e-4
                dem_reg_lwght = np.ones(nt, dtype=np.float64)
                # mask: positive and above threshold
                mask = (dr0 > 0) & (dr0 > fcofmax * np.max(dr0))
                dem_reg_lwght[mask] = dr0[mask]

            # smooth + normalise + clamp small
            if dem_reg_lwght.shape[0] >= 7:
                sm = _moving_avg_valid(dem_reg_lwght[1:-1], 5)
                # align back to nt length (simple pad at ends)
                out = np.ones(nt, dtype=np.float64)
                out[1:1+sm.shape[0]] = sm
                dem_reg_lwght = out / max(out.max(), 1e-12)
            else:
                m = dem_reg_lwght.max()
                if m > 0:
                    dem_reg_lwght = dem_reg_lwght / m
            _clamp_min_inplace(dem_reg_lwght, 1e-8)
        else:
            dem_reg_lwght = dem_norm0.astype(np.float64, copy=True)

        # Main regularisation with weighting
        if l_emd:
            L = np.diag(1.0 / np.abs(dem_reg_lwght))
        else:
            L = np.diag(np.sqrt(dlogt) / np.sqrt(np.abs(dem_reg_lwght)))

        sva, svb, U, V, W = dem_inv_gsvd(rmatrixin.T, L)

        # positivity loop
        nf_loc = U.shape[0]
        while (ndem > 0) and (piter < max_iter):
            lamb = dem_reg_map(sva, svb, U, W, dn, edn, rgt, nmu)

            nf = rmatrix.shape[1]

            if U.shape[0] == nf:
                U_nf = U[:nf, :nf]
                W_nf = W[:, :nf] if W.shape[1] >= nf else W
                filt = np.zeros((nf, nf))
                diags = min(len(sva), len(svb), nf)
                for kk in range(diags):
                    filt[kk, kk] = sva[kk] / (sva[kk]**2 + svb[kk]**2 * lamb)

                kdag = W_nf @ (filt.T @ U_nf)

            else:
                U_nf = U[:, :nf]  # (nt x nf)
                filt = np.zeros((U.shape[0], U.shape[0]))
                diags = min(len(sva), len(svb), nf, U.shape[0])
                for kk in range(diags):
                    filt[kk, kk] = sva[kk] / (sva[kk]**2 + svb[kk]**2 * lamb)

                kdag = W @ (filt.T @ U_nf)

            dem_reg_out = (kdag @ dn).squeeze()
            ndem = int(np.sum(dem_reg_out < 0.0))
            rgt = rgt_fact * rgt
            piter += 1

        if (warn and (piter == max_iter)):
            print("Warning, positivity loop hit max iterations, so increase max_iter? Or rgt_fact too small?")

        dem = dem_reg_out

        # synthetic dn and chi^2
        dn_reg = (rmatrix.T @ dem).squeeze()
        residuals = (dnin - dn_reg) / ednin
        chisq = np.sum(residuals**2) / nf

        # errors
        delxi2 = kdag @ kdag.T
        edem = np.sqrt(np.diag(delxi2))

        kdagk = kdag @ rmatrixin.T

        # horizontal errors (elogt) – keep NumPy logic
        ltt = np.min(logt) + 1e-8 + (np.max(logt) - np.min(logt)) * np.arange(51) / (52.0 - 1.0)
        elogt = np.zeros(nt, dtype=np.float64)
        for kk in range(nt):
            rr = np.interp(ltt, logt, kdagk[:, kk])
            hm_mask = (rr >= np.max(kdagk[:, kk]) / 2.0)
            elogt[kk] = dlogt[kk]
            if np.sum(hm_mask) > 0:
                elogt[kk] = (ltt[hm_mask][-1] - ltt[hm_mask][0]) / 2.0

        if rscl:
            mnrat = np.mean(dnin / dn_reg)
            dem = dem * mnrat
            edem = edem * mnrat
            dn_reg = (rmatrix.T @ dem).squeeze()
            chisq = np.sum(((dnin - dn_reg) / ednin) ** 2) / nf

    return dem, edem, elogt, chisq, dn_reg


def dem_unwrap_numba(dn, ed, rmatrix, logt, dlogt, glc,
                     reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
                     dem_norm0=None, nmu=42, warn=False, l_emd=False, rscl=False):
    """Serial executor (same semantics as dem_unwrap) that calls dem_pix_numba."""
    ndem = dn.shape[0]
    nt = logt.shape[0]
    nf = dn.shape[1]
    dem = np.zeros((ndem, nt))
    edem = np.zeros((ndem, nt))
    elogt = np.zeros((ndem, nt))
    chisq = np.zeros(ndem)
    dn_reg = np.zeros((ndem, nf))

    if dem_norm0 is None:
        dem_norm0 = np.ones((ndem, nt), dtype=np.float64)

    for i in range(ndem):
        result = dem_pix_numba(dn[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
                               reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
                               dem_norm0=dem_norm0[i, :], nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl)
        dem[i, :] = result[0]
        edem[i, :] = result[1]
        elogt[i, :] = result[2]
        chisq[i] = result[3]
        dn_reg[i, :] = result[4]
    return dem, edem, elogt, chisq, dn_reg


def demmap_pos_numba(dd, ed, rmatrix, logt, dlogt, glc,
                     reg_tweak=1.0, max_iter=10, rgt_fact=1.5, dem_norm0=None,
                     nmu=42, warn=False, l_emd=False, rscl=False, parallel_threshold=200, block_size=100):
    """
    Numba-accelerated variant of demmap_pos().
    API is identical, plus two optional knobs:
      - parallel_threshold: minimum number of pixels (na) to trigger process-based parallelism
      - block_size: number of pixels per parallel block
    """
    na = dd.shape[0]
    nf = rmatrix.shape[1]
    nt = logt.shape[0]

    dem = np.zeros((na, nt))
    edem = np.zeros((na, nt))
    elogt = np.zeros((na, nt))
    chisq = np.zeros(na)
    dn_reg = np.zeros((na, nf))

    if dem_norm0 is None:
        dem_norm0 = np.ones((na, nt), dtype=np.float64)

    if na >= parallel_threshold:
        n_par = block_size
        niter = int(np.floor(na / n_par))
        with threadpool_limits(limits=1):
            with ProcessPoolExecutor() as exe:
                futures = [
                    exe.submit(
                        dem_unwrap_numba,
                        dd[i*n_par:(i+1)*n_par, :],
                        ed[i*n_par:(i+1)*n_par, :],
                        rmatrix, logt, dlogt, glc,
                        reg_tweak, max_iter, rgt_fact,
                        dem_norm0[i*n_par:(i+1)*n_par, :],
                        nmu, warn, l_emd, rscl
                    )
                    for i in np.arange(niter)
                ]
                kwargs = {
                    'total': len(futures),
                    'unit': ' x10^2 DEM',
                    'unit_scale': True,
                    'leave': True
                }
                for _ in tqdm(as_completed(futures), **kwargs):
                    pass

        for i, f in enumerate(futures):
            res = f.result()
            dem[i*n_par:(i+1)*n_par, :] = res[0]
            edem[i*n_par:(i+1)*n_par, :] = res[1]
            elogt[i*n_par:(i+1)*n_par, :] = res[2]
            chisq[i*n_par:(i+1)*n_par] = res[3]
            dn_reg[i*n_par:(i+1)*n_par, :] = res[4]

        # remainder serial
        if (na % (niter * n_par)) != 0:
            i_start = niter * n_par
            for i in range(na - i_start):
                result = dem_pix_numba(
                    dd[i_start+i, :], ed[i_start+i, :], rmatrix, logt, dlogt, glc,
                    reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
                    dem_norm0=dem_norm0[i_start+i, :], nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl
                )
                dem[i_start+i, :] = result[0]
                edem[i_start+i, :] = result[1]
                elogt[i_start+i, :] = result[2]
                chisq[i_start+i] = result[3]
                dn_reg[i_start+i, :] = result[4]
    else:
        # serial path
        for i in range(na):
            result = dem_pix_numba(
                dd[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
                reg_tweak=reg_tweak, max_iter=max_iter, rgt_fact=rgt_fact,
                dem_norm0=dem_norm0[i, :], nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl
            )
            dem[i, :] = result[0]
            edem[i, :] = result[1]
            elogt[i, :] = result[2]
            chisq[i] = result[3]
            dn_reg[i, :] = result[4]

    return dem, edem, elogt, chisq, dn_reg
