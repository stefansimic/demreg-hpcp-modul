import cupy as cp
import numpy as np

from dem_inv_gsvd_gpu import dem_inv_gsvd_gpu
from dem_reg_map_gpu import dem_reg_map_gpu

def _ensure_pos_err(ednin_cp):
    # Replace non-positive / non-finite errors by a safe large value
    bad = (ednin_cp <= 0) | ~cp.isfinite(ednin_cp)
    if bad.any():
        mx = cp.where(~bad, ednin_cp, cp.nan).max()
        mx = 1.0 if cp.isnan(mx) else mx
        ednin_cp = cp.where(bad, 10.0 * mx, ednin_cp)
    return ednin_cp

def _build_L(dlogt_cp, glc, dem_reg_lwght_cp, l_emd):
    if l_emd:
        # L = diag(1/abs(dem_reg_lwght))
        return cp.diag(1.0 / cp.abs(dem_reg_lwght_cp))
    else:
        # L = diag(sqrt(dlogt)/sqrt(abs(dem_reg_lwght)))
        return cp.diag(cp.sqrt(dlogt_cp) / cp.sqrt(cp.abs(dem_reg_lwght_cp)))

def demmap_pos_gpu(dd, ed, rmatrix, logt, dlogt, glc,
                   reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
                   dem_norm0=None, nmu=42, warn=False, l_emd=False, rscl=False,
                   batch_size=4096):
    """
    CuPy / GPU implementation of demmap_pos.
    Keeps math equivalent to CPU vectorized path, but runs linalg on the GPU.
    """
    # Host -> Device
    dd_h = np.asarray(dd)
    ed_h = np.asarray(ed)
    rmatrix_h = np.asarray(rmatrix)
    logt_h = np.asarray(logt)
    dlogt_h = np.asarray(dlogt)

    na, nf = dd_h.shape
    nt = rmatrix_h.shape[0]

    dem_out = np.zeros((na, nt), dtype=np.float64)
    edem_out = np.zeros((na, nt), dtype=np.float64)
    elogt_out = np.zeros((na, nt), dtype=np.float64)
    chisq_out = np.zeros((na,), dtype=np.float64)
    dn_reg_out = np.zeros((na, nf), dtype=np.float64)

    # constants on device
    rmatrix_cp = cp.asarray(rmatrix_h, dtype=cp.float64)
    logt_cp = cp.asarray(logt_h, dtype=cp.float64)
    dlogt_cp = cp.asarray(dlogt_h, dtype=cp.float64)
    if np.ndim(glc) == 1:
        L0_glc = cp.diag(cp.asarray(glc, dtype=cp.float64))
    else:
        L0_glc = cp.asarray(glc, dtype=cp.float64)

    # Precompute a uniform ltt for elogt measure (matches CPU logic)
    ltt_cp = cp.linspace(logt_cp.min(), logt_cp.max(), 52)

    # optional initial dem_norm0 on device
    dem_norm0_cp = None
    if dem_norm0 is not None:
        dem_norm0_cp = cp.asarray(dem_norm0, dtype=cp.float64)

    # batch over pixels
    for start in range(0, na, batch_size):
        end = min(start + batch_size, na)
        dn_batch = cp.asarray(dd_h[start:end, :], dtype=cp.float64)   # (B, nf)
        edn_batch = cp.asarray(ed_h[start:end, :], dtype=cp.float64)  # (B, nf)

        # Per-pixel loop (still fast: each does a handful of SVDs on GPU)
        for b in range(end - start):
            dnin = dn_batch[b]
            ednin = _ensure_pos_err(edn_batch[b])

            rmatrixin = rmatrix_cp / ednin[None, :]   # (nt, nf)
            dn = dnin / ednin
            edn = cp.ones_like(dn)

            # --- Initial weighting ---
            if dem_norm0_cp is None or (dem_norm0_cp.ndim == 0 and float(dem_norm0_cp.get()) == 0.0):
                # First GSVD with L from glc/dlogt path (CPU path parity)
                L = L0_glc if not l_emd else cp.diag(cp.sqrt(dlogt_cp))
                sva, svb, U, V, W = dem_inv_gsvd_gpu(rmatrixin.T, L)

                lamb = dem_reg_map_gpu(sva, svb, U, W, dn, edn, reg_tweak, nmu)

                sa2 = sva[:nf]**2
                sb2 = svb[:nf]**2
                denom = sa2 + sb2 * lamb
                denom_safe = cp.where(denom == 0, 1.0, denom)
                filt_vec = sva[:nf] / denom_safe

                U6 = U[:nf, :nf]
                Z = filt_vec[:, None] * U6  # row scaling
                if nt > nf:
                    Kcore = cp.vstack([Z, cp.zeros((nt - nf, nf), dtype=cp.float64)])
                else:
                    Kcore = Z[:nt, :]
                kdag = W @ Kcore

                dr0 = (kdag @ dn).squeeze()
                fcofmax = 1e-4
                mask = (dr0 > 0) & (dr0 > fcofmax * dr0.max())
                dem_reg_lwght = cp.ones(nt, dtype=cp.float64)
                dem_reg_lwght[mask] = dr0[mask]
            else:
                dem_reg_lwght = dem_norm0_cp[start + b].copy()

            # sanitize weights
            finite_mask = cp.isfinite(dem_reg_lwght)
            if not finite_mask.all():
                rep = dem_reg_lwght[finite_mask].min()
                dem_reg_lwght = cp.where(finite_mask, dem_reg_lwght, rep)
            pos_mask = dem_reg_lwght > 0
            rep_pos = dem_reg_lwght[pos_mask].min()
            dem_reg_lwght = cp.where(pos_mask, dem_reg_lwght, rep_pos)

            # Second GSVD with L from dem_reg_lwght / dlogt
            L = _build_L(dlogt_cp, L0_glc, dem_reg_lwght, l_emd)
            sva, svb, U, V, W = dem_inv_gsvd_gpu(rmatrixin.T, L)

            ndem = 1
            piter = 0
            rgt = reg_tweak
            dem_reg_out = cp.zeros(nt, dtype=cp.float64)
            kdag = None  # keep last for error/elogt

            while (ndem > 0) and (piter < max_iter):
                lamb = dem_reg_map_gpu(sva, svb, U, W, dn, edn, rgt, nmu)

                sa2 = sva[:nf]**2
                sb2 = svb[:nf]**2
                denom = sa2 + sb2 * lamb
                denom_safe = cp.where(denom == 0, 1.0, denom)
                filt_vec = sva[:nf] / denom_safe

                U6 = U[:nf, :nf]
                Z = filt_vec[:, None] * U6
                if nt > nf:
                    Kcore = cp.vstack([Z, cp.zeros((nt - nf, nf), dtype=cp.float64)])
                else:
                    Kcore = Z[:nt, :]
                kdag = W @ Kcore

                dem_reg_out = (kdag @ dn).squeeze()
                ndem = int((dem_reg_out < 0).sum().get())
                rgt = rgt_fact * rgt
                piter += 1

            if warn and (piter == max_iter):
                # Optional: print once per X to avoid flood; omitted for perf
                pass

            dem = dem_reg_out
            dn_reg = (rmatrix_cp.T @ dem).squeeze()
            residuals = (dnin - dn_reg) / ednin
            chisq = cp.sum(residuals**2) / nf

            # edem via diag(kdag @ kdag^T)
            delxi2 = kdag @ kdag.T
            edem = cp.sqrt(cp.clip(cp.diag(delxi2), 0.0, cp.inf))

            # elogt estimate
            kdagk = kdag @ rmatrixin.T  # (nt, nt)
            # match CPU loop over kk, but on GPU
            elogt = cp.zeros(nt, dtype=cp.float64)
            colmax = kdagk.max(axis=0)
            half = colmax / 2.0
            for kk in range(nt):
                rr = cp.interp(ltt_cp, logt_cp, kdagk[:, kk])
                hm_mask = rr >= (half[kk])
                elogt[kk] = dlogt_cp[kk]
                if hm_mask.any():
                    idx = cp.where(hm_mask)[0]
                    elogt[kk] = (ltt_cp[idx[-1]] - ltt_cp[idx[0]]) / 2.0

            if rscl:
                mnrat = cp.mean(dnin / dn_reg)
                dem = dem * mnrat
                edem = edem * mnrat
                dn_reg = (rmatrix_cp.T @ dem).squeeze()
                residuals = (dnin - dn_reg) / ednin
                chisq = cp.sum(residuals**2) / nf

            # Device -> Host placement
            dem_out[start + b, :] = dem.get()
            edem_out[start + b, :] = edem.get()
            elogt_out[start + b, :] = elogt.get()
            chisq_out[start + b] = float(chisq.get())
            dn_reg_out[start + b, :] = dn_reg.get()

        # optional: free VRAM between batches
        cp.get_default_memory_pool().free_all_blocks()

    return dem_out, edem_out, elogt_out, chisq_out, dn_reg_out
