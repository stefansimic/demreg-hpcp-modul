import numpy as np
from numpy import diag
from dem_inv_gsvd import dem_inv_gsvd
from dem_reg_map import dem_reg_map
from tqdm import tqdm
from threadpoolctl import threadpool_limits
from dask import delayed, compute

def demmap_pos(dd, ed, rmatrix, logt, dlogt, glc,
               reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
               dem_norm0=None, nmu=42, warn=False,
               l_emd=False, rscl=False):
    """
    demmap_pos
    computes the DEMs for a 1D array of length na with nf filters using
    the dn (g) counts and the temperature response matrix (K) for each filter.

    The regularization is solved via GSVD (dem_inv_gsvd) for each pixel,
    with adaptive regularization parameter selection (dem_reg_map).

    If enough DEMs (na >= 200) are provided, computation is parallelized
    using Dask for efficient shared-memory parallelism.
    """
    na = dd.shape[0]
    nf = rmatrix.shape[1]
    nt = logt.shape[0]

    # preallocate outputs
    dem = np.zeros([na, nt])
    edem = np.zeros([na, nt])
    elogt = np.zeros([na, nt])
    chisq = np.zeros([na])
    dn_reg = np.zeros([na, nf])

    # use Dask parallelization if enough DEMs
    if na >= 200:
        # to keep Dask task overhead reasonable
        chunk_size = 128

        tasks = []

        with threadpool_limits(limits=1):
            rmatrix_d = delayed(rmatrix)
            logt_d = delayed(logt)
            dlogt_d = delayed(dlogt)
            glc_d = delayed(glc)

            for i in range(0, na, chunk_size):
                end = min(i + chunk_size, na)
                tasks.append(
                    delayed(dem_unwrap)(
                        dd[i:end, :], ed[i:end, :],
                        rmatrix_d, logt_d, dlogt_d, glc_d,
                        reg_tweak=reg_tweak, max_iter=max_iter,
                        rgt_fact=rgt_fact,
                        dem_norm0=dem_norm0[i:end, :],
                        nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl
                    )
                )


            # compute all delayed tasks in parallel
            results = compute(*tasks, scheduler='processes')

        # collect results
        offset = 0
        for block in results:
            ndem = block[0].shape[0]
            dem[offset:offset + ndem, :] = block[0]
            edem[offset:offset + ndem, :] = block[1]
            elogt[offset:offset + ndem, :] = block[2]
            chisq[offset:offset + ndem] = block[3]
            dn_reg[offset:offset + ndem, :] = block[4]
            offset += ndem

    else:
        # serial fallback for small input
        for i in range(na):
            result = dem_pix(dd[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
                             reg_tweak=reg_tweak, max_iter=max_iter,
                             rgt_fact=rgt_fact, dem_norm0=dem_norm0[i, :],
                             nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl)
            dem[i, :] = result[0]
            edem[i, :] = result[1]
            elogt[i, :] = result[2]
            chisq[i] = result[3]
            dn_reg[i, :] = result[4]

    return dem, edem, elogt, chisq, dn_reg


def dem_unwrap(dn, ed, rmatrix, logt, dlogt, glc,
               reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
               dem_norm0=0, nmu=42, warn=False, l_emd=False, rscl=False):
    """Wrapper to compute a block of DEMs serially."""
    ndem = dn.shape[0]
    nt = logt.shape[0]
    nf = dn.shape[1]
    dem = np.zeros([ndem, nt])
    edem = np.zeros([ndem, nt])
    elogt = np.zeros([ndem, nt])
    chisq = np.zeros([ndem])
    dn_reg = np.zeros([ndem, nf])
    for i in range(ndem):
        result = dem_pix(dn[i, :], ed[i, :], rmatrix, logt, dlogt, glc,
                         reg_tweak=reg_tweak, max_iter=max_iter,
                         rgt_fact=rgt_fact, dem_norm0=dem_norm0[i, :],
                         nmu=nmu, warn=warn, l_emd=l_emd, rscl=rscl)
        dem[i, :] = result[0]
        edem[i, :] = result[1]
        elogt[i, :] = result[2]
        chisq[i] = result[3]
        dn_reg[i, :] = result[4]
    return dem, edem, elogt, chisq, dn_reg


def dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc,
            reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
            dem_norm0=0, nmu=42, warn=True, l_emd=False, rscl=False):
    """
    Core per-pixel DEM regularization.
    (Unverändert zur Originalversion außer minimalem Style cleanup.)
    """
    nf = rmatrix.shape[1]
    nt = logt.shape[0]
    ltt = min(logt) + 1e-8 + (max(logt) - min(logt)) * np.arange(51) / (52 - 1.0)
    dem = np.zeros(nt)
    edem = np.zeros(nt)
    elogt = np.zeros(nt)
    chisq = 0
    dn_reg = np.zeros(nf)

    rmatrixin = np.zeros([nt, nf])
    filt = np.zeros([nf, nt])

    for kk in np.arange(nf):
        rmatrixin[:, kk] = rmatrix[:, kk] / ednin[kk]
    dn = dnin / ednin
    edn = ednin / ednin

    if (sum(np.isnan(dn)) == 0 and sum(np.isinf(dn)) == 0 and np.prod(dn) > 0):
        ndem = 1
        piter = 0
        rgt = reg_tweak
        L = np.zeros([nt, nt])
        if (np.prod(dem_norm0) == 1.0 or dem_norm0[0] == 0):
            if (np.sum(glc) > 0.0):
                gdglc = (glc > 0).nonzero()[0]
                emloci = np.zeros((nt, gdglc.shape[0]))
                for ee in np.arange(gdglc.shape[0]):
                    emloci[:, ee] = dnin[gdglc[ee]] / (rmatrix[:, gdglc[ee]])
                dem_model = np.zeros(nt)
                for ttt in np.arange(nt):
                    dem_model[ttt] = np.min(emloci[ttt, np.nonzero(emloci[ttt, :])])
                dem_reg_lwght = dem_model
            else:
                L = np.diag(1.0 / np.sqrt(dlogt[:]))
                sva, svb, U, V, W = dem_inv_gsvd(rmatrixin.T, L)
                lamb = dem_reg_map(sva, svb, U, W, dn, edn, rgt, nmu)
                for kk in np.arange(nf):
                    filt[kk, kk] = (sva[kk] / (sva[kk] ** 2 + svb[kk] ** 2 * lamb))
                kdag = W @ (filt.T @ U[:nf, :nf])
                dr0 = np.ravel(kdag @ dn)
                fcofmax = 1e-4
                mask = (dr0 > 0) & (dr0 > fcofmax * np.max(dr0))
                dem_reg_lwght = np.ones(nt)
                dem_reg_lwght[mask] = dr0[mask]
            dem_reg_lwght = (np.convolve(dem_reg_lwght[1:-1], np.ones(5) / 5))[1:-1] / np.max(dem_reg_lwght[:])
            dem_reg_lwght[dem_reg_lwght <= 1e-8] = 1e-8
        else:
            dem_reg_lwght = dem_norm0

        if l_emd:
            L = np.diag(1 / abs(dem_reg_lwght))
        else:
            L = np.diag(np.sqrt(dlogt) / np.sqrt(abs(dem_reg_lwght)))
        sva, svb, U, V, W = dem_inv_gsvd(rmatrixin.T, L)
        while (ndem > 0) and (piter < max_iter):
            lamb = dem_reg_map(sva, svb, U, W, dn, edn, rgt, nmu)
            for kk in np.arange(nf):
                filt[kk, kk] = (sva[kk] / (sva[kk] ** 2 + svb[kk] ** 2 * lamb))
            kdag = W @ (filt.T @ U[:nf, :nf])
            dem_reg_out = (kdag @ dn).squeeze()
            ndem = len(dem_reg_out[dem_reg_out < 0])
            rgt = rgt_fact * rgt
            piter += 1
        if warn and (piter == max_iter):
            print("Warning: positivity loop hit max iterations")
        dem = dem_reg_out
        dn_reg = (rmatrix.T @ dem_reg_out).squeeze()
        residuals = (dnin - dn_reg) / ednin
        chisq = np.sum(residuals ** 2) / nf
        delxi2 = kdag @ kdag.T
        edem = np.sqrt(np.diag(delxi2))
        kdagk = kdag @ rmatrixin.T
        elogt = np.zeros(nt)
        for kk in np.arange(nt):
            rr = np.interp(ltt, logt, kdagk[:, kk])
            hm_mask = (rr >= max(kdagk[:, kk]) / 2.)
            elogt[kk] = dlogt[kk]
            if np.sum(hm_mask) > 0:
                elogt[kk] = (ltt[hm_mask][-1] - ltt[hm_mask][0]) / 2
        if rscl:
            mnrat = np.mean(dnin / dn_reg)
            dem = dem * mnrat
            edem = edem * mnrat
            dn_reg = (rmatrix.T @ dem).squeeze()
            chisq = np.sum(((dnin - dn_reg) / ednin) ** 2) / nf
    return dem, edem, elogt, chisq, dn_reg
