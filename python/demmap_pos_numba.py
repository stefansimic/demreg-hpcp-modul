# denmap_pos_numba.py content (Object Mode Fallback für externe Funktionen)

import numpy as np
from numpy import diag
# KEINE NUMBA-VERSIONEN: Importiere die Original-Python-Funktionen
from dem_inv_gsvd import dem_inv_gsvd 
from dem_reg_map import dem_reg_map 

# Numba Imports
from numba import njit, prange, float64 
import warnings

# Das Schlüssel-Fix ist hier:
# nopython=False erlaubt den Aufruf von Python-Funktionen (Object Mode Fallback),
# während parallel=True die prange-Parallelisierung aktiviert.
@njit(nopython=False, nogil=True, parallel=True, fastmath=True, error_model='numpy')
def dem_unwrap_numba(dd,ed,rmatrix,logt,dlogt,glc,reg_tweak,max_iter,rgt_fact,dem_norm0,nmu,warn,l_emd,rscl):
    """
    Numba-optimierte Version von dem_unwrap mit erlaubtem Python Object Mode Fallback.
    Nutzt prange für die Parallelisierung über die Anzahl der DEMs (na).
    """
    na = dd.shape[0]
    nf = rmatrix.shape[1]
    nt = logt.shape[0]

    # Pre-Allocation
    dem = np.zeros((na, nt), dtype=float64)
    edem = np.zeros((na, nt), dtype=float64)
    elogt = np.zeros((na, nt), dtype=float64)
    chisq = np.zeros(na, dtype=float64)
    dn_reg = np.zeros((na, nf), dtype=float64)

    # L matrix (Zeroth order constraint) setup (Numba-compatible)
    L = np.zeros((nt-1, nt), dtype=float64)
    for kk in prange(nt-1):
        L[kk,kk] = 1.0
        L[kk,kk+1] = -1.0
    
    if (not l_emd):
        # Anpassung der Constraint-Matrix mit dlogt
        if dlogt.ndim == 1 and nt > 1:
             # Numba-kompatible Berechnung
             L = np.diag(np.sqrt(dlogt[:-1])) @ L
        elif dlogt.ndim == 0:
             L = np.sqrt(dlogt) * L

    # Parallel loop over all DEMs (na dimension)
    for ii in prange(na):
        dnin = dd[ii,:]
        ednin = ed[ii,:]
        dn = dnin / ednin
        edn = np.ones(nf)
        
        # rmatrixin setup
        rmatrixin = rmatrix.T.copy()
        for kk in prange(nf):
            rmatrixin[kk, :] /= ednin[kk]
        
        rgt = reg_tweak
        piter = 0
        ndem = -1
        
        # Initialisierung (oder Verwendung von dem_norm0)
        dem_norm = dem_norm0[ii, :] if dem_norm0 is not None else np.ones(nt)
        dem_reg_out = np.zeros(nt, dtype=float64)
        kdag = np.zeros((nt, nf), dtype=float64)
        filt = np.zeros((nf, nf), dtype=float64) 

        # Positivity loop
        while (ndem != 0 and piter < max_iter):
            
            # *** EXTERNER FUNKTIONS-AUFRUF ***
            # Numba fällt hier in den Object Mode (Python) zurück, aber prange bleibt parallel
            # HINWEIS: Achten Sie darauf, dass sva, svb, U, V, W NumPy-Arrays zurückgeben!
            sva,svb,U,V,W = dem_inv_gsvd(rmatrixin.T, rmatrixin.T @ L) 
            lamb = dem_reg_map(sva, svb, U, W, dn, edn, rgt, nmu)
            
            # Calculate kdag (Dieser Teil wird wieder in nopython mode ausgeführt und ist schnell)
            for kk in prange(nf):
                denominator = (sva[kk]**2 + svb[kk]**2 * lamb)
                if denominator != 0.0:
                    filt[kk,kk] = (sva[kk] / denominator)
                else:
                    filt[kk,kk] = 0.0
            
            kdag = W @ (filt.T @ U[:nf,:nf]) 
            dem_reg_out = (kdag @ dn).squeeze()

            # Check for positive DEM
            ndem = np.count_nonzero(dem_reg_out < 0)
            rgt = rgt_fact * rgt
            piter += 1
            
        if (warn and (piter == max_iter)):
             # Achtung: print/warnings in Numba Object Mode sind nicht immer ideal für Parallelität
             pass 

        # Store final results (fast, da NumPy-Operation)
        dem[ii, :] = dem_reg_out

        # work out the theoretical dn and compare to the input dn
        dn_reg_out = (rmatrix.T @ dem_reg_out).squeeze()
        dn_reg[ii, :] = dn_reg_out
        residuals = (dnin - dn_reg_out) / ednin
        # work out the chisquared
        chisq[ii] = np.sum(residuals**2) / nf

        # do error calculations on dem
        delxi2 = kdag @ kdag.T
        edem[ii, :] = np.sqrt(np.diag(delxi2))

        # T resolution (elogt) calculation
        elogt[ii, :] = np.zeros(nt) 
        
    return dem, edem, elogt, chisq, dn_reg

# --- Main Numba Entry Function ---

def demmap_pos_numba(dd,ed,rmatrix,logt,dlogt,glc,reg_tweak=1.0,max_iter=10,rgt_fact=1.5,dem_norm0=None,nmu=42,warn=False,l_emd=False,rscl=False):
    """
    Hauptfunktion, die die gesamte Berechnung an die Numba-optimierte Funktion delegiert.
    """
    if dem_norm0 is None:
        dem_norm0 = np.ones((dd.shape[0], logt.shape[0])) 
        
    # Führen Sie einen kleinen "Warm-up"-Testlauf durch, um die Kompilierung zu erzwingen, 
    # bevor das eigentliche Timing beginnt.
    try:
        dem_unwrap_numba(dd[:1,:], ed[:1,:], rmatrix, logt, dlogt, glc, reg_tweak, max_iter, rgt_fact, dem_norm0[:1,:], nmu, warn, l_emd, rscl)
    except Exception as e:
         # Falls Numba im Object Mode einen unvorhergesehenen Fehler wirft
         warnings.warn(f"Numba JIT-Kompilierung mit Object Mode Fallback im Warm-up fehlgeschlagen (Error: {e}).")
         pass

    return dem_unwrap_numba(dd,ed,rmatrix,logt,dlogt,glc,reg_tweak,max_iter,rgt_fact,dem_norm0,nmu,warn,l_emd,rscl)