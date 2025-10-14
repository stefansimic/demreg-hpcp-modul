import numpy as np
from numpy import diag
from dem_inv_gsvd import dem_inv_gsvd
from dem_reg_map import dem_reg_map
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from threadpoolctl import threadpool_limits

def demmap_pos(dd,ed,rmatrix,logt,dlogt,glc,reg_tweak=1.0,max_iter=10,rgt_fact=1.5,dem_norm0=None,nmu=42,warn=False,l_emd=False,rscl=False):
    dd=np.array(dd)
    ed=np.array(ed)
    rmatrix=np.array(rmatrix)
    logt=np.array(logt)
    dlogt=np.array(dlogt)

    na=dd.shape[0]
    nf=dd.shape[1]
    nt=rmatrix.shape[0]

    dem=np.zeros([na,nt])
    edem=np.zeros([na,nt])
    elogt=np.zeros([na,nt])
    chisq=np.zeros([na])
    dn_reg=np.zeros([na,nf])

    if (na>=200):
        n_par = 100
        niter=int(np.floor(na/n_par))
        with threadpool_limits(limits=1):
            with ProcessPoolExecutor() as exe:
                futures=[
                    exe.submit(
                        dem_unwrap,
                        dd[i*n_par:(i+1)*n_par,:],
                        ed[i*n_par:(i+1)*n_par,:],
                        rmatrix,logt,dlogt,glc,
                        reg_tweak=reg_tweak,max_iter=max_iter,rgt_fact=rgt_fact,
                        dem_norm0=(None if dem_norm0 is None else dem_norm0[i*n_par:(i+1)*n_par,:]),
                        nmu=nmu,warn=warn,l_emd=l_emd,rscl=rscl
                    )
                    for i in np.arange(niter)
                ]
                if na > niter*n_par:
                    futures.append(
                        exe.submit(
                            dem_unwrap,
                            dd[niter*n_par:na,:],
                            ed[niter*n_par:na,:],
                            rmatrix,logt,dlogt,glc,
                            reg_tweak=reg_tweak,max_iter=max_iter,rgt_fact=rgt_fact,
                            dem_norm0=(None if dem_norm0 is None else dem_norm0[niter*n_par:na,:]),
                            nmu=nmu,warn=warn,l_emd=l_emd,rscl=rscl
                        )
                    )
                for _ in tqdm(as_completed(futures), total=len(futures)):
                    pass
            base = 0
            for f in futures:
                dem1,edem1,elogt1,chisq1,dn_reg1 = f.result()
                j0 = base*n_par
                j1 = j0 + dem1.shape[0]
                dem[j0:j1,:]=dem1
                edem[j0:j1,:]=edem1
                elogt[j0:j1,:]=elogt1
                chisq[j0:j1]=chisq1
                dn_reg[j0:j1,:]=dn_reg1
                base += 1
    else:
        with threadpool_limits(limits=1):
            for i in np.arange(na):
                dem[i,:],edem[i,:],elogt[i,:],chisq[i],dn_reg[i,:] = dem_pix(
                    dd[i,:],ed[i,:],rmatrix,logt,dlogt,glc,
                    reg_tweak=reg_tweak,max_iter=max_iter,rgt_fact=rgt_fact,
                    dem_norm0=(None if dem_norm0 is None else dem_norm0[i,:]),
                    nmu=nmu,warn=warn,l_emd=l_emd,rscl=rscl
                )

    return dem,edem,elogt,chisq,dn_reg

def dem_unwrap(dn,ed,rmatrix,logt,dlogt,glc,reg_tweak=1.0,max_iter=10,rgt_fact=1.5,dem_norm0=None,nmu=42,warn=False,l_emd=False,rscl=False):
    na=dn.shape[0]
    nt=logt.shape[0]
    nf=rmatrix.shape[1]
    dem=np.zeros([na,nt])
    edem=np.zeros([na,nt])
    elogt=np.zeros([na,nt])
    chisq=np.zeros([na])
    dn_reg=np.zeros([na,nf])
    for i in np.arange(na):
        dem[i,:],edem[i,:],elogt[i,:],chisq[i],dn_reg[i,:]=dem_pix(
            dn[i,:],ed[i,:],rmatrix,logt,dlogt,glc,
            reg_tweak=reg_tweak,max_iter=max_iter,rgt_fact=rgt_fact,
            dem_norm0=(None if dem_norm0 is None else dem_norm0[i,:]),
            nmu=nmu,warn=warn,l_emd=l_emd,rscl=rscl
        )
    return dem,edem,elogt,chisq,dn_reg

def dem_pix(dnin,ednin,rmatrix,logt,dlogt,glc,reg_tweak=1.0,max_iter=10,rgt_fact=1.5,dem_norm0=None,nmu=42,warn=True,l_emd=False,rscl=False):
    nf=rmatrix.shape[1]
    nt=logt.shape[0]
    ltt=min(logt)+1e-8+(max(logt)-min(logt))*np.arange(51)/(52-1.0)

    dem=np.zeros(nt)
    edem=np.zeros(nt)
    elogt=np.zeros(nt)
    chisq=0.0
    dn_reg=np.zeros(nf)

    ednin = np.array(ednin, dtype=float)
    dnin = np.array(dnin, dtype=float)
    bad = (ednin <= 0) | ~np.isfinite(ednin)
    if np.any(bad):
        mx = np.max(ednin[~bad]) if np.any(~bad) else 1.0
        ednin = np.where(bad, 10.0*mx, ednin)

    rmatrixin = rmatrix / ednin[None, :]
    dn = dnin / ednin
    edn = np.ones_like(dn)

    if dem_norm0 is None or (np.size(dem_norm0)==1 and dem_norm0==0):
        if l_emd:
            L=np.diag(np.sqrt(dlogt))
        else:
            if len(np.shape(glc))==1:
                L=np.diag(glc)
            else:
                L=glc

        rgt=reg_tweak
        sva,svb,U,V,W=dem_inv_gsvd(rmatrixin.T,L)
        lamb=dem_reg_map(sva,svb,U,W,dn,edn,rgt,nmu)

        sa2 = sva[:nf]**2
        sb2 = svb[:nf]**2
        filt_vec = np.divide(sva[:nf], sa2 + sb2*lamb, where=(sa2 + sb2*lamb)!=0)

        U6 = U[:nf, :nf]
        # ROW scaling: diag(filt_vec) @ U6
        Z = filt_vec[:, None] * U6
        if nt > nf:
            Kcore = np.vstack([Z, np.zeros((nt - nf, nf))])
        else:
            Kcore = Z[:nt, :]
        kdag = W @ Kcore

        dr0=(kdag@dn).squeeze()
        fcofmax=1e-4
        mask=(dr0 > 0) & (dr0 > fcofmax*np.max(dr0))
        dem_reg_lwght=np.ones(nt)
        dem_reg_lwght[mask]=dr0[mask]
    else:
        dem_reg_lwght=np.array(dem_norm0, dtype=float)

    dem_reg_lwght[~np.isfinite(dem_reg_lwght)] = np.nanmin(dem_reg_lwght[np.isfinite(dem_reg_lwght)])
    dem_reg_lwght[dem_reg_lwght<=0] = np.min(dem_reg_lwght[dem_reg_lwght>0])

    if l_emd:
        L=np.diag(1/abs(dem_reg_lwght))
    else:
        L=np.diag(np.sqrt(dlogt)/np.sqrt(abs(dem_reg_lwght)))

    sva,svb,U,V,W = dem_inv_gsvd(rmatrixin.T,L)

    ndem=1
    piter=0
    rgt=reg_tweak
    dem_reg_out = np.zeros(nt)

    while((ndem > 0) and (piter < max_iter)):
        lamb=dem_reg_map(sva,svb,U,W,dn,edn,rgt,nmu)

        sa2 = sva[:nf]**2
        sb2 = svb[:nf]**2
        denom = sa2 + sb2*lamb
        filt_vec = np.divide(sva[:nf], denom, where=denom!=0)

        U6 = U[:nf, :nf]
        # ROW scaling again
        Z = filt_vec[:, None] * U6
        if nt > nf:
            Kcore = np.vstack([Z, np.zeros((nt - nf, nf))])
        else:
            Kcore = Z[:nt, :]
        kdag = W @ Kcore

        dem_reg_out=(kdag@dn).squeeze()
        ndem=int(np.sum(dem_reg_out < 0))
        rgt=rgt_fact*rgt
        piter+=1

    if (warn and (piter == max_iter)):
        print('Warning - max iterations reached in positivity regularisation')

    dem=dem_reg_out
    dn_reg=(rmatrix.T @ dem_reg_out).squeeze()
    residuals=(dnin-dn_reg)/ednin
    chisq=np.sum(residuals**2)/(nf)

    delxi2=kdag@kdag.T
    edem=np.sqrt(np.clip(np.diag(delxi2), 0.0, np.inf))

    kdagk=kdag@rmatrixin.T

    ltt=np.linspace(logt.min(),logt.max(),52)
    elogt=np.zeros(nt)
    for kk in np.arange(nt):
        rr=np.interp(ltt,logt,kdagk[:,kk])
        hm_mask=(rr >= np.max(kdagk[:,kk])/2.)
        elogt[kk]=dlogt[kk]
        if np.any(hm_mask):
            elogt[kk]=(ltt[hm_mask][-1]-ltt[hm_mask][0])/2

    if rscl:
        mnrat=np.mean(dnin/dn_reg)
        dem=dem*mnrat
        edem=edem*mnrat
        dn_reg=(rmatrix.T @ dem).squeeze()
        chisq=np.sum(((dnin-dn_reg)/ednin)**2)/nf

    return dem,edem,elogt,chisq,dn_reg
