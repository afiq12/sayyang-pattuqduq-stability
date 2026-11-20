import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from floquet_core import hentakan_coeffs
def hill_curves(betas,phis,N=40):
    out={}
    for parity,name in ((1,'4pi'),(0,'2pi')):
        ns=np.arange(-2*N+parity,2*N+1,2)   # n odd -> e^{i n tau/2} period 4pi ; n even -> 2pi
        idx={n:i for i,n in enumerate(ns)}; H=np.diag((ns/2.0)**2).astype(complex)
        for k,(b,p) in enumerate(zip(betas,phis),1):
            for n in ns:
                if n-2*k in idx: H[idx[n],idx[n-2*k]]-=b/2*np.exp(1j*p)
                if n+2*k in idx: H[idx[n],idx[n+2*k]]-=b/2*np.exp(-1j*p)
        out[name]=np.sort(np.linalg.eigvalsh(H))
    return out
def trace(alpha,betas,phis):
    rhs=lambda t,s:[s[1],-(alpha+sum(b*np.cos((k+1)*t+p) for k,(b,p) in enumerate(zip(betas,phis))))*s[0]]
    M=np.array([solve_ivp(rhs,[0,2*np.pi],ic,rtol=1e-11,atol=1e-13).y[:,-1] for ic in ([1,0],[0,1])]).T
    return np.trace(M)
ak,ph=hentakan_coeffs(K=6,width=0.05)
l=0.30; pp=0.08
betas=[(k**2)*pp*a/l for k,a in enumerate(ak,1)]
print('betas',np.round(betas,4))
hc=hill_curves(betas,list(ph))
for k in (1,2,3):
    c=k*k/4; name='4pi' if k%2 else '2pi'
    ev=hc[name]; near=ev[np.argsort(abs(ev-c))[:2]]; near=np.sort(near)
    target=-2 if k%2 else 2
    f=lambda a: trace(a,betas,list(ph))-target
    fl=[brentq(f,x-0.02,x+0.02) for x in near]
    print(f'k={k}: Hill determinant {np.round(near,5)}  Floquet {np.round(fl,5)}')
