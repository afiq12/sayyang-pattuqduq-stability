import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import sys
import numpy as np
from floquet_core import floquet_mu, hentakan_coeffs
from scipy.optimize import brentq
l=0.30; f0=1.25; Om0sq=(2*np.pi*f0)**2; zeta=0.03
ak,ph=hentakan_coeffs(K=6,width=0.05)
print('A1/pp=%.4f A2/pp=%.4f'%(ak[0],ak[1]))
def mumax(pp):
    f=np.linspace(1.18,1.32,57)   # around f=f0 (k=2 tongue at f = f0)
    mu=floquet_mu(Om0sq,zeta,l,f,[np.full_like(f,pp*x) for x in ak],list(ph),nsteps=1500)
    return mu.max()
pp_num=brentq(mumax,0.05,0.14,xtol=1e-4)
A1=lambda p:ak[0]*p; A2=lambda p:ak[1]*p
first=zeta*l/ak[1]
g=lambda p: abs(A2(p)*np.exp(1j*ph[1]) - A1(p)**2/(8*l)*np.exp(2j*ph[0])) - zeta*l
second=brentq(g,0.01,0.2)
print(f'k=2 threshold pp: numeric {100*pp_num:.2f} cm, first order {100*first:.2f} cm, second order {100*second:.2f} cm')
