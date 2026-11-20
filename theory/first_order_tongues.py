import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from floq import rho_max
from scipy.optimize import brentq
eps=0.04
for k in (1,2,3):
    b=[0,0,0]; b[k-1]=eps
    f=lambda a: rho_max(a,b)-1-1e-7
    a0=k**2/4
    # scan for edges
    al=np.linspace(a0-0.06,a0+0.06,241); r=np.array([rho_max(a,b) for a in al])
    un=al[r>1+1e-6]
    print(f"k={k}: numeric tongue [{un.min():.4f},{un.max():.4f}]  first-order [{a0-eps/2:.4f},{a0+eps/2:.4f}]")
# damping threshold: at alpha=k^2/4, find min beta_k for instability with zeta
for zeta in (0.05,0.1):
  for k in (1,2,3):
    def g(beta):
        b=[0,0,0]; b[k-1]=beta
        # maximize over alpha near tongue center
        return max(rho_max(a,b,zeta=zeta) for a in np.linspace(k**2/4-0.02,k**2/4+0.02,9))-1
    bc=brentq(g,1e-3,3.0,xtol=1e-4)
    print(f"zeta={zeta} k={k}: beta_crit={bc:.4f}  predicted zeta*k^2={zeta*k*k:.4f}")
