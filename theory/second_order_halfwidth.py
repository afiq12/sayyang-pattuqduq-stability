import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
def trM(alpha,b,ph,gam=0.0):
    def rhs(t,s):
        p=alpha+sum(bk*np.cos((k+1)*t+pk) for k,(bk,pk) in enumerate(zip(b,ph)))
        return [s[1],-gam*s[1]-p*s[0]]
    M=np.array([solve_ivp(rhs,[0,2*np.pi],ic,rtol=1e-11,atol=1e-13).y[:,-1] for ic in ([1,0],[0,1])]).T
    return M
def edges2(b,ph):
    # tongue near alpha=1 (k=2): instability where |tr|>2 (period 2pi solutions -> tr=+2)
    f=lambda a: np.trace(trM(a,b,ph))-2
    al=np.linspace(0.97,1.06,181); v=np.array([f(a) for a in al])
    idx=np.where(np.diff(np.sign(v)))[0]
    return [brentq(f,al[i],al[i+1],xtol=1e-9) for i in idx]
b1=0.2
for (b2,p1,p2) in [(0,0,0),(0.01,0,0),(0.01,0,np.pi),(0.01,np.pi,np.pi),(0.01,np.pi/2,0),(0.01,0,np.pi/2)]:
    e=edges2([b1,b2],[p1,p2]); w=(e[-1]-e[0])/2 if len(e)>=2 else 0
    c1=abs(b2/2*np.exp(1j*p2)+b1**2/4*np.exp(2j*p1)); c2=abs(b2/2*np.exp(1j*p2)-b1**2/4*np.exp(2j*p1))
    print(f"b2={b2} p1={p1:.2f} p2={p2:.2f} edges={np.round(e,5)} halfwidth={w:.5f} plus={c1:.5f} minus={c2:.5f} center={np.mean(e):.5f} pred_center={1+b1**2/6:.5f}")
