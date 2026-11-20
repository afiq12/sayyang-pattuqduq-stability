import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, sys

from floquet_core import hentakan_coeffs
from scipy.optimize import brentq
g=9.81
def mu_c(Om0sq,c,l,f,Ak,phik,nsteps=800):
    w=2*np.pi*f; T=1/f; h=T/nsteps
    def ydd(t): return sum(-A*(k*w)**2*np.cos(k*w*t+ph) for k,(A,ph) in enumerate(zip(Ak,phik),1))
    def acc(t,x,v): return -c*v-(Om0sq-ydd(t)/l)*x
    Y=np.array([[1.0,0.0],[0.0,1.0]]); t=0.0  # rows: (x,v) for two ICs
    X=Y[:,0].copy(); V=Y[:,1].copy()
    X=np.array([1.0,0.0]); V=np.array([0.0,1.0])
    for _ in range(nsteps):
        k1x=V; k1v=acc(t,X,V)
        k2x=V+h/2*k1v; k2v=acc(t+h/2,X+h/2*k1x,V+h/2*k1v)
        k3x=V+h/2*k2v; k3v=acc(t+h/2,X+h/2*k2x,V+h/2*k2v)
        k4x=V+h*k3v; k4v=acc(t+h,X+h*k3x,V+h*k3v)
        X=X+h/6*(k1x+2*k2x+2*k3x+k4x); V=V+h/6*(k1v+2*k2v+2*k3v+k4v); t+=h
    M=np.array([[X[0],X[1]],[V[0],V[1]]]); return np.log(np.abs(np.linalg.eigvals(M)).max())/T
a_h,ph_h=hentakan_coeffs(6,0.05); a_s=np.array([0.5]); ph_s=np.array([0.0])
if 'lit' in sys.argv:
    for l in [0.25,0.30,0.45]:
      for b in [0.15,0.18]:
        kap=1.4; Om0sq=(kap-1)*g/l; c=b*g/l; f0=np.sqrt(Om0sq)/2/np.pi; z=c/2/np.sqrt(Om0sq)
        res=[]
        for name,(a,ph) in [('stamp',(a_h,ph_h)),('sine',(a_s,ph_s))]:
            best=(None,9)
            for f in np.arange(0.8,3.61,0.05):
                F=lambda pp: mu_c(Om0sq,c,l,f,a*pp,ph,500)
                if F(1.0)<=0: continue
                pp=brentq(F,0.005,1.0,xtol=1e-3)
                if pp<best[1]: best=(f,pp)
            res.append((name,best))
        print(f'l={l} b={b} f0={f0:.2f} zeta={z:.2f} zeta*l={100*z*l:.1f}cm', [(n,round(f,2) if f else None,round(100*p,1)) for n,(f,p) in res])
if 'kmin' in sys.argv:
    l=0.30
    for f,pp in [(2.0,0.06),(2.5,0.06),(3.0,0.06),(3.0,0.10)]:
        w=2*np.pi*f; A=a_h*pp; k=np.arange(1,7)
        vy2=np.sum(A**2*(k*w)**2/2); eqw=1-vy2/(g*l); eqs=1-(pp/2)**2*w**2/2/(g*l)
        out=[]
        for b in [0.05,0.18]:
            c=b*g/l
            km=brentq(lambda kap: mu_c((kap-1)*g/l,c,l,f,A,ph_h,800),0.5,1.0,xtol=1e-5)
            out.append(km)
        print(f,pp,[round(x,3) for x in out],'eq(waveform)',round(eqw,3),'eq(sine same pp)',round(eqs,3))
