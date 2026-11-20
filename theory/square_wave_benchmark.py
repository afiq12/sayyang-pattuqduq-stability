"""Exact benchmark: square-wave support acceleration (Sec. 3.3).
theta'' + c theta' + (Om0^2 - yb''/l) theta = 0 with yb'' = +a_s, -a_s in alternate half periods.
The monodromy matrix is the product of two matrix exponentials. Displacement harmonics: A_k = 4 a_s / (pi k^3 omega^2), k odd."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, minimize_scalar
g=9.81; l=0.30
def mu(Om0sq,c,a_s,f):
    Th=0.5/f
    M1=expm(np.array([[0,1],[-(Om0sq-a_s/l),-c]])*Th); M2=expm(np.array([[0,1],[-(Om0sq+a_s/l),-c]])*Th)
    return np.log(np.abs(np.linalg.eigvals(M2@M1)).max())*f
def A1(a_s,f): return 4*a_s/(np.pi*(2*np.pi*f)**2)
print('(a) kappa_min: exact vs averaging bound 1 - <ydot^2>/(g l)')
for f,a_s in [(2.5,8.0),(3.0,12.0),(2.0,6.0),(3.0,20.0)]:
    V=a_s*(0.5/f)/2; pred=1-V**2/3/(g*l)
    km=brentq(lambda k: mu((k-1)*g/l,1e-3,a_s,f)-1e-6,0.3,1.2)
    print(f'  f={f} Hz  pp={100*a_s*(0.5/f)**2/4:.1f} cm  exact {km:.4f}  bound {pred:.4f}')
f0=1.25; Om0=2*np.pi*f0
print('(b),(c) thresholds on A1, minimized over drive frequency (l=0.30 m, f0=1.25 Hz)')
for zeta in [0.03,0.10]:
    c=2*zeta*Om0
    for k,fc in [(1,2*f0),(2,f0)]:
        def thr(f):
            F=lambda a: mu(Om0**2,c,a,f)
            return A1(brentq(F,1e-5,400,xtol=1e-7),f) if F(400)>0 else 1e9
        fs=np.linspace(fc*0.8,fc*1.05,251); v=[thr(f) for f in fs]; i=int(np.argmin(v))
        r=minimize_scalar(thr,bracket=(fs[i-1],fs[i],fs[i+1]),tol=1e-7)
        pred=zeta*l if k==1 else l*np.sqrt(8*zeta)
        print(f'  zeta={zeta} k={k}: exact A1={100*r.fun:.2f} cm at f={r.x:.3f} Hz;  prediction {100*pred:.2f} cm')
