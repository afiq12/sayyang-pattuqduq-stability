import numpy as np
g=9.81
def floquet_mu(Om0sq, zeta, l, f, Ak, phik, nsteps=600):
    """Vectorized: arrays broadcast. y_b = sum_k Ak[k]*cos(k w t + phik[k]).
    eq: th'' + c th' + [Om0^2 - yddot/l] th = 0, c = 2 zeta sqrt(|Om0^2|).
    returns Floquet exponent mu (1/s) = ln max|rho| / T"""
    f=np.asarray(f,float); w=2*np.pi*f; T=1/f; h=T/nsteps
    c=2*zeta*np.sqrt(np.abs(Om0sq))
    def ydd(t):
        s=0
        for k,(A,ph) in enumerate(zip(Ak,phik),1):
            s=s - A*(k*w)**2*np.cos(k*w*t+ph)
        return s
    def acc(t,x,v): return -c*v-(Om0sq-ydd(t)/l)*x
    sh=np.broadcast(f,*Ak).shape
    X=np.stack([np.ones(sh),np.zeros(sh)]); V=np.stack([np.zeros(sh),np.ones(sh)])
    t=np.zeros(sh)
    for _ in range(nsteps):
        k1x=V; k1v=acc(t,X,V)
        k2x=V+h/2*k1v; k2v=acc(t+h/2,X+h/2*k1x,V+h/2*k1v)
        k3x=V+h/2*k2v; k3v=acc(t+h/2,X+h/2*k2x,V+h/2*k2v)
        k4x=V+h*k3v; k4v=acc(t+h,X+h*k3x,V+h*k3v)
        X=X+h/6*(k1x+2*k2x+2*k3x+k4x); V=V+h/6*(k1v+2*k2v+2*k3v+k4v); t=t+h
    tr=X[0]+V[1]; det=X[0]*V[1]-X[1]*V[0]
    disc=np.sqrt((tr**2/4-det).astype(complex))
    r=np.maximum(abs(tr/2+disc),abs(tr/2-disc))
    return np.log(r)/T

def hentakan_coeffs(K=6, width=0.08):
    """'Hentakan' waveform: acceleration = periodic train of smoothed impacts (Gaussian, width as fraction of period)
    minus mean. Displacement harmonic magnitudes a_k ~ exp(-(2pi k width)^2/2)/k^2, phases pi (impact at t=0 => trough).
    Normalized to unit peak-to-peak displacement."""
    k=np.arange(1,K+1); a=np.exp(-(2*np.pi*k*width)**2/2)/k**2
    tt=np.linspace(0,1,4000); y=sum(ai*np.cos(2*np.pi*ki*tt+np.pi) for ai,ki in zip(a,k))
    a=a/(y.max()-y.min()); return a, np.full(K,np.pi)
if __name__=="__main__":
    a,ph=hentakan_coeffs(); print("hentakan A_k (per unit pp):",np.round(a,4))
