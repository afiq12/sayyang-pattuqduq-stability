import numpy as np
from scipy.integrate import solve_ivp
def rho_max(alpha, betas, phis=None, zeta=0.0, N=None):
    phis = phis if phis is not None else [0]*len(betas)
    c = 2*zeta*np.sqrt(max(alpha,1e-12))
    def rhs(t,s):
        p = alpha + sum(b*np.cos((n+1)*t+ph) for n,(b,ph) in enumerate(zip(betas,phis)))
        return [s[1], -c*s[1]-p*s[0]]
    M=np.zeros((2,2))
    for j,ic in enumerate(([1,0],[0,1])):
        sol=solve_ivp(rhs,[0,2*np.pi],ic,rtol=1e-10,atol=1e-12,method='DOP853')
        M[:,j]=sol.y[:,-1]
    return max(abs(np.linalg.eigvals(M)))
