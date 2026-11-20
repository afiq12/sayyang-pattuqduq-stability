import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from floquet_core import floquet_mu, hentakan_coeffs
g=9.81
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#555','axes.labelcolor':'#222','xtick.color':'#444','ytick.color':'#444'})
seq=LinearSegmentedColormap.from_list('seq',['#fde2d4','#f08a5d','#b83b18','#6b1e08'])

# ---- Fig 1: Ince-Strutt with horse regime
a=np.linspace(-2.0,3.0,501); q=np.linspace(0,1.6,321)
AA,QQ=np.meshgrid(a,q)
mu=floquet_mu(AA,0.0,1.0,np.full_like(AA,1/np.pi),[-QQ/2],[0.0],nsteps=500)
fig,ax=plt.subplots(figsize=(7,4.6),dpi=200)
stab=(mu<1e-4)
ax.contourf(AA,QQ,stab.astype(float),levels=[0.5,1.5],colors=['#cfe3f5'])
ax.contour(AA,QQ,stab.astype(float),levels=[0.5],colors='#2a6fb0',linewidths=1.2)
qq=np.linspace(0,1.6,200); ax.plot(-qq**2/2,qq,'--',color='#333',lw=1.2)
ax.text(-1.95,1.05,'Kapitza boundary (dashed)\n a = −q²/2',fontsize=9,color='#333')
# horse regime: l 0.25-0.45 m, f 1.5-3.5 Hz, A (amplitude) 0.015-0.06 m
rng=np.random.default_rng(1)
L=rng.uniform(0.25,0.45,600); F=rng.uniform(1.5,3.5,600); Am=rng.uniform(0.015,0.06,600)
ah=-4*g/(L*(2*np.pi*F)**2); qh=2*Am/L
ax.scatter(ah,qh,s=4,color='#b83b18',alpha=0.5,lw=0)
ax.text(-1.95,0.55,'horse regime\n(l 0.25 to 0.45 m;\nf 1.5 to 3.5 Hz;\nA 1.5 to 6 cm)',fontsize=8.5,color='#b83b18')
ax.text(1.7,0.25,'stable',color='#2a6fb0',fontsize=10)
ax.set_xlabel('a = −4g/(lω²)   (a > 0 with hip stiffness: 4Ω₀²/ω²)')
ax.set_ylabel('|q| = 2A/l')
ax.set_title('Ince-Strutt diagram: the horse regime lies outside the Kapitza stable region',fontsize=10.5,loc='left')
ax.set_xlim(-2,3); ax.set_ylim(0,1.6)
fig.tight_layout(); fig.savefig(str(ROOT/'figures'/'fig1.png')); print('fig1 ok')

# ---- Fig 2
l=0.30; f0=1.25; Om0sq=(2*np.pi*f0)**2
f=np.linspace(0.8,4.0,161); App=np.linspace(0.0,0.14,113)
FF,PP=np.meshgrid(f,App)
ak,ph=hentakan_coeffs(K=6,width=0.05)
cases=[('Sine, \u03b6 = 0.10',0.10,[PP/2],[0.0]),('Impulsive stamping, \u03b6 = 0.10',0.10,[PP*x for x in ak],list(ph)),('Impulsive stamping, \u03b6 = 0.03',0.03,[PP*x for x in ak],list(ph))]
mus=[floquet_mu(Om0sq,z,l,FF,A,p) for _,z,A,p in cases]
vmax=max(m.max() for m in mus)
fig,axs=plt.subplots(1,3,figsize=(12,4.2),dpi=200,sharey=True)
for ax,mu,(tt,z,_,_) in zip(axs,mus,cases):
    m=np.ma.masked_less_equal(mu,0); ax.set_facecolor('#eef3f8')
    im=ax.pcolormesh(FF,PP*100,m,cmap=seq,vmin=0,vmax=vmax,shading='auto')
    ax.contour(FF,PP*100,mu,levels=[0],colors='#6b1e08',linewidths=0.8)
    for k,yy in ((1,0.6),(2,1.6),(3,0.6)):
        fk=2*f0/k; ax.axvline(fk,color='#555',lw=0.7,ls=':')
        ax.text(fk+0.04,yy if k!=2 else 0.3,f'k={k}',fontsize=8,color='#444')
    ax.axhline(2*z*l*100/0.5*0.5,color='#2a6fb0',lw=0.8,ls='--')
    ax.set_title(tt,fontsize=10,loc='left'); ax.set_xlabel('stamping frequency f (Hz)')
axs[0].set_ylabel('peak-to-peak amplitude (cm)')
cb=fig.colorbar(im,ax=axs,shrink=0.85,pad=0.02); cb.set_label('Floquet exponent \u03bc (1/s); light blue = stable')
fig.suptitle(f'Rider stability (l = {l} m, f\u2080 = {f0} Hz). Dotted: tongue k at f = 2f\u2080/k. Dashed blue: sine threshold 2\u03b6l (peak to peak)',fontsize=10,x=0.02,ha='left')
fig.savefig(str(ROOT/'figures'/'fig2.png'),bbox_inches='tight'); print('fig2 ok')
mu_s,mu_h,mu_h3=mus
# numbers for text
i=np.argmin(abs(f-2*f0))
print('sine: min A_pp unstable at f=2f0:', App[np.where(mu_s[:,i]>0)[0][0]] if (mu_s[:,i]>0).any() else None)
print('hent: min A_pp unstable at f=2f0:', App[np.where(mu_h[:,i]>0)[0][0]] if (mu_h[:,i]>0).any() else None)
j=np.argmin(abs(f-f0))
print('sine at f=f0 (k=2 tongue):', (mu_s[:,j]>0).any(), ' hent:', App[np.where(mu_h[:,j]>0)[0][0]] if (mu_h[:,j]>0).any() else None)
print('max mu', mu_s.max(), mu_h.max()); jj=np.argmin(abs(f-f0)); print('zeta.03 k=2 tongue min pp:', App[np.where(mu_h3[:,jj]>0)[0][0]] if (mu_h3[:,jj]>0).any() else None)
print('hentakan ak*pp=6cm:',np.round(ak*6,3))
