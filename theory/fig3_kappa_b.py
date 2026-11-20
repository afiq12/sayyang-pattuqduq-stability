import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from floquet_core import floquet_mu, hentakan_coeffs
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
seq=LinearSegmentedColormap.from_list('seq',['#fde2d4','#f08a5d','#b83b18','#6b1e08'])
g=9.81; heff=0.30; f=2.5; App=0.06
ak,ph=hentakan_coeffs(K=6,width=0.05)
kap=np.linspace(0.85,4.0,190); bd=np.linspace(0.01,0.40,140)
KK,BB=np.meshgrid(kap,bd)
Om=(KK-1)*g/heff; c=BB*g/heff
z=c/(2*np.sqrt(np.abs(Om)+1e-9))
mu=floquet_mu(Om,z,heff,np.full_like(KK,f),[np.full_like(KK,App*x) for x in ak],list(ph))
fig,ax=plt.subplots(figsize=(7.2,4.6),dpi=200)
ax.set_facecolor('#eef3f8')
m=np.ma.masked_less_equal(mu,0)
im=ax.pcolormesh(KK,BB,m,cmap=seq,shading='auto',vmin=0)
ax.contour(KK,BB,mu,levels=[0],colors='#6b1e08',linewidths=0.9)
kc=1-(App/2)**2*(2*np.pi*f)**2/(2*g*heff)
ax.axvline(kc,color='#333',ls='--',lw=0.9); ax.text(kc+0.03,0.37,f'Kapitza: κ > {kc:.3f}',fontsize=8.5)
kres=1+(2*np.pi*f/2)**2*heff/g; ax.axvline(kres,color='#555',ls=':',lw=0.9); ax.text(kres+0.03,0.37,'2f₀ = f',fontsize=8.5,color='#444')
ax.add_patch(plt.Rectangle((1.3,0.15),0.2,0.03,fill=False,ec='#2a6fb0',lw=1.6))
ax.annotate('',xy=(1.4,0.30),xytext=(1.4,0.185),arrowprops=dict(arrowstyle='->',color='#2a6fb0',lw=1.2))
ax.text(1.45,0.31,'Ref. [13]: children with CP;\nadults have larger total damping',fontsize=8,color='#2a6fb0',va='bottom')
ax.text(0.87,0.2,'falls',rotation=90,fontsize=9,color='#6b1e08')
ax.annotate('parametric resonance:\nstiff rider,\nlow damping',xy=(2.9,0.04),xytext=(3.15,0.16),fontsize=8.5,arrowprops=dict(arrowstyle='->',color='#444'))
ax.set_xlabel('trunk stiffness κ = K/(mgh)'); ax.set_ylabel('damping b = B/(mgh)  (s)')
ax.set_title('Impulsive stamping, f = 2.5 Hz, 6 cm peak to peak, l = 0.30 m',fontsize=9.5,loc='left')
cb=fig.colorbar(im,ax=ax,pad=0.02); cb.set_label('Floquet exponent μ (1/s)')
fig.tight_layout(); fig.savefig(str(ROOT/'figures'/'fig3.png'))
# report tongue minimum damping location
print('kappa_crit',kc,'kres',kres)
col=np.argmin(abs(kap-kres)); print('max b unstable near kres:', bd[mu[:,col]>0].max() if (mu[:,col]>0).any() else None)
