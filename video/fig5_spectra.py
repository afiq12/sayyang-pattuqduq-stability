"""Figure 5: amplitude spectra of rider head motion, shots B to G (reads data/riders/)."""
import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data'/'riders'
S={r['tag']:r for r in json.load(open(D/'summary.json'))}
lab={'b076':'Shot B, V1 (low quality)','b134':'Shot C, V1','b282':'Shot D, V1','p787':'Shot E, V2','p833':'Shot F, V2','p1105':'Shot G, V2'}
fig,axs=plt.subplots(2,3,figsize=(9,5),sharex=True,sharey=True)
for ax,t in zip(axs.ravel(),lab):
    tt,y=np.loadtxt(D/f'head_{t}.csv',delimiter=',',skiprows=1).T; fr=np.arange(1.0,6.0,0.01)
    def amp(f):
        X=np.column_stack([np.cos(2*np.pi*f*tt),np.sin(2*np.pi*f*tt),np.ones_like(tt)]); c,*_=np.linalg.lstsq(X,y,rcond=None); return np.hypot(c[0],c[1])
    A=10*np.array([amp(f) for f in fr])/S[t]['gain']
    ax.axvspan(2.62,3.33,color='#d9d9d9',lw=0); ax.axvspan(1.31,1.67,color='#ececec',lw=0)
    ax.semilogy(fr,A,color='#1f4e79',lw=1.1); ax.axhline(0.6,color='#888888',lw=0.8,ls='--')
    ax.set_title(lab[t],fontsize=9); ax.set_ylim(0.02,60); ax.grid(alpha=0.25,lw=0.5)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.text(1.49,35,'½ beat',fontsize=7,color='#555555',ha='center'); ax.text(2.975,35,'beat',fontsize=7,color='#555555',ha='center')
for ax in axs[1]: ax.set_xlabel('Frequency (Hz)')
for ax in axs[:,0]: ax.set_ylabel('Head amplitude (mm)')
fig.tight_layout(); fig.savefig(ROOT/'figures'/'fig5.png',dpi=200)
