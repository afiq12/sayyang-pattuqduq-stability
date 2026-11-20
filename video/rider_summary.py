import numpy as np,json
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'data'/'riders'
rows=[]
for t in ['b076','b134','b282','p787','p833','p1105']:
    a=np.loadtxt(D/f'head_{t}.csv',delimiter=',',skiprows=1).T; b=np.loadtxt(D/f'head_{t}i.csv',delimiter=',',skiprows=1).T; n=min(a.shape[1],b.shape[1]); tt=a[0,:n]; y=a[1,:n]; d=b[1,:n]-y
    def amp(z,f):
        X=np.column_stack([np.cos(2*np.pi*f*tt),np.sin(2*np.pi*f*tt),np.ones_like(tt)]); c,*_=np.linalg.lstsq(X,z,rcond=None); return np.hypot(c[0],c[1])
    g=amp(d,2.9)/0.1
    fb=np.arange(2.62,3.335,0.005); Ab=np.array([amp(y,f) for f in fb])
    fh=np.arange(1.31,1.67,0.005); Ah=np.array([amp(y,f) for f in fh])
    fc=np.arange(4.0,6.0,0.01); Ac=np.array([amp(y,f) for f in fc])
    fa=np.arange(1.0,6.0,0.01); Aa=np.array([amp(y,f) for f in fa])
    r=json.load(open(D/f'res_{t}.json'))
    nb=np.r_[np.arange(2.0,2.62,0.01),np.arange(3.34,4.0,0.01)]; nh=np.r_[np.arange(1.0,1.31,0.01),np.arange(1.68,2.2,0.01)]
    rb=Ab.max()/np.percentile([amp(y,f) for f in nb],95); rh=Ah.max()/np.percentile([amp(y,f) for f in nh],95)
    rows.append(dict(ratio_beat=round(rb,2),ratio_half=round(rh,2),tag=t,dur=round(tt[-1]-tt[0],1),resp=round(r['resp_med'],2),gain=round(g,2),
      beat_max_mm=round(10*Ab.max()/g,2),f_beat=round(fb[Ab.argmax()],2),half_max_mm=round(10*Ah.max()/g,2),f_half=round(fh[Ah.argmax()],2),
      ctrl_p95_mm=round(10*np.percentile(Ac,95)/g,2),rms16_mm=round(10*r['rms_1_6']/g,2),fdom=round(fa[Aa.argmax()],2),Adom_mm=round(10*Aa.max()/g,2)))
for r in rows: print(r)
json.dump(rows,open(D/'summary.json','w'),indent=1)
