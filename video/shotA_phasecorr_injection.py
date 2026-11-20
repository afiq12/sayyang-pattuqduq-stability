import cv2,numpy as np,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VID=str(ROOT/'videos'/'V1_1080p.mp4'); T0,T1=221.0,256.0
amp_cm=float(sys.argv[1]); fin=float(sys.argv[2]); tag=sys.argv[3]; save=len(sys.argv)>4
base=np.genfromtxt(str(ROOT/'data'/'shotA'/'face_track.csv'),delimiter=',',names=True)
cmpp=20/np.median(base['fh']); amp_px=amp_cm/cmpp
def bbox(tt):
    k=np.argmin(abs(base['t']-tt)); return base['fx'][k],base['fy'][k],base['fh'][k]
def inj(im,tt):
    if amp_cm==0: return im
    fx,fy,fh=bbox(tt); d=amp_px*np.sin(2*np.pi*fin*tt)
    x0=int(max(fx-1.6*fh,0)); x1=int(min(fx+1.6*fh,im.shape[1])); y0=int(max(fy-1.6*fh,0)); y1=int(min(fy+2.2*fh,im.shape[0]))
    patch=im[y0:y1,x0:x1].astype(np.float32)
    sh=cv2.warpAffine(patch,np.float32([[1,0,0],[0,1,-d]]),(patch.shape[1],patch.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    h,w=patch.shape[:2]; yy,xx=np.mgrid[0:h,0:w]
    m=np.clip(np.minimum.reduce([xx,w-1-xx,yy,h-1-yy]).astype(np.float32)/25.0,0,1)[...,None]
    out=im.copy(); out[y0:y1,x0:x1]=(m*sh+(1-m)*patch).astype(np.uint8); return out
cap=cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_MSEC,T0*1000)
ok,im=cap.read(); t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
prev=cv2.cvtColor(inj(im,t),cv2.COLOR_BGR2GRAY).astype(np.float32)
W=None; rows=[]; Yf=0.0; Yb=0.0
def win(fx,fy,fh):
    s=int(fh*0.55); return int(fx-s),int(fy-s),2*s,2*s
while True:
    t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    ok,im=cap.read()
    if not ok or t>T1: break
    g=cv2.cvtColor(inj(im,t),cv2.COLOR_BGR2GRAY).astype(np.float32)
    fx,fy,fh=bbox(t); x,y,w,h=win(fx,fy,fh)
    x=max(0,min(x,g.shape[1]-w)); y=max(0,min(y,g.shape[0]-h))
    a=prev[y:y+h,x:x+w]; b=g[y:y+h,x:x+w]
    han=cv2.createHanningWindow((a.shape[1],a.shape[0]),cv2.CV_32F)
    (dx,dy),resp=cv2.phaseCorrelate(a,b,han)
    # background: band 90-390 rows, exclude face columns, phase-correlate two side blocks
    bs=[]
    for bx in (60, 1400):
        if abs(bx+200-fx)<500: continue
        A_=prev[120:380,bx:bx+400]; B_=g[120:380,bx:bx+400]
        (bdx,bdy),r2=cv2.phaseCorrelate(A_,B_,cv2.createHanningWindow((400,260),cv2.CV_32F)); bs.append((bdy,r2))
    bdy=np.average([b_[0] for b_ in bs],weights=[b_[1] for b_ in bs]) if bs else 0.0
    Yf+=dy; Yb+=bdy
    rows.append([t,Yf,Yb,fh,resp]); prev=g
r=np.array(rows); tt=r[:,0]; cm=20/np.median(r[:,3])
Y=-(r[:,1]-r[:,2])*cm; Y=Y-np.polyval(np.polyfit(tt,Y,3),tt)
if save: np.save(f'pc_{tag}.npy',np.vstack([tt,Y,-r[:,1]*cm,-r[:,2]*cm]))
def amp(f):
    X=np.column_stack([np.cos(2*np.pi*f*tt),np.sin(2*np.pi*f*tt),np.ones_like(tt)]); c,*_=np.linalg.lstsq(X,Y,rcond=None); return np.hypot(c[0],c[1])
rng=np.random.default_rng(1); noise=[amp(f) for f in rng.uniform(1.5,6,300) if abs(f-fin)>0.1]
res=dict(tag=tag,amp_in=amp_cm,f=fin,amp_out=amp(fin),noise_med=float(np.median(noise)),noise_p95=float(np.percentile(noise,95)),resp_med=float(np.median(r[:,4])))
print(json.dumps(res))
