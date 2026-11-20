"""Phase-correlation measurement of a rider's vertical head motion in one shot.
Usage: python rider_phasecorr.py VIDEO T0 T1 X,Y,W,H INJECT_CM INJECT_HZ TAG
  X,Y,W,H: initial face box in pixels; INJECT_CM=0 for the measurement, e.g. 0.1 for the injection test (run after the 0 run).
Outputs go to data/riders/."""
import cv2,numpy as np,sys,json
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'data'/'riders'
vid,t0,t1=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]); box=tuple(int(x) for x in sys.argv[4].split(','))
inj_cm=float(sys.argv[5]); fin=float(sys.argv[6]); tag=sys.argv[7]
cap=cv2.VideoCapture(vid); cap.set(cv2.CAP_PROP_POS_MSEC,t0*1000); fps=cap.get(5)
ok,im=cap.read(); H,W=im.shape[:2]
# overlay masks
ov=np.ones((H,W),np.uint8)*255
if W>1500: ov[:100,:]=0; ov[990:,:]=0; ov[:260,1500:]=0
else: ov[420:,820:]=0; ov[:30,:]=0
cmpp0=20.0/box[3]; amp_px=inj_cm/cmpp0
pos=[ (box[0]+box[2]/2, box[1]+box[3]/2, box[3]) ]
def inject(img,tt,c):
    if inj_cm==0: return img
    fx,fy,fh=c; d=amp_px*np.sin(2*np.pi*fin*tt)
    x0=int(max(fx-1.6*fh,0)); x1=int(min(fx+1.6*fh,W)); y0=int(max(fy-1.6*fh,0)); y1=int(min(fy+2.2*fh,H))
    p=img[y0:y1,x0:x1].astype(np.float32)
    sh=cv2.warpAffine(p,np.float32([[1,0,0],[0,1,-d]]),(p.shape[1],p.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    h,w=p.shape[:2]; yy,xx=np.mgrid[0:h,0:w]; m=np.clip(np.minimum.reduce([xx,w-1-xx,yy,h-1-yy]).astype(np.float32)/25,0,1)[...,None]
    o=img.copy(); o[y0:y1,x0:x1]=(m*sh+(1-m)*p).astype(np.uint8); return o
t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
im=inject(im,t,pos[0])
tr=cv2.TrackerCSRT_create(cv2.TrackerCSRT_Params()); tr.init(im,box)
prev=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); prevf=prev.astype(np.float32)
c=pos[0]; rows=[]; Y=0.0; lost=0
import os
TR=np.loadtxt(OUT/f'facebox_{tag.rstrip("i")}.csv',delimiter=',',skiprows=1) if inj_cm>0 else None; k=0
rows_c=[]
while True:
    t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    ok,im=cap.read()
    if not ok or t>t1: break
    im=inject(im,t,c)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); gf=g.astype(np.float32)
    # camera transform prev->curr from features outside face region
    if TR is not None: c=tuple(TR[min(k,len(TR)-1)])
    k+=1
    rows_c.append(c)
    m=ov.copy(); fx,fy,fh=c; cv2.rectangle(m,(int(fx-2.2*fh),int(fy-2.0*fh)),(int(fx+2.2*fh),H),0,-1)
    cv2.setRNGSeed(k)
    p0=cv2.goodFeaturesToTrack(prev,400,0.01,10,mask=m); A=None
    if p0 is not None and len(p0)>=10:
        p1,st,_=cv2.calcOpticalFlowPyrLK(prev,g,p0,None); s=st.ravel()==1
        if s.sum()>=10: A,inl=cv2.estimateAffinePartial2D(p0[s],p1[s],method=cv2.RANSAC,ransacReprojThreshold=2.0)
    if A is None: A=np.array([[1,0,0],[0,1,0]],float); lost+=1
    camdy=(A@np.array([fx,fy,1.0]))[1]-fy
    # face shift by phase correlation, window at previous face position
    s_=int(fh*0.55); x=int(max(0,min(fx-s_,W-2*s_))); y=int(max(0,min(fy-s_,H-2*s_)))
    a=prevf[y:y+2*s_,x:x+2*s_]; b=gf[y:y+2*s_,x:x+2*s_]
    (dx,dy),resp=cv2.phaseCorrelate(a,b,cv2.createHanningWindow((a.shape[1],a.shape[0]),cv2.CV_32F))
    if TR is None:
        ok2,bb=tr.update(im)
        if ok2: c=(bb[0]+bb[2]/2,bb[1]+bb[3]/2,bb[3])
    Y+= -(dy-camdy)*cmpp0     # cm, up positive
    rows.append([t,Y,resp,c[2],c[0],c[1],dy,camdy]); prev=g; prevf=gf
r=np.array(rows)
if inj_cm==0: np.savetxt(OUT/f'facebox_{tag}.csv',np.array(rows_c),delimiter=',',header='cx_px,cy_px,h_px',comments='',fmt='%.2f')
tt=r[:,0]; from scipy.signal import butter as _b,filtfilt as _f
_bb,_aa=_b(3,0.8/(fps/2),'high'); Yv=_f(_bb,_aa,r[:,1]-r[0,1])
def amp(f):
    X=np.column_stack([np.cos(2*np.pi*f*tt),np.sin(2*np.pi*f*tt),np.ones_like(tt)]); cc,*_=np.linalg.lstsq(X,Yv,rcond=None); return np.hypot(cc[0],cc[1])
fr=np.arange(1.3,3.45,0.01); A_=np.array([amp(f) for f in fr])
rng=np.random.default_rng(2); noise=np.array([amp(f) for f in rng.uniform(1.3,6,300)])
from scipy.signal import butter,filtfilt
b_,a_=butter(3,[1/(fps/2),6/(fps/2)],'band'); rms=np.std(filtfilt(b_,a_,Yv))
res=dict(tag=tag,dur=float(tt[-1]-tt[0]),inj=inj_cm,amp_at_inj=float(amp(fin)) if inj_cm else None,
 max_amp_13_34=float(A_.max()),f_at_max=float(fr[A_.argmax()]),noise_med=float(np.median(noise)),noise_p95=float(np.percentile(noise,95)),
 rms_1_6=float(rms),resp_med=float(np.median(r[:,2])),cam_fail=lost,box_h_med=float(np.median(r[:,3])))
print(json.dumps(res)); open(OUT/f'res_{tag}.json','w').write(json.dumps(res))
np.savetxt(OUT/f'head_{tag}.csv',np.column_stack([tt,Yv]),delimiter=',',header='t_s,head_vertical_cm',comments='',fmt='%.5f')
