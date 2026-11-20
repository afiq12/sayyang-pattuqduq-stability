import cv2,numpy as np,csv,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VID=str(ROOT/'videos'/'V1_1080p.mp4'); T0,T1=221.0,256.0
amp_cm=float(sys.argv[1]); fin=float(sys.argv[2]); tag=sys.argv[3]
base=np.genfromtxt(str(ROOT/'data'/'shotA'/'face_track.csv'),delimiter=',',names=True)
cmpp=20/np.median(base['fh'])
amp_px=amp_cm/cmpp
cap=cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_MSEC,T0*1000)
ok,im=cap.read(); t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
def bbox(tt):
    k=np.argmin(abs(base['t']-tt)); return base['fx'][k],base['fy'][k],base['fh'][k]
def inj(im,tt):
    if amp_cm==0: return im
    fx,fy,fh=bbox(tt); d=amp_px*np.sin(2*np.pi*fin*tt)
    x0=int(max(fx-1.6*fh,0)); x1=int(min(fx+1.6*fh,im.shape[1])); y0=int(max(fy-1.6*fh,0)); y1=int(min(fy+2.2*fh,im.shape[0]))
    patch=im[y0:y1,x0:x1].astype(np.float32)
    M=np.float32([[1,0,0],[0,1,-d]])   # move up when d>0 (image y down)
    sh=cv2.warpAffine(patch,M,(patch.shape[1],patch.shape[0]),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    # feathered mask so edges blend
    h,w=patch.shape[:2]; yy,xx=np.mgrid[0:h,0:w]
    m=np.minimum.reduce([xx,w-1-xx,yy,h-1-yy]).astype(np.float32); m=np.clip(m/25.0,0,1)[...,None]
    out=im.copy(); out[y0:y1,x0:x1]=(m*sh+(1-m)*patch).astype(np.uint8); return out
im=inj(im,t)
fx,fy,fh=bbox(t)
tr=cv2.TrackerCSRT_create(cv2.TrackerCSRT_Params()); tr.init(im,(int(fx-75),int(fy-fh/2),150,int(fh)))
prev=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); M=np.eye(3); rows=[]
def mask_for(fx,fy,fh):
    m=np.zeros(prev.shape,np.uint8); m[90:390,:]=255; m[:,1680:]=0
    cv2.rectangle(m,(int(fx-180),int(fy-fh)),(int(fx+180),int(fy+2*fh)),0,-1); return m
while True:
    t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    ok,im=cap.read()
    if not ok or t>T1: break
    im=inj(im,t); g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ok2,b=tr.update(im); cx,cy,h=b[0]+b[2]/2,b[1]+b[3]/2,b[3]
    p0=cv2.goodFeaturesToTrack(prev,300,0.01,15,mask=mask_for(cx,cy,h)); A=None
    if p0 is not None and len(p0)>=8:
        p1,st,_=cv2.calcOpticalFlowPyrLK(prev,g,p0,None); s=st.ravel()==1
        if s.sum()>=8: A,_=cv2.estimateAffinePartial2D(p1[s],p0[s],method=cv2.RANSAC,ransacReprojThreshold=3.0)
    if A is None: A=np.array([[1,0,0],[0,1,0]],float)
    M=M@np.vstack([A,[0,0,1]]); pt=M@np.array([cx,cy,1.0])
    rows.append([t,pt[1],h*np.sqrt(abs(np.linalg.det(M[:2,:2])))]); prev=g
r=np.array(rows); tt=r[:,0]; Y=-r[:,1]*20/np.median(r[:,2])
Y=Y-np.polyval(np.polyfit(tt,Y,3),tt)
def amp(f):
    X=np.column_stack([np.cos(2*np.pi*f*tt),np.sin(2*np.pi*f*tt),np.ones_like(tt)]); c,*_=np.linalg.lstsq(X,Y,rcond=None); return np.hypot(c[0],c[1])
rng=np.random.default_rng(1); noise=[amp(f) for f in rng.uniform(1.5,6,300) if abs(f-fin)>0.1]
res=dict(tag=tag,amp_in=amp_cm,f=fin,amp_out=amp(fin),noise_med=float(np.median(noise)),noise_p95=float(np.percentile(noise,95)))
print(json.dumps(res)); open(f'inj_{tag}.json','w').write(json.dumps(res))
