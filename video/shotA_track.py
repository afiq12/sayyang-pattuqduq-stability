import cv2,numpy as np,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cap=cv2.VideoCapture(str(ROOT/'videos'/'V1_1080p.mp4')); cap.set(cv2.CAP_PROP_POS_MSEC,208.7*1000)
ok,im=cap.read()
boxes={'face':(415,205,150,175)}
trk={}
for k,b in boxes.items():
    p=cv2.TrackerCSRT_Params(); t=cv2.TrackerCSRT_create(p); t.init(im,b); trk[k]=t
rows=[]; n=0; snaps=[]
while True:
    tt=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    ok,im=cap.read()
    if not ok or tt>256.0: break
    r=[tt]
    for k in ('face',):
        ok2,b=trk[k].update(im)
        r+= [b[0]+b[2]/2,b[1]+b[3]/2,b[3],int(ok2)] if ok2 else [np.nan]*3+[0]
    rows.append(r)
    n+=1
with open(ROOT/'data'/'shotA'/'face_track.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['t','fx','fy','fh','fok']); w.writerows(rows)
print(len(rows))
