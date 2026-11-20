import cv2,numpy as np,sys
v=sys.argv[1]; cap=cv2.VideoCapture(v); fps=cap.get(5); prev=None; cuts=[0.0]; i=0
while True:
    ok=cap.grab()
    if not ok: break
    if i%2==0:
        ok,im=cap.retrieve(); g=cv2.cvtColor(cv2.resize(im,(160,90)),cv2.COLOR_BGR2GRAY).astype(float)
        if prev is not None and np.mean(np.abs(g-prev))>22: cuts.append(i/fps)
        prev=g
    i+=1
cuts.append(i/fps); c=[]
for x in cuts:
    if not c or x-c[-1]>0.5: c.append(x)
shots=[(a,b) for a,b in zip(c[:-1],c[1:]) if b-a>=8]
print(v.split('/')[-1][:20],'long shots:',[(round(a,1),round(b,1)) for a,b in shots])
