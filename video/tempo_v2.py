"""Live rebana tempo from the audio of video V2 (spectral-flux autocorrelation).
Usage: python tempo_v2.py audio.wav  (mono or stereo WAV extracted from V2, e.g. ffmpeg -i V2.mp4 -ac 1 -ar 22050 a.wav)
10 s windows, 5 s step; a window is kept if the normalized autocorrelation at the beat lag exceeds 0.5."""
import sys, numpy as np, scipy.io.wavfile as w
from scipy.signal import stft
sr,x=w.read(sys.argv[1] if len(sys.argv)>1 else 'a.wav'); x=x.astype(float); x=x.mean(1) if x.ndim>1 else x
def window_tempo(t0,t1):
    seg=x[int(t0*sr):int(t1*sr)]
    f,t,Z=stft(seg,sr,nperseg=1024,noverlap=1024-256); S=np.log1p(np.abs(Z))
    fl=np.maximum(np.diff(S,axis=1),0).sum(0); fl=fl-fl.mean(); hop=256/sr
    ac=np.correlate(fl,fl,'full')[len(fl)-1:]; ac=ac/ac[0]
    lags=np.arange(len(ac))*hop; sel=(lags>60/220)&(lags<60/140)
    idx=np.where(sel)[0]; j=idx[np.argmax(ac[sel])]
    y0,y1,y2=ac[j-1],ac[j],ac[j+1]; d=0.5*(y0-y2)/(y0-2*y1+y2)
    return 60/((j+d)*hop), ac[j]
if __name__=="__main__":
    dur=len(x)/sr; rows=[]
    for t0 in np.arange(0,dur-10,5.0):
        bpm,c=window_tempo(t0,t0+10)
        if c>0.5: rows.append((t0,bpm,c)); print(f"{t0:7.1f}-{t0+10:7.1f} s  {bpm:6.1f} BPM  {bpm/60:5.2f} Hz  r={c:.2f}")
    b=np.array([r[1] for r in rows]); print(f"kept {len(rows)} windows: {b.min():.0f} to {b.max():.0f} BPM, mean {b.mean():.0f}")
