import time
from collections import deque
import numpy as np

# please tune in the values if needed cause it would take a while to adjust these
# weights
W_BLINK = 0.34
W_FACE_JIT = 0.33
W_PITCH_JIT = 0.33




Z_MAX = 3.0
SMOOTH_WIN = 1.0
MIN_STD =1e-6


class Calib:
    def __init__(self, secs=15.0):
        self.secs = secs
        self.start = None
        self.samples = {"blink": [], "face_jit": [], "pitch_jit": []}
        self.done = False
        self.base = {}
    def add(self, blink, face_jit, pitch_jit, voiced):
        if self.start is None:
            self.start = time.time()
        
        self.samples["blink"].append(blink)
        self.samples["face_jit"].append(face_jit)
        if voiced:
            self.samples["pitch_jit"].append(pitch_jit)

        if time.time() - self.start >= self.secs:
            self._finish()
        return self.done
    



    def left(self):
        if self.start is None:
            return self.secs
        return max(0.0, self.secs - (time.time() - self.start))
    def _finish(self):
        for k, vals in self.samples.items():
            if len(vals)<2:
                self.base[k] = (0.0, 0.0)
            else:
                self.base[k] = (float(np.mean(vals)), float(np.std(vals)))
        self.done= True

def normalise(x, base):
    # ahh mathsssssss
    mean, std = base
    if std < MIN_STD:
        return 0.0
    z = (x - mean)/std
    if z <0:
        return 0.0
    if z > Z_MAX:
        return Z_MAX
    return z




class Fusion:
    def __init__(self, base):
        self.base = base
        self.hist = deque()
        self.parts = {}
    
    def update(self,blink, f, p, voiced):
        nb = normalise(blink,self.base["blink"])
        nf = normalise(f, self.base["face_jit"])
        np_ = normalise(p, self.base["pitch_jit"])

        if voiced:
            wsum = W_BLINK+W_FACE_JIT+W_PITCH_JIT
            total = (W_BLINK*nb) + (W_FACE_JIT*nf) + (W_PITCH_JIT*np_)
        else:
            wsum = W_BLINK + W_FACE_JIT
            total = (W_BLINK*nb) + (W_FACE_JIT*nf)
        z = total /wsum if wsum>0 else 0.0
        r = 100.0 * (z/Z_MAX)
        r = max(0.0, min(100.0, r))

        self.parts= {
            "blink": W_BLINK*nb / wsum * 100.0 / Z_MAX if wsum else 0.0,
            "face_jit": W_FACE_JIT *nf / wsum * 100.0 / Z_MAX if wsum else 0.0,
            "pitch_jit": (W_PITCH_JIT*np_/wsum * 100.0/ Z_MAX) if ( wsum and voiced) else 0.0,
            "z_blink": nb,
            "z_face": nf,
            "z_pitch": np_ if voiced else None,
        }


        now = time.time()
        self.hist.append((now, r))
        while self.hist and now - self.hist[0][0] > SMOOTH_WIN:
            self.hist.popleft()
        return float(np.mean([s for _, s in self.hist]))
    
if __name__ == "__main__":
    print("smth is going on gng")
    base = { "blink" : (20.0, 4.0),
            "face_jit": (0.01, 0.002),
            "pitch_jit": (0.02, 0.005),
            }
    f = Fusion(base)
    print("baseline.  ->", round(f.update(20.0, 0.01, 0.02, True), 1), "(want ~0)")
    f = Fusion(base)
    print("calmer.  ->", round(f.update(10.0, 0.005, 0.01, True), 1), "(want 0 but not neg)")
    f = Fusion(base)
    print("1 sig up.  ->", round(f.update(24.0, 0.012, 0.025, True), 1), "(want ~33)")
    f = Fusion(base)
    print("3 sig up.  ->", round(f.update(32.0, 0.016, 0.035, True), 1), "(want 100)")
    f = Fusion(base)
    print("cooked.  ->", round(f.update(99.0, 0.9, 0.9, True), 1), "(want 100 MAX)")
    


    print("lets try smth else")
    f = Fusion(base)
    a = f.update(24.0, 0.012, 0.025, True)
    f = Fusion(base)
    b = f.update(24.0, 0.012, 0.025, False)
    print("talking 1 sig ", round(a,1))
    print("silent 1 sig ", round(b,1))