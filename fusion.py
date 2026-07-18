import time
from collections import deque
import numpy as np

# please tune in the values if needed cause it would take a while to adjust these

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

