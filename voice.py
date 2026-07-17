import threading
import time
from collections import deque
import numpy as np
import librosa
import sounddevice as sd


SR = 22050
BUF = 1.0
HOP=0.4
FMIN=80
FMAX=200 # I might change this later
RMS_MIN = 0.01 # I need to change this asw
JITTER_WIN = 3.0







# rms silent is = 0.005
# rms talking is 0.05
# jitter for normal talking is 0.03
# jitter for jittering is also near 0.02


class VoiceSig:
    def __init__(self, sr=SR):
        self.sr = sr
        self.buf = deque(maxlen=int(sr*BUF))
        self.lock = threading.Lock()
        self.running = True
        self.pitch_mean = 0.0
        self.pitch_jitter = 0.0
        self.voiced = False
        self.rms = 0.0
        self.jits = deque()
        junk = np.zeros(int(sr*0.2), dtype=np.float32)
        librosa.pyin(junk, fmin=FMIN, fmax=FMAX, sr=sr)
        self.stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._mic,
        )
        self.stream.start()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
    
    def _mic(self, indata, f, t, stat):
        with self.lock:
            self.buf.extend(indata[:,0])
    def _loop(self):
        while self.running:
            time.sleep(HOP)
            with self.lock:
                if len(self.buf)<int(self.sr * 0.5):
                    continue
                y = np.array(self.buf, dtype=np.float32)
            self._do_pitch(y)
    def _do_pitch(self, y):
        rms = float(np.sqrt(np.mean(y**2)))
        self.rms = rms
        if rms < RMS_MIN:
            self.voiced = False
            return
        f0, vflag, prob = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=self.sr, resolution=0.01, max_transition_rate=100)
        good = f0[~np.isnan(f0)]
        if len(good) <3:
            self.voiced = False
            return
        self.voiced = True
        self.pitch_mean = float(np.mean(good))






        #i hope this works
        steps = np.abs(np.diff(good))
        self.pitch_jitter = float(np.mean(steps)/self.pitch_mean)
        now = time.time()
        self.jits.append((now, self.pitch_jitter))
        while self.jits and now - self.jits[0][0] > JITTER_WIN:
            self.jits.popleft()

    def get(self):
        if self.jits:
            jit = float(np.mean([v for _, v in self.jits]))
        else:
            jit = 0.0
        return {
            "pitch_mean": self.pitch_mean,
            "pitch_jitter": jit,
            "voiced": self.voiced,
            "rms": self.rms,
        }
    def close(self):
        self.running = False
        self.stream.stop()
        self.stream.close()

if __name__ == "__main__":
    print("listening")
    print("warming up")
    v = VoiceSig()
    print("ready")
    try: 
        while True:
            time.sleep(0.3)
            o = v.get()
            s = "TALKING" if o["voiced"] else "quiet "
            print("\r%s rms %0.4f pitch %6.1f hz jitter %.4f" % (s, o["rms"], o["pitch_mean"], o["pitch_jitter"]), end="", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        v.close()
        print("done")