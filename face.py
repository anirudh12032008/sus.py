import time
from collections import deque
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_P = "models.face_landmarker.task"

REYE = (33, 160, 158, 133, 153, 144)
LEYE = (362, 385, 387, 263, 373, 380)
MOUTHP = (61, 291, 78, 308)
BROWP = (70, 63, 105, 66, 107, 300, 293, 3334, 296, 336)
JITTERP = MOUTHP + BROWP
IODP = (33, 263)
BLINK_EAR = 0.21
MIN_BLINK = 1
MAX_BLINK = 12
BLINK_WIN = 10.0
JITTER_WIN = 2.0


def eye_ration(pts):
    p1, p2, p3, p4, p5, p6 = pts
    vert = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
    hor = np.linalg.norm(p1-p4)
    if hor < 1e-6:
        return 0.0
    return vert / (2.0*hor)

class FaceSignals:
    def __init__(self, model=MODEL_P):
        base = mp_python.BaseOptions(model_asset_path=model)
        self._landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options = base,
                running_mode = vision.RunningMode.VIDEO,
                num_faces = 1,
            )
        )
        self._blinks = deque()
        self._jitters = deque()
        self._closedf = 0
        self._prevpts = None
        self._start = time.monotonic()

    def _topix(self, landmarks, w, h):
        return np.array([[lm.x*w, lm.y *h] for lm in landmarks], dtype=np.float64)
    
    def _uptblinks(self, ear, now):
        if ear < BLINK_EAR:
            self._closedf += 1
        else:
            if MIN_BLINK <= self._closedf <= MAX_BLINK:
                self._blinks.append(now)
            self._closedf=0
        while self._blinks and now - self._blinks[0] > BLINK_WIN:
            self._blinks.popleft()
    def _blinkrate(self, now):
        elp = min(now - self._start, BLINK_WIN)
        if elp < 1.0:
            return 0.0
        return len(self._blinks) * (60.0/elp)