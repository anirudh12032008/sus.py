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

