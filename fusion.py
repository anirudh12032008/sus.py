import time
from collections import deque
import numpy as np

W_BLINK = 0.34
W_FACE_JIT = 0.33
W_PITCH_JIT = 0.33
Z_MAX = 3.0
SMOOTH_WIN = 1.0