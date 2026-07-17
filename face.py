import time
from collections import deque
import numpy as np
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mptask
from mediapipe.tasks.python import vision

MODEL_P = "models/m.task"

REYE = (33, 160, 158, 133, 153, 144)
LEYE = (362, 385, 387, 263, 373, 380)
MOUTHP = (61, 291, 78, 308)
BROWP = (70, 63, 105, 66, 107, 300, 293, 334, 296, 336)
JITTERP = MOUTHP + BROWP
IODP = (33, 263)
BLINK_EAR = 0.21
MIN_BLINK = 2
MAX_BLINK = 12
BLINK_WIN = 10.0
JITTER_WIN = 2.0


def eye_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    vert = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
    hor = np.linalg.norm(p1-p4)
    if hor < 1e-6:
        return 0.0
    return vert / (2.0*hor)


def topix(landmarks, w, h):
    return np.array([[lm.x*w, lm.y *h] for lm in landmarks])
    

opts = vision.FaceLandmarkerOptions(
    base_options=mptask.BaseOptions(model_asset_path=MODEL_P),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)

lm = vision.FaceLandmarker.create_from_options(opts)

import cv2
c = cv2.VideoCapture(0)

if not c.isOpened():
    print("LHDFS")
    exit()
print("q to quit")
t0 = time.time()
lowest = 999.0
t0 = time.time()
blinks =0
closedf = 0
fps = 0.0
lastt = time.time()
blinkt = deque()


prevpts = None
jitters = deque()
jscore = 0.0

while True:
    ok, f = c.read()
    if not ok:
        print("not ok")
        break
    h, w = f.shape[:2]
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ms = int((time.time() -t0)*1000)
    res = lm.detect_for_video(img, ms)

    now = time.time()
    gap = now - lastt
    lastt = now
    if gap >0:
        fps = ( fps * 0.9) + ((1.0/gap) * 0.1)

    if res.face_landmarks:
        pts = res.face_landmarks[0]
        pix = topix(pts, w, h)
        ear = (eye_ratio(pix[list(REYE )]) + eye_ratio(pix[list(LEYE)])) /2.0
        if ear < BLINK_EAR:
            closedf += 1
        else:
            if MIN_BLINK <= closedf <= MAX_BLINK:
                blinks += 1
                blinkt.append(now)
                print("%d for %d frames" % (blinks, closedf)) 
            closedf = 0


        while blinkt and now - blinkt[0] > BLINK_WIN:
            blinkt.popleft()
        secs = min(now - t0, BLINK_WIN)
        if secs < 1.0:
            rate = 0.0
        else:
            rate = len(blinkt) * (60.0/secs)

        











        # holy interesting part
        reg = pix[list(JITTERP)]
        iod = np.linalg.norm(pix[IODP[0]] - pix[IODP[1]])

        if prevpts is not None and iod > 1e-6:
            d = reg - prevpts
            d = d - d.mean(axis=0)
            jit = np.linalg.norm(d, axis=1).mean()/iod
            jitters.append((now, jit))

        prevpts=reg
        while jitters and now - jitters[0][0]> JITTER_WIN:
            jitters.popleft()
        
        if jitters:
            jscore = float(np.mean([v for _, v in jitters]))
        else:
            jscore=0.0
        


        
        # drawing time
        for p in pts:
            x = int(p.x *w)
            y = int(p.y * h)
            cv2.circle(f, (x,y), 1, (0, 255, 0), -1)
        col = (0,0, 255) if ear < BLINK_EAR else (0,255,0)
        cv2.putText(f, "EAR %.3f" % ear, (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        # cv2.putText(f, "EAR %.3f" % ear, (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.putText(f, "blinks %d" % blinks, (20,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.putText(f, "blinks/min %.1f" % rate, (20,160), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.putText(f, "jitter %.4f" % jscore, (20,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(f, "noface", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        prevpts = None
    cv2.putText(f, "fps %.0f" % fps, (20,120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow("sus", f)
    k = cv2.waitKey(1)
    
    if k == ord("q"):
        break
    if k == ord("r"):
        lowest = 999.0
c.release()
cv2.destroyAllWindows()
lm.close()