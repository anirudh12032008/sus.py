import time
import cv2
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
MIN_BLINK = 1
MAX_BLINK = 12
BLINK_WIN = 10.0
JITTER_WIN = 2.0



class FaceSig:
    def __init__(self):
        opts = vision.FaceLandmarkerOptions(
    base_options=mptask.BaseOptions(model_asset_path=MODEL_P),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
        self.lm = vision.FaceLandmarker.create_from_options(opts)

        self.blinks =0
        self.closedf = 0
        self.jscore = 0.0
        self.jitters = deque()
        self.start = time.time()
        self.blinkt = deque()
        self.prevpts = None
    def blink_rate(self, now):
        while self.blinkt and now - self.blinkt[0] > BLINK_WIN:
            self.blinkt.popleft()
        secs = min(now - self.start, BLINK_WIN)
        if secs < 1.0:
            return 0.0
        return len(self.blinkt) * (60.0/secs)
    

    def do_blinks(self, ear, now):
        if ear < BLINK_EAR:
            self.closedf += 1
        else:
            if MIN_BLINK <= self.closedf <= MAX_BLINK:
                self.blinks += 1
                self.blinkt.append(now)
            self.closedf = 0
    
    def do_jitter(self, pix, now):
        # holy interesting part
        reg = pix[list(JITTERP)]
        iod = np.linalg.norm(pix[IODP[0]] - pix[IODP[1]])

        if self.prevpts is not None and iod > 1e-6:
            d = reg - self.prevpts
            d = d - d.mean(axis=0)
            jit = np.linalg.norm(d, axis=1).mean()/iod
            self.jitters.append((now, jit))

        self.prevpts=reg
        while self.jitters and now - self.jitters[0][0]> JITTER_WIN:
            self.jitters.popleft()
        
        if self.jitters:
            self.jscore = float(np.mean([v for _, v in self.jitters]))
        else:
            self.jscore=0.0

    











    def process(self, f, ms):
        h, w = f.shape[:2]
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.lm.detect_for_video(img, ms)
        now = time.time()

        if not res.face_landmarks:
            self.prevpts=None
            return{ 
                "blink_rate": self.blink_rate(now),
                "jitter_score": self.jscore,
                "ear": None,
                "face": False,
                "pts": None,
            }
        pts = res.face_landmarks[0]
        pix = topix(pts, w, h)
        ear = (eye_ratio(pix[list(REYE )]) + eye_ratio(pix[list(LEYE)])) /2.0

        self.do_blinks(ear, now)
        self.do_jitter(pix, now)

        return{
                "blink_rate": self.blink_rate(now),
                "jitter_score": self.jscore,
                "ear": ear,
                "face": True,
                "pts": pts,        }
    def close(self):
        self.lm.close()
        



def eye_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    vert = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
    hor = np.linalg.norm(p1-p4)
    if hor < 1e-6:
        return 0.0
    return vert / (2.0*hor)


def topix(landmarks, w, h):
    return np.array([[lm.x*w, lm.y *h] for lm in landmarks])
    





if __name__=="__main__":

    c = cv2.VideoCapture(0)

    if not c.isOpened():
        print("LHDFS")
        exit()
    face = FaceSig()
    print("q to quit")
    t0 = time.time()
    fps=0.0
    lastt = time.time()


    while True:
        ok, f = c.read()
        if not ok:
            print("not ok")
            break
        

        ms = int((time.time()-t0)*1000)
        out = face.process(f, ms)
        
        now=time.time()
        gap = now - lastt
        lastt = now
        if gap >0:
            fps = ( fps * 0.9) + ((1.0/gap) * 0.1)
        h, w = f.shape[:2]
        if out["face"]:
            # drawing time
            for p in out["pts"]:
                x = int(p.x *w)
                y = int(p.y * h)
                cv2.circle(f, (x,y), 1, (0, 255, 0), -1)
            col = (0,0, 255) if out["ear"] < BLINK_EAR else (0,255,0)
            cv2.putText(f, "EAR %.3f" % out["ear"], (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            # cv2.putText(f, "EAR %.3f" % ear, (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(f, "blinks %d" % face.blinks, (20,80), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(f, "blinks/min %.1f" % out["blink_rate"], (20,120), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.putText(f, "jitter %.4f" % out["jitter_score"], (20,160), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(f, "noface", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            prevpts = None
        cv2.putText(f, "fps %.0f" % fps, (20,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow("sus", f)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
    c.release()
    cv2.destroyAllWindows()
    face.close()