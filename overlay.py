import time
from collections import deque
import cv2
import numpy as np

G_SECS = 30.0
G_H = 150
WARN_H = 26


# I need to tune in this laater
SPIKE = 60.0

WARN = "FOR FUN - NOT 100% ACCURATE"

class Graph:
    def __init__(self):
        self.hist = deque()
    def add(self, score):
        now = time.time()
        self.hist.append((now, score))
        while self.hist and now - self.hist[0][0] > G_SECS:
            self.hist.popleft()
    def draw(self, f):
        h, w = f.shape[:2]
        b = h - WARN_H
        t = b - G_H

        panel = f[t:b, 0:w]
        black = np.zeros_like(panel)
        cv2.addWeighted(black, 0.6, panel, 0.4, 0, panel)
        for pct in (0,25,50,75,100):
            y = int(b - (pct/100.0)*G_H)
            cv2.line(f, (0,y),(w,y), (40,40,40), 1)
            cv2.putText(f, str(pct), (4, y-3),
                        cv2.FONT_HERSHEY_DUPLEX, 0.35, (90,90,90), 1)
        ys = int(b-(SPIKE/100.0) * G_H)
        cv2.line(f, (0, ys), (w, ys), (0,0,120), 1)
        if len(self.hist)<2:
            return
        now = time.time()










        pts = []
        for t,s in self.hist:
            age = now - t
            x = int(w - (age/G_SECS)*w)
            y = int(b- (s/100.0) * G_H)
            pts.append((x,y))

        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(f, [arr], False, (0,255,100), 2)
        cv2.circle(f, pts[-1], 4, (255, 255, 255), -1)

def draw(f, score, parts, voiced, calib=None):
    h, w = f.shape[:2]
    if calib is not None:
        cv2.rectangle(f, (0,0), (w,110), (0,0,0), -1)
        cv2.putText(f, "CALIBRATING %.0fs" % calib, (20, 45), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(f, "act normal + talk normal", (20, 85), cv2.FONT_HERSHEY_COMPLEX, 0.8, (200, 200, 200), 2)
        _warn(f)
        return f
    
    spiking = score >=SPIKE
    col = (0, 0, 255) if spiking else (0, 255, 100)
    cv2.putText(f, "%.0f" % score, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 2.0, col, 4)
    cv2.putText(f, "stress_score", (20,95), cv2.FONT_HERSHEY_COMPLEX, 0.5, (150, 150, 150), 1)
    if spiking:
        cv2.rectangle(f, (0,0), (w -1, h-1), (0,0,255), 12)
        cv2.putText(f, "SUS", (w-160, 70), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0,255), 4)
    # find the culprit
    if parts:
        y = 130
        for name in ("blink", "face_jit", "pitch_jit"):
            v = parts.get(name, 0.0)
            cv2.putText(f, "%-9s %5.1f" % (name, v), (20,y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (180,180,180), 1)
            cv2.rectangle(f, (140, y-8), (140 + int(v*1.5), y-2), (0,200,255), -1)
            y += 22
    if not voiced:
        cv2.putText(f, "no voice ", (20, y+5), cv2.FONT_HERSHEY_COMPLEX, 0.45, (100, 100, 100), 1)
    _warn(f)
    return f












