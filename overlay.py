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

