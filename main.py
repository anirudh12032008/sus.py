import time
import cv2 
from face import FaceSig
from fusion import Calib, Fusion
from overlay import Graph, draw

USE_VOICE = True


def main():
    c = cv2.VideoCapture(0)
    if not c.isOpened():
        print("CAMERA NOT OK")
        return
    face = FaceSig()


    voice = None
    if USE_VOICE:
        from voice import VoiceSig
        print("warming up")
        voice = VoiceSig()
    
    calib = Calib(secs=15.0)
    fus = None
    graph =Graph()
    print("q to quit")
    t0 = time.time()
    score = 0.0

    while True:
        ok, f = c.read()
        if not ok:
            print("FRAME NOT OK")
            break

        ms = int((time.time() -t0) *1000)
        x = face.process(f, ms)


        if voice is not None:
            v = voice.get()
        else:
            v= {"pitch_jitter": 0.0, "voiced": False, "pitch_mean": 0.0}
        if not calib.done:
            if x["face"]:
                calib.add(x["blink_rate"], x["jitter_score"], v["pitch_jitter"], v["voiced"])
            draw(f, 0, None, v["voiced"], calib=calib.left())
            if calib.done:
                fus = Fusion(calib.base)
                print("calibrated")
                for k, (m,s ) in calib.base.items():
                    print(" %-10s mean %.4f std %.4f" % (k,m,s))
                    if s < 1e-6:
                        print("did u talk??")
        else:
            score = fus.update(x["blink_rate"], x["jitter_score"], v["pitch_jitter"], v["voiced"])
            graph.add(score)
            draw(f, score, fus.parts, v["voiced"])
            graph.draw(f)

        cv2.imshow("sus", f)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        if k == ord("c"):
            print("recalib")
            calib = Calib(secs=15.0)
            fus = None
            graph = Graph()



    c.release()
    cv2.destroyAllWindows()
    face.close()
    if voice is not None:
        voice.close()

if __name__ == "__main__":
    main()