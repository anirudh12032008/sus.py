import time
import cv2 
from face import FaceSig



def main():
    c = cv2.VideoCapture(0)
    if not c.isOpened():
        print("CAMERA NOT OK")
        return
    face = FaceSig()

    t0 = time.time()
    score = 0.0

    while True:
        ok, f = c.read()
        if not ok:
            print("FRAME NOT OK")
            break
        ms = int((time.time() -t0) *1000)
        f = face.process(f, ms)









        cv2.imshow("sus", f)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
    c.release()
    cv2.destroyAllWindows()
    face.close()

if __name__ == "__main__":
    main()