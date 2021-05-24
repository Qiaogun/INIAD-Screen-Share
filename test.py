
import cv2
import time
print(cv2.__version__)

print(chr(27))

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FPS, 30)
pTime = 0
cTime = 0
fps = int(cap.get(5))
print("fps:", fps)
while True:
    x,img = cap.read()
    # print(x)
    img = cv2.flip(img, 1)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv2.imshow("FMS", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()


