import cv2
import mediapipe as mp
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
#from threading import Thread
import time

data_list = ['up_down', 'click', 'press_hold', 'relax', 'circle']
time_list = [15, 15, 10, 10, 10]

# from threading import Thread
# import time
# import threading

# class TestThread(threading.Thread):
#     def __init__(self, name='TestThread'):
#         """ constructor, setting initial variables """
#         self._stopevent = threading.Event()
#         self._sleepperiod = 1.0
#         threading.Thread.__init__(self, name=name)

#     def run(self):
#         """ main control loop """
#         print("%s starts" % (self.getName(),))
#         count = 0
#         while not self._stopevent.isSet():
#             count += 1
#             print("loop %d" % (count,))
#             self._stopevent.wait(self._sleepperiod)
#         print("%s ends" % (self.getName(),))

#     def join(self, timeout=None):
#         """ Stop the thread and wait for it to end. """
#         self._stopevent.set()
#         threading.Thread.join(self, timeout)


def get_data(filename, timepr):
    dataframe = pd.DataFrame()
    cap = cv2.VideoCapture(2,cv2.CAP_DSHOW)
    #width, height = mouse.size()
    #cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  #设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mpDraw = mp.solutions.drawing_utils
    #mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    start = time.time()
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.8) as hands:
        while True:
            success,img = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #转换为rgb
            results = hands.process(imgRGB)
            if results.multi_hand_landmarks:
                #start = time.time()
                keypoints = []
                for handLms in results.multi_hand_landmarks:
                    #hand_landmarks = results.multi_hand_landmarks[0]
                    for _, data_point in enumerate(handLms.landmark):
                        keypoints += [data_point.x, data_point.y, data_point.z]
                        mpDraw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
                        dataframe = pd.DataFrame(keypoints)
            else:
                keypoints = [0] * 63
                

            dataframe = pd.DataFrame(keypoints)
            (dataframe.T).to_csv('./' + filename + '.csv',
                                 mode='a',
                                 header=False,
                                index=False)
            cv2.imshow("FMS", img)
            nowtime = time.time()
            #print(nowtime - start)
            if (nowtime - start) > timepr:
                break
        cap.release()

        # if cv2.waitKey(2) & 0xFF == 27:
        #     cap.release()


if __name__ == "__main__":
    for i in range(len(data_list)):
        print(data_list[i], time_list[i])
        get_data(data_list[i], time_list[i])
        print("Next")

    print("辛苦了，我的宝儿")
