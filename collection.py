import cv2
import mediapipe as mp
import time
import pandas as pd

data_list = ['Up_down', 'Click', 'Rock', 'Relax', 'Ok_pose', 'Chinese_seven']
time_list = [16, 15, 15, 15, 15, 15]


def get_data(filename, timepr):

    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  #设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)

    mpDraw = mp.solutions.drawing_utils

    mp_hands = mp.solutions.hands
    # start = time.time()
    counters = 0 
    pTime = 0
    cTime = 0
    frame_count = 0
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.8) as hands:

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #转换为rgb
            results = hands.process(imgRGB)
            if results.multi_hand_landmarks:
                keypoints = []

                for handLms in results.multi_hand_landmarks:
                    for _, data_point in enumerate(handLms.landmark):
                        keypoints += [data_point.x, data_point.y, data_point.z]
                        mpDraw.draw_landmarks(img, handLms,
                                              mp_hands.HAND_CONNECTIONS)
                        dataframe = pd.DataFrame(keypoints)
                    mpDraw.draw_landmarks(img, handLms,
                                          mp_hands.HAND_CONNECTIONS)
            else:
                keypoints = [0] * 63

            dataframe = pd.DataFrame(keypoints)
            (dataframe.T).to_csv('./data/' + filename + '.csv',
                                 mode='a',
                                 header=False,
                                 index=False)
            cTime = time.time()
            fps = round((1 / (cTime - pTime) / 30), 2)
            pTime = cTime
            cv2.putText(img, str(int(frame_count)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 0, 255), 3)
            cv2.imshow("FMS", img)
            # end = time.time()
            # print(frame_count)
            # print(end - start)
            counters += 1
            print(counters)
            frame_count = fps + frame_count
            if (counters) > 450:
                break
            # if (frame_count) > 450:
            #     break
            if cv2.waitKey(2) & 0xFF == 27:
                break
        cap.release()


if __name__ == "__main__":
    for i in range(len(data_list)):
        time.sleep(1)
        print(data_list[i], time_list[i])
        get_data(data_list[i], time_list[i])
        print("Next")

    print("お疲れ様です")
