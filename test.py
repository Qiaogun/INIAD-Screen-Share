from os import path
import cv2
import time
import mediapipe as mp
import pandas as pd

mpDraw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

video = 'D:\\handpose\\zoom\\2021-06-30_05-22-05.mp4'

import cv2
for video in vidoefilepath_list:
    cap = cv2.VideoCapture(video)
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                keypoints = []
                for handLms in results.multi_hand_landmarks:
                    for _, data_point in enumerate(handLms.landmark):
                        keypoints += [data_point.x, data_point.y, data_point.z]
                        mpDraw.draw_landmarks(image, handLms,
                                            mp_hands.HAND_CONNECTIONS)
                        dataframe = pd.DataFrame(keypoints)
                    mpDraw.draw_landmarks(image, handLms,
                                        mp_hands.HAND_CONNECTIONS)
            else:
                keypoints = [0] * 63
            dataframe = pd.DataFrame(keypoints)
            (dataframe.T).to_csv('./data/' + video.split('\\')[2] + '.csv',
                                mode='a',
                                header=False,
                                index=False)
            cv2.imshow('Frame', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()

# if results.multi_hand_landmarks:
#     keypoints = []
#     for handLms in results.multi_hand_landmarks:
#         for _, data_point in enumerate(handLms.landmark):
#             keypoints += [data_point.x, data_point.y, data_point.z]
#             mpDraw.draw_landmarks(frame, handLms,
#                                     mp_hands.HAND_CONNECTIONS)
#             dataframe = pd.DataFrame(keypoints)
#         mpDraw.draw_landmarks(frame, handLms,
#                                 mp_hands.HAND_CONNECTIONS)
# else:
#     keypoints = [0] * 63
# dataframe = pd.DataFrame(keypoints)
# (dataframe.T).to_csv('./data/' + video.split('\\')[2] + '.csv',
#                         mode='a',
#                         header=False,
#                         index=False)
# cv2.imshow('Frame', frame)