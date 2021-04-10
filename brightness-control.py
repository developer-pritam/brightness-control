import cv2
import mediapipe as mp
import time
import numpy as np
import math
import screen_brightness_control as sb 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
 
pTime = 0
cTime = 0

minBright = 1
maxBright = 100
bright = 0
BrightBar = 350
BrightPer = 0
 
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    lmList = []
 
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])     
            if len(lmList) !=0:
                #print(lmList[4])
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                length = math.hypot(x2 - x1, y2 - y1)

                if length < 50:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                bright = np.interp(length, [50, 180], [minBright, maxBright])
                BrightBar = np.interp(length, [50, 180], [350, 200])
                BrightPer = np.interp(length, [50, 180], [0, 100])
                print(int(length), bright)
                sb.set_brightness(int(bright))

                #print(length)
 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cv2.rectangle(img, (50, 200), (85, 350), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(BrightBar)), (85, 350), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(BrightPer)} %', (50, 390), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime) 
    pTime = cTime
 
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 255), 2)
    cv2.putText(img, 'Pritam Kumar' , (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 0), 2)
 
    cv2.imshow("Pritam", img)
    cv2.waitKey(1)
