import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

cam = cv2.VideoCapture(0)
HandM = mp.solutions.hands
hands = HandM.Hands()
draw = mp.solutions.drawing_utils
w_screen,h_screen = pyautogui.size()
prev_x,prev_y=None,None
lastGesture = 0
breakTime = 3
threshold = 65
while True:
    success, frame = cam.read()
    if not success:
        break
    h,w,c = frame.shape
    frame = cv2.flip(frame,1)
    mp_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    res = hands.process(mp_rgb)
    if res.multi_hand_landmarks:
        for landM in res.multi_hand_landmarks:
            draw.draw_landmarks(frame,landM,HandM.HAND_CONNECTIONS)
            landmarks = []
            for id,lm in enumerate(landM.landmark):
                cx,cy=int(lm.x*w),int(lm.y*h)
                landmarks.append((cx,cy))
            indexTip = 8
            ix,iy = landmarks[8]
            if prev_x is None and prev_y is None:
                prev_x,prev_y=ix,iy
            dx = ix - prev_x
            dy = iy - prev_y
            now = time.time()
            if now - lastGesture > breakTime:
                if dx > threshold:
                    pyautogui.press("right")
                    prev_x, prev_y = None, None
                    lastGesture = now
                elif dx < -threshold:
                    pyautogui.press("left")
                    prev_x, prev_y = None, None
                    lastGesture = now
                elif dy > threshold:
                    pyautogui.press("down")
                    prev_x, prev_y = None, None
                    lastGesture = now
                elif dy < -threshold:
                    pyautogui.press("up")
                    prev_x, prev_y = None, None
                    lastGesture = now
    cv2.imshow("Presentation slide controller",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
