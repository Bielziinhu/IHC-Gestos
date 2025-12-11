import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

pyautogui.FAILSAFE = False

wCam, hCam = 640, 480
frame_r = 5
smooth = 4

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

wScr, hScr = pyautogui.size()

pX, pY = 0, 0

left_down = False
dragging = False
last_right = 0

pinch_prev_state = False
pinch_last_open_time = 0
DOUBLE_CLICK_WINDOW = 0.25

last_scroll_y = None
scroll_cooldown = 0


def distancia(a, b, lm):
    return math.hypot(lm[a][1] - lm[b][1], lm[a][2] - lm[b][2])


def dedo_levantado(lm, tip, pip):
    return lm[tip][2] < lm[pip][2]


while True:
    ok, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handlm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handlm, mp.solutions.hands.HAND_CONNECTIONS)

            lm = []
            h, w, _ = img.shape
            for id, pt in enumerate(handlm.landmark):
                lm.append([id, int(pt.x * w), int(pt.y * h)])

            ind_up = dedo_levantado(lm, 8, 6)
            med_up = dedo_levantado(lm, 12, 10)
            ane_up = dedo_levantado(lm, 16, 14)
            min_up = dedo_levantado(lm, 20, 18)

            if ind_up and not med_up and not ane_up and not min_up:
                x, y = lm[8][1], lm[8][2]
                xMapped = np.interp(x, (frame_r, wCam - frame_r), (0, wScr))
                yMapped = np.interp(y, (frame_r, hCam - frame_r), (0, hScr))

                cX = pX + (xMapped - pX) / smooth
                cY = pY + (yMapped - pY) / smooth

                pyautogui.moveTo(cX, cY)
                pX, pY = cX, cY

            pinch = distancia(4, 8, lm)
            pinch_closed = pinch < 28

            if pinch_closed:
                now = time.time()

                if not pinch_prev_state:
                    if now - pinch_last_open_time <= DOUBLE_CLICK_WINDOW:
                        pyautogui.doubleClick()
                    else:
                        pyautogui.mouseDown()
                        left_down = True
                        dragging = True

                pinch_prev_state = True

            else:
                if pinch_prev_state:
                    pinch_last_open_time = time.time()

                    if left_down:
                        pyautogui.mouseUp()
                        left_down = False
                        dragging = False

                pinch_prev_state = False

            is_v = ind_up and med_up and not ane_up and not min_up

            if is_v and time.time() - last_right > 0.4:
                pyautogui.rightClick()
                last_right = time.time()

            is_open = ind_up and med_up and ane_up and min_up

            if is_open:
                y = lm[9][2]

                if last_scroll_y is None:
                    last_scroll_y = y

                dy = y - last_scroll_y
                last_scroll_y = y

                if abs(dy) > 12 and time.time() - scroll_cooldown > 0.05:
                    if dy > 0:
                        pyautogui.scroll(-80)
                    else:
                        pyautogui.scroll(80)

                    scroll_cooldown = time.time()
            else:
                last_scroll_y = None

    cv2.imshow("Mouse por Gestos", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
