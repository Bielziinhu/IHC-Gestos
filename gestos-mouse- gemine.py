import cv2
import mediapipe as mp
import numpy as np
import time
import math
import ctypes # Chamadas de sistema nativas

# Configurações de constantes do Windows para Mouse
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

# Função para mover o mouse via Sistema Operacional
def move_mouse_native(x, y):
    # O Windows usa uma grade de 0 a 65535 para movimento absoluto
    nx = int(x * 65535 / wScr)
    ny = int(y * 65535 / hScr)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, nx, ny, 0, 0)

def click_native(event_down, event_up=None):
    if event_up:
        ctypes.windll.user32.mouse_event(event_down, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(event_up, 0, 0, 0, 0)
    else:
        ctypes.windll.user32.mouse_event(event_down, 0, 0, 0, 0)

def scroll_native(amount):
    # No Windows, o scroll é multiplicado por 120 (unidade padrão)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, amount, 0)

# --- Configurações iniciais ---
wCam, hCam = 640, 480
frame_r = 40  # Aumentado para dar mais margem de manobra
smooth = 5    # Fator de suavização

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Obtendo resolução da tela de forma nativa
wScr = ctypes.windll.user32.GetSystemMetrics(0)
hScr = ctypes.windll.user32.GetSystemMetrics(1)

pX, pY = 0, 0
pinch_prev_state = False
pinch_last_open_time = 0
last_right = 0
last_scroll_y = None
scroll_cooldown = 0
DOUBLE_CLICK_WINDOW = 0.25

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

            # Identificação de dedos
            ind_up = dedo_levantado(lm, 8, 6)
            med_up = dedo_levantado(lm, 12, 10)
            ane_up = dedo_levantado(lm, 16, 14)
            min_up = dedo_levantado(lm, 20, 18)

            # 1. MOVIMENTO (Apenas indicador levantado)
            if ind_up and not med_up:
                x, y = lm[8][1], lm[8][2]
                xMapped = np.interp(x, (frame_r, wCam - frame_r), (0, wScr))
                yMapped = np.interp(y, (frame_r, hCam - frame_r), (0, hScr))

                cX = pX + (xMapped - pX) / smooth
                cY = pY + (yMapped - pY) / smooth

                move_mouse_native(cX, cY)
                pX, pY = cX, cY

            # 2. CLIQUE ESQUERDO E DRAG (Pinch: Polegar + Indicador)
            pinch = distancia(4, 8, lm)
            pinch_closed = pinch < 30

            if pinch_closed:
                now = time.time()
                if not pinch_prev_state:
                    if now - pinch_last_open_time <= DOUBLE_CLICK_WINDOW:
                        # Double click nativo
                        click_native(MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
                        time.sleep(0.05)
                        click_native(MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
                    else:
                        click_native(MOUSEEVENTF_LEFTDOWN) # Segura o clique
                pinch_prev_state = True
            else:
                if pinch_prev_state:
                    click_native(MOUSEEVENTF_LEFTUP) # Solta o clique
                    pinch_last_open_time = time.time()
                pinch_prev_state = False

            # 3. CLIQUE DIREITO (Sinal de V: Indicador + Médio)
            is_v = ind_up and med_up and not ane_up
            if is_v and time.time() - last_right > 0.5:
                click_native(MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)
                last_right = time.time()

            # 4. SCROLL (Mão aberta movendo verticalmente)
            is_open = ind_up and med_up and ane_up and min_up
            if is_open:
                y = lm[9][2]
                if last_scroll_y is None: last_scroll_y = y
                dy = y - last_scroll_y
                last_scroll_y = y

                if abs(dy) > 10 and time.time() - scroll_cooldown > 0.05:
                    scroll_amount = -120 if dy > 0 else 120
                    scroll_native(scroll_amount)
                    scroll_cooldown = time.time()
            else:
                last_scroll_y = None

    cv2.imshow("Mouse Nativo OS", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()