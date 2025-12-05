import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

pyautogui.FAILSAFE = False 

wCam, hCam = 640, 480
frame_r = 90           # Sensibilidade Inicial
suavizacao = 5          # Suavização
distancia_click = 30    # Distância do gatilho

p_loc_x, p_loc_y = 0, 0
c_loc_x, c_loc_y = 0, 0
clique_ativo = False    
ultimo_click_dir = 0    

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hScr = pyautogui.size()

def calcular_distancia(p1, p2, lm_list, img, draw=True):
    x1, y1 = lm_list[p1][1], lm_list[p1][2]
    x2, y2 = lm_list[p2][1], lm_list[p2][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)

    if draw and length < 50:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    
    return length, cx, cy

print("W/S: sensi | Q: sair")

try:
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        texto_acao = "" 

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                if len(lm_list) != 0:
                    x1, y1 = lm_list[8][1], lm_list[8][2]
                    
                    cv2.rectangle(img, (frame_r, frame_r), (wCam - frame_r, hCam - frame_r), (255, 0, 255), 2)
                    
                    x3 = np.interp(x1, (frame_r, wCam - frame_r), (0, wScr))
                    y3 = np.interp(y1, (frame_r, hCam - frame_r), (0, hScr))
                    
                    c_loc_x = p_loc_x + (x3 - p_loc_x) / suavizacao
                    c_loc_y = p_loc_y + (y3 - p_loc_y) / suavizacao
                    
                    pyautogui.moveTo(c_loc_x, c_loc_y)
                    p_loc_x, p_loc_y = c_loc_x, c_loc_y

                    dist_esq, cx, cy = calcular_distancia(8, 4, lm_list, img)
                    
                    if dist_esq < distancia_click:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                        texto_acao = "Clique esquerdo"
                        
                        if not clique_ativo:
                            pyautogui.mouseDown()
                            clique_ativo = True
                    else:
                        if clique_ativo:
                            pyautogui.mouseUp()
                            clique_ativo = False

                    dist_dir, cx, cy = calcular_distancia(12, 4, lm_list, img)
                    
                    if dist_dir < distancia_click:
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
                        texto_acao = "Clique direito"
                        
                        if (time.time() - ultimo_click_dir) > 0.5:
                            pyautogui.rightClick()
                            ultimo_click_dir = time.time()

                    # Adicionar depois logica de dois cliques
                    #if dist_dir >= distancia_click:
                     #   ultimo_click_dir = 0

        else:
            if clique_ativo:
                pyautogui.mouseUp()
                clique_ativo = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'): 
            if frame_r > 20: frame_r -= 5
            texto_acao = f"Sensibilidade: {frame_r}"
        elif key == ord('s'): 
            if frame_r < 200: frame_r += 5
            texto_acao = f"Sensibilidade: {frame_r}"
        elif key == ord('q'):
            break

        if texto_acao != "":
            cv2.putText(img, texto_acao, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        cv2.imshow("IHC - Sistema Mouse", img)

finally:
    pyautogui.mouseUp()
    cap.release()
    cv2.destroyAllWindows()
    print("Fechando sistema")