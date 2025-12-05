import cv2
import mediapipe as mp
import numpy as np
import math

width, height = 1280, 720

horizon_y = 250
goal_width = 280
goal_height = 140
goal_x_center = width // 2
goal_top_y = horizon_y - goal_height
goal_bottom_y = horizon_y

ball_start_x = width // 2
ball_start_y = height - 80
ball_radius = 25
BALL_HITBOX_RADIUS = 40 # Hitbox da bola
FOOT_HITBOX_RADIUS = 30

gravity = 0.6
friction = 0.98

MIN_KICK_SPEED = 15
MAX_KICK_POWER = 55
RESET_COOLDOWN = 50

VISIBILITY_THRESHOLD = 0.6
SMOOTHING_FACTOR = 0.4

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)

ball_pos = np.array([float(ball_start_x), float(ball_start_y)])
ball_velocity = np.array([0.0, 0.0])
ball_in_air = False
cooldown_timer = 0 

smooth_r_ankle = None
smooth_l_ankle = None

score_goals = 0
score_misses = 0
game_msg = "Ajuste a camera..."
msg_timer = 0
show_skeleton = True # mostrar esqueleto

def check_circle_collision(center1, radius1, center2, radius2):
    dist = np.linalg.norm(center1 - center2)
    return dist < (radius1 + radius2)

def draw_field_perspective(img):
    pts = np.array([[0, height], [width, height], [width // 2 + 600, horizon_y], [width // 2 - 600, horizon_y]])
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (50, 100, 50))
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    color_line = (200, 200, 200)
    pt1 = (goal_x_center - goal_width // 2, goal_top_y) 
    pt2 = (goal_x_center + goal_width // 2, goal_top_y) 
    pt3 = (goal_x_center + goal_width // 2 + 20, goal_bottom_y) 
    pt4 = (goal_x_center - goal_width // 2 - 20, goal_bottom_y) 

    cv2.line(img, pt1, pt2, (220, 220, 220), 6) 
    cv2.line(img, pt1, pt4, (220, 220, 220), 6) 
    cv2.line(img, pt2, pt3, (220, 220, 220), 6) 
    
    for i in range(1, 8):
        x = int(pt1[0] + (goal_width / 8) * i)
        cv2.line(img, (x, goal_top_y), (x, goal_bottom_y), (150,150,150), 1)

    return pt4[0], pt3[0] 

def reset_ball(result_text):
    global ball_pos, ball_velocity, ball_in_air, game_msg, msg_timer, cooldown_timer
    ball_in_air = False
    ball_pos = np.array([float(ball_start_x), float(ball_start_y)])
    ball_velocity = np.array([0.0, 0.0])
    game_msg = result_text
    msg_timer = 40 
    cooldown_timer = RESET_COOLDOWN

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    goal_min_x, goal_max_x = draw_field_perspective(frame)

    if show_skeleton and results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if cooldown_timer > 0: cooldown_timer -= 1

    touching_ball = False 

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        raw_r_ankle = np.array([lm[28].x * width, lm[28].y * height])
        raw_l_ankle = np.array([lm[27].x * width, lm[27].y * height])
        
        vis_r = lm[28].visibility
        vis_l = lm[27].visibility

        if smooth_r_ankle is None: 
            smooth_r_ankle = raw_r_ankle
            smooth_l_ankle = raw_l_ankle

        if vis_r > VISIBILITY_THRESHOLD:
            smooth_r_ankle = raw_r_ankle * SMOOTHING_FACTOR + smooth_r_ankle * (1 - SMOOTHING_FACTOR)
        
        if vis_l > VISIBILITY_THRESHOLD:
            smooth_l_ankle = raw_l_ankle * SMOOTHING_FACTOR + smooth_l_ankle * (1 - SMOOTHING_FACTOR)

        vec_r = raw_r_ankle - smooth_r_ankle
        vec_l = raw_l_ankle - smooth_l_ankle
        
        speed_r = np.linalg.norm(vec_r)
        speed_l = np.linalg.norm(vec_l)

        valid_r = speed_r < 150 
        valid_l = speed_l < 150

        hit_r = check_circle_collision(smooth_r_ankle, FOOT_HITBOX_RADIUS, ball_pos, BALL_HITBOX_RADIUS)
        hit_l = check_circle_collision(smooth_l_ankle, FOOT_HITBOX_RADIUS, ball_pos, BALL_HITBOX_RADIUS)

        if not ball_in_air and cooldown_timer == 0:
            if hit_r or hit_l: touching_ball = True

            kick_vec = None
            speed = 0

            if hit_r and speed_r > MIN_KICK_SPEED and vis_r > VISIBILITY_THRESHOLD and valid_r:
                kick_vec = vec_r
                speed = speed_r
            elif hit_l and speed_l > MIN_KICK_SPEED and vis_l > VISIBILITY_THRESHOLD and valid_l:
                kick_vec = vec_l
                speed = speed_l

            if kick_vec is not None:
                kick_dir = kick_vec / speed 
                power = min(speed * 2.0, MAX_KICK_POWER)
                
                ball_velocity[0] = kick_dir[0] * power 
                ball_velocity[1] = -abs(power * 1.3)
                ball_velocity[0] += kick_dir[0] * 3 

                ball_in_air = True
                game_msg = ""

        if not show_skeleton:
            
            color_r = (0, 255, 255) if hit_r else (0, 0, 255) 
            if vis_r < VISIBILITY_THRESHOLD: color_r = (50, 50, 50)
            cv2.circle(frame, (int(smooth_r_ankle[0]), int(smooth_r_ankle[1])), FOOT_HITBOX_RADIUS, color_r, 2)
            
            color_l = (0, 255, 255) if hit_l else (0, 0, 255)
            if vis_l < VISIBILITY_THRESHOLD: color_l = (50, 50, 50)
            cv2.circle(frame, (int(smooth_l_ankle[0]), int(smooth_l_ankle[1])), FOOT_HITBOX_RADIUS, color_l, 2)

    if ball_in_air:
        ball_pos += ball_velocity
        ball_velocity[1] += gravity * 0.5 
        ball_velocity[0] *= friction 

        scale = max(0.2, (ball_pos[1] - horizon_y) / (ball_start_y - horizon_y))
        draw_radius = int(ball_radius * scale)

        if ball_pos[1] <= goal_bottom_y:
            if goal_min_x < ball_pos[0] < goal_max_x and ball_pos[1] > goal_top_y:
                score_goals += 1
                reset_ball("GOL!!!")
            else:
                score_misses += 1
                reset_ball("PRA FORA!")
        
        if ball_pos[0] < 0 or ball_pos[0] > width or ball_pos[1] < 0:
             score_misses += 1
             reset_ball("ISOLOU!")
    else:
        draw_radius = ball_radius

    shadow_pos = (int(ball_pos[0]), int(min(height, ball_pos[1] + draw_radius)))
    if cooldown_timer > 0:
        ball_color = (200, 200, 200)
    elif touching_ball:
        ball_color = (0, 165, 255) 
    else:
        ball_color = (255, 255, 255)

    cv2.ellipse(frame, shadow_pos, (draw_radius, draw_radius//2), 0, 0, 360, (0,0,0, 100), -1)
    cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), draw_radius, ball_color, -1) 
    if not ball_in_air and cooldown_timer == 0:
        cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), BALL_HITBOX_RADIUS, (0, 255, 0), 1)

    cv2.putText(frame, f"GOLS: {score_goals}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    if msg_timer > 0:
        cv2.putText(frame, game_msg, (width//2 - 100, height//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
        msg_timer -= 1

    cv2.imshow('Sistema IHC - Penalti', frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'): break
    if key == ord('s'): show_skeleton = not show_skeleton

cap.release()
cv2.destroyAllWindows()