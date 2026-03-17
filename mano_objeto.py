"""
Mano_objeto.py - Esfera wireframe AR - Control dos manos
=========================================================
Mano 1 (pinch cerca esfera)  -> agarrar y arrastrar
Mano 2 (pinch en cualquier lugar) -> escalar
    abrir dedos  = agrandar
    cerrar dedos = achicar

Controles:
    R  -> resetear
    Q  -> salir
"""

import cv2
import mediapipe as mp
import urllib.request
import os
import time
import math
import numpy as np
import logging
import absl.logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"]      = "3"
logging.getLogger("mediapipe").setLevel(logging.CRITICAL)
absl.logging.set_verbosity(absl.logging.FATAL)

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarkerResult, RunningMode

# --- Modelo ------------------------------------------------------------------

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo (~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Listo.")

# --- Constantes --------------------------------------------------------------

FINGER_TIPS = [4, 8, 12, 16, 20]
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# --- Esfera ------------------------------------------------------------------

def build_sphere_lines(lat=9, lon=14):
    lines = []
    for i in range(1, lat):
        y     = math.sin(math.pi * i / lat - math.pi/2)
        r_lat = math.cos(math.pi * i / lat - math.pi/2)
        for j in range(lon):
            a1 = 2*math.pi*j/lon
            a2 = 2*math.pi*(j+1)/lon
            lines.append(((r_lat*math.cos(a1), y, r_lat*math.sin(a1)),
                          (r_lat*math.cos(a2), y, r_lat*math.sin(a2))))
    for j in range(lon):
        la = 2*math.pi*j/lon
        for i in range(lat*2):
            la1 = math.pi*i/(lat*2) - math.pi/2
            la2 = math.pi*(i+1)/(lat*2) - math.pi/2
            lines.append(((math.cos(la1)*math.cos(la), math.sin(la1), math.cos(la1)*math.sin(la)),
                          (math.cos(la2)*math.cos(la), math.sin(la2), math.cos(la2)*math.sin(la))))
    return lines

SPHERE_LINES = build_sphere_lines()

def project(p, cx, cy, radius, rx, ry):
    x, y, z = p
    cy_r, sy_r = math.cos(ry), math.sin(ry)
    x2 = x*cy_r - z*sy_r
    z2 = x*sy_r + z*cy_r
    cx_r, sx_r = math.cos(rx), math.sin(rx)
    y2 = y*cx_r - z2*sx_r
    z3 = y*sx_r + z2*cx_r
    fov = 600
    zd  = fov / (fov + z3 * radius * 0.3)
    return int(cx + x2*radius*zd), int(cy + y2*radius*zd), z3

def draw_sphere(frame, cx, cy, radius, rx, ry, grabbed, tips):
    fw, fh = frame.shape[1], frame.shape[0]
    base   = (60, 200, 255) if grabbed else (40, 120, 200)
    glow   = (140, 240, 255)

    for p1, p2 in SPHERE_LINES:
        x1, y1, z1 = project(p1, cx, cy, radius, rx, ry)
        x2, y2, z2 = project(p2, cx, cy, radius, rx, ry)
        if not (0<=x1<fw and 0<=y1<fh and 0<=x2<fw and 0<=y2<fh):
            continue
        bright = ((z1+z2)/2 + 1) / 2
        mid_x, mid_y = (x1+x2)//2, (y1+y2)//2
        near = min((math.hypot(mid_x-t[0], mid_y-t[1]) for t in tips), default=9999)
        gf   = max(0, 1 - near/(radius*1.3)) if grabbed else 0
        r = int((base[0]+(glow[0]-base[0])*gf) * (0.3+0.7*bright))
        g = int((base[1]+(glow[1]-base[1])*gf) * (0.3+0.7*bright))
        b = int((base[2]+(glow[2]-base[2])*gf) * (0.3+0.7*bright))
        cv2.line(frame, (x1,y1), (x2,y2), (r,g,b), 2 if gf>0.3 else 1, cv2.LINE_AA)

# --- Mano helpers ------------------------------------------------------------

def get_pos(lm, idx, w, h):
    return int(lm[idx].x*w), int(lm[idx].y*h)

def pinch_center(lm, w, h):
    tx, ty = get_pos(lm, 4, w, h)
    ix, iy = get_pos(lm, 8, w, h)
    return (tx+ix)//2, (ty+iy)//2

def pinch_dist(lm, w, h):
    tx, ty = get_pos(lm, 4, w, h)
    ix, iy = get_pos(lm, 8, w, h)
    return math.hypot(tx-ix, ty-iy)

def draw_skeleton(frame, lm, w, h, color):
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1)
    for i, pt in enumerate(pts):
        cv2.circle(frame, pt, 4 if i in FINGER_TIPS else 2, color, -1)

def draw_tip_glow(frame, pos, color, active):
    if not pos: return
    cv2.circle(frame, pos, 7, color, -1)
    if active:
        cv2.circle(frame, pos, 15, color, 2, cv2.LINE_AA)

# --- Main --------------------------------------------------------------------

def main():
    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la camara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    ret, f0 = cap.read()
    H, W    = f0.shape[:2] if ret else (720, 1280)

    latest    = {"data": None}
    prev_time = 0

    def callback(result: HandLandmarkerResult, img: mp.Image, ts: int):
        latest["data"] = result

    options = HandLandmarkerOptions(
        base_options    = mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode    = RunningMode.LIVE_STREAM,
        num_hands       = 2,   # <-- dos manos
        min_hand_detection_confidence = 0.7,
        min_hand_presence_confidence  = 0.6,
        min_tracking_confidence       = 0.5,
        result_callback = callback,
    )

    # Estado esfera
    sx, sy = float(W//2), float(H//2)
    sr     = 130.0
    rx, ry = 0.2, 0.0
    smx, smy = sx, sy
    smr      = sr
    smrx, smry = rx, ry

    # Agarre (mano 1)
    grabbed       = False
    grab_anchor_x = 0.0
    grab_anchor_y = 0.0
    grab_sx       = sx
    grab_sy       = sy

    # Escala (mano 2)
    scale_ref_dist = None
    scale_ref_r    = sr

    GRAB_THRESHOLD  = 55   # px - pinch cerrado para agarrar
    SCALE_THRESHOLD = 80   # px - pinch activo para escalar (puede estar mas abierto)

    with mp_vision.HandLandmarker.create_from_options(options) as det:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(time.time()*1000)
            )

            result = latest["data"]
            all_tips = []

            grab_hand  = None   # landmarks de la mano que agarra
            scale_hand = None   # landmarks de la mano que escala

            if result and result.hand_landmarks:
                hands = result.hand_landmarks
                handedness_list = result.handedness

                # Identificar cada mano
                for lm, hd in zip(hands, handedness_list):
                    label = hd[0].category_name  # "Left" / "Right"
                    pd    = pinch_dist(lm, w, h)
                    pcx, pcy = pinch_center(lm, w, h)
                    dist_to_sphere = math.hypot(pcx - smx, pcy - smy)
                    near_sphere    = dist_to_sphere < smr * 1.3

                    # Mano que agarra: pinch cerrado Y cerca de la esfera
                    if pd < GRAB_THRESHOLD and near_sphere:
                        grab_hand = lm
                    # Mano que escala: pinch activo en cualquier lugar
                    # (si no esta agarrando)
                    elif grab_hand is None or lm is not grab_hand:
                        if pd < SCALE_THRESHOLD * 3:  # cualquier estado de pinch
                            scale_hand = lm

                # Si solo hay una mano, puede agarrar o escalar, no las dos
                if len(hands) == 1:
                    lm = hands[0]
                    pd = pinch_dist(lm, w, h)
                    pcx, pcy = pinch_center(lm, w, h)
                    dist_to_sphere = math.hypot(pcx - smx, pcy - smy)
                    near_sphere    = dist_to_sphere < smr * 1.3
                    if pd < GRAB_THRESHOLD and near_sphere:
                        grab_hand  = lm
                        scale_hand = None
                    else:
                        grab_hand  = None
                        scale_hand = None

                # --- Dibujar esqueletos ---
                for i, (lm, hd) in enumerate(zip(hands, handedness_list)):
                    if lm is grab_hand:
                        color = (0, 200, 100)    # verde = agarrando
                    elif lm is scale_hand:
                        color = (0, 180, 255)    # azul = escalando
                    else:
                        color = (60, 80, 100)
                    draw_skeleton(frame, lm, w, h, color)
                    tip4 = get_pos(lm, 4, w, h)
                    tip8 = get_pos(lm, 8, w, h)
                    all_tips += [tip4, tip8]
                    active = (lm is grab_hand or lm is scale_hand)
                    draw_tip_glow(frame, tip4, (80,220,255), active)
                    draw_tip_glow(frame, tip8, (120,200,255), active)

                # --- Logica agarre ---
                if grab_hand is not None:
                    pcx, pcy = pinch_center(grab_hand, w, h)
                    if not grabbed:
                        grabbed       = True
                        grab_anchor_x = pcx
                        grab_anchor_y = pcy
                        grab_sx       = sx
                        grab_sy       = sy
                    # Mover: desplazamiento desde el punto de agarre
                    sx = grab_sx + (pcx - grab_anchor_x)
                    sy = grab_sy + (pcy - grab_anchor_y)
                else:
                    grabbed = False

                # --- Logica escala (mano 2) ---
                if scale_hand is not None:
                    pd2 = pinch_dist(scale_hand, w, h)
                    if scale_ref_dist is None:
                        scale_ref_dist = pd2
                        scale_ref_r    = sr
                    ratio = pd2 / max(scale_ref_dist, 1)
                    sr    = max(40, min(scale_ref_r * ratio, 420))
                else:
                    scale_ref_dist = None

            else:
                grabbed        = False
                scale_ref_dist = None

            # Rotacion automatica cuando libre
            if not grabbed:
                ry += 0.008

            # Suavizado
            LERP = 0.22
            smx  += (sx  - smx)  * LERP
            smy  += (sy  - smy)  * LERP
            smr  += (sr  - smr)  * LERP
            smrx += (rx  - smrx) * 0.12
            smry += (ry  - smry) * 0.12

            draw_sphere(frame, int(smx), int(smy), smr,
                        smrx, smry, grabbed, all_tips)

            # FPS
            curr = time.time()
            fps  = 1/(curr - prev_time + 1e-9)
            prev_time = curr
            cv2.putText(frame, f"FPS:{int(fps)}", (w-75, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50,65,80), 1, cv2.LINE_AA)

            cv2.imshow("Esfera AR", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord("r"):
                sx=float(w//2); sy=float(h//2)
                sr=130.0; rx=0.2; ry=0.0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()