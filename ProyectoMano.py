"""
Hand Tracking v1.0 - MediaPipe 0.10+
======================================
Version inicial funcional.
Detecta la mano en tiempo real y muestra:
  - 21 landmarks (puntos clave)
  - Que dedos estan levantados
  - Angulos de servos listos para Arduino

Instalacion:
    pip install opencv-python mediapipe

Controles:
    Q  -> salir
    S  -> mostrar angulos en consola
"""

import cv2
import mediapipe as mp
import urllib.request
import os
import time

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
        print("Descargando modelo (~8MB)... (solo la primera vez)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo descargado.")

# --- Constantes --------------------------------------------------------------

FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_PIPS  = [3, 6, 10, 14, 18]
FINGER_NAMES = ["Pulgar", "Indice", "Medio", "Anular", "Menique"]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# --- Logica de dedos ---------------------------------------------------------

def fingers_up(landmarks, handedness_label):
    lm = landmarks
    fingers = []
    if handedness_label == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers

def get_servo_angles(landmarks):
    BASES = [1, 5, 9, 13, 17]
    angles = []
    for tip_idx, base_idx in zip(FINGER_TIPS, BASES):
        tip  = landmarks[tip_idx]
        base = landmarks[base_idx]
        dist = ((tip.x - base.x)**2 + (tip.y - base.y)**2) ** 0.5
        angles.append(min(180, int(dist * 900)))
    return angles

# --- Dibujar esqueleto -------------------------------------------------------

def draw_hand(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 100), 2)
    for i, pt in enumerate(points):
        if i in FINGER_TIPS:
            cv2.circle(frame, pt, 8, (0, 200, 255), -1)
            cv2.circle(frame, pt, 8, (255, 255, 255), 1)
        else:
            cv2.circle(frame, pt, 4, (200, 200, 200), -1)

# --- Main --------------------------------------------------------------------

def main():
    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la camara.")
        return

    show_coords   = False
    prev_time     = 0
    latest_result = {"data": None}

    def result_callback(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        latest_result["data"] = result

    options = HandLandmarkerOptions(
        base_options    = mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode    = RunningMode.LIVE_STREAM,
        num_hands       = 1,
        min_hand_detection_confidence = 0.6,
        min_hand_presence_confidence  = 0.5,
        min_tracking_confidence       = 0.5,
        result_callback = result_callback,
    )

    print("Iniciando... (Q = salir, S = consola)")

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)
            detector.detect_async(mp_image, timestamp_ms)

            result = latest_result["data"]

            if result and result.hand_landmarks:
                for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                    hand_label = handedness[0].category_name

                    draw_hand(frame, hand_lm, w, h)

                    up     = fingers_up(hand_lm, hand_label)
                    angles = get_servo_angles(hand_lm)
                    count  = sum(up)

                    for i, (name, is_up, angle) in enumerate(zip(FINGER_NAMES, up, angles)):
                        color = (0, 220, 100) if is_up else (80, 80, 80)
                        icon  = "^ " if is_up else "v "
                        cv2.putText(frame, f"{icon}{name}: {angle:3d}",
                            (10, 28 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                    cv2.putText(frame, f"Dedos: {count}/5  ({hand_label})",
                        (10, 28 + 5 * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2, cv2.LINE_AA)

                    if show_coords:
                        print(f"\rServos -> {angles}   ", end="", flush=True)
            else:
                cv2.putText(frame, "Sin mano detectada",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2, cv2.LINE_AA)

            curr_time = time.time()
            fps       = 1 / (curr_time - prev_time + 1e-9)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, "Q: salir  |  S: consola", (w // 2 - 120, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

            cv2.imshow("Hand Tracking - Mano Robotica", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                show_coords = not show_coords
                if not show_coords:
                    print()

    cap.release()
    cv2.destroyAllWindows()
    print("\nCerrado.")

if __name__ == "__main__":
    main()