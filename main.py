# main.py

import cv2
import mediapipe as mp
import numpy as np

from fatigue_metrics import eye_openness, mouth_aspect_ratio
from config import *

# -------------------------------
# MediaPipe setup
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

cap = cv2.VideoCapture(0)

# -------------------------------
# STATE VARIABLES (CRITICAL)
# -------------------------------
eye_closed_frames = 0
yawn_frames = 0
status = "ALERT"

# -------------------------------
# MAIN LOOP
# -------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x * w, lm.y * h])
        landmarks = np.array(landmarks)

        # -------------------------------
        # DRAW LANDMARK DOTS (DEBUG)
        # -------------------------------
        for (x, y) in landmarks.astype(int):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # -------------------------------
        # EYE CLOSURE LOGIC
        # -------------------------------
        eye_open = eye_openness(landmarks)

        if eye_open < EYE_CLOSED_THRESHOLD:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0

        eye_closed = eye_closed_frames >= EYE_CLOSED_FRAMES

        # -------------------------------
        # YAWN LOGIC
        # -------------------------------
        mar = mouth_aspect_ratio(landmarks)

        if mar > MAR_THRESHOLD:
            yawn_frames += 1
        else:
            yawn_frames = 0

        yawn = yawn_frames >= YAWN_FRAMES

        # -------------------------------
        # FINAL STATUS LOGIC
        # -------------------------------
        if eye_closed or yawn:
            status = "DROWSY"
        else:
            status = "ALERT"

        # -------------------------------
        # DEBUG INFO ON SCREEN
        # -------------------------------
        cv2.putText(frame, f"EyeOpen: {eye_open:.2f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(frame, f"EyeFrames: {eye_closed_frames}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(frame, f"YawnFrames: {yawn_frames}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # -------------------------------
    # STATUS DISPLAY
    # -------------------------------
    cv2.putText(frame, f"STATUS: {status}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,0,255) if status == "DROWSY" else (0,255,0), 2)

    cv2.imshow("Fatigue Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
