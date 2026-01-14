# main.py

import cv2
import mediapipe as mp
import numpy as np

from fatigue_metrics import compute_ear, mouth_aspect_ratio
from smoothing import ExponentialSmoother
from config import *

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

ear_smoother = ExponentialSmoother(SMOOTHING_ALPHA)
ear_counter = 0
yawn_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "ALERT"
    fatigue_score = 0.0

    if results.multi_face_landmarks:

        landmarks = []
        h, w, _ = frame.shape

        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x * w, lm.y * h])

        landmarks = np.array(landmarks)

        for (x, y) in landmarks.astype(int):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        ear = compute_ear(landmarks)
        mar = mouth_aspect_ratio(landmarks)

        raw_ear = ear
        ear = ear_smoother.smooth(raw_ear)

        # Eye closure logic
        eye_closed_frames = 0
        eye_closed = 0
        if raw_ear < EAR_THRESHOLD:
            ear_closed_frames += 1
        else:
            ear_closed_frames = 0

        #prolonged closure detection
        if eye_closed_frames >= 20:
            eye_closed = 1

        # Yawn logic
        yawn = 0
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= YAWN_CONSEC_FRAMES:
                yawn = 1
        else:
            yawn_counter = 0

        fatigue_score = (
            0.7 * eye_closed +
            0.3 * yawn
        )


        if fatigue_score > FATIGUE_THRESHOLD:
            status = "DROWSY"

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Status: {status}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0,0,255) if status=="DROWSY" else (0,255,0), 2)

    cv2.imshow("Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
