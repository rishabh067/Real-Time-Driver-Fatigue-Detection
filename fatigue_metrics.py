# fatigue_metrics.py

import numpy as np
from scipy.spatial.distance import euclidean

# MediaPipe Face Mesh indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 81, 13, 311, 291, 308, 402, 14]

def eye_aspect_ratio(landmarks, eye_indices):
    eye = np.array([landmarks[i] for i in eye_indices])
    return (euclidean(eye[1], eye[5]) +
            euclidean(eye[2], eye[4])) / (2.0 * euclidean(eye[0], eye[3]))

def mouth_aspect_ratio(landmarks):
    mouth = np.array([landmarks[i] for i in MOUTH])
    return (euclidean(mouth[1], mouth[7]) +
            euclidean(mouth[2], mouth[6])) / (2.0 * euclidean(mouth[0], mouth[4]))

def compute_ear(landmarks):
    left = eye_aspect_ratio(landmarks, LEFT_EYE)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    return (left + right) / 2.0
