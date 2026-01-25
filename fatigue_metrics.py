# fatigue_metrics.py

import numpy as np
from scipy.spatial.distance import euclidean

# MediaPipe mouth landmarks (stable)
MOUTH = [61, 81, 13, 311, 291, 308, 402, 14]

def mouth_aspect_ratio(landmarks):
    mouth = np.array([landmarks[i] for i in MOUTH])
    return (euclidean(mouth[1], mouth[7]) +
            euclidean(mouth[2], mouth[6])) / (2.0 * euclidean(mouth[0], mouth[4]))

def eye_openness(landmarks):
    # Eyelid distance (MediaPipe reliable method)
    left = abs(landmarks[159][1] - landmarks[145][1])
    right = abs(landmarks[386][1] - landmarks[374][1])
    return (left + right) / 2.0
