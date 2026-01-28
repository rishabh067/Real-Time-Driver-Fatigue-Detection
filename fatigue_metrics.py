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

def head_tilt_and_nod(landmarks, prev_nose_y):
    """
    Returns:
    - head_tilt (bool)
    - nod (bool)
    - current_nose_y
    """

    # Key landmarks
    left_eye_y = landmarks[159][1]
    right_eye_y = landmarks[386][1]
    eye_center_y = (left_eye_y + right_eye_y) / 2.0

    nose_y = landmarks[1][1]

    # Head tilt (nose much lower than eyes)
    head_tilt = (nose_y - eye_center_y) > 18

    # Sudden nod (fast downward motion)
    nod = False
    if prev_nose_y is not None:
        if (nose_y - prev_nose_y) > 15:
            nod = True

    return head_tilt, nod, nose_y
