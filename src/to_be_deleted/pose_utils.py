"""pose_utils.py
Normalization and simple geometry helpers for pose + face landmarks.
Includes mouth aspect ratio (MAR) and eye aspect ratio (EAR) helpers.
"""

import numpy as np
from typing import List, Tuple


def landmarks_to_np(landmarks: List[Tuple[float, float, float]], frame_shape: Tuple[int, int]):
    """Convert normalized landmarks (x,y in [0,1]) to pixel coordinates (x,y).
    Returns Nx2 float array.
    """
    h, w = frame_shape
    pts = np.array([[lm[0] * w, lm[1] * h] for lm in landmarks], dtype=float)
    return pts


def normalize_by_center(points: np.ndarray) -> np.ndarray:
    """Shift points to their centroid and scale by their RMS distance to centroid.
    """
    c = points.mean(axis=0)
    pts = points - c
    rms = np.sqrt((pts ** 2).sum() / pts.shape[0])
    if rms < 1e-6:
        return pts
    return pts / rms


def angle(a, b, c):
    """Angle at point b between a-b and c-b in degrees."""
    ab = a - b
    cb = c - b
    dot = (ab * cb).sum()
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom == 0:
        return 0.0
    cosv = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


# --- Face-based heuristics ---
# landmarks: face mesh 468 points. We'll use typical indices for lips and eyes.

# Common FaceMesh indices (approx):
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_OUTER = [61, 291, 78, 308, 13, 14]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """A simple EAR using 6 points: vertical distances divided by horizontal.
    eye_pts is (6,2) array.
    """
    # vertical distances
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth_pts: np.ndarray) -> float:
    """A simple MAR using top/bottom and left/right mouth points.
    mouth_pts may be an array of key mouth coordinates; we'll fallback to top/bottom pair.
    """
    # try using top and bottom center if provided
    if mouth_pts.shape[0] >= 2:
        top = mouth_pts[0]
        bottom = mouth_pts[1]
        height = np.linalg.norm(top - bottom)
        # width approximate using two lateral mouth points if available
        if mouth_pts.shape[0] >= 4:
            left = mouth_pts[2]
            right = mouth_pts[3]
            width = np.linalg.norm(left - right)
            if width == 0:
                return 0.0
            return height / width
        else:
            return height
    return 0.0


def extract_face_metrics(face_landmarks: List[Tuple[float, float, float]], frame_shape: Tuple[int, int]):
    """Return EAR (avg of both eyes), MAR, and nose position (x,y) in pixels.
    If face_landmarks is None, returns None.
    """
    if face_landmarks is None:
        return None
    pts = landmarks_to_np(face_landmarks, frame_shape)

    try:
        left_eye = pts[LEFT_EYE]
        right_eye = pts[RIGHT_EYE]
    except Exception:
        # index failure, bail
        return None

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0

    # simple mouth: top (13) bottom (14) left(61) right(291) approximate
    top = pts[MOUTH_TOP]
    bottom = pts[MOUTH_BOTTOM]
    left = pts[61]
    right = pts[291]
    mar = mouth_aspect_ratio(np.stack([top, bottom, left, right]))

    # nose (use landmark 1 or 4) fallback
    nose = pts[1] if pts.shape[0] > 1 else pts[0]

    return {
        "ear": float(ear),
        "mar": float(mar),
        "nose_xy": (float(nose[0]), float(nose[1])),
    }


def extract_upper_body_metrics(pose_landmarks: List[Tuple[float, float, float]], frame_shape: Tuple[int, int]):
    """DISABLED: No longer using arm detection. Kept for compatibility.
    Returns None to indicate no upper body metrics are extracted.
    """
    return None
