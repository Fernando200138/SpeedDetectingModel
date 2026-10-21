"""capture.py
Wraps webcam capture and MediaPipe Pose + FaceMesh processing into a simple class.
Returns raw BGR frames, normalized pose landmarks, and face mesh landmarks.
"""

import time
from typing import Optional, Dict, Any, Tuple, List

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeCapture:
    def __init__(self, cam_index: int = 0, webcam_width: int = 1280, webcam_height: int = 720):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)

        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh

        # Pose: we only need upper-body but use full model for robustness
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=1,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        # Face mesh for mouth/eye measurements
        self.face = self.mp_face.FaceMesh(static_image_mode=False,
                                          max_num_faces=1,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

    def read(self) -> Optional[Dict[str, Any]]:
        """Read a frame and return processed landmarks + frame.

        Returns dict with keys:
         - frame_bgr: np.ndarray
         - pose_landmarks: List[(x,y,z), ...] or None (normalized in image coordinates)
         - face_landmarks: List[(x,y,z), ...] or None (normalized)
         - timestamp: float
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        timestamp = time.time()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_res = self.pose.process(img_rgb)
        face_res = self.face.process(img_rgb)

        pose_landmarks = None
        if pose_res.pose_landmarks:
            pose_landmarks = [(lm.x, lm.y, lm.z) for lm in pose_res.pose_landmarks.landmark]

        face_landmarks = None
        if face_res.multi_face_landmarks and len(face_res.multi_face_landmarks) > 0:
            # taking the first face
            face_landmarks = [(lm.x, lm.y, lm.z) for lm in face_res.multi_face_landmarks[0].landmark]

        return {
            "frame_bgr": frame,
            "pose_landmarks": pose_landmarks,
            "face_landmarks": face_landmarks,
            "timestamp": timestamp,
        }

    def release(self):
        self.cap.release()
        self.pose.close()
        self.face.close()
