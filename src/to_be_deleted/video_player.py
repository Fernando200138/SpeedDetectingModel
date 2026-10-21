"""video_player.py
Simple video playback manager. Each clip plays in a loop of 3 seconds per request.
If target_label is None (neutral) this module will continue to return the last active clip.
"""

import cv2
import time
from typing import Dict, Optional
import numpy as np


class VideoPlayer:
    def __init__(self, label_to_path: Dict[str, str], panel_size=(320, 240)):
        self.label_to_path = label_to_path
        self.panel_w, self.panel_h = panel_size
        self.active_label = None
        self.active_cap = None
        self.active_start = None
        self.clip_duration = 3.0  # seconds

    def set_target(self, label: Optional[str]):
        # If label is neutral or None, keep playing last active
        if label is None or label == "neutral":
            return
        if label == self.active_label:
            return
        # switch to new label
        path = self.label_to_path.get(label)
        if path is None:
            return
        # release old
        if self.active_cap is not None:
            self.active_cap.release()
        cap = cv2.VideoCapture(path)
        # reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.active_cap = cap
        self.active_label = label
        self.active_start = time.time()

    def get_panel_frame(self) -> Optional[np.ndarray]:
        """Return the next frame to overlay (resized to panel). If no active clip, return None.
        The clip will loop every self.clip_duration seconds and the playback uses the file's fps.
        """
        if self.active_cap is None:
            return None
        now = time.time()
        elapsed = now - self.active_start if self.active_start else 0.0
        if elapsed > self.clip_duration:
            # restart
            try:
                self.active_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.active_start = now
            except Exception:
                pass

        ret, f = self.active_cap.read()
        if not ret:
            # try restarting or returning None
            try:
                self.active_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, f = self.active_cap.read()
            except Exception:
                return None
        # resize to panel
        if f is None:
            return None
        panel = cv2.resize(f, (self.panel_w, self.panel_h))
        return panel

    def release(self):
        if self.active_cap is not None:
            self.active_cap.release()
