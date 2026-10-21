"""classifier.py
A lightweight rule-based classifier that uses facial metrics over a short temporal window
to classify into two expressions, plus NEUTRAL.

Labels: 'messi_yell', 'mom_homeless_laugh', 'neutral'

This classifier focuses on eye and facial expressions only (no arm/body detection).
"""

from collections import deque
from typing import Optional, Dict, Any
import time


class ExpressionClassifier:
    def __init__(self, window_seconds: float = 0.6, fps_assume: int = 30):
        # time window to smooth features across frames
        self.win = window_seconds
        self.buffer = deque()
        self.last_time = None

        # thresholds (tweak as needed)
        self.MAR_YELL_THRESHOLD = 0.45   # mouth aspect ratio for wide open mouth
        self.EAR_CLOSED_THRESHOLD = 0.20 # eye aspect ratio indicates closed/squinted eyes

        # classification hold (frames) to avoid jitter
        self.debounce_seconds = 0.25
        self.last_label = "neutral"
        self.last_label_time = 0

    def update(self, metrics: Dict[str, Any], timestamp: float) -> str:
        """metrics: dict with key 'face' from pose_utils (upper body metrics no longer used)
        returns: label string
        """
        # push into buffer
        self.buffer.append((timestamp, metrics))
        # pop old
        while len(self.buffer) > 0 and (timestamp - self.buffer[0][0]) > self.win:
            self.buffer.popleft()

        label = self._classify_from_buffer()

        # debounce: require label hold for debounce_seconds before switching
        now = timestamp
        if label != self.last_label:
            if now - self.last_label_time >= self.debounce_seconds:
                self.last_label = label
                self.last_label_time = now
            else:
                # keep previous label
                label = self.last_label
        else:
            self.last_label_time = now

        return label

    def _classify_from_buffer(self) -> str:
        # aggregate simple stats over buffer
        ears = []
        mars = []

        for _, m in self.buffer:
            if m is None:
                continue
            f = m.get("face")
            if f:
                ears.append(f.get("ear", 0.0))
                mars.append(f.get("mar", 0.0))

        # compute simple aggregates
        mean_ear = sum(ears) / len(ears) if ears else 1.0
        mean_mar = sum(mars) / len(mars) if mars else 0.0

        # Heuristics (eyes and face only):
        # 1) "messi_yell": mouth very open (MAR large)
        if mean_mar > self.MAR_YELL_THRESHOLD:
            return "messi_yell"

        # 2) "mom_homeless_laugh": eyes closed/squinted with clenched face
        # Detects when someone is holding in laughter - eyes closed, face tense
        if mean_ear < self.EAR_CLOSED_THRESHOLD:
            return "mom_homeless_laugh"

        # If no significant changes: neutral
        return "neutral"
