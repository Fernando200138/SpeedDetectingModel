"""ui.py
Main GUI and event loop. Glues together capture, classifier, and video player.
Displays the main camera feed in fullscreen/large window with a video panel overlay in the corner.
"""

import cv2
import numpy as np
from typing import Dict, Optional

from . import pose_utils


class UIManager:
    def __init__(self, capture, classifier, video_player, window_name="MemePoseMatcher"):
        """
        Args:
            capture: MediaPipeCapture instance
            classifier: ExpressionClassifier instance
            video_player: VideoPlayer instance
            window_name: name of the OpenCV window
        """
        self.capture = capture
        self.classifier = classifier
        self.video_player = video_player
        self.window_name = window_name

        # UI layout
        self.panel_w, self.panel_h = 320, 240
        self.panel_x, self.panel_y = 10, 10  # top-left corner of overlay
        self.show_metrics = True  # toggle display of metrics overlay

    def draw_frame_with_overlay(self, frame: np.ndarray,
                               pose_landmarks=None,
                               face_landmarks=None,
                               current_label: str = "neutral",
                               metrics: Optional[Dict] = None) -> np.ndarray:
        """Draw face landmarks and text info on frame.

        Args:
            frame: BGR frame to draw on
            pose_landmarks: (unused, kept for compatibility)
            face_landmarks: list of (x,y,z) normalized tuples
            current_label: current expression label
            metrics: dict with 'face' metrics (optional)

        Returns:
            annotated frame
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        # No longer drawing upper body skeleton - focusing on face only

        # Draw face mesh (sparse key points)
        if face_landmarks is not None:
            self._draw_face_keypoints(output, face_landmarks, (h, w))

        # Draw label and metrics text
        label_text = f"Expression: {current_label.upper()}"
        cv2.putText(output, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2)

        if self.show_metrics and metrics:
            self._draw_metrics_text(output, metrics)

        # Draw meme clip panel overlay
        panel_frame = self.video_player.get_panel_frame()
        if panel_frame is not None:
            self._overlay_panel(output, panel_frame)

        return output

    def _draw_pose_skeleton(self, frame, pose_landmarks, frame_shape):
        """DISABLED: No longer drawing pose skeleton (arm detection removed).
        Method kept for compatibility but does nothing.
        """
        pass

    def _draw_face_keypoints(self, frame, face_landmarks, frame_shape):
        """Draw key facial landmarks (eyes, nose, mouth)."""
        h, w = frame_shape
        pts = pose_utils.landmarks_to_np(face_landmarks, frame_shape)

        # key indices
        indices = [
            # Eyes
            33, 160, 158, 133, 153, 144,  # left eye
            263, 387, 385, 362, 380, 373,  # right eye
            # Mouth
            61, 291, 78, 308, 13, 14,
            # Nose
            1, 4,
        ]

        for idx in indices:
            if idx < len(pts):
                center = tuple(map(int, pts[idx]))
                cv2.circle(frame, center, 3, (255, 0, 0), -1)

    def _draw_metrics_text(self, frame, metrics):
        """Draw face metrics as text on frame."""
        y_offset = 70
        line_height = 25

        face_metrics = metrics.get("face")
        if face_metrics:
            text_ear = f"EAR: {face_metrics.get('ear', 0.0):.3f}"
            text_mar = f"MAR: {face_metrics.get('mar', 0.0):.3f}"
            cv2.putText(frame, text_ear, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 0), 1)
            cv2.putText(frame, text_mar, (10, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 0), 1)

        # No longer displaying arm metrics

    def _overlay_panel(self, frame, panel):
        """Overlay the video panel in the corner of the main frame."""
        h, w = frame.shape[:2]
        # ensure panel fits
        panel_h, panel_w = panel.shape[:2]
        x_end = min(self.panel_x + panel_w, w)
        y_end = min(self.panel_y + panel_h, h)
        frame[self.panel_y:y_end, self.panel_x:x_end] = panel[:y_end - self.panel_y, :x_end - self.panel_x]

    def show_frame(self, frame: np.ndarray):
        """Display frame in OpenCV window."""
        cv2.imshow(self.window_name, frame)

    def get_key(self, timeout_ms: int = 1) -> Optional[int]:
        """Get keyboard input. Returns key code or None if timeout."""
        key = cv2.waitKey(timeout_ms)
        return key if key != -1 else None

    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
