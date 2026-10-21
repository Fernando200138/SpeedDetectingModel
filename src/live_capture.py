"""Live webcam capture, landmark processing, and triggered overlays."""

from collections import Counter, deque
from pathlib import Path
import time

import cv2
import mediapipe as mp

from .gesture_recognizer import GestureRecognizer
from .inspect_landmarks import extract_hand_features


class OneShotVideoOverlay:
    """Play a video once as a live-preview overlay after it is triggered."""

    def __init__(self, video_path):
        self.video_path = Path(video_path)
        self.capture = None

    def start(self):
        self.stop()
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open gesture video: {self.video_path}")
        self.capture = capture

    def stop(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def overlay_next_frame(self, frame):
        if self.capture is None:
            return

        success, video_frame = self.capture.read()
        if not success:
            self.stop()
            return

        frame_height, frame_width = frame.shape[:2]
        video_height, video_width = video_frame.shape[:2]
        max_width = min(480, frame_width // 2)
        max_height = max(1, frame_height - 100)
        scale = min(max_width / video_width, max_height / video_height)
        panel_width = max(1, int(video_width * scale))
        panel_height = max(1, int(video_height * scale))
        panel = cv2.resize(video_frame, (panel_width, panel_height))

        x = frame_width - panel_width - 16
        y = 70
        if y + panel_height > frame_height:
            y = max(0, frame_height - panel_height)
        frame[y:y + panel_height, x:x + panel_width] = panel
        cv2.rectangle(
            frame,
            (x - 1, y - 1),
            (x + panel_width, y + panel_height),
            (255, 255, 255),
            1,
        )


class TimedImageOverlay:
    """Show a still image as a live-preview overlay for a fixed duration."""

    def __init__(self, image_path, duration_seconds=3.0):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise RuntimeError(f"Could not open gesture image: {self.image_path}")
        self.duration_seconds = duration_seconds
        self.started_at = None

    def start(self):
        self.started_at = time.monotonic()

    def stop(self):
        self.started_at = None

    def overlay_next_frame(self, frame):
        if self.started_at is None:
            return
        if time.monotonic() - self.started_at >= self.duration_seconds:
            self.started_at = None
            return

        frame_height, frame_width = frame.shape[:2]
        image_height, image_width = self.image.shape[:2]
        max_width = min(480, frame_width // 2)
        max_height = max(1, frame_height - 100)
        scale = min(max_width / image_width, max_height / image_height)
        panel_width = max(1, int(image_width * scale))
        panel_height = max(1, int(image_height * scale))
        panel = cv2.resize(self.image, (panel_width, panel_height))

        x = frame_width - panel_width - 16
        y = 70
        if y + panel_height > frame_height:
            y = max(0, frame_height - panel_height)
        frame[y:y + panel_height, x:x + panel_width] = panel
        cv2.rectangle(
            frame,
            (x - 1, y - 1),
            (x + panel_width, y + panel_height),
            (255, 255, 255),
            1,
        )


class LiveLandmarkCapture:
    """Read webcam frames and detect face, pose, and hand landmarks."""

    gesture_history_frames = 30
    gesture_vote_threshold = 15

    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720,
        show_landmarks=True,
        video_path=None,
        image_path=None,
        chin_video_path=None,
        open_mouth_video_path=None,
    ):
        self.show_landmarks = show_landmarks
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils
        self.recognizer = GestureRecognizer()
        self.gesture_history = deque(maxlen=self.gesture_history_frames)
        self.triggered_gestures = set()
        self.active_gesture = None
        self.gesture_recognition_enabled = True

        project_root = Path(__file__).resolve().parent.parent
        video_path = video_path or project_root / "data" / "test_videos" / "diem_archive_tanner.mp4"
        image_path = image_path or project_root / "data" / "test_videos" / "absolute-cinema-cinema.jpg"
        chin_video_path = chin_video_path or project_root / "data" / "test_videos" / "diemarchive_roy.mp4"
        open_mouth_video_path = open_mouth_video_path or project_root / "data" / "test_videos" / "PinGrab_1791239090712.mp4"

        self.video_overlay = OneShotVideoOverlay(video_path)
        self.image_overlay = TimedImageOverlay(image_path)
        self.chin_video_overlay = OneShotVideoOverlay(chin_video_path)
        self.open_mouth_video_overlay = OneShotVideoOverlay(open_mouth_video_path)

    def read(self):
        success, frame = self.capture.read()
        if not success:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(frame_rgb)
        face_result = self.face_mesh.process(frame_rgb)
        hand_result = self.hands.process(frame_rgb)

        if self.show_landmarks and face_result.multi_face_landmarks:
            self.drawing.draw_landmarks(
                image=frame,
                landmark_list=face_result.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.drawing.DrawingSpec(
                    color=(80, 220, 80), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=self.drawing.DrawingSpec(
                    color=(80, 120, 80), thickness=1, circle_radius=1
                ),
            )

        if self.show_landmarks and pose_result.pose_landmarks:
            self.drawing.draw_landmarks(
                image=frame,
                landmark_list=pose_result.pose_landmarks,
                connections=mp.solutions.pose.POSE_CONNECTIONS,
            )

        detected_hands = hand_result.multi_hand_landmarks or []
        hands_finger_extended = []
        hands_landmarks = []
        frame_height, frame_width = frame.shape[:2]
        for hand_landmarks in detected_hands:
            if self.show_landmarks:
                self.drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                )
            if self.gesture_recognition_enabled:
                image_landmarks = [
                    {"x": landmark.x * frame_width, "y": landmark.y * frame_height}
                    for landmark in hand_landmarks.landmark
                ]
                features = extract_hand_features(image_landmarks)
                if features is not None:
                    hands_finger_extended.append(features["finger_extended"])
                    hands_landmarks.append(hand_landmarks.landmark)

        detected_gesture = None
        face_landmarks = (
            face_result.multi_face_landmarks[0].landmark
            if face_result.multi_face_landmarks
            else None
        )
        if self.gesture_recognition_enabled:
            detected_gesture = self.recognizer.recognize_open_hands(
                hands_finger_extended
            )
            if detected_gesture is None and face_landmarks is not None:
                detected_gesture = self.recognizer.recognize_look_up_open_mouth(
                    face_landmarks, frame_width, frame_height
                )
            if detected_gesture is None and face_landmarks is not None:
                for finger_extended, hand_landmarks in zip(
                    hands_finger_extended, hands_landmarks
                ):
                    detected_gesture = self.recognizer.recognize_index_finger_on_chin(
                        finger_extended,
                        hand_landmarks,
                        face_landmarks,
                        frame_width,
                        frame_height,
                    )
                    if detected_gesture is not None:
                        break
            if detected_gesture is None:
                for finger_extended in hands_finger_extended:
                    detected_gesture = self.recognizer.recognize_pointing_fist(
                        finger_extended
                    )
                    if detected_gesture is not None:
                        break

        triggered_gesture = self._update_gesture_state(detected_gesture)
        if triggered_gesture == "pointing_fist":
            self.video_overlay.start()
        elif triggered_gesture == "index_finger_on_chin":
            self.chin_video_overlay.start()
        elif triggered_gesture == "open_hands":
            self.image_overlay.start()
        elif triggered_gesture == "look_up_open_mouth":
            self.open_mouth_video_overlay.start()
        self.video_overlay.overlay_next_frame(frame)
        self.chin_video_overlay.overlay_next_frame(frame)
        self.open_mouth_video_overlay.overlay_next_frame(frame)
        self.image_overlay.overlay_next_frame(frame)
        self._draw_status(frame, detected_hands)
        return frame

    def _update_gesture_state(self, detected_gesture):
        self.gesture_history.append(detected_gesture)
        if len(self.gesture_history) < self.gesture_history_frames:
            self.active_gesture = None
            return None

        gesture_counts = Counter(
            gesture for gesture in self.gesture_history if gesture is not None
        )
        self.triggered_gestures.intersection_update(
            gesture
            for gesture in self.triggered_gestures
            if gesture_counts[gesture] >= self.gesture_vote_threshold
        )

        qualifying_gestures = [
            gesture
            for gesture, count in gesture_counts.items()
            if count >= self.gesture_vote_threshold
        ]
        self.active_gesture = max(
            qualifying_gestures,
            key=gesture_counts.get,
            default=None,
        )

        if (
            detected_gesture is not None
            and gesture_counts[detected_gesture] >= self.gesture_vote_threshold
            and detected_gesture not in self.triggered_gestures
        ):
            self.triggered_gestures.add(detected_gesture)
            return detected_gesture
        return None

    def set_gesture_recognition_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self.gesture_recognition_enabled:
            return

        self.gesture_recognition_enabled = enabled
        self.gesture_history.clear()
        self.triggered_gestures.clear()
        self.active_gesture = None
        if not enabled:
            self.video_overlay.stop()
            self.chin_video_overlay.stop()
            self.open_mouth_video_overlay.stop()
            self.image_overlay.stop()

    def _draw_status(self, frame, detected_hands, show_status=False):
        if show_status:
            if self.active_gesture:
                text = f"Gesture recognized: {self.active_gesture}"
                color = (40, 220, 40)
            elif detected_hands:
                text = "Hand detected - make pointing_fist"
                color = (0, 220, 255)
            else:
                text = "Show a hand to the camera"
                color = (240, 240, 240)

            cv2.rectangle(frame, (12, 12), (660, 58), (20, 20, 20), -1)
            cv2.putText(
                frame,
                text,
                (24, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            "Press q to quit",
            (24, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def release(self):
        self.video_overlay.stop()
        self.chin_video_overlay.stop()
        self.open_mouth_video_overlay.stop()
        self.capture.release()
        self.pose.close()
        self.face_mesh.close()
        self.hands.close()
        cv2.destroyAllWindows()