"""Live webcam preview with MediaPipe landmarks and hand gesture recognition."""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pyvirtualcam

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


class HandGestureRecognizer:
    """Recognize hand poses from the per-frame finger-extension features."""

    def recognize(self, finger_extended):
        if not finger_extended:
            return None

        if (
            finger_extended.get("thumb_extended") is True
            and all(
                finger_extended.get(f"{finger}_extended") is False
                for finger in ("index", "middle", "ring", "pinky")
            )
        ):
            return "pointing_fist"

        return None


class LiveLandmarkCapture:
    """Read webcam frames and detect face, pose, and hand landmarks."""

    def __init__(
        self,
        camera_index=0,
        width=1280,
        height=720,
        show_landmarks=True,
        video_path=None,
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
        self.recognizer = HandGestureRecognizer()
        self.candidate_gesture = None
        self.candidate_frames = 0
        self.active_gesture = None
        self.required_stable_frames = 3
        if video_path is None:
            project_root = Path(__file__).resolve().parent.parent
            video_path = project_root / "data" / "test_videos" / "diem_archive_tanner.mp4"
        self.video_overlay = OneShotVideoOverlay(video_path)

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
        detected_gesture = None
        frame_height, frame_width = frame.shape[:2]
        for hand_landmarks in detected_hands:
            if self.show_landmarks:
                self.drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                )
            image_landmarks = [
                {"x": landmark.x * frame_width, "y": landmark.y * frame_height}
                for landmark in hand_landmarks.landmark
            ]
            features = extract_hand_features(image_landmarks)
            gesture = self.recognizer.recognize(features["finger_extended"])
            if gesture is not None:
                detected_gesture = gesture
                break

        if self._update_gesture_state(detected_gesture):
            self.video_overlay.start()
        self.video_overlay.overlay_next_frame(frame)
        self._draw_status(frame, detected_hands)
        return frame

    def _update_gesture_state(self, detected_gesture):
        previous_gesture = self.active_gesture
        if detected_gesture != self.candidate_gesture:
            self.candidate_gesture = detected_gesture
            self.candidate_frames = 1
        else:
            self.candidate_frames += 1

        if self.candidate_frames >= self.required_stable_frames:
            self.active_gesture = self.candidate_gesture
        return (
            self.active_gesture == "pointing_fist"
            and previous_gesture != "pointing_fist"
        )

    def _draw_status(self, frame, detected_hands):
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
        self.capture.release()
        self.pose.close()
        self.face_mesh.close()
        self.hands.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Show live MediaPipe landmarks and recognize a pointing_fist hand pose."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument(
        "--no-landmarks",
        action="store_true",
        help="Hide face, pose, and hand landmark drawings while keeping gesture recognition active",
    )
    parser.add_argument(
        "--virtual-camera",
        action="store_true",
        help="Send processed frames to the OBS Virtual Camera device",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a local preview window (optional when using --virtual-camera)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Virtual camera output frame rate (default: 30)",
    )
    args = parser.parse_args()
    if not args.virtual_camera and not args.preview:
        parser.error("choose at least one output: --virtual-camera and/or --preview")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    app = LiveLandmarkCapture(
        args.camera,
        args.width,
        args.height,
        show_landmarks=not args.no_landmarks,
    )
    if not app.capture.isOpened():
        app.release()
        parser.error(
            f"Could not open camera index {args.camera}; try another value with --camera."
        )

    virtual_camera = None
    try:
        if args.virtual_camera:
            camera_width = int(app.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
            camera_height = int(app.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
            try:
                virtual_camera = pyvirtualcam.Camera(
                    width=camera_width,
                    height=camera_height,
                    fps=args.fps,
                    fmt=pyvirtualcam.PixelFormat.BGR,
                )
            except RuntimeError as error:
                parser.error(
                    "Could not open a virtual camera. Start OBS Virtual Camera first; "
                    f"details: {error}"
                )
            print(
                f"Virtual camera started: {virtual_camera.device} "
                f"({camera_width}x{camera_height} at {args.fps:g} FPS)."
            )
        if args.preview:
            print("Local preview enabled. Press 'q' in the preview window to quit.")
        elif args.virtual_camera:
            print("Virtual camera output active. Press Ctrl+C to stop.")

        while True:
            frame = app.read()
            if frame is None:
                print("[ERROR] Could not read a frame from the camera.")
                break
            if virtual_camera is not None:
                if frame.shape[1] != virtual_camera.width or frame.shape[0] != virtual_camera.height:
                    frame = cv2.resize(
                        frame, (virtual_camera.width, virtual_camera.height)
                    )
                virtual_camera.send(frame)
                virtual_camera.sleep_until_next_frame()
            if args.preview:
                cv2.imshow("Live Hand Gesture Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            if virtual_camera is not None:
                virtual_camera.close()
        finally:
            app.release()


if __name__ == "__main__":
    main()
