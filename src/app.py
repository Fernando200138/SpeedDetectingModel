"""app.py
Main entry point for MemePoseMatcher. Orchestrates capture, classification, and UI.
Run with: python -m src.app
"""

import os
from pathlib import Path

from .capture import MediaPipeCapture
import cv2
from .classifier import ExpressionClassifier
from .video_player import VideoPlayer
from .ui import UIManager
from . import pose_utils


def main():
    """Main application loop."""
    # Set up video paths (adjust these to match your files)
    base_dir = Path(__file__).parent.parent
    videos_dir = base_dir / "data" / "videos"

    label_to_path = {
        "messi_yell": str(videos_dir / "speed_messi.mp4"),
        "mom_homeless_laugh": str(videos_dir / "speed_mom_homeless.mp4"),
    }

    # Check if videos exist (warning only, not blocking)
    missing_videos = []
    for label, path in label_to_path.items():
        if not os.path.exists(path):
            missing_videos.append(f"{label}: {path}")

    if missing_videos:
        print("[WARNING] Some video files not found:")
        for mv in missing_videos:
            print(f"  - {mv}")
        print("Please add video files to data/videos/ directory.")
        print("The app will run but no overlays will play until videos are added.")

    # Initialize components
    # Try to auto-detect a working camera index (some systems map the active camera to /dev/video1)
    def _find_working_cam(max_idx=3):
        for i in range(0, max_idx + 1):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    continue
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return i
            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass
        return 0

    cam_index = _find_working_cam(0)
    print(f"[INFO] Initializing MediaPipe capture on camera index {cam_index}...")
    capture = MediaPipeCapture(cam_index=cam_index, webcam_width=1280, webcam_height=720)

    print("[INFO] Initializing expression classifier...")
    classifier = ExpressionClassifier(window_seconds=0.6)

    print("[INFO] Initializing video player...")
    video_player = VideoPlayer(label_to_path=label_to_path, panel_size=(640, 360))

    print("[INFO] Initializing UI manager...")
    ui = UIManager(capture, classifier, video_player)

    print("\n[READY] Starting main loop. Press 'q' to quit, 'm' to toggle metrics overlay.")
    print("-" * 60)

    frame_count = 0
    try:
        while True:
            # Capture frame with landmarks
            data = capture.read()
            if data is None:
                print("[ERROR] Failed to read frame from camera.")
                break

            frame_bgr = data["frame_bgr"]
            pose_landmarks = data["pose_landmarks"]
            face_landmarks = data["face_landmarks"]
            timestamp = data["timestamp"]
            frame_h, frame_w = frame_bgr.shape[:2]

            # Extract metrics
            face_metrics = pose_utils.extract_face_metrics(face_landmarks, (frame_h, frame_w))
            upper_metrics = pose_utils.extract_upper_body_metrics(pose_landmarks, (frame_h, frame_w))
            metrics = {
                "face": face_metrics,
                "upper": upper_metrics,
            }

            # Classify
            label = classifier.update(metrics, timestamp)

            # Update video player
            video_player.set_target(label)

            # Draw and display
            annotated_frame = ui.draw_frame_with_overlay(
                frame_bgr,
                pose_landmarks=pose_landmarks,
                face_landmarks=face_landmarks,
                current_label=label,
                metrics=metrics
            )
            ui.show_frame(annotated_frame)

            # Check for key press
            key = ui.get_key(timeout_ms=1)
            if key is not None:
                if key == ord('q'):
                    print("[INFO] Quit key pressed. Exiting...")
                    break
                elif key == ord('m'):
                    ui.show_metrics = not ui.show_metrics
                    print(f"[INFO] Metrics overlay: {'ON' if ui.show_metrics else 'OFF'}")

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[FRAME {frame_count}] Label: {label:20s} | "
                      f"EAR: {face_metrics.get('ear', 0) if face_metrics else 0:.3f} | "
                      f"MAR: {face_metrics.get('mar', 0) if face_metrics else 0:.3f}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Cleaning up...")
        capture.release()
        video_player.release()
        ui.cleanup()
        print("[INFO] Done!")


if __name__ == "__main__":
    main()
