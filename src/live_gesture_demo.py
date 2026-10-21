"""CLI entry point for the live webcam gesture demo."""

import argparse
from queue import Empty, Queue
from threading import Thread

import cv2
import pyvirtualcam

from .live_capture import LiveLandmarkCapture


def main():
    parser = argparse.ArgumentParser(
        description="Show live MediaPipe landmarks and recognize hand and face gestures."
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
    command_queue = Queue()

    def read_commands():
        while True:
            try:
                command = input().strip().lower()
            except EOFError:
                return
            command_queue.put(command)

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
            print("Local preview enabled. Press 'q' to quit or 'm' to toggle meme triggers.")
        elif args.virtual_camera:
            print("Virtual camera output active. Press Ctrl+C to stop.")
        print(
            "Terminal commands: 'stop recognition', 'start recognition', "
            "or 'toggle recognition'."
        )
        Thread(target=read_commands, daemon=True).start()

        while True:
            while True:
                try:
                    command = command_queue.get_nowait()
                except Empty:
                    break

                if command in ("stop", "stop recognition", "pause recognition"):
                    app.set_gesture_recognition_enabled(False)
                elif command in ("start", "start recognition", "resume recognition"):
                    app.set_gesture_recognition_enabled(True)
                elif command in ("toggle", "toggle recognition"):
                    app.set_gesture_recognition_enabled(
                        not app.gesture_recognition_enabled
                    )
                else:
                    print("Unknown command. Use 'stop recognition', 'start recognition', or 'toggle recognition'.")
                    continue
                state = "enabled" if app.gesture_recognition_enabled else "paused"
                print(f"Meme recognition {state}; camera output continues.")

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
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    app.set_gesture_recognition_enabled(
                        not app.gesture_recognition_enabled
                    )
                    state = "enabled" if app.gesture_recognition_enabled else "paused"
                    print(f"Meme recognition {state}; camera output continues.")
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