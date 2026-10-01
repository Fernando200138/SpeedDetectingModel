"""Run MediaPipe face, pose, and hand landmark detection on one image."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp


HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

FINGER_JOINTS = {
    "index": ((5, 6, 7), (6, 7, 8)),
    "middle": ((9, 10, 11), (10, 11, 12)),
    "ring": ((13, 14, 15), (14, 15, 16)),
    "pinky": ((17, 18, 19), (18, 19, 20)),
}

FINGER_LABELS = {
    "index": "IDX",
    "middle": "MID",
    "ring": "RNG",
    "pinky": "PNK",
}


def _angle_degrees(point_a, point_b, point_c):
    """Return the angle at point_b, where a straight joint is 180 degrees."""
    vector_a = [point_a[axis] - point_b[axis] for axis in ("x", "y")]
    vector_c = [point_c[axis] - point_b[axis] for axis in ("x", "y")]
    length_a = math.sqrt(sum(value * value for value in vector_a))
    length_c = math.sqrt(sum(value * value for value in vector_c))
    if length_a == 0 or length_c == 0:
        return None
    cosine = sum(a * c for a, c in zip(vector_a, vector_c)) / (length_a * length_c)
    return math.degrees(np.arccos(cosine))


def _distance(point_a, point_b):
    return math.sqrt(sum(
        (point_a[axis] - point_b[axis]) ** 2 for axis in ("x", "y")
    ))


def extract_hand_features(image_landmarks):
    """Derive hand angles and scale-normalized features in image pixel space."""
    if len(image_landmarks) != len(HAND_LANDMARK_NAMES):
        return None

    joint_angles = {}
    extended_fingers = {}
    for finger, (pip_points, dip_points) in FINGER_JOINTS.items():
        pip_angle = _angle_degrees(*(image_landmarks[index] for index in pip_points))
        dip_angle = _angle_degrees(*(image_landmarks[index] for index in dip_points))
        joint_angles[f"{finger}_pip"] = pip_angle
        joint_angles[f"{finger}_dip"] = dip_angle
        extended_fingers[f"{finger}_extended"] = (
            pip_angle is not None
            and dip_angle is not None
            and (pip_angle + dip_angle) / 2.0 >= 120.0
        )

    thumb_mcp_angle = _angle_degrees(
        image_landmarks[1], image_landmarks[2], image_landmarks[3]
    )
    palm_length = _distance(image_landmarks[0], image_landmarks[9])
    thumb_separation = (
        _distance(image_landmarks[4], image_landmarks[5]) / palm_length
        if palm_length > 0
        else None
    )
    joint_angles["thumb_mcp"] = thumb_mcp_angle
    extended_fingers["thumb_extended"] = (
        thumb_mcp_angle is not None and thumb_mcp_angle > 165.0
    )

    return {
        "coordinate_space": "image_pixels_2d",
        "finger_extended": extended_fingers,
        "joint_angles_degrees": joint_angles,
        "thumb_tip_to_index_mcp_over_palm_length": thumb_separation,
        "thresholds": {
            "finger_average_pip_dip_angle_min_degrees": 120.0,
            "thumb_mcp_angle_min_degrees_strictly_greater_than": 165.0,
        },
    }


def serialize_hands(hand_result, image_shape):
    """Convert MediaPipe hand results to JSON-serializable landmark data."""
    image_height, image_width = image_shape[:2]
    hands = []
    detected_hands = hand_result.multi_hand_landmarks or []
    handedness_results = hand_result.multi_handedness or []
    world_results = hand_result.multi_hand_world_landmarks or []

    for hand_index, hand_landmarks in enumerate(detected_hands):
        classification = None
        if hand_index < len(handedness_results):
            classifications = handedness_results[hand_index].classification
            if classifications:
                classification = classifications[0]

        landmarks = []
        image_landmarks = []
        for landmark_index, landmark in enumerate(hand_landmarks.landmark):
            pixel_x = landmark.x * image_width
            pixel_y = landmark.y * image_height
            image_landmarks.append({"x": pixel_x, "y": pixel_y})
            landmarks.append({
                "id": landmark_index,
                "name": HAND_LANDMARK_NAMES[landmark_index],
                "x": landmark.x,
                "y": landmark.y,
                "pixel_x": round(pixel_x),
                "pixel_y": round(pixel_y),
            })

        world_landmarks = []
        if hand_index < len(world_results):
            world_landmarks = [
                {
                    "id": landmark_index,
                    "name": HAND_LANDMARK_NAMES[landmark_index],
                    "x": landmark.x,
                    "y": landmark.y,
                }
                for landmark_index, landmark in enumerate(
                    world_results[hand_index].landmark
                )
            ]

        hands.append({
            "label": classification.label if classification else None,
            "score": classification.score if classification else None,
            "landmarks": landmarks,
            "world_landmarks_meters": world_landmarks,
            "features": extract_hand_features(image_landmarks),
        })

    connections = [
        {
            "start_id": start,
            "start_name": HAND_LANDMARK_NAMES[start],
            "end_id": end,
            "end_name": HAND_LANDMARK_NAMES[end],
        }
        for start, end in sorted(mp.solutions.hands.HAND_CONNECTIONS)
    ]

    return {
        "coordinate_notes": {
            "x_y": "Normalized image coordinates in [0, 1]; multiply by image width or height for pixels.",
            "z": "Relative depth estimate in image-landmark coordinates; not centimeters.",
            "world_landmarks_meters": "MediaPipe's hand-centered 3D coordinates in meters, when available.",
        },
        "connections": connections,
        "hands": hands,
    }


def annotate_image(image, show_joint_labels=True):
    """Return an annotated image, detection counts, and serialized hand data."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose_model, mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_model, mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hand_model:
        pose_result = pose_model.process(image_rgb)
        face_result = face_model.process(image_rgb)
        hand_result = hand_model.process(image_rgb)

    annotated = image.copy()
    drawing = mp.solutions.drawing_utils

    face_landmarks = None
    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0]
        drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(
                color=(80, 220, 80), thickness=1, circle_radius=1
            ),
            connection_drawing_spec=drawing.DrawingSpec(
                color=(80, 120, 80), thickness=1, circle_radius=1
            ),
        )

    pose_landmarks = pose_result.pose_landmarks
    if pose_landmarks:
        drawing.draw_landmarks(
            image=annotated,
            landmark_list=pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
        )

    hands = hand_result.multi_hand_landmarks or []
    hand_data = serialize_hands(hand_result, image.shape)
    image_height, image_width = image.shape[:2]
    label_scale = max(0.45, min(0.8, image_height / 1200.0))*0.5
    line_height = max(20, int(30 * label_scale))
    panel_y = 12
    for hand_index, hand_landmarks in enumerate(hands):
        drawing.draw_landmarks(
            image=annotated,
            landmark_list=hand_landmarks,
            connections=mp.solutions.hands.HAND_CONNECTIONS,
        )
        if not show_joint_labels:
            continue

        features = hand_data["hands"][hand_index]["features"]
        angle_labels = [("TH MCP", 2, features["joint_angles_degrees"].get("thumb_mcp"))]
        for finger, (pip_points, dip_points) in FINGER_JOINTS.items():
            angle_labels.extend((
                (f"{FINGER_LABELS[finger]} PIP", pip_points[1],
                 features["joint_angles_degrees"].get(f"{finger}_pip")),
                (f"{FINGER_LABELS[finger]} DIP", dip_points[1],
                 features["joint_angles_degrees"].get(f"{finger}_dip")),
            ))

        legend_lines = []
        for joint_name, landmark_index, angle in angle_labels:
            if angle is None:
                continue
            landmark = hand_landmarks.landmark[landmark_index]
            point_x = int(landmark.x * image_width)
            point_y = int(landmark.y * image_height)
            legend_lines.append(f"{joint_name} #{landmark_index}: {angle:.0f} deg")
            text_origin = (min(point_x + 7, image_width - 20), max(18, point_y - 7))
            cv2.putText(
                annotated,
                f"{joint_name}"+str(landmark_index),
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                label_scale,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )


        handedness = hand_data["hands"][hand_index]["label"] or f"Hand {hand_index + 1}"
        legend_lines.insert(0, f"{handedness}: joint angle (landmark ID)")
        panel_width = max(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)[0][0]
            for line in legend_lines
        ) + 20
        panel_height = line_height * len(legend_lines) + 12
        panel_x = 12
        panel_bottom = min(panel_y + panel_height, image_height - 1)
        panel_right = min(panel_x + panel_width, image_width - 1)
        panel_roi = annotated[panel_y:panel_bottom, panel_x:panel_right]
        if panel_roi.size:
            dark_panel = panel_roi.copy()
            dark_panel[:] = (20, 20, 20)
            cv2.addWeighted(dark_panel, 0.78, panel_roi, 0.22, 0, panel_roi)
            for line_index, line in enumerate(legend_lines):
                text_y = panel_y + 8 + (line_index + 1) * line_height
                if text_y >= panel_bottom:
                    break
                cv2.putText(
                    annotated,
                    line,
                    (panel_x + 8, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        panel_y = panel_bottom + 10

    counts = {
        "face": len(face_landmarks.landmark) if face_landmarks else 0,
        "pose": len(pose_landmarks.landmark) if pose_landmarks else 0,
        "hands": [len(hand.landmark) for hand in hands],
    }
    return annotated, counts, hand_data


def main():
    parser = argparse.ArgumentParser(
        description="Display and save MediaPipe face, pose, and hand landmarks for an image."
    )
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the annotated image (defaults to <image>_landmarks.<ext>)",
    )
    parser.add_argument(
        "--landmarks-output",
        type=Path,
        help="Path for hand landmark JSON (defaults to <image>_hand_landmarks.json)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save the annotated image without opening a display window",
    )
    parser.add_argument(
        "--no-joint-labels",
        action="store_true",
        help="Hide joint number and angle text while keeping landmark points and lines",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Could not read image: {args.image}")

    annotated, counts, hand_data = annotate_image(
        image, show_joint_labels=not args.no_joint_labels
    )
    test_images_dir = Path(__file__).parent.parent / "data" / "test_images"
    output_path = args.output or (
        test_images_dir
        / "test_annotated_image"
        / f"{args.image.stem}_landmarks{args.image.suffix}"
    )
    landmarks_path = args.landmarks_output or (
        test_images_dir
        / "test_json"
        / f"{args.image.stem}_hand_landmarks.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        parser.error(f"Could not save annotated image: {output_path}")
    try:
        landmarks_path.write_text(
            json.dumps(hand_data, indent=2), encoding="utf-8"
        )
    except OSError as error:
        parser.error(f"Could not save hand landmarks to {landmarks_path}: {error}")

    print(f"Annotated image saved to: {output_path}")
    print(f"Hand landmark data saved to: {landmarks_path}")
    print(f"Face landmarks: {counts['face']}")
    print(f"Pose landmarks: {counts['pose']}")
    print(f"Hands detected: {len(counts['hands'])}")
    for hand_index, landmark_count in enumerate(counts["hands"], start=1):
        print(f"  Hand {hand_index} landmarks: {landmark_count}")

    if not args.no_display:
        cv2.imshow("MediaPipe landmarks", annotated)
        print("Press any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()