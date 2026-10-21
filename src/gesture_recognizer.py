"""Gesture rules for hand poses and face expressions."""

from math import hypot


class GestureRecognizer:
    """Recognize gestures from per-frame hand and face landmarks."""

    chin_distance_face_width_ratio = 0.20
    open_mouth_mar_threshold = 0.25
    upward_gaze_ratio_threshold = 0.39

    def recognize_pointing_fist(self, finger_extended):
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

    def recognize_open_hands(self, hands_finger_extended):
        if len(hands_finger_extended) != 2:
            return None

        fingers = ("thumb", "index", "middle", "ring", "pinky")
        if all(
            all(hand.get(f"{finger}_extended") is True for finger in fingers)
            for hand in hands_finger_extended
        ):
            return "open_hands"

        return None

    def recognize_look_up_open_mouth(self, face_landmarks, frame_width, frame_height):
        if face_landmarks is None or len(face_landmarks) < 478:
            return None

        def pixel_point(index):
            landmark = face_landmarks[index]
            return landmark.x * frame_width, landmark.y * frame_height

        mouth_width = hypot(
            pixel_point(61)[0] - pixel_point(291)[0],
            pixel_point(61)[1] - pixel_point(291)[1],
        )
        if mouth_width == 0:
            return None
        mouth_height = hypot(
            pixel_point(13)[0] - pixel_point(14)[0],
            pixel_point(13)[1] - pixel_point(14)[1],
        )
        mouth_aspect_ratio = mouth_height / mouth_width

        eye_landmark_sets = (
            (468, (160, 158), (144, 153)),
            (473, (387, 385), (373, 380)),
        )
        gaze_ratios = []
        for iris_index, upper_lids, lower_lids in eye_landmark_sets:
            upper_y = sum(face_landmarks[index].y for index in upper_lids) / 2
            lower_y = sum(face_landmarks[index].y for index in lower_lids) / 2
            eye_height = lower_y - upper_y
            if eye_height <= 0:
                return None
            gaze_ratios.append(
                (face_landmarks[iris_index].y - upper_y) / eye_height
            )

        average_gaze_ratio = sum(gaze_ratios) / len(gaze_ratios)
        if (
            mouth_aspect_ratio >= self.open_mouth_mar_threshold
            and average_gaze_ratio <= self.upward_gaze_ratio_threshold
        ):
            return "look_up_open_mouth"
        return None

    def recognize_index_finger_on_chin(
        self,
        finger_extended,
        hand_landmarks,
        face_landmarks,
        frame_width,
        frame_height,
    ):
        if not finger_extended or hand_landmarks is None or face_landmarks is None:
            return None

        if (
            finger_extended.get("index_extended") is not True
            or any(
                finger_extended.get(f"{finger}_extended") is not False
                for finger in ("thumb", "middle", "ring", "pinky")
            )
        ):
            return None

        if len(hand_landmarks) <= 8 or len(face_landmarks) <= 454:
            return None

        index_tip = hand_landmarks[8]
        chin = face_landmarks[152]
        face_left = face_landmarks[234]
        face_right = face_landmarks[454]
        index_x = index_tip.x * frame_width
        index_y = index_tip.y * frame_height
        chin_x = chin.x * frame_width
        chin_y = chin.y * frame_height
        face_width = abs(face_right.x - face_left.x) * frame_width
        if face_width == 0:
            return None

        chin_distance = hypot(index_x - chin_x, index_y - chin_y)
        if chin_distance <= face_width * self.chin_distance_face_width_ratio:
            return "index_finger_on_chin"
        return None