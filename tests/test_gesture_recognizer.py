from types import SimpleNamespace
import unittest

from src.gesture_recognizer import GestureRecognizer


def landmark(x=0.5, y=0.5):
    return SimpleNamespace(x=x, y=y)


def open_mouth_looking_up_face():
    face = [landmark() for _ in range(478)]
    face[61] = landmark(0.4, 0.6)
    face[291] = landmark(0.6, 0.6)
    face[13] = landmark(0.5, 0.64)
    face[14] = landmark(0.5, 0.56)

    for index in (160, 158, 387, 385):
        face[index] = landmark(y=0.3)
    for index in (144, 153, 373, 380):
        face[index] = landmark(y=0.4)
    face[468] = landmark(y=0.32)
    face[473] = landmark(y=0.32)
    return face


class GestureRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = GestureRecognizer()

    def test_pointing_fist_requires_only_thumb_extended(self):
        pointing_fist = {
            "thumb_extended": True,
            "index_extended": False,
            "middle_extended": False,
            "ring_extended": False,
            "pinky_extended": False,
        }
        wrong_finger_pattern = dict(pointing_fist, index_extended=True)

        self.assertEqual(
            self.recognizer.recognize_pointing_fist(pointing_fist), "pointing_fist"
        )
        self.assertIsNone(
            self.recognizer.recognize_pointing_fist(wrong_finger_pattern)
        )
        self.assertIsNone(self.recognizer.recognize_pointing_fist(None))

    def test_open_hands_requires_two_fully_extended_hands(self):
        open_hand = {
            f"{finger}_extended": True
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        }
        curled_finger_hand = dict(open_hand, pinky_extended=False)

        self.assertEqual(
            self.recognizer.recognize_open_hands([open_hand, open_hand]), "open_hands"
        )
        self.assertIsNone(
            self.recognizer.recognize_open_hands([open_hand, curled_finger_hand])
        )
        self.assertIsNone(self.recognizer.recognize_open_hands([open_hand]))

    def test_look_up_open_mouth_rejects_closed_mouth_and_downward_gaze(self):
        face = open_mouth_looking_up_face()

        self.assertEqual(
            self.recognizer.recognize_look_up_open_mouth(face, 1000, 1000),
            "look_up_open_mouth",
        )

        closed_mouth = list(face)
        closed_mouth[14] = landmark(0.5, 0.62)
        self.assertIsNone(
            self.recognizer.recognize_look_up_open_mouth(
                closed_mouth, 1000, 1000
            )
        )

        downward_gaze = list(face)
        downward_gaze[468] = landmark(y=0.38)
        downward_gaze[473] = landmark(y=0.38)
        self.assertIsNone(
            self.recognizer.recognize_look_up_open_mouth(
                downward_gaze, 1000, 1000
            )
        )
        self.assertIsNone(
            self.recognizer.recognize_look_up_open_mouth(face[:477], 1000, 1000)
        )

    def test_index_finger_on_chin_rejects_fingertip_too_far_away(self):
        index_only = {
            "thumb_extended": False,
            "index_extended": True,
            "middle_extended": False,
            "ring_extended": False,
            "pinky_extended": False,
        }
        hand = [landmark() for _ in range(9)]
        hand[8] = landmark(0.5, 0.7)
        face = [landmark() for _ in range(455)]
        face[152] = landmark(0.5, 0.7)
        face[234] = landmark(0.3, 0.5)
        face[454] = landmark(0.7, 0.5)

        self.assertEqual(
            self.recognizer.recognize_index_finger_on_chin(
                index_only, hand, face, 1000, 1000
            ),
            "index_finger_on_chin",
        )

        hand[8] = landmark(0.5, 0.8)
        self.assertIsNone(
            self.recognizer.recognize_index_finger_on_chin(
                index_only, hand, face, 1000, 1000
            )
        )


if __name__ == "__main__":
    unittest.main()