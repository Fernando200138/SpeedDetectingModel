from collections import deque
import unittest

from src.live_capture import LiveLandmarkCapture


class FakeOverlay:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1


def make_capture_state():
    capture = LiveLandmarkCapture.__new__(LiveLandmarkCapture)
    capture.gesture_history = deque(maxlen=30)
    capture.triggered_gestures = set()
    capture.active_gesture = None
    capture.gesture_recognition_enabled = True
    capture.video_overlay = FakeOverlay()
    capture.image_overlay = FakeOverlay()
    capture.chin_video_overlay = FakeOverlay()
    capture.open_mouth_video_overlay = FakeOverlay()
    return capture


class GestureVotingTests(unittest.TestCase):
    def test_waits_for_full_window_before_triggering(self):
        capture = make_capture_state()

        results = [capture._update_gesture_state("pointing_fist") for _ in range(15)]

        self.assertEqual(results, [None] * 15)

    def test_fourteen_votes_do_not_trigger_but_fifteen_do(self):
        capture = make_capture_state()

        results = [capture._update_gesture_state(None) for _ in range(16)]
        results.extend(
            capture._update_gesture_state("pointing_fist") for _ in range(14)
        )
        self.assertTrue(all(result is None for result in results))

        result = capture._update_gesture_state("pointing_fist")

        self.assertEqual(result, "pointing_fist")

    def test_triggers_once_and_rearms_after_vote_count_drops_below_threshold(self):
        capture = make_capture_state()

        results = [capture._update_gesture_state("pointing_fist") for _ in range(30)]
        self.assertEqual(results.count("pointing_fist"), 1)
        self.assertTrue(all(result is None for result in results[:29]))

        results = [capture._update_gesture_state(None) for _ in range(15)]
        self.assertTrue(all(result is None for result in results))
        self.assertIn("pointing_fist", capture.triggered_gestures)

        result = capture._update_gesture_state(None)
        self.assertNotIn("pointing_fist", capture.triggered_gestures)
        self.assertIsNone(result)

        results = [
            capture._update_gesture_state("pointing_fist") for _ in range(15)
        ]
        self.assertEqual(results.count("pointing_fist"), 1)

    def test_new_gesture_triggers_after_its_votes_reach_threshold(self):
        capture = make_capture_state()
        results = [capture._update_gesture_state("pointing_fist") for _ in range(30)]
        self.assertEqual(results.count("pointing_fist"), 1)

        results = [capture._update_gesture_state("open_hands") for _ in range(15)]

        self.assertTrue(all(result is None for result in results[:14]))
        self.assertEqual(results[14], "open_hands")


class RecognitionToggleTests(unittest.TestCase):
    def test_disabling_clears_vote_state_and_stops_every_overlay(self):
        capture = make_capture_state()
        capture.gesture_history.extend(["pointing_fist"] * 20)
        capture.triggered_gestures.add("pointing_fist")
        capture.active_gesture = "pointing_fist"

        capture.set_gesture_recognition_enabled(False)

        self.assertFalse(capture.gesture_recognition_enabled)
        self.assertEqual(list(capture.gesture_history), [])
        self.assertEqual(capture.triggered_gestures, set())
        self.assertIsNone(capture.active_gesture)
        for overlay in (
            capture.video_overlay,
            capture.image_overlay,
            capture.chin_video_overlay,
            capture.open_mouth_video_overlay,
        ):
            self.assertEqual(overlay.stop_calls, 1)

    def test_enabling_recognition_does_not_restart_overlays(self):
        capture = make_capture_state()
        capture.gesture_recognition_enabled = False
        capture.gesture_history.extend(["pointing_fist"] * 20)
        capture.triggered_gestures.add("pointing_fist")

        capture.set_gesture_recognition_enabled(True)

        self.assertTrue(capture.gesture_recognition_enabled)
        self.assertEqual(list(capture.gesture_history), [])
        self.assertEqual(capture.triggered_gestures, set())
        for overlay in (
            capture.video_overlay,
            capture.image_overlay,
            capture.chin_video_overlay,
            capture.open_mouth_video_overlay,
        ):
            self.assertEqual(overlay.start_calls, 0)


if __name__ == "__main__":
    unittest.main()