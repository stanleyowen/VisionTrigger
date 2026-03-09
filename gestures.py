"""
gestures.py – Hand gesture recognition using MediaPipe Tasks HandLandmarker.

Supported gestures:
  THUMBS_UP   – thumb extended upward, all fingers curled
  THUMBS_DOWN – thumb extended downward, all fingers curled
  OPEN_PALM   – all five fingers extended
  FIST        – all fingers curled
  PEACE       – index + middle extended (V sign)
  POINT       – only index finger extended
  THREE       – index + middle + ring extended
  FOUR        – index + middle + ring + pinky extended
  ROCK        – index + pinky extended (horns)
  CALL        – thumb + pinky extended (phone sign)
"""

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


class GestureRecognizer:
    # Map of finger-state tuples → gesture name.
    # Tuple order: (thumb, index, middle, ring, pinky)  True = extended
    _GESTURE_MAP: dict[tuple, str] = {
        (False, True,  False, False, False): "POINT",
        (False, True,  True,  False, False): "PEACE",
        (True,  True,  True,  True,  True): "OPEN_PALM",
        (False, True,  True,  True,  False): "THREE",
        (False, True,  True,  True,  True): "FOUR",
        (False, False, False, False, False): "FIST",
        (False, True,  False, False, True): "ROCK",
        (True,  False, False, False, True): "CALL",
    }

    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(
            options)
        self._connections = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
        self._draw = mp_vision.drawing_utils

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, rgb_frame, timestamp_ms: int):
        """Run the landmarker on an RGB frame. Returns HandLandmarkerResult."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def classify(self, landmarks, handedness_label: str) -> str:
        """
        Classify the gesture from a list of NormalizedLandmark objects.

        Args:
            landmarks:        hand_landmarks list from HandLandmarkerResult.
            handedness_label: "Right" or "Left" (as reported by MediaPipe,
                              which is mirrored relative to the viewer when
                              the frame is flipped).

        Returns:
            One of the gesture name strings, or "UNKNOWN".
        """
        states = self._finger_states(landmarks, handedness_label)
        thumb, index, middle, ring, pinky = states

        # Thumbs up / down: thumb extended, all fingers curled
        if thumb and not index and not middle and not ring and not pinky:
            # Landmark 9 = middle finger MCP (rough palm centre).
            # y increases downward in image coords.
            if landmarks[4].y < landmarks[9].y:
                return "THUMBS_UP"
            else:
                return "THUMBS_DOWN"

        return self._GESTURE_MAP.get(tuple(states), "UNKNOWN")

    def draw_landmarks(self, bgr_frame, landmark_list) -> None:
        """Overlay hand skeleton onto a BGR frame (in-place)."""
        self._draw.draw_landmarks(
            bgr_frame,
            landmark_list,
            self._connections,
        )

    def close(self) -> None:
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finger_states(landmarks, handedness_label: str) -> list[bool]:
        """
        Return [thumb, index, middle, ring, pinky] True if each is extended.

        Thumb extension is checked on the x-axis (horizontal) because it
        extends sideways relative to the palm.  All other fingers are
        checked on the y-axis (tip above PIP joint → extended).

        MediaPipe reports handedness relative to the *original* (unflipped)
        image, so after a horizontal flip the labels swap.
        """
        lm = landmarks

        # Thumb: tip vs. IP (interphalangeal) joint on the x-axis.
        # For a "Right" result in the flipped-mirror view the thumb tip
        # sits to the left of lm[3] when extended.
        if handedness_label == "Right":
            thumb_ext = lm[4].x < lm[3].x
        else:
            thumb_ext = lm[4].x > lm[3].x

        # Other fingers: tip y < PIP y  (smaller y = higher in image)
        # Tip IDs:  8, 12, 16, 20
        # PIP IDs:  6, 10, 14, 18
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        other_ext = [lm[tip].y < lm[pip].y for tip, pip in finger_pairs]

        return [thumb_ext] + other_ext
