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

# Single source of truth for built-in gesture finger patterns.
# Tuple order: (thumb, index, middle, ring, pinky)  True = extended.
# THUMBS_UP and THUMBS_DOWN share the same finger pattern; direction is
# disambiguated at classification time using the hand's orientation vector.
BUILTIN_GESTURES: dict[str, tuple] = {
    "THUMBS_UP":   (True,  False, False, False, False),
    "THUMBS_DOWN": (True,  False, False, False, False),
    "OPEN_PALM":   (True,  True,  True,  True,  True),
    "PEACE":       (False, True,  True,  False, False),
    "FIST":        (False, False, False, False, False),
    "POINT":       (False, True,  False, False, False),
    "THREE":       (False, True,  True,  True,  False),
    "FOUR":        (False, True,  True,  True,  True),
    "ROCK":        (False, True,  False, False, True),
    "CALL":        (True,  False, False, False, True),
}

_BUILTIN_GESTURE_MAP: dict[tuple, str] = {
    v: k for k, v in BUILTIN_GESTURES.items()
    if k not in ("THUMBS_UP", "THUMBS_DOWN")
}


class GestureRecognizer:
    def __init__(
        self,
        model_path: str,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        custom_gestures: dict | None = None,
    ):
        # Start with built-ins; custom gestures overlay (and can override) them.
        self._gesture_map: dict[tuple, str] = dict(_BUILTIN_GESTURE_MAP)
        if custom_gestures:
            for name, fingers in custom_gestures.items():
                if len(fingers) != 5:
                    raise ValueError(
                        f"Custom gesture '{name}': 'fingers' must have exactly "
                        f"5 boolean values [thumb, index, middle, ring, pinky], "
                        f"got {len(fingers)}."
                    )
                key = tuple(bool(f) for f in fingers)
                self._gesture_map[key] = name

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

    def classify(self, landmarks, handedness_label: str = "") -> tuple[str, tuple]:
        """Returns (gesture_name, finger_states_tuple)."""
        states = self._finger_states(landmarks)
        finger_tuple = tuple(states)
        thumb, index, middle, ring, pinky = states

        if thumb and not index and not middle and not ring and not pinky:
            # Rotation-robust THUMBS_UP/DOWN: use the dot product of
            # (wrist→thumb_tip) with (wrist→middle_MCP).
            # Positive → thumb in same direction as fingers → THUMBS_UP.
            hx = landmarks[9].x - landmarks[0].x
            hy = landmarks[9].y - landmarks[0].y
            tx = landmarks[4].x - landmarks[0].x
            ty = landmarks[4].y - landmarks[0].y
            gesture = "THUMBS_UP" if hx * tx + hy * ty > 0 else "THUMBS_DOWN"
            return gesture, finger_tuple

        return self._gesture_map.get(finger_tuple, "UNKNOWN"), finger_tuple

    def draw_landmarks(self, bgr_frame, landmark_list) -> None:
        """Overlay hand skeleton onto a BGR frame (in-place)."""
        self._draw.draw_landmarks(
            bgr_frame,
            landmark_list,
            self._connections,
        )

    def finger_states(self, landmarks, handedness_label: str = "") -> tuple:
        """Return (thumb, index, middle, ring, pinky) True if each finger is extended."""
        return tuple(self._finger_states(landmarks))

    def close(self) -> None:
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _finger_states(landmarks) -> list[bool]:
        """
        Return [thumb, index, middle, ring, pinky] True if each is extended.

        Thumb extension is checked on the x-axis. The thumb side is inferred
        from whether the thumb MCP (lm[2]) sits left or right of the index
        MCP (lm[5]), so this works for both palm-facing and back-facing
        orientations without needing a handedness label.

        Other fingers use the y-axis (tip above PIP joint → extended).
        """
        lm = landmarks

        # Find which side the thumb is on, then check if the tip has moved
        # further in that direction than the IP joint.
        if lm[2].x > lm[5].x:   # thumb MCP is to the right of index MCP
            thumb_ext = lm[4].x > lm[3].x
        else:                    # thumb MCP is to the left of index MCP
            thumb_ext = lm[4].x < lm[3].x

        # Other fingers: tip y < PIP y  (smaller y = higher in image)
        # Tip IDs:  8, 12, 16, 20
        # PIP IDs:  6, 10, 14, 18
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        other_ext = [lm[tip].y < lm[pip].y for tip, pip in finger_pairs]

        return [thumb_ext] + other_ext
