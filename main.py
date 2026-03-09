"""
main.py – VisionTrigger entry point.

Reads the camera, detects hand gestures with MediaPipe, and fires the
corresponding macOS action configured in config.yaml.

Controls:
  q / Esc – quit
  l       – toggle landmark overlay
"""

import logging
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import yaml

from gestures import GestureRecognizer
from mac_trigger import MacTrigger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Colours (BGR)
_GREEN = (60, 220, 60)
_YELLOW = (0, 200, 255)
_GREY = (140, 140, 140)
_WHITE = (240, 240, 240)
_BLACK = (0, 0, 0)
_CYAN = (255, 210, 0)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def ensure_model(path: Path, url: str) -> None:
    """Download the hand landmarker model if it isn't already present."""
    if path.exists():
        return
    logger.info("Downloading hand landmarker model (~7 MB)…")
    urllib.request.urlretrieve(url, path)
    logger.info("Model saved to %s", path)


def load_config(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def display_label(gesture: str, cfg: dict) -> str:
    """Return a human-readable label for a gesture's bound action."""
    entry = cfg.get(gesture, {})
    if not entry:
        return ""
    if label := entry.get("label"):
        return label
    kind = entry.get("action", "")
    if kind == "shortcut":
        return entry.get("name", "Shortcut")
    if kind == "shell":
        cmd = entry.get("command", "")
        return cmd[:28] + "…" if len(cmd) > 28 else cmd
    if kind == "applescript":
        script = entry.get("script", "")
        return script[:28] + "…" if len(script) > 28 else script
    return kind


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _put(frame, text, pos, scale=0.65, color=_WHITE, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_overlay(
    frame,
    gesture: str,
    action_label: str,
    fps: float,
    hold_progress: float,
    flash_active: bool,
    show_fps: bool,
):
    h, w = frame.shape[:2]

    # ── top bar ────────────────────────────────────────────────────────────
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 68), _BLACK, -1)
    cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    if show_fps:
        _put(frame, f"FPS {fps:4.1f}", (10, 26), scale=0.55, color=_GREEN)

    gesture_color = _GREEN if gesture not in ("UNKNOWN", "NONE", "") else _GREY
    _put(frame, f"Gesture:  {gesture}", (10, 56),
         scale=0.80, color=gesture_color)

    if action_label:
        _put(frame, f"Action:  {action_label}",
             (w - 310, 32), scale=0.65, color=_YELLOW)

    # ── hold-progress bar (bottom) ─────────────────────────────────────────
    if hold_progress > 0.0:
        bx1, by1, bx2, by2 = 10, h - 22, w - 10, h - 6
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
        fill_w = int((bx2 - bx1) * hold_progress)
        cv2.rectangle(frame, (bx1, by1), (bx1 + fill_w, by2), _CYAN, -1)

    # ── trigger flash ──────────────────────────────────────────────────────
    if flash_active:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), _GREEN, -1)
        cv2.addWeighted(flash, 0.15, frame, 0.85, 0, frame)
        label = action_label or gesture
        _put(frame, f"✓ {label}", (w // 2 - 120, h // 2 + 10),
             scale=1.2, color=_GREEN, thickness=3)

    # ── hint ───────────────────────────────────────────────────────────────
    _put(frame, "q / Esc = quit   l = toggle landmarks",
         (10, h - 34), scale=0.45, color=_GREY, thickness=1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ensure_model(MODEL_PATH, MODEL_URL)

    config = load_config(CONFIG_PATH)
    settings = config.get("settings", {})
    gesture_cfgs = config.get("gestures", {})

    recognizer = GestureRecognizer(
        model_path=str(MODEL_PATH),
        min_detection_confidence=settings.get("detection_confidence", 0.7),
        min_tracking_confidence=settings.get("tracking_confidence", 0.5),
    )
    trigger = MacTrigger()
    start_time = time.monotonic()

    cam_idx = settings.get("camera_index", 0)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", cam_idx)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hold_required: int = settings.get("gesture_hold_frames", 20)
    show_landmarks: bool = settings.get("show_landmarks", True)
    show_fps: bool = settings.get("show_fps", True)

    hold_counts: dict[str, int] = {}
    last_trigger_ts: dict[str, float] = {}

    flash_gesture = ""
    flash_ts = 0.0

    fps = 0.0
    fps_frame_cnt = 0
    fps_tick = time.time()

    logger.info("VisionTrigger started. Show your hand to the camera.")
    logger.info("Press  q / Esc  to quit,  l  to toggle landmarks.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        frame = cv2.flip(frame, 1)                       # mirror view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        results = recognizer.process(rgb, timestamp_ms)

        current_gesture = "NONE"
        hold_progress = 0.0
        action_label = ""

        if results.hand_landmarks and results.handedness:
            # list[NormalizedLandmark]
            hand_lm = results.hand_landmarks[0]
            # "Right" or "Left"
            handedness = results.handedness[0][0].category_name

            if show_landmarks:
                recognizer.draw_landmarks(frame, hand_lm)

            current_gesture = recognizer.classify(hand_lm, handedness)
            action_label = display_label(current_gesture, gesture_cfgs)

            if current_gesture not in ("NONE", "UNKNOWN"):
                # Reset counts for any other gesture that was accumulating
                for g in list(hold_counts):
                    if g != current_gesture:
                        hold_counts.pop(g)

                hold_counts[current_gesture] = hold_counts.get(
                    current_gesture, 0) + 1
                hold_progress = min(
                    hold_counts[current_gesture] / hold_required, 1.0)

                if hold_counts[current_gesture] >= hold_required:
                    g_cfg = gesture_cfgs.get(current_gesture, {})
                    cooldown = g_cfg.get("cooldown", 2.0)
                    now = time.time()

                    if g_cfg and now - last_trigger_ts.get(current_gesture, 0.0) >= cooldown:
                        last_trigger_ts[current_gesture] = now
                        hold_counts[current_gesture] = 0
                        flash_gesture = current_gesture
                        flash_ts = now
                        logger.info("Triggered: %s → %s",
                                    current_gesture, action_label)

                        # Run action in background so the camera loop stays smooth
                        threading.Thread(
                            target=trigger.execute,
                            args=(g_cfg,),
                            daemon=True,
                        ).start()
            else:
                hold_counts.clear()
        else:
            # No hand visible – clear all hold counts
            hold_counts.clear()

        # ── FPS ────────────────────────────────────────────────────────────
        fps_frame_cnt += 1
        if fps_frame_cnt >= 30:
            fps = fps_frame_cnt / (time.time() - fps_tick)
            fps_tick = time.time()
            fps_frame_cnt = 0

        flash_active = bool(flash_gesture) and (time.time() - flash_ts < 1.0)

        draw_overlay(
            frame,
            current_gesture,
            action_label,
            fps,
            hold_progress,
            flash_active,
            show_fps,
        )

        cv2.imshow("VisionTrigger", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):          # q or Esc
            break
        if key == ord("l"):
            show_landmarks = not show_landmarks
            logger.info("Landmarks: %s", "ON" if show_landmarks else "OFF")

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    logger.info("VisionTrigger stopped.")


if __name__ == "__main__":
    main()
