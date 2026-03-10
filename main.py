"""
main.py – VisionTrigger entry point.

Reads the camera, detects hand gestures with MediaPipe, and fires the
corresponding macOS action configured in config.yaml.

Controls:
  q / Esc – quit
  l       – toggle landmark overlay
  g       – show/hide gesture list
  r       – register a new gesture from camera
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
_ORANGE = (0, 140, 255)

# Gesture registration states
REG_IDLE = "idle"
REG_CAPTURE = "capture"
REG_NAME = "name"
REG_ACTION_TYPE = "action_type"
REG_ACTION_DETAIL = "action_detail"
REG_FILE_PICK = "file_pick"
REG_CONFIRM = "confirm"
REG_STABLE_REQUIRED = 40   # consecutive frames of the same pattern needed

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
SCRIPTS_DIR = Path(__file__).parent / "scripts"
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


def save_custom_gesture(
    config_path: Path,
    name: str,
    fingers: tuple,
    action_type: str,
    action_detail: str,
    label: str,
    cooldown: float = 2.0,
) -> dict:
    """Persist a new custom gesture to config.yaml and return the entry dict."""
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config.get("custom_gestures"), dict):
        config["custom_gestures"] = {}
    entry: dict = {
        "fingers": list(fingers),
        "action": action_type,
        "label": label or name,
        "cooldown": cooldown,
    }
    if action_type == "shell":
        entry["command"] = action_detail
    elif action_type == "applescript":
        entry["script"] = action_detail
    elif action_type == "shortcut":
        entry["name"] = action_detail
    config["custom_gestures"][name] = entry
    with open(config_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    return entry


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


# Extensions accepted when browsing the scripts upload folder
_SCRIPT_EXTS: dict[str, set[str]] = {
    "applescript": {".applescript"},
    "shell": {".sh", ".bash", ".zsh", ".command"},
}


def _get_script_files(action_type: str) -> list:
    """Return sorted Path list for script files in SCRIPTS_DIR matching action_type."""
    SCRIPTS_DIR.mkdir(exist_ok=True)
    exts = _SCRIPT_EXTS.get(action_type, set())
    return sorted(
        f for f in SCRIPTS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )


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
    _put(frame, "q/Esc = quit   l = landmarks   g = gesture list   r = register",
         (10, h - 34), scale=0.45, color=_GREY, thickness=1)


def _finger_display_str(fingers: tuple) -> str:
    """Format a 5-bool finger tuple as a compact readable string."""
    labels = ("Th", "Ix", "Md", "Rg", "Pk")
    return "  ".join(f"{l}:{'Y' if v else 'N'}" for l, v in zip(labels, fingers))


def draw_gestures_list_overlay(frame, gesture_cfgs: dict) -> None:
    """Draw a semi-transparent panel listing all configured gestures."""
    h, w = frame.shape[:2]
    rows = list(gesture_cfgs.items())
    row_h = 26
    padding = 16
    header_h = 36
    max_visible = min(len(rows), (h - 120) // row_h)
    panel_h = header_h + max_visible * row_h + padding
    panel_w = 560
    px1 = w - panel_w - 16
    py1 = 80
    px2 = w - 16
    py2 = py1 + panel_h

    panel = frame.copy()
    cv2.rectangle(panel, (px1, py1), (px2, py2), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _CYAN, 2)

    lx = px1 + 14
    y = py1 + 24
    _put(frame, f"GESTURES  ({len(gesture_cfgs)})  —  press g to close",
         (lx, y), scale=0.58, color=_CYAN)
    y += header_h - 6
    cv2.line(frame, (px1 + 8, y - 6), (px2 - 8, y - 6), _GREY, 1)

    for name, cfg in rows[:max_visible]:
        if not isinstance(cfg, dict):
            continue
        label = cfg.get("label") or name
        action = cfg.get("action", "?")
        detail = (
            cfg.get("command") or cfg.get("script") or cfg.get("name") or ""
        )
        detail_short = detail[:30] + ("…" if len(detail) > 30 else "")
        action_color = {
            "shell": _GREEN,
            "applescript": _YELLOW,
            "shortcut": _CYAN,
        }.get(action, _WHITE)
        _put(frame, label, (lx, y), scale=0.60, color=_WHITE)
        tag = f"[{action}]"
        _put(frame, tag, (lx + 180, y), scale=0.52, color=action_color)
        _put(frame, detail_short, (lx + 270, y),
             scale=0.47, color=_GREY, thickness=1)
        y += row_h

    if len(rows) > max_visible:
        _put(frame,
             f"  … and {len(rows) - max_visible} more (scroll not supported)",
             (lx, y + 6), scale=0.44, color=_GREY, thickness=1)


def draw_registration_overlay(
    frame,
    reg_state: str,
    reg_input_buf: str,
    reg_fingers,
    reg_name: str,
    reg_action_type: str,
    reg_action_detail: str,
    reg_stable_count: int,
    current_fingers,
    reg_selected_filename: str = "",
) -> None:
    h, w = frame.shape[:2]
    px1, py1 = w // 2 - 310, h // 2 - 140
    px2, py2 = w // 2 + 310, h // 2 + 155
    panel = frame.copy()
    cv2.rectangle(panel, (px1, py1), (px2, py2), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _ORANGE, 2)

    lx, y = px1 + 18, py1 + 30
    _put(frame, "REGISTER GESTURE", (lx, y), scale=0.75, color=_ORANGE)
    _put(frame, "Esc to cancel", (px2 - 168, y), scale=0.48,
         color=_GREY, thickness=1)
    y += 38

    if reg_state == REG_CAPTURE:
        _put(frame, "Hold your gesture steady...", (lx, y), color=_YELLOW)
        y += 32
        if current_fingers is not None:
            _put(frame, _finger_display_str(current_fingers), (lx, y),
                 scale=0.58, color=_WHITE)
        y += 28
        bx1, by1, bx2, by2 = lx, y, px2 - 18, y + 18
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
        progress = min(reg_stable_count / REG_STABLE_REQUIRED, 1.0)
        cv2.rectangle(frame, (bx1, by1),
                      (bx1 + int((bx2 - bx1) * progress), by2), _ORANGE, -1)
        y += 26
        _put(frame, f"Stable: {reg_stable_count}/{REG_STABLE_REQUIRED} frames",
             (lx, y), scale=0.50, color=_GREY, thickness=1)

    elif reg_state == REG_NAME:
        _put(frame, f"Captured: {_finger_display_str(reg_fingers)}",
             (lx, y), scale=0.58, color=_GREEN)
        y += 32
        _put(frame, "Enter a name for this gesture:", (lx, y))
        y += 32
        _put(frame, f"> {reg_input_buf}_", (lx, y), color=_CYAN)
        y += 36
        _put(frame, "Press Enter to confirm", (lx, y), scale=0.50,
             color=_GREY, thickness=1)

    elif reg_state == REG_ACTION_TYPE:
        _put(frame, f"Name: {reg_name}", (lx, y), color=_GREEN)
        y += 36
        _put(frame, "Choose action type:", (lx, y))
        y += 32
        _put(frame, "  S   Shell script       (pick .sh file from scripts/)",
             (lx, y), scale=0.58)
        y += 28
        _put(frame, "  A   AppleScript        (pick .applescript file from scripts/)",
             (lx, y), scale=0.58)
        y += 28
        _put(frame, "  K   Shortcut           (type Shortcuts.app name)",
             (lx, y), scale=0.58)

    elif reg_state == REG_ACTION_DETAIL:
        _put(frame, f"{reg_name}  [{reg_action_type}]", (lx, y), color=_GREEN)
        y += 32
        _put(frame, "Shortcut name:", (lx, y))
        y += 32
        _put(frame, f"> {reg_input_buf}_", (lx, y), color=_CYAN)
        y += 36
        _put(frame, "Press Enter to confirm", (lx, y), scale=0.50,
             color=_GREY, thickness=1)

    elif reg_state == REG_CONFIRM:
        _put(frame, "Save this gesture?", (lx, y), color=_YELLOW)
        y += 32
        _put(frame, f"Name:    {reg_name}", (lx, y), color=_GREEN, scale=0.60)
        y += 28
        _put(frame,
             f"Pattern: {_finger_display_str(reg_fingers) if reg_fingers else '?'}",
             (lx, y), scale=0.55)
        y += 28
        if reg_selected_filename:
            preview = reg_selected_filename
        else:
            preview = reg_action_detail[:42] + \
                ("..." if len(reg_action_detail) > 42 else "")
        _put(
            frame, f"Action:  [{reg_action_type}]  {preview}", (lx, y), scale=0.55)
        y += 38
        _put(frame, "Y  Save        N  Cancel", (lx, y), color=_YELLOW)


def draw_file_pick_overlay(
    frame,
    action_type: str,
    file_list: list,
    cursor: int,
) -> None:
    """Draw a file-picker panel so the user can select an uploaded script file."""
    h, w = frame.shape[:2]
    px1, py1 = w // 2 - 340, h // 2 - 170
    px2, py2 = w // 2 + 340, h // 2 + 195
    panel = frame.copy()
    cv2.rectangle(panel, (px1, py1), (px2, py2), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _ORANGE, 2)

    lx, y = px1 + 18, py1 + 30
    title = ("SELECT APPLESCRIPT FILE"
             if action_type == "applescript" else "SELECT SHELL SCRIPT FILE")
    _put(frame, title, (lx, y), scale=0.72, color=_ORANGE)
    _put(frame, "Esc to cancel", (px2 - 170, y),
         scale=0.48, color=_GREY, thickness=1)
    y += 34

    folder_str = str(SCRIPTS_DIR)
    if len(folder_str) > 58:
        folder_str = "\u2026" + folder_str[-55:]
    _put(frame, f"Folder: {folder_str}", (lx, y),
         scale=0.44, color=_GREY, thickness=1)
    y += 32

    if not file_list:
        ext_hint = (".applescript"
                    if action_type == "applescript" else ".sh / .bash / .zsh")
        _put(frame, "No script files found.", (lx, y), color=_YELLOW)
        y += 30
        _put(frame, f"Drop {ext_hint} files into the scripts/ folder,",
             (lx, y), scale=0.54, color=_GREY, thickness=1)
        y += 24
        _put(frame, "then press r again to restart registration.",
             (lx, y), scale=0.54, color=_GREY, thickness=1)
    else:
        _put(frame, "j / k  =  navigate          Enter  =  select",
             (lx, y), scale=0.50, color=_GREY, thickness=1)
        y += 30
        row_h = 30
        max_visible = max(1, min(len(file_list), (py2 - y - 24) // row_h))
        start = max(0, min(cursor - max_visible // 2,
                           len(file_list) - max_visible))
        for idx in range(start, min(start + max_visible, len(file_list))):
            fp = file_list[idx]
            color = _CYAN if idx == cursor else _WHITE
            prefix = "\u25b6  " if idx == cursor else "   "
            _put(frame, f"{prefix}{fp.name}", (lx, y), scale=0.62, color=color)
            y += row_h
        if len(file_list) > max_visible:
            _put(frame, f"  \u2026 {len(file_list)} files total",
                 (lx, y + 4), scale=0.44, color=_GREY, thickness=1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ensure_model(MODEL_PATH, MODEL_URL)
    SCRIPTS_DIR.mkdir(exist_ok=True)

    config = load_config(CONFIG_PATH)
    settings = config.get("settings", {})
    gesture_cfgs = config.get("gestures") or {}
    custom_gesture_cfgs = config.get("custom_gestures") or {}

    # Build the finger-pattern dict consumed by GestureRecognizer.
    custom_patterns = {
        name: cfg["fingers"]
        for name, cfg in custom_gesture_cfgs.items()
        if isinstance(cfg, dict) and "fingers" in cfg
    }

    # Merge custom action configs so the trigger/display logic finds them.
    gesture_cfgs.update(custom_gesture_cfgs)

    recognizer = GestureRecognizer(
        model_path=str(MODEL_PATH),
        min_detection_confidence=settings.get("detection_confidence", 0.7),
        min_tracking_confidence=settings.get("tracking_confidence", 0.5),
        custom_gestures=custom_patterns,
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

    show_gestures: bool = False

    # -- Gesture registration state ----------------------------------------
    reg_state = REG_IDLE
    reg_fingers = None   # captured finger-state tuple (bool * 5)
    reg_stable_count = 0
    reg_prev_fingers = None
    reg_current_fingers = None
    reg_name = ""
    reg_action_type = ""
    reg_action_detail = ""
    reg_input_buf = ""
    reg_file_list: list = []
    reg_file_cursor: int = 0
    reg_selected_filename: str = ""

    fps = 0.0
    fps_frame_cnt = 0
    fps_tick = time.time()

    logger.info("VisionTrigger started. Show your hand to the camera.")
    logger.info(
        "Press  q / Esc  to quit,  l  to toggle landmarks,  r  to register a gesture.")

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

            # ── Registration capture ─────────────────────────────────────
            if reg_state == REG_CAPTURE:
                fingers = recognizer.finger_states(hand_lm, handedness)
                if fingers == reg_prev_fingers:
                    reg_stable_count += 1
                else:
                    reg_stable_count = 0
                    reg_prev_fingers = fingers
                reg_current_fingers = fingers
                if reg_stable_count >= REG_STABLE_REQUIRED:
                    reg_fingers = fingers
                    reg_stable_count = 0
                    reg_prev_fingers = None
                    reg_input_buf = ""
                    reg_state = REG_NAME

            # ── Normal action triggering (only in idle mode) ─────────────
            elif reg_state == REG_IDLE:
                if current_gesture not in ("NONE", "UNKNOWN"):
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
            if reg_state == REG_CAPTURE:
                reg_stable_count = 0
                reg_current_fingers = None

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

        if reg_state not in (REG_IDLE, REG_FILE_PICK):
            draw_registration_overlay(
                frame,
                reg_state,
                reg_input_buf,
                reg_fingers,
                reg_name,
                reg_action_type,
                reg_action_detail,
                reg_stable_count,
                reg_current_fingers,
                reg_selected_filename,
            )

        if reg_state == REG_FILE_PICK:
            draw_file_pick_overlay(
                frame, reg_action_type, reg_file_list, reg_file_cursor)

        if show_gestures and reg_state == REG_IDLE:
            draw_gestures_list_overlay(frame, gesture_cfgs)

        cv2.imshow("VisionTrigger", frame)

        key = cv2.waitKey(1) & 0xFF

        # ── Registration key handling ──────────────────────────────────────
        if reg_state != REG_IDLE:
            if key == 27:   # Esc → cancel
                reg_state = REG_IDLE
                reg_input_buf = ""
                reg_stable_count = 0
                reg_prev_fingers = None
                reg_current_fingers = None
                reg_file_list = []
                reg_file_cursor = 0
                reg_selected_filename = ""
                hold_counts.clear()
                logger.info("Gesture registration cancelled.")
            elif reg_state in (REG_NAME, REG_ACTION_DETAIL):
                if key == 13:   # Enter
                    text = reg_input_buf.strip()
                    if text:
                        if reg_state == REG_NAME:
                            reg_name = text.upper()
                            reg_input_buf = ""
                            reg_state = REG_ACTION_TYPE
                        else:   # REG_ACTION_DETAIL
                            reg_action_detail = text
                            reg_input_buf = ""
                            reg_state = REG_CONFIRM
                elif key in (8, 127):   # Backspace / Delete
                    reg_input_buf = reg_input_buf[:-1]
                elif 32 <= key <= 126:  # Printable ASCII
                    reg_input_buf += chr(key)
            elif reg_state == REG_ACTION_TYPE:
                if key == ord("s"):
                    reg_action_type = "shell"
                    reg_file_list = _get_script_files("shell")
                    reg_file_cursor = 0
                    reg_state = REG_FILE_PICK
                elif key == ord("a"):
                    reg_action_type = "applescript"
                    reg_file_list = _get_script_files("applescript")
                    reg_file_cursor = 0
                    reg_state = REG_FILE_PICK
                elif key == ord("k"):
                    reg_action_type = "shortcut"
                    reg_input_buf = ""
                    reg_state = REG_ACTION_DETAIL
            elif reg_state == REG_FILE_PICK:
                if key == ord("j"):
                    reg_file_cursor = min(reg_file_cursor + 1,
                                          max(0, len(reg_file_list) - 1))
                elif key == ord("k"):
                    reg_file_cursor = max(reg_file_cursor - 1, 0)
                elif key == 13 and reg_file_list:   # Enter – select file
                    selected = reg_file_list[reg_file_cursor]
                    try:
                        content = selected.read_text(encoding="utf-8").strip()
                    except Exception as exc:
                        logger.error("Cannot read %s: %s", selected.name, exc)
                    else:
                        reg_action_detail = content
                        reg_selected_filename = selected.name
                        reg_input_buf = ""
                        reg_state = REG_CONFIRM
            elif reg_state == REG_CONFIRM:
                if key == ord("y"):
                    label = reg_name.replace("_", " ").title()
                    entry = save_custom_gesture(
                        CONFIG_PATH,
                        reg_name,
                        reg_fingers,
                        reg_action_type,
                        reg_action_detail,
                        label=label,
                    )
                    recognizer._gesture_map[reg_fingers] = reg_name
                    gesture_cfgs[reg_name] = entry
                    flash_gesture = reg_name
                    flash_ts = time.time()
                    logger.info("Registered gesture '%s' → %s: %s",
                                reg_name, reg_action_type, reg_action_detail)
                    reg_state = REG_IDLE
                    reg_fingers = None
                    reg_name = ""
                    reg_action_type = ""
                    reg_action_detail = ""
                    reg_input_buf = ""
                    reg_file_list = []
                    reg_file_cursor = 0
                    reg_selected_filename = ""
                    hold_counts.clear()
                elif key == ord("n"):
                    reg_state = REG_IDLE
                    reg_fingers = None
                    reg_name = ""
                    reg_action_type = ""
                    reg_action_detail = ""
                    reg_input_buf = ""
                    reg_file_list = []
                    reg_file_cursor = 0
                    reg_selected_filename = ""

        # ── Normal key handling ────────────────────────────────────────────
        else:
            if key in (ord("q"), 27):   # q or Esc
                break
            if key == ord("l"):
                show_landmarks = not show_landmarks
                logger.info("Landmarks: %s", "ON" if show_landmarks else "OFF")
            if key == ord("g"):
                show_gestures = not show_gestures
                logger.info("Gesture list: %s",
                            "ON" if show_gestures else "OFF")
            if key == ord("r"):
                show_gestures = False
                reg_state = REG_CAPTURE
                reg_stable_count = 0
                reg_prev_fingers = None
                reg_current_fingers = None
                hold_counts.clear()
                logger.info(
                    "Gesture registration started. Hold your gesture steady.")

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    logger.info("VisionTrigger stopped.")


if __name__ == "__main__":
    main()
