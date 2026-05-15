"""
main.py – VisionTrigger entry point.

Reads the camera, detects hand gestures with MediaPipe, and fires the
corresponding macOS action configured in config.yaml.

Controls:
  q / Esc – quit
  l       – toggle landmark overlay
  g       – show/hide gesture list
  r       – register a new gesture from camera
  v       – toggle voice command listener
"""

import json
import logging
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import yaml

from gestures import GestureRecognizer
from mac_trigger import MacTrigger
from voice import VoiceListener

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

# Built-in gesture finger states: (thumb, index, middle, ring, pinky) True = extended
_BUILTIN_FINGER_STATES: dict[str, tuple] = {
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

# Gesture registration states
REG_IDLE = "idle"
REG_CAPTURE = "capture"
REG_NAME = "name"
REG_ACTION_TYPE = "action_type"
REG_ACTION_DETAIL = "action_detail"
REG_FILE_PICK = "file_pick"
REG_CONFIRM = "confirm"
REG_DELETE_CONFIRM = "delete_confirm"
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


def update_gesture_action(
    config_path: Path,
    name: str,
    action_type: str,
    action_detail: str,
    is_custom: bool,
) -> dict:
    """Update only the action fields of an existing gesture, preserving all others."""
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh) or {}
    section = "custom_gestures" if is_custom else "gestures"
    entry = dict((config.get(section) or {}).get(name, {}))
    for k in ("command", "script", "name"):  # remove old action-specific keys
        entry.pop(k, None)
    entry["action"] = action_type
    if action_type == "shell":
        entry["command"] = action_detail
    elif action_type == "applescript":
        entry["script"] = action_detail
    elif action_type == "shortcut":
        entry["name"] = action_detail
    if not isinstance(config.get(section), dict):
        config[section] = {}
    config[section][name] = entry
    with open(config_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)
    return entry


def delete_gesture_from_config(config_path: Path, name: str) -> None:
    """Remove a gesture entry from whichever section of config.yaml it lives in."""
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh) or {}
    for section in ("gestures", "custom_gestures"):
        d = config.get(section)
        if isinstance(d, dict) and name in d:
            del d[name]
            break
    with open(config_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


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
# File-upload helpers  (no tkinter – uses osascript + folder watcher)
# ---------------------------------------------------------------------------

def _browse_and_copy(action_type: str) -> list:
    """
    Open a native macOS file-chooser via osascript; copy chosen scripts
    into SCRIPTS_DIR; return the list of copied Path objects.
    No tkinter or extra dependencies required.
    """
    if action_type == "applescript":
        ext_filter = '{"applescript"}'
    else:
        ext_filter = '{"sh", "bash", "zsh", "command"}'

    script = "\n".join([
        f'set chosen to (choose file with prompt "Select {action_type} files"'
        f' of type {ext_filter} with multiple selections allowed)',
        'set out to ""',
        'repeat with f in chosen',
        '  set out to out & (POSIX path of f) & "\\n"',
        'end repeat',
        'return out',
    ])
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        logger.error("Browse dialog error: %s", exc)
        return []

    if result.returncode != 0:
        return []   # user cancelled – not an error

    valid = _SCRIPT_EXTS.get(action_type, set())
    copied: list = []
    for line in result.stdout.splitlines():
        p = Path(line.strip())
        if p.suffix.lower() in valid and p.exists():
            dest = SCRIPTS_DIR / p.name
            try:
                shutil.copy2(p, dest)
                copied.append(dest)
                logger.info("Copied '%s' → scripts/", p.name)
            except Exception as exc:
                logger.error("Cannot copy %s: %s", p.name, exc)
    return copied


def _open_scripts_folder() -> None:
    """Reveal the scripts/ directory in macOS Finder."""
    subprocess.Popen(["open", str(SCRIPTS_DIR)])


class _FolderWatcher:
    """
    Polls the scripts/ directory in a background thread.
    When new matching files appear (dragged in via Finder), poll() returns them.
    No tkinter or extra dependencies required.
    """

    def __init__(self, action_type: str):
        self.action_type = action_type
        self._new_files: list = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        valid = _SCRIPT_EXTS.get(action_type, set())
        try:
            self._seen: set = {
                f for f in SCRIPTS_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in valid
            }
        except Exception:
            self._seen = set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        valid = _SCRIPT_EXTS.get(self.action_type, set())
        while not self._stop.wait(0.5):
            try:
                current = {
                    f for f in SCRIPTS_DIR.iterdir()
                    if f.is_file() and f.suffix.lower() in valid
                }
                new = current - self._seen
                if new:
                    with self._lock:
                        self._new_files.extend(new)
                    self._seen = current
            except Exception:
                pass

    def poll(self) -> list:
        """Return any files added since the last call."""
        with self._lock:
            files = self._new_files[:]
            self._new_files.clear()
        return files

    def close(self) -> None:
        self._stop.set()


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
    live_fingers=None,
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

    # Small hand icon to the right of the gesture label
    icon_fingers = live_fingers
    if icon_fingers is None and gesture not in ("UNKNOWN", "NONE", ""):
        icon_fingers = _BUILTIN_FINGER_STATES.get(gesture)
    if icon_fingers is not None:
        draw_hand_icon(frame, 360, 14, tuple(icon_fingers), gesture_name=gesture)

    if action_label:
        _put(frame, f"Action:  {action_label}",
             (w - 310, 32), scale=0.65, color=_YELLOW)

    # ── hold-progress bar (above button bar) ──────────────────────────────
    if hold_progress > 0.0:
        bx1, by1, bx2, by2 = 10, h - 56, w - 10, h - 52
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


def _finger_display_str(fingers: tuple) -> str:
    """Format a 5-bool finger tuple as a compact readable string."""
    labels = ("Th", "Ix", "Md", "Rg", "Pk")
    return "  ".join(f"{l}:{'Y' if v else 'N'}" for l, v in zip(labels, fingers))


def draw_hand_icon(
    frame, left: int, top: int, fingers: tuple, gesture_name: str = ""
) -> None:
    """Render a 46×42 px schematic hand icon.

    Layout (left→right):  thumb | gap | index | middle | ring | pinky
                                         [========= palm =========]

    Extended fingers are drawn in bright green; curled ones as short dark stubs.
    THUMBS_UP / THUMBS_DOWN are handled with directional arrows on the thumb.
    """
    thumb, index, middle, ring, pinky = fingers

    ON  = (75, 215, 75)    # extended – bright green
    OFF = (38, 50, 38)     # curled   – dark stub
    PLM = (52, 76, 52)     # palm background

    fw  = 7    # finger bar width (px)
    fg  = 2    # gap between finger bars
    ph  = 9    # palm height (px)
    mfh = 26   # max bar height when finger is extended
    sfh = 5    # stub height when finger is curled

    # 4-finger block: index → pinky, each fw wide with fg gaps
    palm_x = left + 11          # leave room for thumb (7px) + gap (2px) + 2px margin
    palm_w = 4 * fw + 3 * fg   # = 34 px
    palm_y = top + 33           # top edge of the palm rect

    # Palm rect
    cv2.rectangle(frame, (palm_x, palm_y), (palm_x + palm_w, palm_y + ph), PLM, -1)
    cv2.rectangle(frame, (palm_x, palm_y), (palm_x + palm_w, palm_y + ph), (80, 108, 80), 1)

    # Draw index, middle, ring, pinky
    for i, ext in enumerate([index, middle, ring, pinky]):
        fx = palm_x + i * (fw + fg)
        fh = mfh if ext else sfh
        fy = palm_y - fh
        c  = ON if ext else OFF
        cv2.rectangle(frame, (fx, fy), (fx + fw, palm_y), c, -1)
        cv2.circle(frame, (fx + fw // 2, fy), fw // 2, c, -1)   # rounded fingertip

    # Thumb column (always left of the 4-finger block)
    thumb_up   = (gesture_name == "THUMBS_UP")
    thumb_down = (gesture_name == "THUMBS_DOWN")
    tc = ON if thumb else OFF
    tx = left + 1

    if thumb_down:
        # Thumb bar extends *downward* below the palm
        fh = mfh if thumb else sfh
        ty = palm_y + ph
        cv2.rectangle(frame, (tx, ty), (tx + fw, ty + fh), tc, -1)
        cv2.circle(frame, (tx + fw // 2, ty + fh), fw // 2, tc, -1)
        if thumb:
            pts = np.array([[tx + fw // 2, ty + fh + 5],
                            [tx - 1,        ty + fh    ],
                            [tx + fw + 1,   ty + fh    ]], dtype=np.int32)
            cv2.fillPoly(frame, [pts], ON)
    else:
        # Thumb bar extends *upward* (normal hand or THUMBS_UP)
        fh = mfh if thumb else sfh
        ty = palm_y - fh
        cv2.rectangle(frame, (tx, ty), (tx + fw, palm_y), tc, -1)
        cv2.circle(frame, (tx + fw // 2, ty), fw // 2, tc, -1)
        if thumb and thumb_up:
            pts = np.array([[tx + fw // 2, ty - 5],
                            [tx - 1,        ty    ],
                            [tx + fw + 1,   ty    ]], dtype=np.int32)
            cv2.fillPoly(frame, [pts], ON)


def draw_gestures_list_overlay(
    frame,
    gesture_cfgs: dict,
    cursor: int = -1,
    mouse_pos: tuple = (0, 0),
) -> dict:
    """Draw gesture list panel; return {key: rect} of clickable elements."""
    h, w = frame.shape[:2]
    rects: dict = {}
    rows = list(gesture_cfgs.items())
    row_h    = 44          # taller rows to fit hand icon + 2-line text
    header_h = 36
    padding  = 16
    footer_h = 50
    available_h = h - 120 - header_h - padding - footer_h
    max_visible = min(len(rows), max(1, available_h // row_h))
    panel_h = header_h + max_visible * row_h + padding + footer_h
    panel_w = 580
    px1 = w - panel_w - 16
    py1 = 80
    px2 = w - 16
    py2 = py1 + panel_h

    panel = frame.copy()
    cv2.rectangle(panel, (px1, py1), (px2, py2), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _CYAN, 2)

    lx = px1 + 14
    _put(frame, f"GESTURES  ({len(gesture_cfgs)})", (lx, py1 + 24), scale=0.58, color=_CYAN)

    # Close button (top-right of panel)
    close_rect = (px2 - 72, py1 + 6, px2 - 6, py1 + 28)
    draw_button(frame, close_rect, "Close", mouse_pos)
    rects["close"] = close_rect

    sep_y = py1 + header_h - 2
    cv2.line(frame, (px1 + 8, sep_y), (px2 - 8, sep_y), _GREY, 1)

    scroll_top = 0
    if cursor >= 0:
        scroll_top = max(0, min(cursor - max_visible // 2,
                                len(rows) - max_visible))

    row_y      = sep_y + 4    # top pixel of the first row
    icon_w     = 52            # pixels reserved for the hand icon + gap

    for list_idx in range(scroll_top, min(scroll_top + max_visible, len(rows))):
        name, cfg = rows[list_idx]
        if not isinstance(cfg, dict):
            row_y += row_h
            continue
        label  = cfg.get("label") or name
        action = cfg.get("action", "?")
        detail = cfg.get("command") or cfg.get("script") or cfg.get("name") or ""
        detail_short = detail[:30] + ("…" if len(detail) > 30 else "")
        action_color = {
            "shell": _GREEN, "applescript": _YELLOW, "shortcut": _CYAN,
        }.get(action, _WHITE)
        is_selected = (list_idx == cursor)
        row_rect = (px1 + 4, row_y, px2 - 4, row_y + row_h - 2)
        rects[f"row_{list_idx}"] = row_rect

        hovered = _hit(row_rect, mouse_pos)
        if is_selected:
            cv2.rectangle(frame, (px1 + 4, row_y), (px2 - 4, row_y + row_h - 2),
                          (40, 40, 60), -1)
        elif hovered:
            cv2.rectangle(frame, (px1 + 4, row_y), (px2 - 4, row_y + row_h - 2),
                          (30, 30, 46), -1)

        # ── Hand shape icon ──────────────────────────────────────────────────
        raw_fingers = cfg.get("fingers")
        if raw_fingers is not None:
            icon_fingers = tuple(bool(f) for f in raw_fingers)
        else:
            icon_fingers = _BUILTIN_FINGER_STATES.get(name, (False,) * 5)
        draw_hand_icon(frame, lx, row_y + 1, icon_fingers, gesture_name=name)

        # ── Two-line text (label on top, action+detail below) ────────────────
        row_color = _CYAN if is_selected else _WHITE
        prefix = "▶ " if is_selected else ""
        tx = lx + icon_w
        _put(frame, f"{prefix}{label}", (tx, row_y + 16), scale=0.55, color=row_color)
        _put(frame, f"[{action}]",      (tx, row_y + 34), scale=0.48, color=action_color)
        _put(frame, detail_short, (tx + 90, row_y + 34), scale=0.42, color=_GREY, thickness=1)

        row_y += row_h

    if len(rows) > max_visible:
        _put(frame,
             f"  … {len(rows)} total  ({scroll_top+1}–{min(scroll_top+max_visible, len(rows))})",
             (lx, row_y + 4), scale=0.44, color=_GREY, thickness=1)
        row_y += footer_h - 28

    cv2.line(frame, (px1 + 8, row_y + 2), (px2 - 8, row_y + 2), _GREY, 1)

    # Edit / Delete buttons
    btn_y1, btn_y2 = row_y + 10, row_y + 36
    edit_rect = (lx,       btn_y1, lx + 72,  btn_y2)
    del_rect  = (lx + 80,  btn_y1, lx + 158, btn_y2)
    draw_button(frame, edit_rect, "Edit",   mouse_pos)
    draw_button(frame, del_rect,  "Delete", mouse_pos, danger=True)
    rects["edit"]   = edit_rect
    rects["delete"] = del_rect

    return rects


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
    reg_is_edit: bool = False,
    mouse_pos: tuple = (0, 0),
) -> dict:
    """Draw registration overlay; return {key: rect} of clickable elements."""
    h, w = frame.shape[:2]
    rects: dict = {}
    px1, py1 = w // 2 - 310, h // 2 - 140
    px2, py2 = w // 2 + 310, h // 2 + 165
    panel = frame.copy()
    cv2.rectangle(panel, (px1, py1), (px2, py2), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _ORANGE, 2)

    lx, y = px1 + 18, py1 + 30
    title = "EDIT GESTURE" if reg_is_edit else "REGISTER GESTURE"
    _put(frame, title, (lx, y), scale=0.75, color=_ORANGE)

    # Cancel button (always visible, top-right)
    cancel_rect = (px2 - 86, py1 + 8, px2 - 8, py1 + 32)
    draw_button(frame, cancel_rect, "Cancel", mouse_pos)
    rects["cancel"] = cancel_rect
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
        y += 36
        # Clickable action-type buttons
        shell_rect = (lx,       y, lx + 140, y + 34)
        apple_rect = (lx + 152, y, lx + 310, y + 34)
        short_rect = (lx + 322, y, lx + 470, y + 34)
        draw_button(frame, shell_rect, "Shell (.sh)",   mouse_pos)
        draw_button(frame, apple_rect, "AppleScript",   mouse_pos)
        draw_button(frame, short_rect, "Shortcut",      mouse_pos)
        rects["shell"]       = shell_rect
        rects["applescript"] = apple_rect
        rects["shortcut"]    = short_rect
        y += 44
        _put(frame, "Shell/AppleScript: pick a file from scripts/",
             (lx, y), scale=0.46, color=_GREY, thickness=1)
        y += 22
        _put(frame, "Shortcut: type the Shortcuts.app name",
             (lx, y), scale=0.46, color=_GREY, thickness=1)

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
        save_label = "Update this gesture?" if reg_is_edit else "Save this gesture?"
        _put(frame, save_label, (lx, y), color=_YELLOW)
        y += 32
        _put(frame, f"Name:    {reg_name}", (lx, y), color=_GREEN, scale=0.60)
        y += 28
        _put(frame,
             f"Pattern: {_finger_display_str(reg_fingers) if reg_fingers else '?'}",
             (lx, y), scale=0.55)
        y += 28
        preview = reg_selected_filename or (
            reg_action_detail[:42] + ("..." if len(reg_action_detail) > 42 else ""))
        _put(frame, f"Action:  [{reg_action_type}]  {preview}", (lx, y), scale=0.55)
        y += 42
        save_rect   = (lx,       y, lx + 90,  y + 34)
        cancel2_rect = (lx + 102, y, lx + 200, y + 34)
        draw_button(frame, save_rect,    "Save",   mouse_pos, active=True)
        draw_button(frame, cancel2_rect, "Cancel", mouse_pos)
        rects["save"]    = save_rect
        rects["cancel2"] = cancel2_rect

    return rects


def draw_delete_confirm_overlay(
    frame,
    gesture_name: str,
    mouse_pos: tuple = (0, 0),
) -> dict:
    """Draw deletion confirmation; return {key: rect} of clickable elements."""
    h, w = frame.shape[:2]
    rects: dict = {}
    px1, py1 = w // 2 - 250, h // 2 - 90
    px2, py2 = w // 2 + 250, h // 2 + 100
    panel = frame.copy()
    cv2.rectangle(panel, (px1, py1), (px2, py2), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 60, 200), 2)

    lx, y = px1 + 20, py1 + 34
    _put(frame, "DELETE GESTURE", (lx, y), scale=0.72, color=(0, 80, 255))
    y += 38
    _put(frame, f"Name:  {gesture_name}", (lx, y), scale=0.65, color=_WHITE)
    y += 34
    _put(frame, "This cannot be undone.", (lx, y), scale=0.52,
         color=_YELLOW, thickness=1)
    y += 38
    del_rect    = (lx,       y, lx + 120, y + 34)
    cancel_rect = (lx + 132, y, lx + 240, y + 34)
    draw_button(frame, del_rect,    "Yes, Delete", mouse_pos, danger=True)
    draw_button(frame, cancel_rect, "Cancel",      mouse_pos)
    rects["delete"] = del_rect
    rects["cancel"] = cancel_rect
    return rects


def draw_file_pick_overlay(
    frame,
    action_type: str,
    file_list: list,
    cursor: int,
    mouse_pos: tuple = (0, 0),
    watching: bool = True,
) -> dict:
    """Draw file-picker panel; return {key: rect} of clickable elements."""
    h, w = frame.shape[:2]
    rects: dict = {}
    px1, py1 = w // 2 - 340, h // 2 - 185
    px2, py2 = w // 2 + 340, h // 2 + 200
    panel = frame.copy()
    cv2.rectangle(panel, (px1, py1), (px2, py2), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _ORANGE, 2)

    lx, y = px1 + 18, py1 + 30
    title = ("ADD APPLESCRIPT FILE"
             if action_type == "applescript" else "ADD SHELL SCRIPT FILE")
    _put(frame, title, (lx, y), scale=0.72, color=_ORANGE)

    cancel_rect = (px2 - 88, py1 + 8, px2 - 8, py1 + 32)
    draw_button(frame, cancel_rect, "Cancel", mouse_pos)
    rects["cancel"] = cancel_rect
    y += 34

    folder_str = str(SCRIPTS_DIR)
    if len(folder_str) > 60:
        folder_str = "\u2026" + folder_str[-57:]
    _put(frame, f"scripts/: {folder_str}", (lx, y),
         scale=0.44, color=_GREY, thickness=1)
    y += 28

    # \u2500\u2500 Upload row: Browse + Open-in-Finder buttons \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    browse_rect = (lx, y, lx + 148, y + 32)
    finder_rect = (lx + 158, y, lx + 158 + 160, y + 32)
    draw_button(frame, browse_rect, "Browse Files\u2026",      mouse_pos)
    draw_button(frame, finder_rect, "Open scripts/ folder", mouse_pos)
    rects["browse"] = browse_rect
    rects["finder"] = finder_rect
    y += 40

    # \u2500\u2500 Folder-watcher status \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    dot_x, dot_y = lx + 6, y + 6
    if watching:
        cv2.circle(frame, (dot_x, dot_y), 5, _GREEN, -1)
        _put(frame, "Watching for new files \u2014 drag files to scripts/ in Finder",
             (lx + 18, y + 10), scale=0.45, color=_GREEN, thickness=1)
    else:
        cv2.circle(frame, (dot_x, dot_y), 5, _GREY, 1)
        _put(frame, "Not watching",
             (lx + 18, y + 10), scale=0.45, color=_GREY, thickness=1)
    y += 22

    cv2.line(frame, (px1 + 8, y), (px2 - 8, y), (60, 60, 60), 1)
    y += 12

    # \u2500\u2500 File list \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if not file_list:
        ext_hint = (".applescript"
                    if action_type == "applescript" else ".sh / .bash / .zsh")
        _put(frame, "No script files in scripts/ yet.", (lx, y), color=_YELLOW)
        y += 28
        _put(frame, f"Browse above or drop {ext_hint} files into the drop zone.",
             (lx, y), scale=0.52, color=_GREY, thickness=1)
    else:
        _put(frame, "Click a file to select it",
             (lx, y), scale=0.50, color=_GREY, thickness=1)
        y += 28
        row_h = 32
        max_visible = max(1, min(len(file_list), (py2 - y - 16) // row_h))
        start = max(0, min(cursor - max_visible // 2,
                           len(file_list) - max_visible))
        for idx in range(start, min(start + max_visible, len(file_list))):
            fp = file_list[idx]
            row_rect = (px1 + 8, y - 4, px2 - 8, y + row_h - 6)
            rects[f"file_{idx}"] = row_rect
            hovered  = _hit(row_rect, mouse_pos)
            selected = idx == cursor
            if selected:
                cv2.rectangle(frame, (px1 + 8, y - 4), (px2 - 8, y + row_h - 6),
                              (40, 40, 60), -1)
                cv2.rectangle(frame, (px1 + 8, y - 4), (px2 - 8, y + row_h - 6),
                              _CYAN, 1)
            elif hovered:
                cv2.rectangle(frame, (px1 + 8, y - 4), (px2 - 8, y + row_h - 6),
                              (32, 32, 50), -1)
            color  = _CYAN if selected else _WHITE
            prefix = "\u25b6  " if selected else "   "
            _put(frame, f"{prefix}{fp.name}", (lx, y + 12), scale=0.62, color=color)
            y += row_h
        if len(file_list) > max_visible:
            _put(frame, f"  \u2026 {len(file_list)} files total",
                 (lx, y + 4), scale=0.44, color=_GREY, thickness=1)
    return rects


# ---------------------------------------------------------------------------
# Mouse / button utilities
# ---------------------------------------------------------------------------

_mouse_state: dict = {"pos": (0, 0), "click": None}


def _mouse_cb(event, x, y, flags, param):
    _mouse_state["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_state["click"] = (x, y)


def _hit(rect: tuple, point) -> bool:
    if not point or not rect:
        return False
    x, y = point
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def draw_button(
    frame,
    rect: tuple,
    label: str,
    mouse_pos: tuple = (0, 0),
    active: bool = False,
    danger: bool = False,
) -> None:
    x1, y1, x2, y2 = rect
    hovered = _hit(rect, mouse_pos)
    if danger:
        bg     = (90, 40, 40) if hovered else (55, 20, 20)
        border = (160, 70, 70)
    elif active:
        bg     = (30, 130, 30) if hovered else (20, 100, 20)
        border = _GREEN
    elif hovered:
        bg     = (55, 55, 85)
        border = _CYAN
    else:
        bg     = (32, 32, 32)
        border = (75, 75, 75)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
    tx = x1 + max(4, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    _put(frame, label, (tx, ty), scale=0.50, color=_WHITE, thickness=1)


def draw_main_buttons(
    frame,
    mouse_pos: tuple,
    show_landmarks: bool,
    show_gestures: bool,
    voice_on: bool = False,
) -> dict:
    """Draw the clickable bottom bar; return {key: rect}."""
    h, w = frame.shape[:2]
    y1, y2 = h - 46, h - 8
    specs = [
        ("quit",      "Quit",                                          65),
        ("landmarks", f"Landmarks: {'ON ' if show_landmarks else 'OFF'}", 150),
        ("gestures",  "Gestures",                                      94),
        ("register",  "Register",                                      94),
        ("camera",    "Camera",                                        84),
        ("voice",     f"Voice: {'ON ' if voice_on else 'OFF'}",        110),
    ]
    rects: dict = {}
    x = 12
    for key, label, bw in specs:
        rect = (x, y1, x + bw, y2)
        is_active = (key == "landmarks" and show_landmarks) or \
                    (key == "gestures"  and show_gestures) or \
                    (key == "voice"     and voice_on)
        draw_button(frame, rect, label, mouse_pos, active=is_active)
        rects[key] = rect
        x += bw + 8
    return rects


def draw_voice_status(frame, status: str, status_text: str, wake_word: str) -> None:
    """Draw a small voice-status indicator below the top bar (top-left)."""
    if not status or status == "off":
        return
    color = {
        "listening":     _CYAN,
        "loading":       _YELLOW,
        "transcribing":  _YELLOW,
        "heard":         _GREEN,
        "executing":     _GREEN,
        "error":         (0, 80, 255),
    }.get(status, _WHITE)

    label_for = {
        "listening":    f"listening for \"{wake_word}\"…",
        "loading":      status_text or "loading model…",
        "transcribing": "transcribing…",
        "heard":        f"heard: {status_text}" if status_text else "heard",
        "executing":    f"running: {status_text}" if status_text else "running",
        "error":        f"error: {status_text}" if status_text else "error",
    }
    msg = label_for.get(status, status)
    # Truncate to keep it on one line
    if len(msg) > 70:
        msg = msg[:67] + "…"
    _put(frame, f"VOICE  {msg}", (10, 92), scale=0.55, color=color, thickness=1)


# ---------------------------------------------------------------------------
# Camera selector UI
# ---------------------------------------------------------------------------

def draw_camera_selector(
    frame,
    cam_names: list[str],
    cursor: int,
    availability: dict,
    has_active: bool,
    mouse_pos: tuple = (0, 0),
) -> dict:
    """Draw camera selector; return {key: rect} of clickable elements."""
    h, w = frame.shape[:2]
    rects: dict = {}

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), _BLACK, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    _put(frame, "SELECT CAMERA", (w // 2 - 140, 72), scale=0.95, color=_ORANGE)
    _put(frame, "Click a camera to open it",
         (w // 2 - 130, 106), scale=0.50, color=_GREY, thickness=1)

    # Bottom buttons
    btn_y1, btn_y2 = h - 54, h - 14
    rescan_rect = (w // 2 - 116, btn_y1, w // 2 - 8,   btn_y2)
    close_rect  = (w // 2 + 8,   btn_y1, w // 2 + 116, btn_y2)
    draw_button(frame, rescan_rect, "Rescan", mouse_pos)
    draw_button(frame, close_rect, "Cancel" if has_active else "Quit", mouse_pos)
    rects["rescan"] = rescan_rect
    rects["close"]  = close_rect

    if not cam_names:
        _put(frame, "No cameras detected.",
             (w // 2 - 160, h // 2 - 16), color=_YELLOW)
        _put(frame, "Connect a camera then click  Rescan.",
             (w // 2 - 210, h // 2 + 26), scale=0.58, color=_GREY, thickness=1)
        return rects

    row_h = 56
    total_h = len(cam_names) * row_h
    start_y = max(140, h // 2 - total_h // 2)

    for idx, name in enumerate(cam_names):
        y = start_y + idx * row_h
        selected = idx == cursor
        avail    = availability.get(idx)
        row_rect = (w // 2 - 310, y - 28, w // 2 + 310, y + 20)
        rects[f"cam_{idx}"] = row_rect

        hovered = _hit(row_rect, mouse_pos)
        if selected:
            bg = (30, 30, 58)
        elif hovered:
            bg = (42, 42, 58)
        else:
            bg = (18, 18, 18)
        cv2.rectangle(frame, (w // 2 - 310, y - 28), (w // 2 + 310, y + 20), bg, -1)
        border = _CYAN if selected else ((90, 90, 120) if hovered else (35, 35, 35))
        cv2.rectangle(frame, (w // 2 - 310, y - 28), (w // 2 + 310, y + 20), border, 1)

        name_color = _CYAN if selected else _WHITE
        prefix = "▶  " if selected else "   "
        _put(frame, f"{prefix}{idx}:  {name}",
             (w // 2 - 298, y), scale=0.70, color=name_color)

        if avail is True:
            _put(frame, "● available",
                 (w // 2 + 148, y), scale=0.52, color=_GREEN, thickness=1)
        elif avail is False:
            _put(frame, "○ unavailable",
                 (w // 2 + 148, y), scale=0.52, color=_GREY, thickness=1)
    return rects


# ---------------------------------------------------------------------------
# Camera detection
# ---------------------------------------------------------------------------

def _macos_camera_names() -> list[str]:
    """Return camera display names in AVFoundation order (macOS only)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        )
        data = json.loads(result.stdout)
        return [c.get("_name", "") for c in data.get("SPCameraDataType", [])]
    except Exception:
        return []


def _open_index(idx: int) -> cv2.VideoCapture | None:
    """Try to open a camera index and confirm it can deliver frames."""
    cap = cv2.VideoCapture(idx)
    if cap.isOpened() and cap.grab():
        return cap
    cap.release()
    return None


def find_camera(
    preferred: int,
    max_probe: int = 6,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple:
    """
    Return (VideoCapture, index, display_name) for the best available camera.

    Tries `preferred` first.  If that fails, probes known device indices and
    prefers any device whose name contains "iPhone" or "Continuity".
    Retries up to `retries` times with `retry_delay` seconds between attempts
    so a transiently sleeping Continuity Camera gets a chance to reconnect.
    """
    names = _macos_camera_names()
    # Cap probe range to known device count to avoid OpenCV out-of-bound noise.
    n_devices = len(names) if names else max_probe
    probe_range = range(n_devices)

    def label(idx: int) -> str:
        return names[idx] if idx < len(names) else f"Camera {idx}"

    for attempt in range(retries):
        # Preferred index first
        cap = _open_index(preferred)
        if cap:
            logger.info("Camera %d opened: %s", preferred, label(preferred))
            return cap, preferred, label(preferred)

        if attempt == 0:
            logger.warning(
                "Camera index %d unavailable – scanning %d device(s)…",
                preferred, n_devices,
            )

        # Collect all working cameras
        candidates: list[tuple[int, cv2.VideoCapture]] = []
        for idx in probe_range:
            if idx == preferred:
                continue
            cap = _open_index(idx)
            if cap:
                candidates.append((idx, cap))

        if candidates:
            # Prefer Continuity Camera / iPhone camera by name
            for idx, cap in candidates:
                name = label(idx)
                if any(k in name for k in ("iPhone", "Continuity", "iSight")):
                    for other_idx, other_cap in candidates:
                        if other_idx != idx:
                            other_cap.release()
                    logger.info(
                        "Built-in camera unavailable – using %s (index %d)",
                        name, idx,
                    )
                    return cap, idx, name

            # Fall back to first working camera
            idx, cap = candidates[0]
            for _, other_cap in candidates[1:]:
                other_cap.release()
            logger.info("Falling back to camera %d: %s", idx, label(idx))
            return cap, idx, label(idx)

        if attempt < retries - 1:
            logger.warning(
                "No camera available – retrying in %.0fs… (attempt %d/%d)",
                retry_delay, attempt + 1, retries,
            )
            time.sleep(retry_delay)

    return None, -1, ""


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
    cap, cam_idx, cam_name = find_camera(preferred=cam_idx, retries=1)

    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Camera selector state ──────────────────────────────────────────────
    all_cam_names: list[str] = _macos_camera_names()
    cam_select_mode: bool = cap is None
    cam_select_cursor: int = max(0, cam_idx)
    cam_availability: dict[int, bool] = {}

    def rescan_cameras() -> None:
        nonlocal all_cam_names, cam_availability
        all_cam_names = _macos_camera_names()
        cam_availability = {}
        active = cam_idx if cap is not None else -1
        for i in range(len(all_cam_names)):
            if i == active:
                cam_availability[i] = True
            else:
                test = _open_index(i)
                if test:
                    cam_availability[i] = True
                    test.release()
                else:
                    cam_availability[i] = False

    if cam_select_mode:
        logger.info("No camera found – opening camera selector.")
        rescan_cameras()

    hold_required: int = settings.get("gesture_hold_frames", 20)
    show_landmarks: bool = settings.get("show_landmarks", True)
    show_fps: bool = settings.get("show_fps", True)

    hold_counts: dict[str, int] = {}
    last_trigger_ts: dict[str, float] = {}
    current_live_fingers = None   # finger states for the current frame's hand

    flash_gesture = ""
    flash_ts = 0.0

    # ── Voice listener ────────────────────────────────────────────────────
    voice_settings = (settings.get("voice") or {}) if isinstance(settings, dict) else {}
    voice_commands_cfg = config.get("voice_commands") or {}
    voice_wake_word = str(voice_settings.get("wake_word", "hey vision"))

    def _on_voice_command(name: str, cfg: dict) -> None:
        nonlocal flash_gesture, flash_ts
        flash_gesture = name
        flash_ts = time.time()
        logger.info("Voice triggered: %s → %s", name, cfg.get("label", name))
        threading.Thread(
            target=trigger.execute, args=(cfg,), daemon=True,
        ).start()

    voice_listener = VoiceListener(
        wake_word=voice_wake_word,
        commands=voice_commands_cfg,
        on_command=_on_voice_command,
        model_size=str(voice_settings.get("model", "base.en")),
        language=str(voice_settings.get("language", "en")),
        silence_threshold=float(voice_settings.get("silence_threshold", 0.01)),
    )
    if bool(voice_settings.get("enabled", False)):
        voice_listener.start()

    show_gestures: bool = False
    gesture_list_cursor: int = 0

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
    reg_is_edit: bool = False
    reg_edit_is_custom: bool = False

    _folder_watcher: _FolderWatcher | None = None

    fps = 0.0
    fps_frame_cnt = 0
    fps_tick = time.time()

    logger.info("VisionTrigger started. Show your hand to the camera.")
    logger.info(
        "Press  q / Esc  to quit,  l  to toggle landmarks,  r  to register a gesture.")

    # Button rects from the previous frame (for click hit-testing)
    _cam_btns:     dict = {}
    _main_btns:    dict = {}
    _gesture_btns: dict = {}
    _reg_btns:     dict = {}
    _file_btns:    dict = {}
    _del_btns:     dict = {}
    _cb_registered = False

    while True:
        # ── Mouse state ────────────────────────────────────────────────────
        mouse_pos = _mouse_state["pos"]
        click     = _mouse_state["click"]
        _mouse_state["click"] = None          # consume

        # ── Frame acquisition ──────────────────────────────────────────────
        if cam_select_mode or cap is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera %d lost – opening camera selector.", cam_idx)
                cap.release()
                cap = None
                cam_select_mode = True
                hold_counts.clear()
                rescan_cameras()
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                frame = cv2.flip(frame, 1)

        current_gesture = "NONE"
        hold_progress = 0.0
        action_label = ""

        # ── Gesture processing (only when a camera is live) ────────────────
        if not cam_select_mode and cap is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            results = recognizer.process(rgb, timestamp_ms)

            if results.hand_landmarks and results.handedness:
                hand_lm = results.hand_landmarks[0]
                handedness = results.handedness[0][0].category_name

                if show_landmarks:
                    recognizer.draw_landmarks(frame, hand_lm)

                current_gesture = recognizer.classify(hand_lm, handedness)
                action_label = display_label(current_gesture, gesture_cfgs)
                current_live_fingers = recognizer.finger_states(hand_lm, handedness)

                # ── Registration capture ─────────────────────────────────
                if reg_state == REG_CAPTURE:
                    fingers = current_live_fingers
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

                # ── Normal action triggering ─────────────────────────────
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
                hold_counts.clear()
                current_live_fingers = None
                if reg_state == REG_CAPTURE:
                    reg_stable_count = 0
                    reg_current_fingers = None

        # ── FPS ────────────────────────────────────────────────────────────
        if not cam_select_mode:
            fps_frame_cnt += 1
            if fps_frame_cnt >= 30:
                fps = fps_frame_cnt / (time.time() - fps_tick)
                fps_tick = time.time()
                fps_frame_cnt = 0

        flash_active = bool(flash_gesture) and (time.time() - flash_ts < 1.0)

        # ── Folder-watcher lifecycle ───────────────────────────────────────
        if reg_state == REG_FILE_PICK:
            if _folder_watcher is None:
                _folder_watcher = _FolderWatcher(reg_action_type)
            if _folder_watcher.poll():
                reg_file_list = _get_script_files(reg_action_type)
                reg_file_cursor = max(0, len(reg_file_list) - 1)
        elif _folder_watcher is not None:
            _folder_watcher.close()
            _folder_watcher = None

        # ── Drawing ────────────────────────────────────────────────────────
        if cam_select_mode:
            _cam_btns = draw_camera_selector(
                frame, all_cam_names, cam_select_cursor,
                cam_availability, cap is not None, mouse_pos,
            )
            _main_btns = _gesture_btns = _reg_btns = _file_btns = _del_btns = {}
        else:
            draw_overlay(
                frame, current_gesture, action_label,
                fps, hold_progress, flash_active, show_fps,
                live_fingers=current_live_fingers,
            )
            voice_status, voice_status_text = voice_listener.status()
            _main_btns = draw_main_buttons(
                frame, mouse_pos, show_landmarks, show_gestures,
                voice_on=voice_listener.running,
            )
            draw_voice_status(frame, voice_status, voice_status_text, voice_wake_word)

            if reg_state not in (REG_IDLE, REG_FILE_PICK, REG_DELETE_CONFIRM):
                _reg_btns = draw_registration_overlay(
                    frame, reg_state, reg_input_buf, reg_fingers,
                    reg_name, reg_action_type, reg_action_detail,
                    reg_stable_count, reg_current_fingers,
                    reg_selected_filename, reg_is_edit, mouse_pos,
                )
            else:
                _reg_btns = {}

            if reg_state == REG_FILE_PICK:
                _file_btns = draw_file_pick_overlay(
                    frame, reg_action_type, reg_file_list,
                    reg_file_cursor, mouse_pos,
                    watching=(_folder_watcher is not None))
            else:
                _file_btns = {}

            if reg_state == REG_DELETE_CONFIRM:
                _del_btns = draw_delete_confirm_overlay(
                    frame, reg_name, mouse_pos)
            else:
                _del_btns = {}

            if show_gestures and reg_state == REG_IDLE:
                _gesture_btns = draw_gestures_list_overlay(
                    frame, gesture_cfgs, gesture_list_cursor, mouse_pos)
            else:
                _gesture_btns = {}

        cv2.imshow("VisionTrigger", frame)
        if not _cb_registered:
            cv2.setMouseCallback("VisionTrigger", _mouse_cb)
            _cb_registered = True

        kbd = cv2.waitKey(1) & 0xFF

        # ── Shared helper: open a camera by index ──────────────────────────
        def _switch_camera(idx: int) -> bool:
            nonlocal cap, cam_idx, cam_name, cam_select_mode, fps_frame_cnt, fps_tick
            new_cap = _open_index(idx)
            if new_cap:
                if cap is not None:
                    cap.release()
                cap = new_cap
                cam_idx = idx
                cam_name = (all_cam_names[idx]
                            if idx < len(all_cam_names) else f"Camera {idx}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cam_select_mode = False
                fps_frame_cnt = 0
                fps_tick = time.time()
                logger.info("Switched to camera %d: %s", cam_idx, cam_name)
                return True
            cam_availability[idx] = False
            logger.warning("Camera %d could not be opened.", idx)
            return False

        # ── Shared helper: cancel registration ─────────────────────────────
        def _cancel_reg() -> None:
            nonlocal reg_state, reg_input_buf, reg_stable_count
            nonlocal reg_prev_fingers, reg_current_fingers, reg_file_list
            nonlocal reg_file_cursor, reg_selected_filename, reg_name
            nonlocal reg_is_edit, reg_edit_is_custom
            reg_state = REG_IDLE
            reg_input_buf = ""
            reg_stable_count = 0
            reg_prev_fingers = None
            reg_current_fingers = None
            reg_file_list = []
            reg_file_cursor = 0
            reg_selected_filename = ""
            reg_name = ""
            reg_is_edit = False
            reg_edit_is_custom = False
            hold_counts.clear()
            logger.info("Gesture registration cancelled.")

        # ── Shared helper: confirm/save gesture ────────────────────────────
        def _save_gesture() -> None:
            nonlocal reg_state, reg_fingers, reg_name, reg_action_type
            nonlocal reg_action_detail, reg_input_buf, reg_file_list
            nonlocal reg_file_cursor, reg_selected_filename, reg_is_edit
            nonlocal reg_edit_is_custom, flash_gesture, flash_ts
            if reg_is_edit:
                is_custom = bool(gesture_cfgs.get(reg_name, {}).get("fingers"))
                entry = update_gesture_action(
                    CONFIG_PATH, reg_name, reg_action_type,
                    reg_action_detail, is_custom,
                )
                gesture_cfgs[reg_name] = entry
                flash_gesture = reg_name
                flash_ts = time.time()
                logger.info("Updated gesture '%s' → %s", reg_name, reg_action_type)
            else:
                lbl = reg_name.replace("_", " ").title()
                entry = save_custom_gesture(
                    CONFIG_PATH, reg_name, reg_fingers,
                    reg_action_type, reg_action_detail, label=lbl,
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
            reg_is_edit = False
            reg_edit_is_custom = False
            hold_counts.clear()

        # ── Camera selector input ──────────────────────────────────────────
        if cam_select_mode:
            if kbd in (ord("q"), 27) or _hit(_cam_btns.get("close", ()), click):
                if cap is not None:
                    cam_select_mode = False
                else:
                    break
            elif kbd == ord("j"):
                cam_select_cursor = min(cam_select_cursor + 1,
                                        max(0, len(all_cam_names) - 1))
            elif kbd == ord("k"):
                cam_select_cursor = max(cam_select_cursor - 1, 0)
            elif kbd == ord("r") or _hit(_cam_btns.get("rescan", ()), click):
                rescan_cameras()
                logger.info("Camera list rescanned.")
            else:
                # Keyboard Enter opens highlighted row; click opens clicked row
                open_idx = None
                if kbd == 13 and all_cam_names:
                    open_idx = cam_select_cursor
                else:
                    for key_name, rect in _cam_btns.items():
                        if key_name.startswith("cam_") and _hit(rect, click):
                            open_idx = int(key_name.split("_")[1])
                            cam_select_cursor = open_idx
                            break
                if open_idx is not None:
                    _switch_camera(open_idx)

        # ── Registration input ─────────────────────────────────────────────
        elif reg_state != REG_IDLE:
            # Cancel (Esc key or Cancel button)
            if kbd == 27 or _hit(_reg_btns.get("cancel", ()), click):
                _cancel_reg()
            elif reg_state in (REG_NAME, REG_ACTION_DETAIL):
                if kbd == 13:
                    text = reg_input_buf.strip()
                    if text:
                        if reg_state == REG_NAME:
                            reg_name = text.upper()
                            reg_input_buf = ""
                            reg_state = REG_ACTION_TYPE
                        else:
                            reg_action_detail = text
                            reg_input_buf = ""
                            reg_state = REG_CONFIRM
                elif kbd in (8, 127):
                    reg_input_buf = reg_input_buf[:-1]
                elif 32 <= kbd <= 126:
                    reg_input_buf += chr(kbd)

            elif reg_state == REG_ACTION_TYPE:
                # Key or button click
                action_chosen = None
                if kbd == ord("s") or _hit(_reg_btns.get("shell", ()), click):
                    action_chosen = "shell"
                elif kbd == ord("a") or _hit(_reg_btns.get("applescript", ()), click):
                    action_chosen = "applescript"
                elif kbd == ord("k") or _hit(_reg_btns.get("shortcut", ()), click):
                    action_chosen = "shortcut"
                if action_chosen == "shortcut":
                    reg_action_type = "shortcut"
                    reg_input_buf = ""
                    reg_state = REG_ACTION_DETAIL
                elif action_chosen in ("shell", "applescript"):
                    reg_action_type = action_chosen
                    reg_file_list = _get_script_files(action_chosen)
                    reg_file_cursor = 0
                    reg_state = REG_FILE_PICK

            elif reg_state == REG_FILE_PICK:
                # Browse button → osascript dialog, copy to scripts/, refresh list
                if _hit(_file_btns.get("browse", ()), click):
                    new_files = _browse_and_copy(reg_action_type)
                    if new_files:
                        reg_file_list = _get_script_files(reg_action_type)
                        reg_file_cursor = max(0, len(reg_file_list) - 1)
                # Open-in-Finder button → reveal scripts/ so user can drag files in
                elif _hit(_file_btns.get("finder", ()), click):
                    _open_scripts_folder()
                elif kbd == ord("j"):
                    reg_file_cursor = min(reg_file_cursor + 1,
                                          max(0, len(reg_file_list) - 1))
                elif kbd == ord("k"):
                    reg_file_cursor = max(reg_file_cursor - 1, 0)
                else:
                    # Enter selects highlighted; click selects clicked row
                    sel_idx = None
                    if kbd == 13 and reg_file_list:
                        sel_idx = reg_file_cursor
                    else:
                        for key_name, rect in _file_btns.items():
                            if key_name.startswith("file_") and _hit(rect, click):
                                sel_idx = int(key_name.split("_")[1])
                                reg_file_cursor = sel_idx
                                break
                    if sel_idx is not None and sel_idx < len(reg_file_list):
                        chosen_file = reg_file_list[sel_idx]
                        try:
                            content = chosen_file.read_text(encoding="utf-8").strip()
                        except Exception as exc:
                            logger.error("Cannot read %s: %s", chosen_file.name, exc)
                        else:
                            reg_action_detail = content
                            reg_selected_filename = chosen_file.name
                            reg_input_buf = ""
                            reg_state = REG_CONFIRM

            elif reg_state == REG_DELETE_CONFIRM:
                if kbd == ord("y") or _hit(_del_btns.get("delete", ()), click):
                    entry = gesture_cfgs.pop(reg_name, None)
                    delete_gesture_from_config(CONFIG_PATH, reg_name)
                    if entry and entry.get("fingers"):
                        recognizer._gesture_map.pop(tuple(entry["fingers"]), None)
                    hold_counts.pop(reg_name, None)
                    last_trigger_ts.pop(reg_name, None)
                    logger.info("Deleted gesture '%s'", reg_name)
                    reg_state = REG_IDLE
                    reg_name = ""
                elif kbd in (ord("n"), 27) or _hit(_del_btns.get("cancel", ()), click):
                    reg_state = REG_IDLE
                    reg_name = ""

            elif reg_state == REG_CONFIRM:
                if kbd == ord("y") or _hit(_reg_btns.get("save", ()), click):
                    _save_gesture()
                elif (kbd == ord("n") or
                      _hit(_reg_btns.get("cancel2", ()), click)):
                    reg_state = REG_IDLE
                    reg_fingers = None
                    reg_name = ""
                    reg_action_type = ""
                    reg_action_detail = ""
                    reg_input_buf = ""
                    reg_file_list = []
                    reg_file_cursor = 0
                    reg_selected_filename = ""
                    reg_is_edit = False
                    reg_edit_is_custom = False

        # ── Normal input ───────────────────────────────────────────────────
        else:
            if kbd in (ord("q"), 27) or _hit(_main_btns.get("quit", ()), click):
                break
            if kbd == ord("c") or _hit(_main_btns.get("camera", ()), click):
                rescan_cameras()
                cam_select_mode = True
                logger.info("Camera selector opened.")
            if kbd == ord("l") or _hit(_main_btns.get("landmarks", ()), click):
                show_landmarks = not show_landmarks
                logger.info("Landmarks: %s", "ON" if show_landmarks else "OFF")
            if kbd == ord("g") or _hit(_main_btns.get("gestures", ()), click):
                show_gestures = not show_gestures
                if show_gestures:
                    gesture_list_cursor = 0
                logger.info("Gesture list: %s", "ON" if show_gestures else "OFF")
            if kbd == ord("r") or _hit(_main_btns.get("register", ()), click):
                if not show_gestures:
                    show_gestures = False
                    reg_state = REG_CAPTURE
                    reg_stable_count = 0
                    reg_prev_fingers = None
                    reg_current_fingers = None
                    hold_counts.clear()
                    logger.info("Gesture registration started.")
            if kbd == ord("v") or _hit(_main_btns.get("voice", ()), click):
                if voice_listener.running:
                    voice_listener.stop()
                    logger.info("Voice listener stopped.")
                else:
                    voice_listener.update_commands(voice_commands_cfg)
                    voice_listener.start()
                    logger.info("Voice listener starting…")
            if show_gestures:
                _gnames = list(gesture_cfgs.keys())
                # Keyboard navigation
                if kbd == ord("j"):
                    gesture_list_cursor = min(
                        gesture_list_cursor + 1, max(0, len(_gnames) - 1))
                elif kbd == ord("k"):
                    gesture_list_cursor = max(gesture_list_cursor - 1, 0)
                # Click on a row → select it
                for key_name, rect in _gesture_btns.items():
                    if key_name.startswith("row_") and _hit(rect, click):
                        gesture_list_cursor = int(key_name.split("_")[1])
                        break
                # Close button
                if kbd == ord("g") or _hit(_gesture_btns.get("close", ()), click):
                    show_gestures = False
                # Edit button / key
                elif (kbd == ord("e") or _hit(_gesture_btns.get("edit", ()), click)) and _gnames:
                    if 0 <= gesture_list_cursor < len(_gnames):
                        edit_name = _gnames[gesture_list_cursor]
                        edit_cfg  = gesture_cfgs[edit_name]
                        reg_name  = edit_name
                        reg_fingers = (tuple(edit_cfg["fingers"])
                                       if edit_cfg.get("fingers") else None)
                        reg_is_edit      = True
                        reg_edit_is_custom = bool(edit_cfg.get("fingers"))
                        show_gestures    = False
                        reg_input_buf    = ""
                        reg_file_list    = []
                        reg_file_cursor  = 0
                        reg_selected_filename = ""
                        reg_state = REG_ACTION_TYPE
                        logger.info("Editing gesture '%s'", edit_name)
                # Delete button / key
                elif (kbd == ord("d") or _hit(_gesture_btns.get("delete", ()), click)) and _gnames:
                    if 0 <= gesture_list_cursor < len(_gnames):
                        del_name = _gnames[gesture_list_cursor]
                        reg_name = del_name
                        show_gestures = False
                        reg_state = REG_DELETE_CONFIRM
                        logger.info("Delete confirm for '%s'", del_name)

    voice_listener.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    logger.info("VisionTrigger stopped.")


if __name__ == "__main__":
    main()
