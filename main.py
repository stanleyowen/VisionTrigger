"""
main.py – VisionTrigger entry point.

Controls:
  q / Esc – quit
  l       – toggle landmark overlay
  g       – show/hide gesture list
  r       – register a new gesture from camera
  v       – toggle voice command listener
  s       – open/close settings panel
  h       – show/hide voice commands panel
  c       – open camera selector
"""

import json
import logging
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from config_io import (
    delete_gesture_from_config,
    flush_pending_save,
    load_config,
    save_config,
    update_gesture_in_config,
)
from draw_utils import (
    _hit,
    draw_camera_selector,
    draw_delete_confirm_overlay,
    draw_file_pick_overlay,
    draw_gestures_list_overlay,
    draw_main_buttons,
    draw_overlay,
    draw_registration_overlay,
    draw_settings_panel,
    draw_voice_commands_panel,
    draw_voice_status,
)
from gestures import GestureRecognizer
from mac_trigger import MacTrigger
from reg_machine import (
    RegistrationStateMachine,
    RegState,
    _browse_and_copy,
    _get_script_files,
    _open_scripts_folder,
)
from voice import VoiceListener

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
MODEL_PATH  = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL   = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def ensure_model(path: Path, url: str) -> None:
    if path.exists():
        return
    logger.info("Downloading hand landmarker model (~7 MB)…")
    urllib.request.urlretrieve(url, path)
    logger.info("Model saved to %s", path)


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
# Camera detection
# ---------------------------------------------------------------------------

def _macos_camera_names() -> list[str]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        )
        data = json.loads(result.stdout)
        return [c.get("_name", "") for c in data.get("SPCameraDataType", [])]
    except Exception:
        return []


def _open_index(idx: int):
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
    names = _macos_camera_names()
    n_devices = len(names) if names else max_probe
    probe_range = range(n_devices)

    def label(idx: int) -> str:
        return names[idx] if idx < len(names) else f"Camera {idx}"

    for attempt in range(retries):
        cap = _open_index(preferred)
        if cap:
            logger.info("Camera %d opened: %s", preferred, label(preferred))
            return cap, preferred, label(preferred)

        if attempt == 0:
            logger.warning(
                "Camera index %d unavailable – scanning %d device(s)…",
                preferred, n_devices,
            )

        candidates = []
        for idx in probe_range:
            if idx == preferred:
                continue
            cap = _open_index(idx)
            if cap:
                candidates.append((idx, cap))

        if candidates:
            for idx, cap in candidates:
                name = label(idx)
                if any(k in name for k in ("iPhone", "Continuity", "iSight")):
                    for other_idx, other_cap in candidates:
                        if other_idx != idx:
                            other_cap.release()
                    logger.info("Built-in camera unavailable – using %s (index %d)", name, idx)
                    return cap, idx, name
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
# Mouse callback
# ---------------------------------------------------------------------------

_mouse_state: dict = {"pos": (0, 0), "click": None}


def _mouse_cb(event, x, y, flags, param):
    _mouse_state["pos"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_state["click"] = (x, y)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ensure_model(MODEL_PATH, MODEL_URL)

    config = load_config(CONFIG_PATH)
    settings = config.get("settings", {})
    gesture_cfgs = config.get("gestures") or {}
    custom_gesture_cfgs = config.get("custom_gestures") or {}
    gesture_cfgs.update(custom_gesture_cfgs)

    def _make_recognizer() -> GestureRecognizer:
        return GestureRecognizer(
            model_path=str(MODEL_PATH),
            min_detection_confidence=settings.get("detection_confidence", 0.7),
            min_tracking_confidence=settings.get("tracking_confidence", 0.5),
            custom_gestures={
                name: cfg["fingers"]
                for name, cfg in gesture_cfgs.items()
                if isinstance(cfg, dict) and "fingers" in cfg
            },
        )

    recognizer = _make_recognizer()
    trigger    = MacTrigger()
    start_time = time.monotonic()

    cam_idx = settings.get("camera_index", 0)
    cap, cam_idx, cam_name = find_camera(preferred=cam_idx, retries=1)

    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    all_cam_names:     list = _macos_camera_names()
    cam_select_mode:   bool = cap is None
    cam_select_cursor: int  = max(0, cam_idx)
    cam_availability:  dict = {}

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

    hold_required:  int  = settings.get("gesture_hold_frames", 20)
    show_landmarks: bool = settings.get("show_landmarks", True)
    show_fps:       bool = settings.get("show_fps", True)

    hold_counts:     dict[str, int]   = {}
    last_trigger_ts: dict[str, float] = {}
    triggered_ts:    dict[str, float] = {}  # when action fired; hold bar stays 100% for 500ms
    current_live_fingers = None

    flash_gesture = ""
    flash_ts      = 0.0

    # ── Voice ──────────────────────────────────────────────────────────────
    voice_settings     = (settings.get("voice") or {}) if isinstance(settings, dict) else {}
    voice_commands_cfg = config.get("voice_commands") or {}
    voice_wake_word    = str(voice_settings.get("wake_word", "hey vision"))

    def _on_voice_command(name: str, cfg: dict) -> None:
        nonlocal flash_gesture, flash_ts
        flash_gesture = name
        flash_ts = time.time()
        logger.info("Voice triggered: %s → %s", name, cfg.get("label", name))
        threading.Thread(target=trigger.execute, args=(cfg,), daemon=True).start()

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

    # ── UI state ───────────────────────────────────────────────────────────
    show_gestures:     bool = False
    gesture_list_cursor: int = 0
    show_settings:     bool = False
    pending_settings:  dict = {}
    show_voice_cmds:   bool = False
    voice_cmds_cursor: int  = 0

    reg = RegistrationStateMachine()

    fps = 0.0
    fps_frame_cnt = 0
    fps_tick = time.time()

    logger.info("VisionTrigger started. Show your hand to the camera.")
    logger.info("Press q/Esc to quit, l landmarks, g gestures, r register, "
                "v voice, s settings, h voice cmds, c camera.")

    _cam_btns:      dict = {}
    _main_btns:     dict = {}
    _gesture_btns:  dict = {}
    _reg_btns:      dict = {}
    _file_btns:     dict = {}
    _del_btns:      dict = {}
    _settings_btns: dict = {}
    _vcmds_btns:    dict = {}
    _cb_registered        = False

    while True:
        mouse_pos = _mouse_state["pos"]
        click     = _mouse_state["click"]
        _mouse_state["click"] = None

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
        hold_progress   = 0.0
        action_label    = ""

        # ── Gesture processing ─────────────────────────────────────────────
        if not cam_select_mode and cap is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            results = recognizer.process(rgb, timestamp_ms)

            if results.hand_landmarks and results.handedness:
                hand_lm = results.hand_landmarks[0]

                if show_landmarks:
                    recognizer.draw_landmarks(frame, hand_lm)

                current_gesture, finger_tuple = recognizer.classify(hand_lm)
                current_live_fingers = finger_tuple
                action_label = display_label(current_gesture, gesture_cfgs)

                # Feed registration state machine (no-op outside CAPTURE state)
                reg.on_fingers(finger_tuple)

                # Normal triggering (only when idle)
                if reg.state == RegState.IDLE and current_gesture not in ("NONE", "UNKNOWN"):
                    for g in list(hold_counts):
                        if g != current_gesture:
                            hold_counts.pop(g)
                    hold_counts[current_gesture] = hold_counts.get(current_gesture, 0) + 1

                    now = time.time()
                    # Keep progress bar at 100% for 500ms after trigger fires
                    if current_gesture in triggered_ts and now - triggered_ts[current_gesture] < 0.5:
                        hold_progress = 1.0
                    else:
                        hold_progress = min(hold_counts[current_gesture] / hold_required, 1.0)

                    if hold_counts[current_gesture] >= hold_required:
                        g_cfg    = gesture_cfgs.get(current_gesture, {})
                        cooldown = float(g_cfg.get("cooldown", 2.0))
                        if g_cfg and now - last_trigger_ts.get(current_gesture, 0.0) >= cooldown:
                            last_trigger_ts[current_gesture] = now
                            triggered_ts[current_gesture]    = now
                            hold_counts[current_gesture]     = 0
                            flash_gesture = current_gesture
                            flash_ts      = now
                            logger.info("Triggered: %s → %s", current_gesture, action_label)

                            def _run_action(_cfg=g_cfg):
                                try:
                                    trigger.execute(_cfg)
                                except Exception as exc:
                                    logger.error("Action error: %s", exc)
                            threading.Thread(target=_run_action, daemon=True).start()
                elif reg.state != RegState.IDLE:
                    hold_counts.clear()
            else:
                hold_counts.clear()
                current_live_fingers = None
                reg.on_fingers(None)  # resets stable_count when no hand visible

        # ── FPS ────────────────────────────────────────────────────────────
        if not cam_select_mode:
            fps_frame_cnt += 1
            if fps_frame_cnt >= 30:
                fps      = fps_frame_cnt / (time.time() - fps_tick)
                fps_tick = time.time()
                fps_frame_cnt = 0

        flash_active = bool(flash_gesture) and (time.time() - flash_ts < 1.0)

        # ── Folder-watcher lifecycle ───────────────────────────────────────
        new_files = reg.tick_file_watcher()
        if new_files:
            reg.file_list   = _get_script_files(reg.action_type)
            reg.file_cursor = max(0, len(reg.file_list) - 1)

        # ── Build cooldowns map for gesture-list display ───────────────────
        cooldowns_map = {
            name: float(cfg.get("cooldown", 2.0))
            for name, cfg in gesture_cfgs.items()
            if isinstance(cfg, dict)
        }

        # ── Drawing ────────────────────────────────────────────────────────
        if cam_select_mode:
            _cam_btns = draw_camera_selector(
                frame, all_cam_names, cam_select_cursor,
                cam_availability, cap is not None, mouse_pos,
            )
            _main_btns = _gesture_btns = _reg_btns = {}
            _file_btns = _del_btns = _settings_btns = _vcmds_btns = {}
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
                show_settings=show_settings,
                show_voice_commands=show_voice_cmds,
            )
            draw_voice_status(frame, voice_status, voice_status_text, voice_wake_word)

            if reg.state not in (RegState.IDLE, RegState.FILE_PICK, RegState.DELETE_CONFIRM):
                _reg_btns = draw_registration_overlay(
                    frame, reg.state, reg.input_buf, reg.fingers,
                    reg.name, reg.action_type, reg.action_detail,
                    reg.stable_count, reg.current_fingers,
                    reg.selected_filename, reg.is_edit, mouse_pos,
                )
            else:
                _reg_btns = {}

            if reg.state == RegState.FILE_PICK:
                _file_btns = draw_file_pick_overlay(
                    frame, reg.action_type, reg.file_list,
                    reg.file_cursor, mouse_pos,
                    watching=(reg._folder_watcher is not None),
                )
            else:
                _file_btns = {}

            if reg.state == RegState.DELETE_CONFIRM:
                _del_btns = draw_delete_confirm_overlay(frame, reg.name, mouse_pos)
            else:
                _del_btns = {}

            if show_gestures and reg.state == RegState.IDLE:
                _gesture_btns = draw_gestures_list_overlay(
                    frame, gesture_cfgs, gesture_list_cursor, mouse_pos,
                    last_trigger_ts=last_trigger_ts,
                    cooldowns=cooldowns_map,
                )
            else:
                _gesture_btns = {}

            if show_settings and reg.state == RegState.IDLE:
                _settings_btns = draw_settings_panel(frame, pending_settings, mouse_pos)
            else:
                _settings_btns = {}

            if show_voice_cmds and reg.state == RegState.IDLE:
                _vcmds_btns = draw_voice_commands_panel(
                    frame, voice_commands_cfg, voice_cmds_cursor, mouse_pos)
            else:
                _vcmds_btns = {}

        cv2.imshow("VisionTrigger", frame)
        if not _cb_registered:
            cv2.setMouseCallback("VisionTrigger", _mouse_cb)
            _cb_registered = True

        kbd = cv2.waitKey(1) & 0xFF

        # ── In-loop helpers ────────────────────────────────────────────────

        def _switch_camera(idx: int) -> bool:
            nonlocal cap, cam_idx, cam_name, cam_select_mode, fps_frame_cnt, fps_tick
            new_cap = _open_index(idx)
            if new_cap:
                if cap is not None:
                    cap.release()
                cap = new_cap
                cam_idx  = idx
                cam_name = all_cam_names[idx] if idx < len(all_cam_names) else f"Camera {idx}"
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cam_select_mode = False
                fps_frame_cnt   = 0
                fps_tick        = time.time()
                logger.info("Switched to camera %d: %s", cam_idx, cam_name)
                return True
            cam_availability[idx] = False
            logger.warning("Camera %d could not be opened.", idx)
            return False

        def _save_gesture() -> None:
            nonlocal flash_gesture, flash_ts
            if reg.is_edit:
                section  = "custom_gestures" if reg.edit_is_custom else "gestures"
                existing = dict((config.get(section) or {}).get(reg.name, {}))
                for k in ("command", "script", "name"):
                    existing.pop(k, None)
                existing["action"] = reg.action_type
                if reg.action_type == "shell":
                    existing["command"] = reg.action_detail
                elif reg.action_type == "applescript":
                    existing["script"]  = reg.action_detail
                elif reg.action_type == "shortcut":
                    existing["name"]    = reg.action_detail
                update_gesture_in_config(config, CONFIG_PATH, reg.name, existing, section)
                gesture_cfgs[reg.name] = existing
            else:
                lbl   = reg.name.replace("_", " ").title()
                entry: dict = {
                    "fingers":  list(reg.fingers),
                    "action":   reg.action_type,
                    "label":    lbl,
                    "cooldown": 2.0,
                }
                if reg.action_type == "shell":
                    entry["command"] = reg.action_detail
                elif reg.action_type == "applescript":
                    entry["script"]  = reg.action_detail
                elif reg.action_type == "shortcut":
                    entry["name"]    = reg.action_detail
                update_gesture_in_config(config, CONFIG_PATH, reg.name, entry)
                recognizer._gesture_map[reg.fingers] = reg.name
                gesture_cfgs[reg.name] = entry
            flash_gesture = reg.name
            flash_ts      = time.time()
            logger.info("Saved gesture '%s' → %s", reg.name, reg.action_type)
            reg.reset()
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
        elif reg.state != RegState.IDLE:
            # Esc or Cancel button always cancels
            if kbd == 27:
                reg.reset()
                hold_counts.clear()
            elif _hit(_reg_btns.get("cancel", ())  or
                      _file_btns.get("cancel", ()) or
                      _del_btns.get("cancel",  ()), click):
                reg.reset()
                hold_counts.clear()
            # Back button (registration overlay states only)
            elif _hit(_reg_btns.get("back", ()), click):
                reg.go_back()
            elif reg.state in (RegState.NAME, RegState.ACTION_DETAIL):
                if kbd == 13:
                    text = reg.input_buf.strip()
                    if text:
                        if reg.state == RegState.NAME:
                            reg.name      = text.upper()
                            reg.input_buf = ""
                            reg.state     = RegState.ACTION_TYPE
                        else:
                            reg.action_detail = text
                            reg.input_buf     = ""
                            reg.state         = RegState.CONFIRM
                elif kbd in (8, 127):
                    reg.input_buf = reg.input_buf[:-1]
                elif 32 <= kbd <= 126:
                    reg.input_buf += chr(kbd)
            elif reg.state == RegState.ACTION_TYPE:
                action_chosen = None
                if kbd == ord("s") or _hit(_reg_btns.get("shell", ()), click):
                    action_chosen = "shell"
                elif kbd == ord("a") or _hit(_reg_btns.get("applescript", ()), click):
                    action_chosen = "applescript"
                elif kbd == ord("k") or _hit(_reg_btns.get("shortcut", ()), click):
                    action_chosen = "shortcut"
                if action_chosen == "shortcut":
                    reg.action_type = "shortcut"
                    reg.input_buf   = ""
                    reg.state       = RegState.ACTION_DETAIL
                elif action_chosen in ("shell", "applescript"):
                    reg.action_type = action_chosen
                    reg.file_list   = _get_script_files(action_chosen)
                    reg.file_cursor = 0
                    reg.state       = RegState.FILE_PICK
            elif reg.state == RegState.FILE_PICK:
                if _hit(_file_btns.get("back", ()), click):
                    reg.go_back()
                elif _hit(_file_btns.get("browse", ()), click):
                    new_files = _browse_and_copy(reg.action_type)
                    if new_files:
                        reg.file_list   = _get_script_files(reg.action_type)
                        reg.file_cursor = max(0, len(reg.file_list) - 1)
                elif _hit(_file_btns.get("finder", ()), click):
                    _open_scripts_folder()
                elif kbd == ord("j"):
                    reg.file_cursor = min(reg.file_cursor + 1, max(0, len(reg.file_list) - 1))
                elif kbd == ord("k"):
                    reg.file_cursor = max(reg.file_cursor - 1, 0)
                else:
                    sel_idx = None
                    if kbd == 13 and reg.file_list:
                        sel_idx = reg.file_cursor
                    else:
                        for key_name, rect in _file_btns.items():
                            if key_name.startswith("file_") and _hit(rect, click):
                                sel_idx = int(key_name.split("_")[1])
                                reg.file_cursor = sel_idx
                                break
                    if sel_idx is not None and sel_idx < len(reg.file_list):
                        chosen_file = reg.file_list[sel_idx]
                        if chosen_file.suffix.lower() in (".scpt", ".scptd"):
                            reg.action_detail     = str(chosen_file)
                            reg.selected_filename = chosen_file.name
                            reg.input_buf         = ""
                            reg.state             = RegState.CONFIRM
                        else:
                            try:
                                content = chosen_file.read_text(encoding="utf-8").strip()
                            except Exception as exc:
                                logger.error("Cannot read %s: %s", chosen_file.name, exc)
                            else:
                                reg.action_detail     = content
                                reg.selected_filename = chosen_file.name
                                reg.input_buf         = ""
                                reg.state             = RegState.CONFIRM
            elif reg.state == RegState.DELETE_CONFIRM:
                if kbd == ord("y") or _hit(_del_btns.get("delete", ()), click):
                    entry = gesture_cfgs.pop(reg.name, None)
                    delete_gesture_from_config(config, CONFIG_PATH, reg.name)
                    if entry and entry.get("fingers"):
                        recognizer._gesture_map.pop(tuple(entry["fingers"]), None)
                    hold_counts.pop(reg.name, None)
                    last_trigger_ts.pop(reg.name, None)
                    triggered_ts.pop(reg.name, None)
                    logger.info("Deleted gesture '%s'", reg.name)
                    reg.reset()
                elif kbd == ord("n") or _hit(_del_btns.get("cancel", ()), click):
                    reg.reset()
            elif reg.state == RegState.CONFIRM:
                if kbd == ord("y") or _hit(_reg_btns.get("save", ()), click):
                    _save_gesture()
                elif kbd == ord("n") or _hit(_reg_btns.get("cancel2", ()), click):
                    reg.reset()
                    hold_counts.clear()

        # ── Settings panel input ───────────────────────────────────────────
        elif show_settings:
            def _hit_s(key: str) -> bool:
                return _hit(_settings_btns.get(key, ()), click)

            if kbd == 27 or _hit_s("close") or _hit_s("cancel") or kbd == ord("s"):
                show_settings = False
            elif _hit_s("save"):
                old_det = settings.get("detection_confidence", 0.7)
                old_trk = settings.get("tracking_confidence",  0.5)
                settings.update(pending_settings)
                hold_required  = int(settings.get("gesture_hold_frames", 20))
                show_fps       = bool(settings.get("show_fps",       True))
                show_landmarks = bool(settings.get("show_landmarks", True))
                save_config(CONFIG_PATH, config)
                if (pending_settings.get("detection_confidence", old_det) != old_det or
                        pending_settings.get("tracking_confidence",  old_trk) != old_trk):
                    recognizer.close()
                    recognizer = _make_recognizer()
                show_settings = False
                logger.info("Settings saved.")
            elif _hit_s("det_minus"):
                pending_settings["detection_confidence"] = round(
                    max(0.1, pending_settings.get("detection_confidence", 0.7) - 0.05), 2)
            elif _hit_s("det_plus"):
                pending_settings["detection_confidence"] = round(
                    min(1.0, pending_settings.get("detection_confidence", 0.7) + 0.05), 2)
            elif _hit_s("track_minus"):
                pending_settings["tracking_confidence"] = round(
                    max(0.1, pending_settings.get("tracking_confidence", 0.5) - 0.05), 2)
            elif _hit_s("track_plus"):
                pending_settings["tracking_confidence"] = round(
                    min(1.0, pending_settings.get("tracking_confidence", 0.5) + 0.05), 2)
            elif _hit_s("hold_minus"):
                pending_settings["gesture_hold_frames"] = max(
                    1, pending_settings.get("gesture_hold_frames", 20) - 1)
            elif _hit_s("hold_plus"):
                pending_settings["gesture_hold_frames"] = min(
                    120, pending_settings.get("gesture_hold_frames", 20) + 1)
            elif _hit_s("fps_toggle"):
                pending_settings["show_fps"] = not pending_settings.get("show_fps", True)
            elif _hit_s("landmarks_toggle"):
                pending_settings["show_landmarks"] = not pending_settings.get("show_landmarks", True)

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
                settings["show_landmarks"] = show_landmarks
                save_config(CONFIG_PATH, config)
                logger.info("Landmarks: %s", "ON" if show_landmarks else "OFF")
            if kbd == ord("g") or _hit(_main_btns.get("gestures", ()), click):
                show_gestures = not show_gestures
                if show_gestures:
                    gesture_list_cursor = 0
                    show_settings  = False
                    show_voice_cmds = False
                logger.info("Gesture list: %s", "ON" if show_gestures else "OFF")
            if kbd == ord("r") or _hit(_main_btns.get("register", ()), click):
                if reg.state == RegState.IDLE:
                    show_gestures  = False
                    show_settings  = False
                    show_voice_cmds = False
                    reg.start_capture()
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
            if kbd == ord("s") or _hit(_main_btns.get("settings", ()), click):
                show_settings = not show_settings
                if show_settings:
                    pending_settings = dict(settings)
                    show_gestures   = False
                    show_voice_cmds = False
                logger.info("Settings panel: %s", "ON" if show_settings else "OFF")
            if kbd == ord("h") or _hit(_main_btns.get("voice_cmds", ()), click):
                show_voice_cmds = not show_voice_cmds
                if show_voice_cmds:
                    voice_cmds_cursor = 0
                    show_settings  = False
                    show_gestures  = False
                logger.info("Voice commands panel: %s", "ON" if show_voice_cmds else "OFF")

            # Gesture-list interactions (only when visible)
            if show_gestures:
                _gnames = list(gesture_cfgs.keys())
                if kbd == ord("j"):
                    gesture_list_cursor = min(gesture_list_cursor + 1,
                                              max(0, len(_gnames) - 1))
                elif kbd == ord("k"):
                    gesture_list_cursor = max(gesture_list_cursor - 1, 0)
                for key_name, rect in _gesture_btns.items():
                    if key_name.startswith("row_") and _hit(rect, click):
                        gesture_list_cursor = int(key_name.split("_")[1])
                        break
                if kbd == ord("g") or _hit(_gesture_btns.get("close", ()), click):
                    show_gestures = False
                elif (kbd == ord("e") or _hit(_gesture_btns.get("edit", ()), click)) \
                        and _gnames:
                    if 0 <= gesture_list_cursor < len(_gnames):
                        edit_name = _gnames[gesture_list_cursor]
                        edit_cfg  = gesture_cfgs[edit_name]
                        show_gestures = False
                        reg.start_edit(edit_name, edit_cfg)
                        logger.info("Editing gesture '%s'", edit_name)
                elif (kbd == ord("d") or _hit(_gesture_btns.get("delete", ()), click)) \
                        and _gnames:
                    if 0 <= gesture_list_cursor < len(_gnames):
                        del_name = _gnames[gesture_list_cursor]
                        show_gestures = False
                        reg.reset()
                        reg.name  = del_name
                        reg.state = RegState.DELETE_CONFIRM
                        logger.info("Delete confirm for '%s'", del_name)

            # Voice-commands-panel interactions (only when visible)
            if show_voice_cmds:
                vcmd_names = list(voice_commands_cfg.keys())
                if kbd == ord("j"):
                    voice_cmds_cursor = min(voice_cmds_cursor + 1,
                                            max(0, len(vcmd_names) - 1))
                elif kbd == ord("k"):
                    voice_cmds_cursor = max(voice_cmds_cursor - 1, 0)
                if kbd == ord("h") or _hit(_vcmds_btns.get("close", ()), click):
                    show_voice_cmds = False

    # ── Cleanup ────────────────────────────────────────────────────────────
    flush_pending_save()
    voice_listener.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    recognizer.close()
    logger.info("VisionTrigger stopped.")


if __name__ == "__main__":
    main()
