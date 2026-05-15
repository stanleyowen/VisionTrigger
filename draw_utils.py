"""
draw_utils.py – All OpenCV drawing helpers for VisionTrigger.
"""

import time

import cv2
import numpy as np

from gestures import BUILTIN_GESTURES
from reg_machine import RegState

# ---------------------------------------------------------------------------
# Colours (BGR)
# ---------------------------------------------------------------------------

_GREEN  = (60, 220, 60)
_YELLOW = (0, 200, 255)
_GREY   = (140, 140, 140)
_WHITE  = (240, 240, 240)
_BLACK  = (0, 0, 0)
_CYAN   = (255, 210, 0)
_ORANGE = (0, 140, 255)

# ---------------------------------------------------------------------------
# Low-level primitives
# ---------------------------------------------------------------------------

def _put(frame, text, pos, scale=0.65, color=_WHITE, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _hit(rect: tuple, point) -> bool:
    if not point or not rect:
        return False
    x, y = point
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def _fill_alpha(
    frame,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    alpha: float,
) -> None:
    """Semi-transparent filled rectangle using only a ROI copy (not a full-frame copy)."""
    roi = frame[y1:y2, x1:x2]
    filled = roi.copy()
    filled[:] = color
    cv2.addWeighted(filled, alpha, roi, 1.0 - alpha, 0, roi)


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


# ---------------------------------------------------------------------------
# Hand icon
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Top overlay
# ---------------------------------------------------------------------------

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
    _fill_alpha(frame, 0, 0, w, 68, _BLACK, 0.55)

    if show_fps:
        _put(frame, f"FPS {fps:4.1f}", (10, 26), scale=0.55, color=_GREEN)

    gesture_color = _GREEN if gesture not in ("UNKNOWN", "NONE", "") else _GREY
    _put(frame, f"Gesture:  {gesture}", (10, 56),
         scale=0.80, color=gesture_color)

    # Small hand icon to the right of the gesture label
    icon_fingers = live_fingers
    if icon_fingers is None and gesture not in ("UNKNOWN", "NONE", ""):
        icon_fingers = BUILTIN_GESTURES.get(gesture)
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


# ---------------------------------------------------------------------------
# Finger display helper
# ---------------------------------------------------------------------------

def _finger_display_str(fingers: tuple) -> str:
    """Format a 5-bool finger tuple as a compact readable string."""
    labels = ("Th", "Ix", "Md", "Rg", "Pk")
    return "  ".join(f"{l}:{'Y' if v else 'N'}" for l, v in zip(labels, fingers))


# ---------------------------------------------------------------------------
# Gesture list panel
# ---------------------------------------------------------------------------

def draw_gestures_list_overlay(
    frame,
    gesture_cfgs: dict,
    cursor: int = -1,
    mouse_pos: tuple = (0, 0),
    last_trigger_ts: dict = None,
    cooldowns: dict = None,
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

    _fill_alpha(frame, px1, py1, px2, py2, (15, 15, 15), 0.82)
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
            icon_fingers = BUILTIN_GESTURES.get(name, (False,) * 5)
        draw_hand_icon(frame, lx, row_y + 1, icon_fingers, gesture_name=name)

        # ── Two-line text (label on top, action+detail below) ────────────────
        row_color = _CYAN if is_selected else _WHITE
        prefix = "▶ " if is_selected else ""
        tx = lx + icon_w
        _put(frame, f"{prefix}{label}", (tx, row_y + 16), scale=0.55, color=row_color)
        _put(frame, f"[{action}]",      (tx, row_y + 34), scale=0.48, color=action_color)
        _put(frame, detail_short, (tx + 90, row_y + 34), scale=0.42, color=_GREY, thickness=1)

        # ── Cooldown indicator ───────────────────────────────────────────────
        if last_trigger_ts and cooldowns:
            now = time.time()
            cd = cooldowns.get(name, 0.0)
            last = last_trigger_ts.get(name, 0.0)
            remaining = cd - (now - last)
            if remaining > 0:
                _put(frame, f"⟳{remaining:.1f}s", (tx + 200, row_y + 34),
                     scale=0.42, color=_ORANGE, thickness=1)

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


# ---------------------------------------------------------------------------
# Registration overlay
# ---------------------------------------------------------------------------

def draw_registration_overlay(
    frame,
    reg_state,
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

    _fill_alpha(frame, px1, py1, px2, py2, (15, 15, 15), 0.82)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _ORANGE, 2)

    lx, y = px1 + 18, py1 + 30
    title = "EDIT GESTURE" if reg_is_edit else "REGISTER GESTURE"
    _put(frame, title, (lx, y), scale=0.75, color=_ORANGE)

    # Cancel button (always visible, top-right)
    cancel_rect = (px2 - 86, py1 + 8, px2 - 8, py1 + 32)
    draw_button(frame, cancel_rect, "Cancel", mouse_pos)
    rects["cancel"] = cancel_rect

    # Back button — shown in NAME, ACTION_TYPE, ACTION_DETAIL, CONFIRM
    # but NOT in CAPTURE or DELETE_CONFIRM, and NOT in edit ACTION_TYPE.
    _states_with_back = (RegState.NAME, RegState.ACTION_TYPE,
                         RegState.ACTION_DETAIL, RegState.CONFIRM)
    show_back = (reg_state in _states_with_back and
                 not (reg_is_edit and reg_state == RegState.ACTION_TYPE))
    if show_back:
        back_rect = (px2 - 174, py1 + 8, px2 - 92, py1 + 32)
        draw_button(frame, back_rect, "< Back", mouse_pos)
        rects["back"] = back_rect

    y += 38

    if reg_state == RegState.CAPTURE:
        _put(frame, "Hold your gesture steady...", (lx, y), color=_YELLOW)
        y += 32
        if current_fingers is not None:
            _put(frame, _finger_display_str(current_fingers), (lx, y),
                 scale=0.58, color=_WHITE)
        y += 28
        stable_required = 40
        bx1, by1, bx2, by2 = lx, y, px2 - 18, y + 18
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
        progress = min(reg_stable_count / stable_required, 1.0)
        cv2.rectangle(frame, (bx1, by1),
                      (bx1 + int((bx2 - bx1) * progress), by2), _ORANGE, -1)
        y += 26
        _put(frame, f"Stable: {reg_stable_count}/{stable_required} frames",
             (lx, y), scale=0.50, color=_GREY, thickness=1)

    elif reg_state == RegState.NAME:
        _put(frame, f"Captured: {_finger_display_str(reg_fingers)}",
             (lx, y), scale=0.58, color=_GREEN)
        y += 32
        _put(frame, "Enter a name for this gesture:", (lx, y))
        y += 32
        _put(frame, f"> {reg_input_buf}_", (lx, y), color=_CYAN)
        y += 36
        _put(frame, "Press Enter to confirm", (lx, y), scale=0.50,
             color=_GREY, thickness=1)

    elif reg_state == RegState.ACTION_TYPE:
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

    elif reg_state == RegState.ACTION_DETAIL:
        _put(frame, f"{reg_name}  [{reg_action_type}]", (lx, y), color=_GREEN)
        y += 32
        _put(frame, "Shortcut name:", (lx, y))
        y += 32
        _put(frame, f"> {reg_input_buf}_", (lx, y), color=_CYAN)
        y += 36
        _put(frame, "Press Enter to confirm", (lx, y), scale=0.50,
             color=_GREY, thickness=1)

    elif reg_state == RegState.CONFIRM:
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
        save_rect    = (lx,       y, lx + 90,  y + 34)
        cancel2_rect = (lx + 102, y, lx + 200, y + 34)
        draw_button(frame, save_rect,    "Save",   mouse_pos, active=True)
        draw_button(frame, cancel2_rect, "Cancel", mouse_pos)
        rects["save"]    = save_rect
        rects["cancel2"] = cancel2_rect

    return rects


# ---------------------------------------------------------------------------
# Delete confirm overlay
# ---------------------------------------------------------------------------

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

    _fill_alpha(frame, px1, py1, px2, py2, (15, 15, 15), 0.85)
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


# ---------------------------------------------------------------------------
# File-pick overlay
# ---------------------------------------------------------------------------

def draw_file_pick_overlay(
    frame,
    action_type: str,
    file_list: list,
    cursor: int,
    mouse_pos: tuple = (0, 0),
    watching: bool = True,
) -> dict:
    """Draw file-picker panel; return {key: rect} of clickable elements."""
    from reg_machine import SCRIPTS_DIR
    h, w = frame.shape[:2]
    rects: dict = {}
    px1, py1 = w // 2 - 340, h // 2 - 185
    px2, py2 = w // 2 + 340, h // 2 + 200

    _fill_alpha(frame, px1, py1, px2, py2, (15, 15, 15), 0.82)
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
        folder_str = "…" + folder_str[-57:]
    _put(frame, f"scripts/: {folder_str}", (lx, y),
         scale=0.44, color=_GREY, thickness=1)
    y += 28

    # Browse + Open-in-Finder buttons
    browse_rect = (lx, y, lx + 148, y + 32)
    finder_rect = (lx + 158, y, lx + 158 + 160, y + 32)
    draw_button(frame, browse_rect, "Browse Files…",      mouse_pos)
    draw_button(frame, finder_rect, "Open scripts/ folder", mouse_pos)
    rects["browse"] = browse_rect
    rects["finder"] = finder_rect
    y += 40

    # Folder-watcher status
    dot_x, dot_y = lx + 6, y + 6
    if watching:
        cv2.circle(frame, (dot_x, dot_y), 5, _GREEN, -1)
        _put(frame, "Watching for new files — drag files to scripts/ in Finder",
             (lx + 18, y + 10), scale=0.45, color=_GREEN, thickness=1)
    else:
        cv2.circle(frame, (dot_x, dot_y), 5, _GREY, 1)
        _put(frame, "Not watching",
             (lx + 18, y + 10), scale=0.45, color=_GREY, thickness=1)
    y += 22

    cv2.line(frame, (px1 + 8, y), (px2 - 8, y), (60, 60, 60), 1)
    y += 12

    # File list
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
            prefix = "▶  " if selected else "   "
            _put(frame, f"{prefix}{fp.name}", (lx, y + 12), scale=0.62, color=color)
            y += row_h
        if len(file_list) > max_visible:
            _put(frame, f"  … {len(file_list)} files total",
                 (lx, y + 4), scale=0.44, color=_GREY, thickness=1)
    return rects


# ---------------------------------------------------------------------------
# Main button bar
# ---------------------------------------------------------------------------

def draw_main_buttons(
    frame,
    mouse_pos: tuple,
    show_landmarks: bool,
    show_gestures: bool,
    voice_on: bool = False,
    show_settings: bool = False,
    show_voice_commands: bool = False,
) -> dict:
    """Draw the clickable bottom bar; return {key: rect}."""
    h, w = frame.shape[:2]
    y1, y2 = h - 46, h - 8
    specs = [
        ("quit",       "Quit",                                           65),
        ("landmarks",  f"Landmarks: {'ON ' if show_landmarks else 'OFF'}", 150),
        ("gestures",   "Gestures",                                       94),
        ("register",   "Register",                                       94),
        ("camera",     "Camera",                                         84),
        ("voice",      f"Voice: {'ON ' if voice_on else 'OFF'}",         110),
        ("settings",   "Settings",                                       80),
        ("voice_cmds", "Voice Cmds",                                     94),
    ]
    rects: dict = {}
    x = 12
    for key, label, bw in specs:
        rect = (x, y1, x + bw, y2)
        is_active = (
            (key == "landmarks"  and show_landmarks) or
            (key == "gestures"   and show_gestures) or
            (key == "voice"      and voice_on) or
            (key == "settings"   and show_settings) or
            (key == "voice_cmds" and show_voice_commands)
        )
        draw_button(frame, rect, label, mouse_pos, active=is_active)
        rects[key] = rect
        x += bw + 8
    return rects


# ---------------------------------------------------------------------------
# Voice status indicator
# ---------------------------------------------------------------------------

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
    if len(msg) > 70:
        msg = msg[:67] + "…"
    _put(frame, f"VOICE  {msg}", (10, 92), scale=0.55, color=color, thickness=1)


# ---------------------------------------------------------------------------
# Camera selector
# ---------------------------------------------------------------------------

def draw_camera_selector(
    frame,
    cam_names: list,
    cursor: int,
    availability: dict,
    has_active: bool,
    mouse_pos: tuple = (0, 0),
) -> dict:
    """Draw camera selector; return {key: rect} of clickable elements."""
    h, w = frame.shape[:2]
    rects: dict = {}

    _fill_alpha(frame, 0, 0, w, h, _BLACK, 0.80)

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
# Settings panel
# ---------------------------------------------------------------------------

def draw_settings_panel(frame, pending: dict, mouse_pos: tuple) -> dict:
    """
    Draw the settings panel (centred, ~500×320 px).
    Returns {key: rect} of clickable elements.

    pending keys: detection_confidence (float), tracking_confidence (float),
                  gesture_hold_frames (int), show_fps (bool), show_landmarks (bool).
    """
    h, w = frame.shape[:2]
    rects: dict = {}
    pw, panel_h = 500, 320
    px1 = w // 2 - pw // 2
    py1 = h // 2 - panel_h // 2
    px2 = px1 + pw
    py2 = py1 + panel_h

    _fill_alpha(frame, px1, py1, px2, py2, (15, 15, 15), 0.88)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _CYAN, 2)

    lx = px1 + 18
    _put(frame, "SETTINGS", (lx, py1 + 28), scale=0.75, color=_CYAN)

    close_rect = (px2 - 72, py1 + 8, px2 - 8, py1 + 32)
    draw_button(frame, close_rect, "Close", mouse_pos)
    rects["close"] = close_rect

    sep_y = py1 + 40
    cv2.line(frame, (px1 + 8, sep_y), (px2 - 8, sep_y), _GREY, 1)

    y = sep_y + 22

    def _row_pm(label: str, value_str: str, key_minus: str, key_plus: str):
        nonlocal y
        _put(frame, label, (lx, y), scale=0.55, color=_WHITE)
        minus_rect = (px2 - 180, y - 16, px2 - 148, y + 4)
        plus_rect  = (px2 - 80,  y - 16, px2 - 48,  y + 4)
        draw_button(frame, minus_rect, "-", mouse_pos)
        draw_button(frame, plus_rect,  "+", mouse_pos)
        _put(frame, value_str, (px2 - 140, y), scale=0.55, color=_CYAN)
        rects[key_minus] = minus_rect
        rects[key_plus]  = plus_rect
        y += 34

    def _row_toggle(label: str, value: bool, key: str):
        nonlocal y
        _put(frame, label, (lx, y), scale=0.55, color=_WHITE)
        state_label = "ON" if value else "OFF"
        tog_rect = (px2 - 100, y - 16, px2 - 8, y + 4)
        draw_button(frame, tog_rect, state_label, mouse_pos,
                    active=value)
        rects[key] = tog_rect
        y += 34

    _row_pm("Detection Confidence",
            f"{pending.get('detection_confidence', 0.7):.2f}",
            "det_minus", "det_plus")
    _row_pm("Tracking Confidence",
            f"{pending.get('tracking_confidence', 0.5):.2f}",
            "track_minus", "track_plus")
    _row_pm("Gesture Hold Frames",
            str(pending.get("gesture_hold_frames", 20)),
            "hold_minus", "hold_plus")
    _row_toggle("Show FPS",       pending.get("show_fps", True),       "fps_toggle")
    _row_toggle("Show Landmarks", pending.get("show_landmarks", True), "landmarks_toggle")

    sep_y2 = py2 - 52
    cv2.line(frame, (px1 + 8, sep_y2), (px2 - 8, sep_y2), _GREY, 1)

    save_rect   = (px2 - 230, py2 - 42, px2 - 110, py2 - 10)
    cancel_rect = (px2 - 100, py2 - 42, px2 - 10,  py2 - 10)
    draw_button(frame, save_rect,   "Save & Apply", mouse_pos, active=True)
    draw_button(frame, cancel_rect, "Cancel",       mouse_pos)
    rects["save"]   = save_rect
    rects["cancel"] = cancel_rect

    return rects


# ---------------------------------------------------------------------------
# Voice commands panel
# ---------------------------------------------------------------------------

def draw_voice_commands_panel(
    frame,
    voice_commands: dict,
    cursor: int,
    mouse_pos: tuple,
) -> dict:
    """
    Draw the voice commands panel (right side, scrollable).
    Returns {key: rect} of clickable elements.
    """
    h, w = frame.shape[:2]
    rects: dict = {}

    panel_w = 600
    px1 = w - panel_w - 16
    py1 = 80
    px2 = w - 16
    py2 = h - 60

    _fill_alpha(frame, px1, py1, px2, py2, (15, 15, 15), 0.85)
    cv2.rectangle(frame, (px1, py1), (px2, py2), _CYAN, 2)

    lx = px1 + 14
    n = len(voice_commands)
    _put(frame, f"VOICE COMMANDS  ({n})", (lx, py1 + 24), scale=0.58, color=_CYAN)

    close_rect = (px2 - 72, py1 + 6, px2 - 6, py1 + 28)
    draw_button(frame, close_rect, "Close", mouse_pos)
    rects["close"] = close_rect

    sep_y = py1 + 36
    cv2.line(frame, (px1 + 8, sep_y), (px2 - 8, sep_y), _GREY, 1)

    row_h = 40
    available_h = py2 - sep_y - 8
    max_visible = max(1, available_h // row_h)

    rows = list(voice_commands.items())
    scroll_top = 0
    if cursor >= 0:
        scroll_top = max(0, min(cursor - max_visible // 2,
                                max(0, len(rows) - max_visible)))

    row_y = sep_y + 4

    for list_idx in range(scroll_top, min(scroll_top + max_visible, len(rows))):
        name, cfg = rows[list_idx]
        if not isinstance(cfg, dict):
            row_y += row_h
            continue
        action = cfg.get("action", "?")
        phrases = cfg.get("phrases") or []
        if isinstance(phrases, str):
            phrases = [phrases]
        first_phrase = phrases[0] if phrases else ""

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

        action_color = {
            "shell": _GREEN, "applescript": _YELLOW, "shortcut": _CYAN,
        }.get(action, _WHITE)
        row_color = _CYAN if is_selected else _WHITE
        prefix = "▶ " if is_selected else ""
        _put(frame, f"{prefix}{name}", (lx, row_y + 14), scale=0.52, color=row_color)
        _put(frame, f"[{action}]", (lx + 190, row_y + 14), scale=0.45, color=action_color)
        phrase_short = first_phrase[:38] + ("…" if len(first_phrase) > 38 else "")
        _put(frame, phrase_short, (lx, row_y + 30), scale=0.40, color=_GREY, thickness=1)

        row_y += row_h

    # Scroll indicators
    if len(rows) > max_visible:
        if scroll_top > 0:
            _put(frame, "▲", (px2 - 20, sep_y + 16), scale=0.5, color=_GREY, thickness=1)
        if scroll_top + max_visible < len(rows):
            _put(frame, "▼", (px2 - 20, py2 - 12), scale=0.5, color=_GREY, thickness=1)
        _put(frame,
             f"j/k to scroll  ({scroll_top+1}–{min(scroll_top+max_visible, len(rows))} of {n})",
             (lx, py2 - 12), scale=0.40, color=_GREY, thickness=1)

    return rects
