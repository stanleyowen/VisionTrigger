# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The hand landmarker model (`hand_landmarker.task`, ~7 MB) is auto-downloaded on first run if missing.

**Runtime keyboard controls:** `q`/`Esc` quit, `l` toggle landmarks, `g` toggle gesture list, `r` register gesture, `v` toggle voice, `c` open camera selector. Inside lists, `j`/`k` navigate rows.

## Architecture

The app is a single-threaded OpenCV display loop (`main.py:main()`) with two daemon threads: one for action execution and one for voice listening.

**Data flow:**
1. `main.py` reads the camera frame, calls `GestureRecognizer.process()` → `classify()` each frame.
2. When a gesture holds steady for `gesture_hold_frames` consecutive frames, `MacTrigger.execute()` is dispatched in a daemon thread.
3. `VoiceListener` runs its own background thread: mic capture → RMS VAD → faster-whisper transcription → wake-word filter → phrase match → `on_command` callback (which also calls `MacTrigger.execute()` in a thread).

**Module responsibilities:**
- `gestures.py` — `GestureRecognizer` wraps MediaPipe's `HandLandmarker` in VIDEO mode. Finger state is a `(thumb, index, middle, ring, pinky)` bool tuple. Custom gestures from `config.yaml` are overlaid onto `_BUILTIN_GESTURE_MAP` at construction time. THUMBS_UP/DOWN share the same finger pattern and are disambiguated by thumb-tip y-coordinate.
- `mac_trigger.py` — `MacTrigger.execute()` dispatches to `shortcuts run`, `osascript -e`, or `subprocess(shell=True)` based on the `action` key in the config dict.
- `voice.py` — `VoiceListener` does RMS-based VAD, transcribes with faster-whisper, matches phrases using compiled regexes with `{placeholder}` capture groups, and substitutes captured values into the action's `command`/`script`/`name` fields before firing.
- `main.py` — Everything else: config I/O, camera detection/switching, all OpenCV drawing (overlay, gesture list panel, registration wizard, file picker, camera selector), mouse hit-testing, and the gesture registration state machine.

## Configuration (`config.yaml`)

Three top-level keys:
- `gestures` — built-in gesture → action mappings.
- `custom_gestures` — user-defined gestures; each requires a `fingers` list `[thumb, index, middle, ring, pinky]` plus an action.
- `voice_commands` — voice command → action mappings; each has a `phrases` list that may contain `{placeholder}` tokens which get substituted into action fields.
- `settings` — camera index, confidence thresholds, `gesture_hold_frames`, display toggles, and a `voice` sub-key.

Action types are `shortcut` (macOS Shortcuts app), `applescript` (inline script string), and `shell` (shell command string). All three accept `label` and `cooldown`.

## Gesture registration state machine

`REG_IDLE → REG_CAPTURE → REG_NAME → REG_ACTION_TYPE → REG_FILE_PICK | REG_ACTION_DETAIL → REG_CONFIRM → REG_IDLE`

All state variables are `reg_*` locals in `main()`. `_save_gesture()` writes to `config.yaml` and hot-patches `recognizer._gesture_map` so the new gesture works immediately without restart.

## Key implementation details

- The frame is horizontally flipped (`cv2.flip(frame, 1)`) so it acts as a mirror. MediaPipe's handedness labels are therefore swapped from the viewer's perspective; `gestures.py` handles this in `_finger_states()`.
- All action execution happens in daemon threads so it never blocks the display loop.
- `find_camera()` probes available AVFoundation devices and prefers names containing "iPhone"/"Continuity"/"iSight" for Continuity Camera support.
- Shell scripts for gesture registration are stored in `scripts/` and loaded into the action field as their full text content (not as a path reference).
- Voice phrase templates use `{name}` placeholders compiled to named regex capture groups; the most-specific match (by literal character count) wins when multiple phrases could match.
