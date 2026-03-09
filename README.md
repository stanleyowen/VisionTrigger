# VisionTrigger

Recognize hand gestures with your MacBook camera and automatically trigger macOS actions — Shortcuts, AppleScript, or shell commands.

---

## How it works

1. Your camera feed is processed by [MediaPipe Hands](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) in real time.
2. Finger states are classified into a recognized gesture.
3. Hold a gesture steady for a configurable number of frames (default: 20 ≈ 0.7 s at 30 fps) to trigger its mapped action.
4. A visual flash and log message confirm the trigger.

---

## Gesture reference

| Gesture       | Hand shape                    | Default action                  |
| ------------- | ----------------------------- | ------------------------------- |
| `THUMBS_UP`   | 👍 thumb up, fingers curled   | Volume Up                       |
| `THUMBS_DOWN` | 👎 thumb down, fingers curled | Volume Down                     |
| `OPEN_PALM`   | 🖐 all five fingers open      | Take Screenshot                 |
| `PEACE`       | ✌️ index + middle extended    | Open Spotlight                  |
| `FIST`        | 👊 all fingers curled         | Lock Screen                     |
| `POINT`       | ☝️ index finger only          | Open Finder                     |
| `ROCK`        | 🤘 index + pinky (horns)      | Open Terminal                   |
| `CALL`        | 🤙 thumb + pinky              | Run a Shortcut                  |
| `THREE`       | three fingers up              | _(unmapped — edit config.yaml)_ |
| `FOUR`        | four fingers up               | _(unmapped — edit config.yaml)_ |

---

## Setup

### Prerequisites

- macOS 12 Monterey or later (for `shortcuts run` CLI support)
- Python 3.9+
- Homebrew is **not** required

### 1 – Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2 – Install dependencies

```bash
pip install -r requirements.txt
```

### 3 – Grant permissions

macOS will prompt for these on first run — approve both:

| Permission        | Required for                                      |
| ----------------- | ------------------------------------------------- |
| **Camera**        | Seeing your hand                                  |
| **Accessibility** | AppleScript keyboard simulation (Spotlight, etc.) |

To grant Accessibility access manually:  
**System Settings → Privacy & Security → Accessibility** → add your Terminal app (or VS Code).

### 4 – Run

```bash
python main.py
```

Press **q** or **Esc** to quit. Press **l** to toggle the hand landmark overlay.

---

## Configuration

Edit `config.yaml` to change what each gesture does.

### Trigger a macOS Shortcut

```yaml
PEACE:
  action: shortcut
  name: "Focus Mode" # exact name of your Shortcut
  label: "Focus Mode"
  cooldown: 2.0
```

Create the Shortcut in the **Shortcuts** app, then use its exact name here.

### Run an AppleScript snippet

```yaml
THUMBS_UP:
  action: applescript
  script: "set volume output volume (output volume of (get volume settings) + 10)"
  label: "Volume Up"
  cooldown: 0.8
```

### Run a shell command

```yaml
ROCK:
  action: shell
  command: "open -a Terminal"
  label: "Open Terminal"
  cooldown: 3.0
```

### Settings

| Key                    | Default | Description                                              |
| ---------------------- | ------- | -------------------------------------------------------- |
| `camera_index`         | `0`     | Camera to use — `0` = built-in FaceTime, `1`+ = external |
| `detection_confidence` | `0.7`   | MediaPipe detection threshold (0–1)                      |
| `tracking_confidence`  | `0.5`   | MediaPipe tracking threshold (0–1)                       |
| `gesture_hold_frames`  | `20`    | Consecutive frames a gesture must be stable to fire      |
| `show_landmarks`       | `true`  | Draw hand skeleton on the video feed                     |
| `show_fps`             | `true`  | Show FPS counter                                         |

Increasing `gesture_hold_frames` reduces accidental triggers; decreasing it makes the app more responsive.

---

## Project structure

```
VisionTrigger/
├── main.py          # Camera loop + UI overlay
├── gestures.py      # MediaPipe hand processing + gesture classification
├── mac_trigger.py   # macOS Shortcut / AppleScript / shell execution
├── config.yaml      # Gesture → action mapping (edit this)
└── requirements.txt
```

---

## Troubleshooting

| Problem                         | Fix                                            |
| ------------------------------- | ---------------------------------------------- |
| Camera not found                | Set `camera_index: 1` in `config.yaml`         |
| Spotlight shortcut doesn't open | Grant **Accessibility** permission to Terminal |
| `shortcuts` command not found   | Upgrade to macOS 12 Monterey or later          |
| Gesture fires too often         | Increase `cooldown` or `gesture_hold_frames`   |
| Gesture not detected            | Ensure good lighting; keep hand fully visible  |
