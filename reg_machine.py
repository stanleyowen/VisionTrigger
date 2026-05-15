"""
reg_machine.py – Gesture registration state machine and file helpers for VisionTrigger.
"""

import logging
import shutil
import subprocess
import threading
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).parent / "scripts"

# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class RegState(Enum):
    IDLE          = "idle"
    CAPTURE       = "capture"
    NAME          = "name"
    ACTION_TYPE   = "action_type"
    ACTION_DETAIL = "action_detail"
    FILE_PICK     = "file_pick"
    CONFIRM       = "confirm"
    DELETE_CONFIRM = "delete_confirm"


# ---------------------------------------------------------------------------
# Script file helpers
# ---------------------------------------------------------------------------

_SCRIPT_EXTS: dict[str, set[str]] = {
    "applescript": {".scpt", ".scptd", ".applescript"},
    "shell": {".sh", ".bash", ".zsh", ".command"},
}


def _is_script_entry(f: Path, valid_exts: set) -> bool:
    """True for regular script files and .scptd bundle directories."""
    return f.suffix.lower() in valid_exts and (f.is_file() or f.suffix.lower() == ".scptd")


def _get_script_files(action_type: str) -> list:
    """Return sorted Path list for script files in SCRIPTS_DIR matching action_type."""
    SCRIPTS_DIR.mkdir(exist_ok=True)
    exts = _SCRIPT_EXTS.get(action_type, set())
    return sorted(
        f for f in SCRIPTS_DIR.iterdir()
        if _is_script_entry(f, exts)
    )


def _browse_and_copy(action_type: str) -> list:
    """
    Open a native macOS file-chooser via osascript; copy chosen scripts
    into SCRIPTS_DIR; return the list of copied Path objects.
    No tkinter or extra dependencies required.
    """
    if action_type == "applescript":
        ext_filter = '{"scpt", "scptd", "applescript"}'
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
        if not _is_script_entry(p, valid):
            continue
        dest = SCRIPTS_DIR / p.name
        try:
            if p.suffix.lower() == ".scptd":
                shutil.copytree(p, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(p, dest)
            copied.append(dest)
            logger.info("Copied '%s' to scripts/", p.name)
        except Exception as exc:
            logger.error("Cannot copy %s: %s", p.name, exc)
    return copied


def _open_scripts_folder() -> None:
    """Reveal the scripts/ directory in macOS Finder."""
    subprocess.Popen(["open", str(SCRIPTS_DIR)])


# ---------------------------------------------------------------------------
# Folder watcher
# ---------------------------------------------------------------------------

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
                if _is_script_entry(f, valid)
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
                    if _is_script_entry(f, valid)
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
# Registration state machine
# ---------------------------------------------------------------------------

class RegistrationStateMachine:
    STABLE_REQUIRED = 40  # consecutive frames needed to capture a gesture

    def __init__(self):
        self.state = RegState.IDLE
        self.fingers: tuple | None = None
        self.stable_count: int = 0
        self.prev_fingers: tuple | None = None
        self.current_fingers: tuple | None = None
        self.name: str = ""
        self.action_type: str = ""
        self.action_detail: str = ""
        self.input_buf: str = ""
        self.file_list: list = []
        self.file_cursor: int = 0
        self.selected_filename: str = ""
        self.is_edit: bool = False
        self.edit_is_custom: bool = False
        self._folder_watcher: "_FolderWatcher | None" = None

    def reset(self):
        """Reset all state back to IDLE."""
        if self._folder_watcher:
            self._folder_watcher.close()
            self._folder_watcher = None
        self.state = RegState.IDLE
        self.fingers = None
        self.stable_count = 0
        self.prev_fingers = None
        self.current_fingers = None
        self.name = ""
        self.action_type = ""
        self.action_detail = ""
        self.input_buf = ""
        self.file_list = []
        self.file_cursor = 0
        self.selected_filename = ""
        self.is_edit = False
        self.edit_is_custom = False

    def start_capture(self):
        """Begin new gesture capture (from IDLE)."""
        self.reset()
        self.state = RegState.CAPTURE

    def start_edit(self, name: str, cfg: dict):
        """Begin editing an existing gesture (jumps to ACTION_TYPE)."""
        self.reset()
        self.name = name
        self.fingers = tuple(cfg["fingers"]) if cfg.get("fingers") else None
        self.is_edit = True
        self.edit_is_custom = bool(cfg.get("fingers"))
        self.state = RegState.ACTION_TYPE

    def on_fingers(self, fingers: tuple | None) -> bool:
        """
        Feed current finger states during CAPTURE phase.
        Must be called every frame when state == CAPTURE.
        Returns True if a stable gesture was just captured (transitions to NAME).
        """
        if self.state != RegState.CAPTURE:
            return False
        self.current_fingers = fingers
        if fingers is None:
            self.stable_count = 0
            self.prev_fingers = None
            return False
        if fingers == self.prev_fingers:
            self.stable_count += 1
        else:
            self.stable_count = 0
            self.prev_fingers = fingers
        if self.stable_count >= self.STABLE_REQUIRED:
            self.fingers = fingers
            self.stable_count = 0
            self.prev_fingers = None
            self.input_buf = ""
            self.state = RegState.NAME
            return True
        return False

    def go_back(self):
        """Navigate to the previous step. No-op if already at IDLE."""
        confirm_prev = (
            RegState.FILE_PICK if self.action_type in ("shell", "applescript")
            else RegState.ACTION_DETAIL
        )
        _prev = {
            RegState.NAME:          RegState.CAPTURE,
            RegState.ACTION_TYPE:   RegState.NAME if not self.is_edit else RegState.IDLE,
            RegState.ACTION_DETAIL: RegState.ACTION_TYPE,
            RegState.FILE_PICK:     RegState.ACTION_TYPE,
            RegState.CONFIRM:       confirm_prev,
        }
        prev = _prev.get(self.state, RegState.IDLE)
        if prev == RegState.IDLE:
            self.reset()
        else:
            self.state = prev
            self.input_buf = ""

    def tick_file_watcher(self) -> list:
        """
        Called each frame when state == FILE_PICK.
        Starts the watcher if needed; returns any newly detected files.
        """
        if self.state != RegState.FILE_PICK:
            if self._folder_watcher:
                self._folder_watcher.close()
                self._folder_watcher = None
            return []
        if self._folder_watcher is None:
            self._folder_watcher = _FolderWatcher(self.action_type)
        return self._folder_watcher.poll()
