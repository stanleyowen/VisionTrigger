"""
mac_trigger.py – Execute macOS actions triggered by gestures.

Supported action types (set in config.yaml):
  shortcut    – run a named macOS Shortcut  (requires macOS 12+)
  applescript – run an inline AppleScript snippet via osascript
  shell       – run a shell command
"""

import logging
import subprocess

logger = logging.getLogger(__name__)

# Shared timeout for all subprocesses (seconds)
_TIMEOUT = 15


class MacTrigger:
    """Dispatch a config action dict to the appropriate macOS mechanism."""

    def execute(self, action_cfg: dict) -> tuple[bool, str]:
        """
        Execute one action.

        Args:
            action_cfg: A dict from config.yaml with at minimum an
                        'action' key ('shortcut', 'applescript', 'shell').

        Returns:
            (success, error_message). Empty error_message means success.
        """
        kind = action_cfg.get("action", "")
        try:
            if kind == "shortcut":
                return self._run_shortcut(action_cfg.get("name", ""))
            if kind == "applescript":
                return self._run_applescript(action_cfg.get("script", ""))
            if kind == "shell":
                return self._run_shell(action_cfg.get("command", ""))
            msg = f"Unknown action type: {kind!r}"
            logger.error(msg)
            return False, msg
        except Exception as exc:
            logger.error("Unexpected error executing '%s': %s", kind, exc)
            return False, str(exc)

    # ------------------------------------------------------------------
    # Private dispatch methods
    # ------------------------------------------------------------------

    def _run_shortcut(self, name: str) -> tuple[bool, str]:
        """Run a macOS Shortcut by name (requires macOS 12 Monterey+)."""
        if not name:
            msg = "Shortcut name is empty"
            logger.error(msg)
            return False, msg
        try:
            result = subprocess.run(
                ["shortcuts", "run", name],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
            if result.returncode != 0:
                msg = result.stderr.strip()
                logger.error("Shortcut '%s' failed: %s", name, msg)
                return False, msg
            logger.info("Shortcut '%s' ran successfully", name)
            return True, ""
        except FileNotFoundError:
            msg = "'shortcuts' CLI not found. macOS 12 Monterey or later is required."
            logger.error(msg)
            return False, msg
        except subprocess.TimeoutExpired:
            msg = f"Shortcut '{name}' timed out after {_TIMEOUT}s"
            logger.error(msg)
            return False, msg

    def _run_applescript(self, script: str) -> tuple[bool, str]:
        """Execute an AppleScript snippet with osascript."""
        if not script:
            msg = "AppleScript is empty"
            logger.error(msg)
            return False, msg
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
            if result.returncode != 0:
                msg = result.stderr.strip()
                logger.error("AppleScript failed: %s", msg)
                return False, msg
            return True, ""
        except subprocess.TimeoutExpired:
            msg = f"AppleScript timed out after {_TIMEOUT}s"
            logger.error(msg)
            return False, msg

    def _run_shell(self, command: str) -> tuple[bool, str]:
        """
        Run a shell command.

        The command comes from the user-controlled config.yaml on the local
        machine, so shell=True is acceptable here.
        """
        if not command:
            msg = "Shell command is empty"
            logger.error(msg)
            return False, msg
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
            if result.returncode != 0:
                msg = result.stderr.strip()
                logger.error("Shell command failed: %s", msg)
                return False, msg
            return True, ""
        except subprocess.TimeoutExpired:
            msg = f"Shell command timed out after {_TIMEOUT}s"
            logger.error(msg)
            return False, msg
