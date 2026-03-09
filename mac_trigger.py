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

    def execute(self, action_cfg: dict) -> bool:
        """
        Execute one action.

        Args:
            action_cfg: A dict from config.yaml with at minimum an
                        'action' key ('shortcut', 'applescript', 'shell').

        Returns:
            True on success, False on failure.
        """
        kind = action_cfg.get("action", "")
        if kind == "shortcut":
            return self._run_shortcut(action_cfg.get("name", ""))
        if kind == "applescript":
            return self._run_applescript(action_cfg.get("script", ""))
        if kind == "shell":
            return self._run_shell(action_cfg.get("command", ""))

        logger.error("Unknown action type: %r", kind)
        return False

    # ------------------------------------------------------------------
    # Private dispatch methods
    # ------------------------------------------------------------------

    def _run_shortcut(self, name: str) -> bool:
        """Run a macOS Shortcut by name (requires macOS 12 Monterey+)."""
        if not name:
            logger.error("Shortcut name is empty")
            return False
        try:
            result = subprocess.run(
                ["shortcuts", "run", name],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
            if result.returncode != 0:
                logger.error("Shortcut '%s' failed: %s",
                             name, result.stderr.strip())
                return False
            logger.info("Shortcut '%s' ran successfully", name)
            return True
        except FileNotFoundError:
            logger.error(
                "'shortcuts' CLI not found. macOS 12 Monterey or later is required."
            )
            return False
        except subprocess.TimeoutExpired:
            logger.error("Shortcut '%s' timed out after %ds", name, _TIMEOUT)
            return False

    def _run_applescript(self, script: str) -> bool:
        """Execute an AppleScript snippet with osascript."""
        if not script:
            logger.error("AppleScript is empty")
            return False
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
            if result.returncode != 0:
                logger.error("AppleScript failed: %s", result.stderr.strip())
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("AppleScript timed out after %ds", _TIMEOUT)
            return False

    def _run_shell(self, command: str) -> bool:
        """
        Run a shell command.

        The command comes from the user-controlled config.yaml on the local
        machine, so shell=True is acceptable here.
        """
        if not command:
            logger.error("Shell command is empty")
            return False
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT,
            )
            if result.returncode != 0:
                logger.error("Shell command failed: %s", result.stderr.strip())
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("Shell command timed out after %ds", _TIMEOUT)
            return False
