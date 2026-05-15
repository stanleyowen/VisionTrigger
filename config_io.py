"""
config_io.py – Config loading and atomic/debounced saving for VisionTrigger.
"""

import logging
import os
import threading
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debounced save state
# ---------------------------------------------------------------------------

_save_timer: threading.Timer | None = None
_save_lock = threading.Lock()
_pending_save_args: tuple | None = None   # (path, config)


def _do_save(path: Path, config: dict) -> None:
    """Write config atomically: tmp file → os.replace."""
    tmp = path.with_suffix(".yaml.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False,
                      allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)
        logger.debug("Config saved to %s", path)
    except Exception as exc:
        logger.error("Failed to save config: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _timer_fired(path: Path, config: dict) -> None:
    global _save_timer, _pending_save_args
    with _save_lock:
        _save_timer = None
        _pending_save_args = None
    _do_save(path, config)


def save_config(path: Path, config: dict, delay: float = 0.5) -> None:
    """
    Schedule a debounced save of *config* to *path*.

    If another save is pending, cancel it and restart the timer with the
    latest config snapshot.  When delay=0 the save happens on the next
    timer tick (still background), which is effectively immediate.
    """
    global _save_timer, _pending_save_args

    # Snapshot the config dict so late mutations don't corrupt the pending write.
    import copy
    config_copy = copy.deepcopy(config)

    with _save_lock:
        if _save_timer is not None:
            _save_timer.cancel()
            _save_timer = None
        _pending_save_args = (path, config_copy)
        t = threading.Timer(delay, _timer_fired, args=(path, config_copy))
        t.daemon = True
        t.start()
        _save_timer = t


def flush_pending_save() -> None:
    """
    Fire any pending debounced save immediately (call before app exit).
    Blocks until the write completes.
    """
    global _save_timer, _pending_save_args
    with _save_lock:
        if _save_timer is not None:
            _save_timer.cancel()
            _save_timer = None
        args = _pending_save_args
        _pending_save_args = None
    if args is not None:
        _do_save(*args)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    """
    Load and lightly validate config.yaml.

    Bad entries (wrong action type, non-numeric cooldowns, wrong fingers
    length) are logged as warnings but skipped rather than crashing.
    Returns the raw dict (same structure as before).
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    _VALID_ACTIONS = {"shortcut", "applescript", "shell"}

    for section in ("gestures", "custom_gestures"):
        block = raw.get(section)
        if not isinstance(block, dict):
            continue
        bad_keys = []
        for name, cfg in block.items():
            if not isinstance(cfg, dict):
                logger.warning("Config: %s/%s is not a dict – skipping", section, name)
                bad_keys.append(name)
                continue
            action = cfg.get("action", "")
            if action not in _VALID_ACTIONS:
                logger.warning(
                    "Config: %s/%s has unknown action %r – skipping",
                    section, name, action,
                )
                bad_keys.append(name)
                continue
            cd = cfg.get("cooldown")
            if cd is not None:
                try:
                    float(cd)
                except (TypeError, ValueError):
                    logger.warning(
                        "Config: %s/%s cooldown %r is not numeric – ignoring field",
                        section, name, cd,
                    )
                    cfg.pop("cooldown", None)
            fingers = cfg.get("fingers")
            if fingers is not None:
                if not (isinstance(fingers, (list, tuple)) and len(fingers) == 5):
                    logger.warning(
                        "Config: %s/%s fingers must be a list of 5 booleans – skipping",
                        section, name,
                    )
                    bad_keys.append(name)
        for k in bad_keys:
            block.pop(k, None)

    return raw


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def update_gesture_in_config(
    config: dict,
    path: Path,
    name: str,
    entry: dict,
    section: str = "custom_gestures",
) -> None:
    """Update config[section][name] = entry and schedule a debounced save."""
    if not isinstance(config.get(section), dict):
        config[section] = {}
    config[section][name] = entry
    save_config(path, config, delay=0.5)


def delete_gesture_from_config(
    config: dict,
    path: Path,
    name: str,
) -> None:
    """Remove *name* from whichever section it lives in, then debounce-save."""
    for section in ("gestures", "custom_gestures"):
        d = config.get(section)
        if isinstance(d, dict) and name in d:
            del d[name]
            break
    save_config(path, config, delay=0.5)
