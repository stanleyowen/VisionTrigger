"""
voice.py – Voice command listener for VisionTrigger.

Runs a background thread that:
  1. Continuously captures audio from the default microphone.
  2. Detects utterances via a simple RMS-based VAD.
  3. Transcribes each utterance locally with faster-whisper.
  4. If the transcript contains the configured wake word, looks for a
     registered command phrase after it and fires `on_command(name, cfg)`.

faster-whisper and sounddevice are imported lazily inside the thread so the
rest of the app still runs even if the voice deps are not installed.

Expected config (config.yaml):

  voice_commands:
    OPEN_TERMINAL:
      phrases:
        - open terminal
        - launch terminal
      action: shell
      command: open -a Terminal
      label: Open Terminal
      cooldown: 2.0

  settings:
    voice:
      enabled: false
      wake_word: "hey"
      language: "en"
      model: "base.en"
      silence_threshold: 0.01
"""

import logging
import queue
import re
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Audio capture parameters
_SAMPLE_RATE = 16000
_BLOCK_DURATION = 0.1                # seconds per audio callback
_MIN_SPEECH_DURATION = 0.3           # ignore utterances shorter than this
_MAX_UTTERANCE_DURATION = 8.0        # force-flush an utterance after this
_SILENCE_AFTER_SPEECH = 0.7          # silence needed to end an utterance

# Phrase template syntax: {name} is a captured parameter that gets substituted
# back into the command/script/name action field.
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_TENS = {20, 30, 40, 50, 60, 70, 80, 90}


def _normalize_numbers(text: str) -> str:
    """Convert spelled-out English numbers (0–99) to digits in place."""
    parts = re.split(r"(\s+)", text)
    out: list = []
    i = 0
    while i < len(parts):
        word = parts[i].lower().strip("'\"-.,")
        if word in _NUM_WORDS:
            num = _NUM_WORDS[word]
            # twenty + five  ->  25
            if num in _TENS and i + 2 < len(parts):
                next_word = parts[i + 2].lower().strip("'\"-.,")
                if next_word in _NUM_WORDS and 1 <= _NUM_WORDS[next_word] < 10:
                    num += _NUM_WORDS[next_word]
                    out.append(str(num))
                    i += 3
                    continue
            out.append(str(num))
        else:
            out.append(parts[i])
        i += 1
    return "".join(out)


def _normalize_text(text: str) -> str:
    """Lowercase + clean up transcribed text so phrase regexes can match."""
    text = text.lower()
    # Collapse "7 a.m." / "7 p.m." (with any punctuation/spacing) before we
    # strip periods.
    text = re.sub(r"\b(\d+)\s*a\.?\s*m\.?\b", r"\1am", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)\s*p\.?\s*m\.?\b", r"\1pm", text, flags=re.IGNORECASE)
    # Drop sentence-ish punctuation but keep colons (times), apostrophes,
    # and hyphens.
    text = re.sub(r"[,!?;.]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _normalize_numbers(text)
    # Re-run once now that word-numbers became digits ("seven am" → "7am").
    text = re.sub(r"\b(\d+)\s+a\s*m\b", r"\1am", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)\s+p\s*m\b", r"\1pm", text, flags=re.IGNORECASE)
    return text


def _phrase_to_pattern(phrase: str) -> tuple:
    """
    Compile a phrase template like 'set a timer for {minutes} minutes' into
    a regex with named capture groups. Returns (pattern, literal_score)
    where literal_score is the number of literal alphanumerics in the phrase
    (used to prioritize more-specific matches).
    """
    phrase = phrase.strip().lower()
    placeholders = list(_PLACEHOLDER_RE.finditer(phrase))
    parts: list = []
    literal_chars = 0
    last = 0
    for i, m in enumerate(placeholders):
        literal = phrase[last:m.start()]
        parts.append(re.escape(literal))
        literal_chars += sum(1 for c in literal if c.isalnum())
        # Greedy capture only when the placeholder is at the very end of the
        # phrase – otherwise we need non-greedy so the trailing literal anchors.
        is_terminal = (i == len(placeholders) - 1 and m.end() == len(phrase))
        quant = ".+" if is_terminal else ".+?"
        parts.append(f"(?P<{m.group(1)}>{quant})")
        last = m.end()
    trailing = phrase[last:]
    parts.append(re.escape(trailing))
    literal_chars += sum(1 for c in trailing if c.isalnum())
    pattern = r"\b" + "".join(parts)
    # Anchor end with a word boundary when no terminal placeholder; otherwise
    # `.+` already grabs to end-of-string.
    if not (placeholders and placeholders[-1].end() == len(phrase)):
        pattern += r"\b"
    return re.compile(pattern, re.IGNORECASE), literal_chars


def _substitute_params(template: str, params: dict) -> str:
    """Replace every {name} in template with params['name']. Leave unmatched."""
    def repl(m):
        return params.get(m.group(1), m.group(0))
    return _PLACEHOLDER_RE.sub(repl, template)


class VoiceListener:
    """Background mic listener that fires a callback on matched voice commands."""

    # Status strings exposed via status()
    STATUS_OFF = "off"
    STATUS_LOADING = "loading"
    STATUS_LISTENING = "listening"
    STATUS_TRANSCRIBING = "transcribing"
    STATUS_HEARD = "heard"
    STATUS_EXECUTING = "executing"
    STATUS_ERROR = "error"

    def __init__(
        self,
        wake_word: str,
        commands: dict,
        on_command: Callable[[str, dict], None],
        model_size: str = "base.en",
        language: str = "en",
        silence_threshold: float = 0.01,
    ):
        self.wake_word = _normalize_text(wake_word or "").strip()
        self.commands = commands or {}
        self.on_command = on_command
        self.language = language or "en"
        self._model_size = model_size or "base.en"
        self._silence_threshold = float(silence_threshold)
        self._model = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = self.STATUS_OFF
        self._status_text = ""
        self._status_lock = threading.Lock()
        self._last_trigger_ts: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> tuple:
        """Return (status_str, status_text)."""
        with self._status_lock:
            return self._status, self._status_text

    def update_commands(self, commands: dict) -> None:
        """Hot-swap the registered command set."""
        self.commands = commands or {}

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="VoiceListener",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._set_status(self.STATUS_OFF, "")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set_status(self, status: str, text: str = "") -> None:
        with self._status_lock:
            self._status = status
            self._status_text = text

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            self._set_status(self.STATUS_LOADING, f"loading {self._model_size}…")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self._model_size, device="cpu", compute_type="int8",
            )
            logger.info("Whisper model '%s' loaded.", self._model_size)
            return True
        except ImportError:
            logger.error(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper sounddevice"
            )
            self._set_status(self.STATUS_ERROR, "faster-whisper not installed")
            return False
        except Exception as exc:
            logger.error("Could not load Whisper model: %s", exc)
            self._set_status(self.STATUS_ERROR, f"model load: {exc}")
            return False

    def _run(self) -> None:
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error(
                "sounddevice/numpy missing. Run: pip install sounddevice"
            )
            self._set_status(self.STATUS_ERROR, "sounddevice not installed")
            return

        if not self._ensure_model():
            return

        audio_q: queue.Queue = queue.Queue()

        def _audio_cb(indata, frames, time_info, status):  # noqa: ARG001
            if status:
                logger.debug("audio status: %s", status)
            audio_q.put(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=int(_SAMPLE_RATE * _BLOCK_DURATION),
                callback=_audio_cb,
            )
        except Exception as exc:
            logger.error("Cannot open microphone: %s", exc)
            self._set_status(self.STATUS_ERROR, f"mic: {exc}")
            return

        with stream:
            self._listen_loop(audio_q, np)

    def _listen_loop(self, audio_q: queue.Queue, np) -> None:
        buffer: list = []
        in_speech = False
        speech_start = 0.0
        last_voice_time = 0.0
        self._set_status(self.STATUS_LISTENING, "")

        while not self._stop.is_set():
            try:
                chunk = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            chunk = chunk.flatten()
            rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            now = time.monotonic()
            voiced = rms > self._silence_threshold

            if voiced:
                if not in_speech:
                    in_speech = True
                    speech_start = now
                    buffer = []
                last_voice_time = now
                buffer.append(chunk)
            elif in_speech:
                buffer.append(chunk)
                if now - last_voice_time >= _SILENCE_AFTER_SPEECH:
                    duration = last_voice_time - speech_start
                    if duration >= _MIN_SPEECH_DURATION:
                        try:
                            audio_np = np.concatenate(buffer)
                            self._process_utterance(audio_np)
                        except Exception as exc:
                            logger.error("Utterance processing failed: %s", exc)
                    in_speech = False
                    buffer = []
                    self._set_status(self.STATUS_LISTENING, "")

            if in_speech and (now - speech_start) > _MAX_UTTERANCE_DURATION:
                try:
                    audio_np = np.concatenate(buffer)
                    self._process_utterance(audio_np)
                except Exception as exc:
                    logger.error("Utterance processing failed: %s", exc)
                in_speech = False
                buffer = []
                self._set_status(self.STATUS_LISTENING, "")

    def _process_utterance(self, audio_np) -> None:
        self._set_status(self.STATUS_TRANSCRIBING, "…")
        try:
            segments, _info = self._model.transcribe(
                audio_np,
                language=self.language,
                beam_size=1,
                vad_filter=False,
            )
            raw_text = " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.error("Transcribe error: %s", exc)
            self._set_status(self.STATUS_LISTENING, "")
            return

        if not raw_text:
            self._set_status(self.STATUS_LISTENING, "")
            return

        text = _normalize_text(raw_text)
        logger.info("Voice heard: %s", text)
        self._set_status(self.STATUS_HEARD, text)

        # Wake-word filter (skipped when no wake word is configured)
        cmd_text = text
        if self.wake_word:
            idx = text.find(self.wake_word)
            if idx < 0:
                return
            cmd_text = text[idx + len(self.wake_word):]
        cmd_text = cmd_text.strip(" \t")

        if not cmd_text:
            return

        match = self._match_command(cmd_text)
        if not match:
            return

        name, cfg, params = match
        cooldown = float(cfg.get("cooldown", 2.0) or 0.0)
        now_real = time.time()
        if now_real - self._last_trigger_ts.get(name, 0.0) < cooldown:
            return
        self._last_trigger_ts[name] = now_real

        # Substitute captured {placeholders} into the action fields.
        if params:
            cfg = dict(cfg)
            for key in ("command", "script", "name"):
                value = cfg.get(key)
                if isinstance(value, str):
                    cfg[key] = _substitute_params(value, params)

        self._set_status(self.STATUS_EXECUTING, name)
        logger.info("Voice command matched: %s  params=%s", name, params)
        try:
            self.on_command(name, cfg)
        except Exception as exc:
            logger.error("on_command callback failed: %s", exc)

    def _match_command(self, cmd_text: str) -> Optional[tuple]:
        """Return (name, cfg, params) for the most specific matching phrase."""
        best: Optional[tuple] = None    # (score, name, cfg, params)
        for name, cfg in self.commands.items():
            phrases = cfg.get("phrases") or []
            if isinstance(phrases, str):
                phrases = [phrases]
            for phrase in phrases:
                if not phrase:
                    continue
                try:
                    pattern, score = _phrase_to_pattern(phrase)
                except re.error:
                    continue
                m = pattern.search(cmd_text)
                if not m:
                    continue
                params = {k: (v or "").strip() for k, v in m.groupdict().items()}
                if best is None or score > best[0]:
                    best = (score, name, cfg, params)
        if best:
            return best[1], best[2], best[3]
        return None
