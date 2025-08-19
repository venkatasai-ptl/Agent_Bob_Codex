import webrtcvad
import audioop
import wave
import tempfile
import os
from typing import Callable

class BufferManager:
    """Collects fixed-size PCM frames and detects end of speech."""

    def __init__(
        self,
        session_id: str,
        callback: Callable[[str, str], None],
        *,
        rate: int = 16000,
        frame_duration: int = 30,
        vad_aggressiveness: int = 3,
        silence_duration: float = 0.5,
        use_vad: bool = True,
        rms_threshold: int = 500,
    ) -> None:
        self.session_id = session_id
        self.callback = callback
        self.rate = rate
        self.frame_duration = frame_duration
        self.frame_size = int(rate * frame_duration / 1000) * 2  # bytes per frame
        self.vad = webrtcvad.Vad(vad_aggressiveness) if use_vad else None
        self.rms_threshold = rms_threshold
        self.silence_frames = int(silence_duration * 1000 / frame_duration)
        self.frames: list[bytes] = []
        self.in_speech = False
        self.silence_count = 0

    def _is_speech(self, frame: bytes) -> bool:
        if self.vad:
            return self.vad.is_speech(frame, self.rate)
        return audioop.rms(frame, 2) > self.rms_threshold

    def add_frame(self, frame: bytes) -> None:
        """Add a 30ms frame and run end-of-speech detection."""
        if len(frame) != self.frame_size:
            return
        self.frames.append(frame)
        if self._is_speech(frame):
            self.in_speech = True
            self.silence_count = 0
        elif self.in_speech:
            self.silence_count += 1
            if self.silence_count >= self.silence_frames:
                self._finalize()

    def flush(self) -> None:
        """Force finalize current buffer."""
        if self.frames:
            self._finalize()

    def _finalize(self) -> None:
        pcm_data = b"".join(self.frames)
        self.frames.clear()
        self.in_speech = False
        self.silence_count = 0

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.rate)
                wf.writeframes(pcm_data)
            path = tmp.name
        try:
            self.callback(self.session_id, path)
        finally:
            try:
                os.remove(path)  # type: ignore[name-defined]
            except Exception:
                pass
