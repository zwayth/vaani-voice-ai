"""Voice activity detection + endpointing.

WebRTC VAD (fast, robust) gated by an adaptive energy floor so steady background noise (fans, traffic)
doesn't register as speech. 20 ms frames at 16 kHz."""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import webrtcvad

FRAME_MS = 20
FRAME = 320  # samples @16k


class VAD:
    def __init__(self, endpoint_ms: int = 550, aggressiveness: int = 2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.endpoint_frames = max(5, endpoint_ms // FRAME_MS)
        self.noise = 0.003
        self.recent = deque(maxlen=6)
        self.preroll = deque(maxlen=15)  # 300 ms before speech onset
        self.speaking = False
        self.silence = 0
        self.voiced = 0
        self.buf: list[np.ndarray] = []
        self._rem = np.zeros(0, np.float32)
        self.last_voice_t = 0.0  # perf_counter() of the most recent voiced frame = "caller stopped talking"

    def set_endpoint(self, ms: int):
        self.endpoint_frames = max(5, ms // FRAME_MS)

    def _is_speech(self, fr: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(fr * fr)) + 1e-9)
        pcm = (np.clip(fr, -1, 1) * 32767).astype("<i2").tobytes()
        v = self.vad.is_speech(pcm, 16000)
        loud = rms > max(0.006, self.noise * 2.8)
        if not v or not loud:  # track noise floor only on non-speech frames
            self.noise = 0.97 * self.noise + 0.03 * min(rms, 0.05)
        return v and loud

    def feed(self, x: np.ndarray):
        """Yield events: ('start',), ('speech_ms', ms), ('end', audio)."""
        x = np.concatenate([self._rem, x])
        n = len(x) // FRAME
        self._rem = x[n * FRAME:]
        for i in range(n):
            fr = x[i * FRAME:(i + 1) * FRAME]
            sp = self._is_speech(fr)
            if sp:
                # frames are processed as they arrive; back-date by the frames still queued in this chunk
                self.last_voice_t = time.perf_counter() - (n - 1 - i) * FRAME_MS / 1000
            self.recent.append(sp)
            if not self.speaking:
                self.preroll.append(fr)
                if sum(self.recent) >= 4:
                    self.speaking, self.silence, self.voiced = True, 0, sum(self.recent)
                    self.buf = list(self.preroll)
                    yield ("start",)
            else:
                self.buf.append(fr)
                if sp:
                    self.voiced += 1
                    self.silence = 0
                    if self.voiced % 5 == 0:
                        yield ("speech_ms", self.voiced * FRAME_MS)
                else:
                    self.silence += 1
                    if self.silence >= self.endpoint_frames:
                        audio = np.concatenate(self.buf)
                        voiced_ms = self.voiced * FRAME_MS
                        self.speaking, self.buf = False, []
                        self.preroll.clear()
                        self.recent.clear()
                        yield ("end", audio, voiced_ms)
