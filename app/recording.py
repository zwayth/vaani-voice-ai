"""Call recording on a wall clock.

The caller's audio is stamped when it arrives and Aria's audio when it is scheduled to play, so the saved
file matches what both sides heard, including barge-in (Aria's audio is cut at the moment she was interrupted).
"""
from __future__ import annotations

import time

import numpy as np

from .audio import resample, to_wav
from .config import IN_RATE, OUT_RATE, ROOT

REC_DIR = ROOT / "data" / "recordings"
REC_DIR.mkdir(parents=True, exist_ok=True)


class CallRecorder:
    def __init__(self):
        self.t0 = time.time()
        self.inp: list[tuple[float, np.ndarray]] = []
        self.out: list[tuple[float, np.ndarray]] = []
        self.clears: list[float] = []

    def caller(self, x16: np.ndarray):
        self.inp.append((time.time() - len(x16) / IN_RATE, x16.copy()))

    def agent(self, x24: np.ndarray, start_wall: float):
        self.out.append((start_wall, x24))

    def clear(self):
        self.clears.append(time.time())

    def render(self) -> np.ndarray:
        n = int((time.time() - self.t0 + 0.5) * OUT_RATE)
        mix = np.zeros(n, np.float32)

        def place(t: float, x: np.ndarray):
            i = int((t - self.t0) * OUT_RATE)
            if i < 0:
                x, i = x[-i:], 0
            j = min(n, i + len(x))
            if j > i:
                mix[i:j] += x[: j - i]

        # caller frames: lay them back to back, but re-anchor to their real arrival time whenever the
        # sender's clock has drifted more than 40 ms (keeps sync without clicks from per-frame jitter)
        run_start, run, expect = None, [], None
        for t, x in self.inp + [(None, None)]:
            if t is not None and expect is not None and abs(t - expect) <= 0.04:
                run.append(x)
                expect += len(x) / IN_RATE
                continue
            if run:
                place(run_start, resample(np.concatenate(run), IN_RATE, OUT_RATE))
            if t is None:
                break
            run_start, run, expect = t, [x], t + len(x) / IN_RATE
        for start, x in self.out:
            cut = next((c for c in self.clears if c > start), None)
            if cut is not None:
                x = x[: max(0, int((cut - start) * OUT_RATE))]
            place(start, x)
        return np.clip(mix, -1, 1)

    def save(self, call_id: str) -> str | None:
        try:
            name = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.t0))}-{call_id}.wav"
            (REC_DIR / name).write_bytes(to_wav(self.render(), OUT_RATE))
            return name
        except Exception:  # noqa: BLE001 - a recording failure must never affect the call
            return None
