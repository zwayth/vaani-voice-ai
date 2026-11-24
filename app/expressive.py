"""Expressive layer.

The LLM writes inline tags: [breath] [sigh] [laugh] [pause] [whisper]..[/whisper] [hmm].
Each TTS provider declares which tags it renders natively (e.g. ElevenLabs v3 does [sighs]/[whispers]).
Everything else is rendered here so EVERY provider - even the free fallback - can sigh, breathe and whisper:
  * breath / sigh / mouth-click: procedurally synthesised from spectrally shaped noise
  * whisper: phase-vocoder "devoicing" of the synthesised speech (random phase, high-tilt spectrum)
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

TAG_RE = re.compile(r"\[(/?)(breath|inhale|sigh|exhale|laugh|laughs|chuckle|pause|whisper|whispers|hmm|smile)\]", re.I)
ANY_TAG_RE = re.compile(r"\[[^\]]{1,24}\]")

CANON = {"inhale": "breath", "exhale": "sigh", "laughs": "laugh", "chuckle": "laugh", "whispers": "whisper"}


@dataclass
class Seg:
    kind: str            # "speech" | "sfx" | "pause"
    text: str = ""
    whisper: bool = False
    name: str = ""
    ms: int = 0


def parse(text: str) -> list[Seg]:
    segs: list[Seg] = []
    whisper = False
    pos = 0
    for m in TAG_RE.finditer(text):
        chunk = text[pos:m.start()]
        if chunk.strip():
            segs.append(Seg("speech", chunk.strip(), whisper))
        closing, name = m.group(1) == "/", CANON.get(m.group(2).lower(), m.group(2).lower())
        if name == "whisper":
            whisper = not closing
        elif name == "pause":
            segs.append(Seg("pause", ms=380))
        elif name == "smile":
            pass
        elif not closing:
            segs.append(Seg("sfx", name=name))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        segs.append(Seg("speech", tail.strip(), whisper))
    # strip any unknown bracket tags the LLM invented so they are never read aloud
    for s in segs:
        if s.kind == "speech":
            s.text = ANY_TAG_RE.sub("", s.text).strip()
    return [s for s in segs if s.kind != "speech" or s.text]


def strip_tags(text: str) -> str:
    return re.sub(r"\s{2,}", " ", ANY_TAG_RE.sub("", text)).strip()


# ---------------- procedural non-verbal sounds ----------------

def _shaped_noise(n: int, rate: int, shape, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n).astype(np.float32))
    f = np.fft.rfftfreq(n, 1.0 / rate)
    spec *= shape(f)
    y = np.fft.irfft(spec, n).astype(np.float32)
    return y / (np.max(np.abs(y)) + 1e-9)


def _env(n: int, attack: float, release_pow: float = 1.6) -> np.ndarray:
    t = np.linspace(0, 1, n, dtype=np.float32)
    a = np.clip(t / attack, 0, 1) ** 1.5
    r = np.clip((1 - t) / (1 - attack), 0, 1) ** release_pow
    return a * r


def _formant(f, centers, width):
    g = np.zeros_like(f)
    for c, amp in centers:
        g += amp * np.exp(-0.5 * ((f - c) / width) ** 2)
    return g


def breath(rate: int, rng) -> np.ndarray:
    n = int(rate * rng.uniform(0.32, 0.45))
    shape = lambda f: _formant(f, [(900, 0.6), (1700, 1.0), (3200, 0.5)], 520) * (f > 180)
    return _shaped_noise(n, rate, shape, rng) * _env(n, 0.55, 2.2) * 0.10


def sigh(rate: int, rng) -> np.ndarray:
    n = int(rate * rng.uniform(0.70, 0.95))
    t = np.arange(n) / rate
    # exhale noise whose spectral centre slides down, plus a faint low voiced "hhaa"
    shape = lambda f: _formant(f, [(700, 1.0), (1200, 0.8), (2500, 0.35)], 420) * (f > 120)
    noise = _shaped_noise(n, rate, shape, rng)
    f0 = 190 * np.exp(-t * 0.9)
    voiced = np.sin(2 * np.pi * np.cumsum(f0) / rate) * np.exp(-t * 3.5) * 0.18
    y = (noise * 0.9 + voiced) * _env(n, 0.18, 1.4)
    inhale = breath(rate, rng) * 0.6
    return np.concatenate([inhale, np.zeros(int(rate * 0.04), np.float32), y * 0.13]).astype(np.float32)


def mouth_click(rate: int, rng) -> np.ndarray:
    n = int(rate * 0.018)
    shape = lambda f: _formant(f, [(2600, 1.0), (4200, 0.6)], 900)
    return _shaped_noise(n, rate, shape, rng) * np.exp(-np.linspace(0, 9, n)).astype(np.float32) * 0.05


def laugh(rate: int, rng) -> np.ndarray:
    """Soft closed-mouth chuckle: 3 breathy voiced pulses with falling pitch."""
    parts = []
    f0 = rng.uniform(210, 250)
    for i in range(3):
        n = int(rate * 0.11)
        t = np.arange(n) / rate
        ph = 2 * np.pi * np.cumsum(np.full(n, f0 * (1 - 0.06 * i))) / rate
        voiced = sum(np.sin(k * ph) / k for k in range(1, 6))
        nz = _shaped_noise(n, rate, lambda f: _formant(f, [(800, 1), (1500, .7)], 500), rng)
        e = np.sin(np.pi * t / t[-1]) ** 2
        parts.append(((0.45 * voiced + 0.55 * nz) * e * (0.09 - 0.02 * i)).astype(np.float32))
        parts.append(np.zeros(int(rate * 0.07), np.float32))
    return np.concatenate(parts)


def hmm(rate: int, rng) -> np.ndarray:
    n = int(rate * 0.42)
    t = np.arange(n) / rate
    f0 = 175 + 25 * np.sin(np.pi * t / t[-1])
    ph = 2 * np.pi * np.cumsum(f0) / rate
    y = sum(np.sin(k * ph) * np.exp(-0.5 * ((k * 175 - 250) / 300) ** 2) for k in range(1, 8))
    return (y * _env(n, 0.2, 1.2) * 0.12).astype(np.float32)


GEN = {"breath": breath, "sigh": sigh, "laugh": laugh, "hmm": hmm, "click": mouth_click}


class SfxBank:
    """Pre-renders several variants of each sound so they never sound identical twice."""

    def __init__(self, rate: int, variants: int = 5):
        self.rate = rate
        self.bank = {k: [fn(rate, np.random.default_rng(i * 97 + 7)) for i in range(variants)] for k, fn in GEN.items()}

    def get(self, name: str) -> np.ndarray:
        v = self.bank.get(name)
        return random.choice(v) if v else np.zeros(0, np.float32)


# ---------------- whisper conversion ----------------

def whisperize(x: np.ndarray, rate: int, gain: float = 0.75) -> np.ndarray:
    """Remove voicing: keep the spectral envelope, randomise phase, tilt energy upward."""
    if len(x) < 1024:
        return x
    n_fft, hop = 512, 128
    win = np.hanning(n_fft).astype(np.float32)
    pad = np.concatenate([np.zeros(n_fft, np.float32), x, np.zeros(n_fft, np.float32)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop] * win
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec)
    # smooth magnitude across frequency to wash out harmonics (=> no pitch)
    k = np.ones(9, np.float32) / 9
    mag = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, mag)
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    tilt = np.clip((f / 1200.0) ** 0.6, 0.15, 1.6) * (f > 250)
    rng = np.random.default_rng(3)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=mag.shape))
    out_frames = np.fft.irfft(mag * tilt * phase, n_fft, axis=1).astype(np.float32) * win
    y = np.zeros(len(pad) + n_fft, np.float32)
    norm = np.zeros_like(y)
    for i, fr in enumerate(out_frames):
        y[i * hop:i * hop + n_fft] += fr
        norm[i * hop:i * hop + n_fft] += win ** 2
    y = (y / np.maximum(norm, 1e-3))[n_fft:n_fft + len(x)]
    peak_in, peak_out = np.max(np.abs(x)) + 1e-9, np.max(np.abs(y)) + 1e-9
    return (y * (peak_in / peak_out) * gain * 0.6).astype(np.float32)
