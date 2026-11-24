"""Audio utilities: resampling, mu-law (telephony), WAV encode, compressed-audio decode via ffmpeg."""
from __future__ import annotations

import asyncio
import io
import shutil
import wave

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def pcm16_to_float(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(x) == 0:
        return x
    n = int(round(len(x) * dst / src))
    if dst < src:  # crude anti-alias: moving average before decimation
        k = max(1, int(round(src / dst)))
        if k > 1:
            x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
    t_src = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_dst = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(t_dst, t_src, x).astype(np.float32)


def to_wav(x: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(float_to_pcm16(x))
    return buf.getvalue()


# ---- G.711 mu-law (Twilio / Exotel media streams) ----
_BIAS, _CLIP = 0x84, 32635


def mulaw_decode(b: bytes) -> np.ndarray:
    u = ~np.frombuffer(b, dtype=np.uint8)
    sign = u & 0x80
    exp = (u >> 4) & 0x07
    mant = u & 0x0F
    mag = ((mant.astype(np.int32) << 3) + _BIAS) << exp
    s = np.where(sign != 0, _BIAS - mag, mag - _BIAS)
    return (s.astype(np.float32) / 32768.0)


def mulaw_encode(x: np.ndarray) -> bytes:
    s = np.clip(x * 32768.0, -32768, 32767).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    s = np.minimum(np.abs(s), _CLIP) + _BIAS
    exp = np.floor(np.log2(np.maximum(s, 1))).astype(np.int32) - 7
    exp = np.clip(exp, 0, 7)
    mant = (s >> (exp + 3)) & 0x0F
    return (~(sign | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


async def decode_to_pcm(data: bytes, rate: int) -> np.ndarray:
    """Decode mp3/ogg/wav bytes to mono float32 at `rate` using ffmpeg."""
    proc = await asyncio.create_subprocess_exec(
        FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-f", "s16le", "-ac", "1", "-ar", str(rate), "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate(data)
    return pcm16_to_float(out)


def trim_silence(x: np.ndarray, rate: int, thresh: float = 0.008, keep_ms: int = 30) -> np.ndarray:
    """Trim leading/trailing near-silence that TTS engines pad onto every clip (kills dead air)."""
    if len(x) == 0:
        return x
    env = np.abs(x)
    idx = np.where(env > thresh)[0]
    if len(idx) == 0:
        return x[:0]
    keep = int(rate * keep_ms / 1000)
    return x[max(0, idx[0] - keep): min(len(x), idx[-1] + keep * 3)]


def fade(x: np.ndarray, rate: int, ms: int = 8) -> np.ndarray:
    n = min(len(x) // 2, int(rate * ms / 1000))
    if n > 0:
        r = np.linspace(0, 1, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= r
        x[-n:] *= r[::-1]
    return x
