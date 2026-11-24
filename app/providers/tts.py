"""Text-to-speech with fallback chain + expressive rendering.

render(sentence) yields float32 PCM chunks at OUT_RATE as soon as each piece is ready.
Non-verbal tags are rendered natively where the vendor supports them, otherwise by app.expressive.
"""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
from typing import AsyncIterator

import httpx
import numpy as np

from .. import expressive as ex
from ..audio import decode_to_pcm, fade, pcm16_to_float, resample, trim_silence
from ..config import KEYS, OUT_RATE, SETTINGS, env
from ..health import health

_client = httpx.AsyncClient(timeout=httpx.Timeout(12.0, connect=4.0),
                            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
TIMEOUT = float(env("TTS_TIMEOUT", "6.0"))
SFX = ex.SfxBank(OUT_RATE)

EL_MODEL = env("ELEVENLABS_MODEL", "eleven_flash_v2_5")
NATIVE = {  # our tag -> vendor tag, per provider
    "elevenlabs": ({"sigh": "[sighs]", "laugh": "[laughs]", "breath": "[inhales]", "whisper": "[whispers]"}
                   if EL_MODEL.startswith("eleven_v3") else {}),
    "cartesia": {"laugh": "[laughter]"},
}


def _kokoro_ok() -> bool:
    from ..config import ROOT
    return any((ROOT / "models" / n).exists() for n in ("kokoro-v1.0.int8.onnx", "kokoro-v1.0.onnx", "kokoro-v1.0.fp16.onnx"))


def configured(name: str) -> bool:
    return {"elevenlabs": bool(KEYS["elevenlabs"]), "cartesia": bool(KEYS["cartesia"]),
            "sarvam": bool(KEYS["sarvam"]), "nvidia": bool(KEYS["nvidia"]),
            "kokoro": _kokoro_ok(), "edge": True, "sapi": os.name == "nt"}.get(name, False)


# ---------------- vendor adapters: (text, lang) -> float32 @ OUT_RATE ----------------

async def _elevenlabs(text: str, lang: str) -> np.ndarray:
    voice = env("ELEVENLABS_VOICE_HI" if lang == "hi" else "ELEVENLABS_VOICE_EN", "") or env(
        "ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    body = {"text": text, "model_id": EL_MODEL,
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.35, "use_speaker_boost": True}}
    if not EL_MODEL.startswith("eleven_v3"):
        body["language_code"] = "hi" if lang == "hi" else "en"
    r = await _client.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=pcm_24000&optimize_streaming_latency=3",
        json=body, headers={"xi-api-key": KEYS["elevenlabs"]})
    r.raise_for_status()
    return pcm16_to_float(r.content)


async def _cartesia(text: str, lang: str) -> np.ndarray:
    voice = env("CARTESIA_VOICE_HI" if lang == "hi" else "CARTESIA_VOICE_EN", "") or env(
        "CARTESIA_VOICE_ID", "f9836c6e-a0bd-460e-9d3c-f7299fa60f94")
    body = {"model_id": env("CARTESIA_MODEL", "sonic-3"), "transcript": text,
            "voice": {"mode": "id", "id": voice}, "language": "hi" if lang == "hi" else "en",
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": OUT_RATE}}
    r = await _client.post("https://api.cartesia.ai/tts/bytes", json=body,
                           headers={"X-API-Key": KEYS["cartesia"], "Cartesia-Version": "2025-04-16"})
    r.raise_for_status()
    return pcm16_to_float(r.content)


async def _sarvam(text: str, lang: str) -> np.ndarray:
    body = {"text": text, "target_language_code": "hi-IN" if lang == "hi" else "en-IN",
            "speaker": env("SARVAM_SPEAKER", "anushka"), "model": env("SARVAM_TTS_MODEL", "bulbul:v2"),
            "speech_sample_rate": OUT_RATE, "enable_preprocessing": True}
    r = await _client.post("https://api.sarvam.ai/text-to-speech", json=body,
                           headers={"api-subscription-key": KEYS["sarvam"]})
    r.raise_for_status()
    wav = base64.b64decode(r.json()["audios"][0])
    return pcm16_to_float(wav[44:])


def _nvidia_sync(text: str, lang: str) -> np.ndarray:
    """NVIDIA Magpie TTS Multilingual via NVIDIA cloud (gRPC/Riva). Requires: pip install nvidia-riva-client."""
    import riva.client  # type: ignore
    auth = riva.client.Auth(uri="grpc.nvcf.nvidia.com:443", use_ssl=True, metadata_args=[
        ["function-id", env("NVIDIA_TTS_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")],
        ["authorization", f"Bearer {KEYS['nvidia']}"]])
    svc = riva.client.SpeechSynthesisService(auth)
    resp = svc.synthesize(text, voice_name=env("NVIDIA_TTS_VOICE", "Magpie-Multilingual.EN-US.Aria"),
                          language_code=env("NVIDIA_TTS_LANG", "en-US"), sample_rate_hz=OUT_RATE)
    return pcm16_to_float(resp.audio)


async def _nvidia(text: str, lang: str) -> np.ndarray:
    if lang == "hi":
        raise RuntimeError("magpie: hindi not routed")
    return await asyncio.to_thread(_nvidia_sync, text, lang)


_kokoro = None
_kokoro_sem = asyncio.Semaphore(int(env("KOKORO_PARALLEL", "2")))


KOKORO_DEVICE = {"name": "cpu"}


def _kokoro_load():
    """GPU (CUDA) when available - ~3x faster and keeps the CPU free for other calls - else int8 on CPU."""
    global _kokoro
    if _kokoro is None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        from ..config import ROOT
        models = ROOT / "models"
        sess = None
        want_gpu = env("KOKORO_DEVICE", "auto") != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
        if want_gpu:
            try:
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
                ort.set_default_logger_severity(3)
                for name in ("kokoro-v1.0.fp16.onnx", "kokoro-v1.0.onnx"):
                    if (models / name).exists():
                        sess = ort.InferenceSession(str(models / name), ort.SessionOptions(),
                                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                        if sess.get_providers()[0] == "CUDAExecutionProvider":
                            KOKORO_DEVICE["name"] = f"gpu ({name})"
                            break
                        sess = None
            except Exception:  # noqa: BLE001 - any CUDA/cuDNN problem => CPU
                sess = None
        if sess is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(env("KOKORO_THREADS", "3"))
            sess = ort.InferenceSession(str(models / "kokoro-v1.0.int8.onnx"), so, providers=["CPUExecutionProvider"])
            KOKORO_DEVICE["name"] = "cpu (int8)"
        _kokoro = Kokoro.from_session(sess, str(models / "voices-v1.0.bin"))
        try:  # first run compiles GPU kernels; do it now, not on a live call
            _kokoro.create("Hello.", voice=SETTINGS["kokoro_en"], lang="en-us")
            _kokoro.create("नमस्ते।", voice=SETTINGS["kokoro_hi"], lang="hi")
        except Exception:  # noqa: BLE001
            pass
    return _kokoro


def _kokoro_sync(text: str, lang: str) -> np.ndarray:
    k = _kokoro_load()
    voice = SETTINGS["kokoro_hi"] if lang == "hi" else SETTINGS["kokoro_en"]
    a, sr = k.create(text, voice=voice, speed=1.05, lang="hi" if lang == "hi" else "en-us")
    return resample(a.astype(np.float32), sr, OUT_RATE)


async def _kokoro_tts(text: str, lang: str) -> np.ndarray:
    async with _kokoro_sem:
        return await asyncio.to_thread(_kokoro_sync, text, lang)


async def _edge(text: str, lang: str, whisper: bool = False) -> np.ndarray:
    import edge_tts
    voice = SETTINGS["voice_hi"] if lang == "hi" else SETTINGS["voice_en"]
    comm = edge_tts.Communicate(text, voice, rate="+6%" if not whisper else "-4%", pitch="+0Hz")
    buf = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    if not buf:
        raise RuntimeError("edge: no audio")
    return await decode_to_pcm(bytes(buf), OUT_RATE)


async def _sapi(text: str, lang: str) -> np.ndarray:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    safe = text.replace("'", "''")
    ps = ("Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{safe}'); $s.Dispose()")
    p = await asyncio.create_subprocess_exec("powershell", "-NoProfile", "-Command", ps,
                                             stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await p.wait()
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return await decode_to_pcm(data, OUT_RATE)


ADAPTERS = {"elevenlabs": _elevenlabs, "cartesia": _cartesia, "nvidia": _nvidia, "sarvam": _sarvam,
            "kokoro": _kokoro_tts, "edge": _edge, "sapi": _sapi}

# phrase-level cache: support calls repeat a lot ("Is there anything else I can help you with?")
_CACHE: dict[tuple, np.ndarray] = {}
_CACHE_MAX = 400


def _voice_of(name: str, lang: str) -> str:
    if name == "kokoro":
        return SETTINGS["kokoro_hi" if lang == "hi" else "kokoro_en"]
    if name == "edge":
        return SETTINGS["voice_hi" if lang == "hi" else "voice_en"]
    return ""


async def _synth(text: str, lang: str, whisper: bool, chain: list[str]) -> tuple[np.ndarray, str]:
    for name in chain:
        h = health("tts", name, configured(name))
        if not h.available or name not in ADAPTERS:
            continue
        key = (name, _voice_of(name, lang), lang, whisper, text)
        if key in _CACHE:
            return _CACHE[key], name
        t0 = time.perf_counter()
        try:
            if name == "edge":
                coro = _edge(text, lang, whisper)
            else:
                coro = ADAPTERS[name](text, lang)
            x = await asyncio.wait_for(coro, timeout=TIMEOUT)
            if len(x) == 0:
                raise RuntimeError("empty audio")
            h.ok((time.perf_counter() - t0) * 1000)
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = x
            return x, name
        except Exception as e:  # noqa: BLE001
            h.fail(f"{type(e).__name__}: {e}")
    return np.zeros(0, np.float32), "none"


def first_available(chain: list[str]) -> str:
    for name in chain:
        if health("tts", name, configured(name)).available:
            return name
    return "none"


async def render(sentence: str, lang: str, chain: list[str] | None = None,
                 stats: dict | None = None) -> AsyncIterator[np.ndarray]:
    """Yield audio for one sentence, piece by piece (speech segments, breaths, sighs, pauses)."""
    chain = chain or SETTINGS["tts_chain"]
    expressive = SETTINGS.get("expressive", True)
    provider = first_available(chain)
    native = NATIVE.get(provider, {})

    if not expressive:
        segs = [ex.Seg("speech", ex.strip_tags(sentence))]
    elif native:
        # fold natively supported tags back into the text for this vendor; render the rest locally
        segs, carry = [], ""
        for s in ex.parse(sentence):
            if s.kind == "sfx" and s.name in native:
                carry += native[s.name] + " "
            elif s.kind == "speech":
                prefix = native.get("whisper", "") + " " if (s.whisper and "whisper" in native) else ""
                segs.append(ex.Seg("speech", carry + prefix + s.text, s.whisper and "whisper" not in native))
                carry = ""
            else:
                segs.append(s)
    else:
        segs = ex.parse(sentence)

    # synthesise every speech segment concurrently; yield strictly in order
    jobs = {i: asyncio.create_task(_synth(s.text, lang, s.whisper, chain))
            for i, s in enumerate(segs) if s.kind == "speech"}
    try:
        async for item in _ordered(segs, jobs, stats):
            yield item
    finally:
        for j in jobs.values():
            j.cancel()


async def _ordered(segs, jobs, stats):
    for i, s in enumerate(segs):
        if s.kind == "pause":
            yield np.zeros(int(OUT_RATE * s.ms / 1000), np.float32)
        elif s.kind == "sfx":
            clip = SFX.get(s.name)
            if len(clip):
                yield clip
                yield np.zeros(int(OUT_RATE * 0.05), np.float32)
        else:
            x, used = await jobs[i]
            if stats is not None:
                stats["tts_provider"] = used
            if used in ("edge", "sapi", "kokoro"):
                x = trim_silence(x, OUT_RATE)
            if s.whisper:
                x = ex.whisperize(x, OUT_RATE)
            if len(x):
                yield fade(x, OUT_RATE)


def resample_out(x: np.ndarray, rate: int) -> np.ndarray:
    return resample(x, OUT_RATE, rate)
