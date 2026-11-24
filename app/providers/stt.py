"""Speech-to-text with fallback chain. Operates per utterance (VAD-endpointed), which with a 0.5s
endpoint keeps total STT cost at ~150-400ms on cloud providers."""
from __future__ import annotations

import asyncio
import re
import time

import httpx
import numpy as np

from ..audio import to_wav
from ..config import IN_RATE, KEYS, env
from ..health import health

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0))
TIMEOUT = float(env("STT_TIMEOUT", "6.0"))


def detect_lang(text: str) -> str:
    return "hi" if DEVANAGARI.search(text or "") else "en"


def configured(name: str) -> bool:
    return {"deepgram": bool(KEYS["deepgram"]), "groq": bool(KEYS["groq"]), "sarvam": bool(KEYS["sarvam"]),
            "google": True, "whisper": True}.get(name, False)


async def _deepgram(wav: bytes, lang: str) -> str:
    language = {"en": "en-IN", "hi": "hi"}.get(lang, "multi")
    r = await _client.post(
        f"https://api.deepgram.com/v1/listen?model=nova-3&language={language}&smart_format=true&punctuate=true",
        content=wav, headers={"Authorization": f"Token {KEYS['deepgram']}", "Content-Type": "audio/wav"})
    r.raise_for_status()
    return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


async def _groq(wav: bytes, lang: str) -> str:
    data = {"model": env("GROQ_STT_MODEL", "whisper-large-v3-turbo"), "response_format": "json", "temperature": "0"}
    if lang in ("en", "hi"):
        data["language"] = lang
    r = await _client.post("https://api.groq.com/openai/v1/audio/transcriptions", data=data,
                           files={"file": ("a.wav", wav, "audio/wav")},
                           headers={"Authorization": f"Bearer {KEYS['groq']}"})
    r.raise_for_status()
    return r.json()["text"]


async def _sarvam(wav: bytes, lang: str) -> str:
    r = await _client.post("https://api.sarvam.ai/speech-to-text",
                           data={"model": env("SARVAM_STT_MODEL", "saarika:v2.5"),
                                 "language_code": {"en": "en-IN", "hi": "hi-IN"}.get(lang, "unknown")},
                           files={"file": ("a.wav", wav, "audio/wav")},
                           headers={"api-subscription-key": KEYS["sarvam"]})
    r.raise_for_status()
    return r.json().get("transcript", "")


HINDI_ROMAN = set("main mai hai hain hun hoon kya aap ap nahi nahin mera meri mujhe rahi raha ka ki ke se ko haan han ji "
                  "namaste accha acha theek thik bhi kar karna karo kab kaise kahan kitna wala wali tha thi abhi "
                  "batao bataiye chahiye paisa order mein mera humko hum".split())


def _google_one(audio, language: str) -> str:
    import speech_recognition as sr
    try:
        return sr.Recognizer().recognize_google(audio, language=language)
    except sr.UnknownValueError:
        return ""


def _google_sync(wav: bytes, lang: str) -> str:
    import speech_recognition as sr
    audio = sr.AudioData(wav[44:], IN_RATE, 2)
    if lang in ("en", "hi"):
        return _google_one(audio, "en-IN" if lang == "en" else "hi-IN")
    # auto: recognise in both languages concurrently, then decide by whether the English pass
    # came back as romanised Hindi ("main order ka status jaanna chahta hoon")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        f_en, f_hi = pool.submit(_google_one, audio, "en-IN"), pool.submit(_google_one, audio, "hi-IN")
        en, hi = f_en.result(), f_hi.result()
    words = re.findall(r"[a-z']+", en.lower())
    hindi_ratio = sum(w in HINDI_ROMAN for w in words) / max(1, len(words))
    return hi if (hindi_ratio >= 0.25 or not en) and hi else en


_whisper_model = None
_whisper_lock = asyncio.Lock()


def _whisper_sync(x: np.ndarray, lang: str) -> str:
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(env("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
    segs, _ = _whisper_model.transcribe(x, language=lang if lang in ("en", "hi") else None, beam_size=1,
                                        vad_filter=False, condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


async def transcribe(x: np.ndarray, chain: list[str], lang: str, on_provider=None) -> tuple[str, str]:
    """Return (text, provider). Empty text means nothing intelligible was said."""
    wav = to_wav(x, IN_RATE)
    for name in chain:
        h = health("stt", name, configured(name))
        if not h.available:
            continue
        t0 = time.perf_counter()
        try:
            if name == "deepgram":
                coro = _deepgram(wav, lang)
            elif name == "groq":
                coro = _groq(wav, lang)
            elif name == "sarvam":
                coro = _sarvam(wav, lang)
            elif name == "google":
                coro = asyncio.to_thread(_google_sync, wav, lang)
            elif name == "whisper":
                async with _whisper_lock:
                    text = await asyncio.wait_for(asyncio.to_thread(_whisper_sync, x, lang), timeout=25)
                h.ok((time.perf_counter() - t0) * 1000)
                return text.strip(), name
            else:
                continue
            text = await asyncio.wait_for(coro, timeout=TIMEOUT)
            h.ok((time.perf_counter() - t0) * 1000)
            if on_provider:
                on_provider(name)
            return (text or "").strip(), name
        except Exception as e:  # noqa: BLE001
            h.fail(f"{type(e).__name__}: {e}")
    return "", "none"
