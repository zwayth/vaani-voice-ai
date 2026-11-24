"""Native speech-to-speech engine: Gemini Live (BidiGenerateContent over WebSocket).

Caller audio streams straight into a speech-native model that hears tone and answers with its own voice
(no separate STT/TTS hops). It does its own turn detection and barge-in; we keep a local VAD only to
measure honest voice-to-voice latency and drive the UI.

Reliability: if the key is missing, setup fails, or the socket drops mid-call, CallHandle swaps the call to
the cascade engine with the conversation so far - the caller keeps talking to "Aria" without a hang-up.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid

import numpy as np
import websockets

from . import expressive as ex
from .audio import float_to_pcm16, pcm16_to_float
from .config import KEYS, OUT_RATE, SETTINGS, env, system_prompt
from .health import health
from .hub import HUB
from .recording import CallRecorder
from .session import CallSession
from .vad import VAD

log = logging.getLogger("voice.s2s")
MODEL = env("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
URL = env("GEMINI_LIVE_URL", "wss://generativelanguage.googleapis.com/ws/"
          "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}")
CONNECT_TIMEOUT = float(env("GEMINI_LIVE_CONNECT_TIMEOUT", "2.0"))
SETUP_TIMEOUT = float(env("GEMINI_LIVE_SETUP_TIMEOUT", "3.0"))


def available() -> bool:
    return bool(KEYS["gemini"]) and health("s2s", "gemini_live", bool(KEYS["gemini"])).available


class GeminiLiveSession:
    def __init__(self, send_audio, send_event, clear_audio, channel="web", caller="browser", language=None,
                 on_fail=None):
        self.id = uuid.uuid4().hex[:8]
        self.channel, self.caller = channel, caller
        self._send_audio, self._send_event, self._clear = send_audio, send_event, clear_audio
        self.on_fail = on_fail
        self.lang_mode = language or SETTINGS["language"]
        self.lang = "hi" if self.lang_mode == "hi" else "en"
        self.vad = VAD(SETTINGS["endpoint_ms"])
        self.transcript: list[dict] = []
        self.turns: list[dict] = []
        self.started = time.time()
        self.rec = CallRecorder()
        self.recording: str | None = None
        self.state = "connecting"
        self.providers = {"stt": "native", "llm": MODEL.split("-preview")[0], "tts": "native", "engine": "gemini_live"}
        self.ws = None
        self.reader: asyncio.Task | None = None
        self.closed = False
        self.play_until = 0.0
        self._in_buf = bytearray()
        self._user_txt, self._agent_txt = "", ""
        self._turn_first_audio: float | None = None
        self._turn_no = 0
        self._was_interrupted = False

    # ---- shared interface with CallSession ----
    def summary(self) -> dict:
        lats = [t["e2e_ms"] for t in self.turns if t.get("e2e_ms")]
        last = self.turns[-1] if self.turns else {}
        return {"id": self.id, "channel": self.channel, "caller": self.caller, "lang": self.lang,
                "state": self.state, "started": self.started, "duration": round(time.time() - self.started),
                "turns": len(self.turns), "last_e2e": last.get("e2e_ms"),
                "avg_e2e": round(sum(lats) / len(lats)) if lats else None, "providers": dict(self.providers),
                "last_text": self.transcript[-1]["text"][:90] if self.transcript else "",
                "recording": self.recording, "rec_t0": self.rec.t0}

    async def event(self, ev: dict):
        ev["call_id"] = self.id
        try:
            await self._send_event(ev)
        except Exception:  # noqa: BLE001
            pass
        HUB.publish(dict(ev))

    def set_state(self, s: str):
        if s != self.state:
            self.state = s
            HUB.publish({"type": "call_state", "call_id": self.id, "state": s})
            asyncio.create_task(self.event({"type": "state", "state": s}))

    # ---- lifecycle ----
    async def start(self):
        """Connect + handshake. Raises on failure so the caller can fall back before the call is live."""
        h = health("s2s", "gemini_live", bool(KEYS["gemini"]))
        t0 = time.perf_counter()
        try:
            self.ws = await asyncio.wait_for(  # short connect budget: caller must never sit in dead air
                websockets.connect(URL.format(key=KEYS["gemini"]), max_size=None, ping_interval=20,
                                   open_timeout=CONNECT_TIMEOUT), CONNECT_TIMEOUT)
            prompt_lang = self.lang_mode if self.lang_mode != "auto" else "auto"
            setup = {"setup": {
                "model": f"models/{MODEL}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "temperature": SETTINGS["temperature"],
                    "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": SETTINGS["gemini_voice"]}}},
                },
                "systemInstruction": {"parts": [{"text": system_prompt(prompt_lang, native_audio=True)}]},
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "contextWindowCompression": {"slidingWindow": {}},
                "realtimeInputConfig": {"automaticActivityDetection": {
                    "silenceDurationMs": SETTINGS["endpoint_ms"], "prefixPaddingMs": 200}},
            }}
            await self.ws.send(json.dumps(setup))
            while True:
                msg = json.loads(await asyncio.wait_for(self.ws.recv(), SETUP_TIMEOUT))
                if "setupComplete" in msg:
                    break
                if "error" in msg:
                    raise RuntimeError(str(msg["error"])[:200])
        except Exception as e:
            h.fail(f"{type(e).__name__}: {e}")
            if self.ws:
                await self.ws.close()
            raise
        h.ok((time.perf_counter() - t0) * 1000)
        HUB.register(self)
        await self.event({"type": "hello", "engine": "gemini_live",
                          "settings": {"language": self.lang_mode, "company": SETTINGS["company"]}})
        self.reader = asyncio.create_task(self._read())
        greet_lang = "hi" if self.lang_mode == "hi" else "en"
        greeting = ex.strip_tags(SETTINGS[f"greeting_{greet_lang}"].format(company=SETTINGS["company"]))
        await self._client_text(f"(The phone call just connected. Greet the caller warmly with: \"{greeting}\")")
        self.set_state("speaking")

    async def _client_text(self, text: str):
        await self.ws.send(json.dumps({"clientContent": {
            "turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}}))

    async def close(self, failed: bool = False):
        if self.closed:
            return
        self.closed = True
        if self.reader and not self.reader.done() and asyncio.current_task() is not self.reader:
            self.reader.cancel()
        try:
            if self.ws:
                await self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        if not failed:
            self.set_state("ended")
            self.recording = await asyncio.to_thread(self.rec.save, self.id)
            HUB.unregister(self)

    # ---- input ----
    async def feed(self, x: np.ndarray):
        self.rec.caller(x)
        for ev in self.vad.feed(x):
            if ev[0] == "start" and time.monotonic() >= self.play_until:
                self.set_state("hearing")
        self._in_buf.extend(float_to_pcm16(x))
        if len(self._in_buf) >= 1280 and self.ws and not self.closed:  # 40 ms @16k
            data, self._in_buf = bytes(self._in_buf), bytearray()
            try:
                await self.ws.send(json.dumps({"realtimeInput": {"audio": {
                    "data": base64.b64encode(data).decode(), "mimeType": "audio/pcm;rate=16000"}}}))
            except Exception as e:  # noqa: BLE001
                await self._failed(e)

    async def user_text(self, text: str):
        self._user_txt = text
        await self._client_text(text)
        self.set_state("thinking")

    async def say(self, text: str):
        await self._client_text(f"(Operator instruction - say this to the caller now, naturally: \"{text}\")")

    async def interrupt(self):
        self.play_until = 0.0
        self.rec.clear()
        await self._clear()

    # ---- model output ----
    async def _read(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                sc = msg.get("serverContent")
                if not sc:
                    if "goAway" in msg:
                        log.info("gemini goAway on call %s", self.id)
                    continue
                if sc.get("inputTranscription", {}).get("text"):
                    self._user_txt += sc["inputTranscription"]["text"]
                    if self.state != "speaking":
                        self.set_state("thinking")
                if sc.get("interrupted"):
                    await self.interrupt()
                    await self.event({"type": "interrupted"})
                    self._finish_turn(interrupted=True)
                    self._was_interrupted = True
                    self.set_state("hearing")
                for part in (sc.get("modelTurn") or {}).get("parts", []):
                    inline = part.get("inlineData")
                    if inline and inline.get("mimeType", "").startswith("audio/pcm"):
                        await self._play(pcm16_to_float(base64.b64decode(inline["data"])))
                if sc.get("outputTranscription", {}).get("text"):
                    self._agent_txt += sc["outputTranscription"]["text"]
                    await self.event({"type": "agent_partial", "text": self._agent_txt.strip()})
                if sc.get("turnComplete"):
                    await self._flush_turn()
            if not self.closed:
                raise ConnectionError("gemini closed the session")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if not self.closed:
                await self._failed(e)

    async def _play(self, x: np.ndarray):
        if self._turn_first_audio is None:
            self._turn_first_audio = time.perf_counter()
            if self._user_txt.strip():
                await self._emit_user()
            self.set_state("speaking")
        now = time.monotonic()
        start = max(self.play_until, now)
        self.rec.agent(x, time.time() + (start - now))
        self.play_until = start + len(x) / OUT_RATE
        await self._send_audio(x)

    async def _emit_user(self):
        text = self._user_txt.strip()
        self._user_txt = ""
        if not text:
            return
        if self.lang_mode == "auto":
            from .providers.stt import detect_lang
            self.lang = detect_lang(text) if any("ऀ" <= c <= "ॿ" for c in text) else "en"
        self.transcript.append({"role": "user", "text": text, "t": time.time()})
        await self.event({"type": "transcript", "role": "user", "text": text, "final": True, "lang": self.lang})

    def _finish_turn(self, interrupted=False):
        m = {"turn": self._turn_no, "engine": "gemini_live", "stt": "native", "llm": "gemini-live", "tts": "native"}
        if self._turn_first_audio and self.vad.last_voice_t and self._turn_no > 0:
            e2e = round((self._turn_first_audio - self.vad.last_voice_t) * 1000)
            if 0 < e2e < 15000:
                m["e2e_ms"] = e2e
                HUB.turn_latencies.append(e2e)
        if interrupted:
            m["interrupted"] = True
        self.turns.append(m)
        self._turn_first_audio = None
        self._turn_no += 1
        return m

    async def _flush_turn(self):
        await self._emit_user()
        text = self._agent_txt.strip()
        self._agent_txt = ""
        if text:
            self.transcript.append({"role": "agent", "text": text, "t": time.time()})
            await self.event({"type": "transcript", "role": "agent", "text": text, "final": True})
        if self._was_interrupted:  # turn already accounted for when the caller cut in
            self._was_interrupted = False
        else:
            m = self._finish_turn()
            if self._turn_no > 1:
                await self.event({"type": "metrics", **m})
        while time.monotonic() < self.play_until and not self.closed:
            await asyncio.sleep(0.05)
        if not self.closed:
            self.set_state("listening")

    async def _failed(self, e: Exception):
        if self.closed:
            return
        log.warning("gemini live call %s failed: %s -> failing over to cascade", self.id, e)
        health("s2s", "gemini_live").fail(f"{type(e).__name__}: {e}")
        await self.close(failed=True)
        if self.on_fail:
            await self.on_fail(self)


class CallHandle:
    """What transports talk to. Picks the engine and hot-swaps to the cascade if the native engine dies."""

    def __init__(self, send_audio, send_event, clear_audio, channel="web", caller="browser", language=None,
                 engine: str | None = None):
        self.args = (send_audio, send_event, clear_audio)
        self.kw = {"channel": channel, "caller": caller, "language": language}
        self.engine = engine or SETTINGS["engine"]
        self.impl = None

    @property
    def id(self):
        return self.impl.id if self.impl else "-"

    async def start(self):
        if self.engine == "gemini_live" and available():
            g = GeminiLiveSession(*self.args, **self.kw, on_fail=self._failover)
            try:
                await g.start()
                self.impl = g
                return
            except Exception as e:  # noqa: BLE001
                log.warning("gemini live unavailable (%s); using cascade", e)
        self.impl = CallSession(*self.args, **self.kw)
        await self.impl.start()
        if self.engine == "gemini_live":
            await self.impl.event({"type": "engine", "engine": "cascade", "failover": True})

    async def _failover(self, prior: GeminiLiveSession):
        cs = CallSession(*self.args, **self.kw)
        await cs.resume(prior)
        self.impl = cs

    async def feed(self, x):
        await self.impl.feed(x)

    async def user_text(self, text):
        await self.impl.user_text(text)

    async def say(self, text):
        await self.impl.say(text)

    async def event(self, ev):
        await self.impl.event(ev)

    async def close(self):
        if self.impl:
            await self.impl.close()
