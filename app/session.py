"""CallSession: one live conversation. Transport-agnostic (browser WebSocket, Twilio, Exotel...).

Pipeline per turn (fully streamed and overlapped):
  mic -> VAD endpoint -> STT -> LLM token stream -> sentence chunker -> TTS (prefetched, ordered) -> speaker
Barge-in: caller speech while the agent talks cancels generation and flushes the client's audio buffer.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Awaitable, Callable

import numpy as np

from . import expressive as ex
from .config import OUT_RATE, SETTINGS, env, system_prompt
from .hub import HUB
from .providers import llm, stt, tts
from .vad import VAD

SENT_END = re.compile(r"([.!?।]+[\"')\]]*)(\s+|$)")
SOFT_BREAK = re.compile(r"([,;:—–])\s+")
JUNK = {"", ".", "you", "thank you.", "thanks for watching!", "bye.", "उपशीर्षक"}

KOKORO_MAX_CALLS = int(env("KOKORO_MAX_CALLS", "2"))
KOKORO_MAX_CALLS_GPU = int(env("KOKORO_MAX_CALLS_GPU", "4"))
FILLERS = {"en": ["Hmm,", "Right,", "Okay,"], "hi": ["हम्म,", "अच्छा,", "जी,"]}
_FILLER_CACHE: dict[str, list[np.ndarray]] = {}
_GREETING_CACHE: dict[str, list[np.ndarray]] = {}


def _chains() -> list[list[str]]:
    full = list(SETTINGS["tts_chain"])
    return [full] + ([[n for n in full if n != "kokoro"]] if "kokoro" in full else [])


async def warm_caches():
    """Pre-render greeting + filler clips (for every routing variant) so the first audio of a call is instant."""
    for chain in _chains():
        eng = tts.first_available(chain)
        for lang in ("en", "hi"):
            clips = []
            for f in FILLERS[lang]:
                try:
                    clips.append(np.concatenate([c async for c in tts.render(f, lang, chain=chain)]))
                except Exception:  # noqa: BLE001
                    pass
            _FILLER_CACHE[f"{lang}:{eng}"] = [c for c in clips if len(c)]
            await greeting_audio(lang, chain, refresh=True)


async def greeting_audio(lang: str, chain: list[str], refresh: bool = False) -> list[np.ndarray]:
    text = SETTINGS[f"greeting_{lang}"].format(company=SETTINGS["company"])
    eng = tts.first_available(chain)
    key = f"{lang}:{text}:{eng}:{SETTINGS['voice_' + lang]}:{SETTINGS['kokoro_' + lang]}"
    if refresh or key not in _GREETING_CACHE:
        _GREETING_CACHE[key] = [c async for c in tts.render(text, lang, chain=chain)]
    return _GREETING_CACHE[key]


JUNK_REPLY = re.compile(r"(?i)(response is invalid|invalid response|as an ai\b|i cannot assist|\[object|^\s*(null|none|undefined)\s*$)")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _reply_ok(text: str, lang: str) -> bool:
    t = ex.strip_tags(text)
    if not t or JUNK_REPLY.search(t):
        return False
    if lang == "hi" and re.search(r"[A-Za-z]{3}", t) and not DEVANAGARI.search(t):
        return False  # Hindi call answered in English
    return True


async def guarded_reply(msgs: list, lang: str, on_provider, attempts: int = 3):
    """Stream the model's reply, but check its opening before anything is spoken. A junk or wrong-language
    opening is discarded and regenerated, so the caller never hears a glitch."""
    for attempt in range(attempts):
        head, gen = "", llm.stream_reply(msgs, SETTINGS["llm_chain"], SETTINGS["temperature"], lang,
                                         on_provider=on_provider)
        buffered: list[str] = []
        done = False
        async for tok in gen:
            buffered.append(tok)
            head += tok
            if len(ex.strip_tags(head)) >= 24:
                break
        else:
            done = True
        if _reply_ok(head, lang):
            for tok in buffered:
                yield tok
            if not done:
                async for tok in gen:
                    yield tok
            return
        await gen.aclose()
        msgs = msgs + [{"role": "system", "content":
                        "Your previous draft was invalid. Reply to the caller naturally"
                        + (" in Hindi (Devanagari script)." if lang == "hi" else ".")}]
    yield ("माफ़ कीजिए, क्या आप एक बार फिर बोल सकते हैं?" if lang == "hi" else "Sorry, could you say that once more?")


class Chunker:
    """Turns an LLM token stream into speakable chunks, flushing as early as is natural."""

    def __init__(self):
        self.buf = ""
        self.first = True

    def push(self, tok: str) -> list[str]:
        self.buf += tok
        out = []
        while True:
            b = self.buf.lstrip()
            # a leading expressive tag (e.g. [breath]) is emitted instantly -> audio starts during LLM gen
            m = re.match(r"^\[(breath|sigh|laugh|hmm|pause|inhale|exhale)\]\s*", b, re.I)
            if m:
                out.append(m.group(0).strip())
                self.buf = b[m.end():]
                continue
            if b.count("[") > b.count("]"):
                break  # inside a tag
            m = SENT_END.search(b)
            if m and len(b[:m.end()].strip()) >= 2:
                out.append(b[:m.end()].strip())
                self.buf = b[m.end():]
                self.first = False
                continue
            limit = 8 if self.first else 70   # tiny first chunk => first audio in ~300ms
            m = SOFT_BREAK.search(b)
            if m and m.start() >= limit:
                out.append(b[:m.end()].strip())
                self.buf = b[m.end():]
                self.first = False
                continue
            if self.first and "[" not in b:
                words = b.split(" ")
                if len(words) > 7:  # long unpunctuated opener: speak the first words now
                    out.append(" ".join(words[:6]))
                    self.buf = " ".join(words[6:])
                    self.first = False
                    continue
            break
        return out

    def flush(self) -> list[str]:
        b, self.buf = self.buf.strip(), ""
        return [b] if b else []


class CallSession:
    def __init__(self, send_audio: Callable[[np.ndarray], Awaitable[None]],
                 send_event: Callable[[dict], Awaitable[None]], clear_audio: Callable[[], Awaitable[None]],
                 channel: str = "web", caller: str = "browser", language: str | None = None):
        self.id = uuid.uuid4().hex[:8]
        self.channel, self.caller = channel, caller
        self._send_audio, self._send_event, self._clear = send_audio, send_event, clear_audio
        self.vad = VAD(SETTINGS["endpoint_ms"])
        self.lang_mode = language or SETTINGS["language"]
        self.lang = "hi" if self.lang_mode == "hi" else "en"
        self.history: list[dict] = []
        self.transcript: list[dict] = []
        self.turns: list[dict] = []
        self.started = time.time()
        self.state = "connecting"
        self.play_until = 0.0
        self.task: asyncio.Task | None = None
        self.turn_id = 0
        self.closed = False
        self.providers = {"stt": "-", "llm": "-", "tts": "-", "engine": "cascade"}
        self._speech_ms = 0
        self._pending_audio: np.ndarray | None = None
        # capacity-aware voice routing: local GPU/CPU voice only while it has headroom, chosen once per call
        chain = list(SETTINGS["tts_chain"])
        local_calls = sum(1 for c in HUB.calls.values() if "kokoro" in getattr(c, "tts_chain", ()))
        limit = KOKORO_MAX_CALLS_GPU if tts.KOKORO_DEVICE["name"].startswith("gpu") else KOKORO_MAX_CALLS
        if local_calls >= limit and "kokoro" in chain:
            chain.remove("kokoro")
        self.tts_chain = chain

    # ---------------- helpers ----------------
    def summary(self) -> dict:
        last = self.turns[-1] if self.turns else {}
        lats = [t["e2e_ms"] for t in self.turns if t.get("e2e_ms") and not t.get("typed")]
        return {"id": self.id, "channel": self.channel, "caller": self.caller, "lang": self.lang,
                "state": self.state, "started": self.started, "duration": round(time.time() - self.started),
                "turns": len(self.turns), "last_e2e": last.get("e2e_ms"),
                "avg_e2e": round(sum(lats) / len(lats)) if lats else None, "providers": dict(self.providers),
                "last_text": self.transcript[-1]["text"][:90] if self.transcript else ""}

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
            asyncio.create_task(self._safe_event({"type": "state", "state": s}))

    async def _safe_event(self, ev):
        try:
            await self._send_event(ev)
        except Exception:  # noqa: BLE001
            pass

    @property
    def agent_speaking(self) -> bool:
        return time.monotonic() < self.play_until or (self.task is not None and not self.task.done())

    async def play(self, x: np.ndarray):
        if len(x) == 0 or self.closed:
            return
        now = time.monotonic()
        self.play_until = max(self.play_until, now) + len(x) / OUT_RATE
        await self._send_audio(x)

    # ---------------- lifecycle ----------------
    async def start(self):
        HUB.register(self)
        self.set_state("speaking")
        await self.event({"type": "hello", "settings": {"language": self.lang_mode, "company": SETTINGS["company"]}})
        greet_lang = "hi" if self.lang_mode == "hi" else "en"
        text = SETTINGS[f"greeting_{greet_lang}"].format(company=SETTINGS["company"])
        self.history.append({"role": "assistant", "content": ex.strip_tags(text)})
        self.transcript.append({"role": "agent", "text": ex.strip_tags(text), "t": time.time()})
        await self.event({"type": "transcript", "role": "agent", "text": ex.strip_tags(text), "final": True})
        self.task = asyncio.create_task(self._play_clips(await greeting_audio(greet_lang, self.tts_chain)))

    async def _play_clips(self, clips):
        for c in clips:
            await self.play(c)
        await self._wait_playback()
        self.set_state("listening")

    async def _wait_playback(self):
        while time.monotonic() < self.play_until and not self.closed:
            await asyncio.sleep(0.05)

    async def close(self):
        if self.closed:
            return
        self.closed = True
        if self.task and not self.task.done():
            self.task.cancel()
        self.set_state("ended")
        HUB.unregister(self)

    # ---------------- input ----------------
    async def feed(self, x: np.ndarray):
        """x: float32 mono @16k from the caller."""
        for ev in self.vad.feed(x):
            if ev[0] == "start":
                self._speech_ms = 0
                if not self.agent_speaking:
                    self.set_state("hearing")
                await self.event({"type": "vad", "speech": True})
            elif ev[0] == "speech_ms":
                self._speech_ms = ev[1]
                # barge-in after ~240ms of real speech: avoids coughs / echo blips cutting the agent off
                if SETTINGS["barge_in"] and self.agent_speaking and ev[1] >= 240:
                    await self.interrupt()
            elif ev[0] == "end":
                _, audio, voiced_ms = ev
                await self.event({"type": "vad", "speech": False})
                if voiced_ms < 200:
                    if not self.agent_speaking:
                        self.set_state("listening")
                    continue
                if self.agent_speaking:
                    if not SETTINGS["barge_in"]:
                        continue
                    await self.interrupt()
                if self._pending_audio is not None:
                    # caller kept talking before we transcribed their previous fragment: merge them
                    audio = np.concatenate([self._pending_audio, audio])
                self.turn_id += 1
                # latency is measured from when the caller actually stopped talking (industry definition),
                # so it includes the end-of-turn silence wait - a stopwatch on the call will agree with it
                self.task = asyncio.create_task(self._turn(self.turn_id, audio, self.vad.last_voice_t))

    async def interrupt(self):
        if self.task and not self.task.done():
            self.task.cancel()
        self.play_until = 0.0
        try:
            await self._clear()
        except Exception:  # noqa: BLE001
            pass
        await self.event({"type": "interrupted"})
        self.set_state("hearing")

    async def user_text(self, text: str):
        """A typed caller turn (silent demos, simulated callers)."""
        await self.interrupt()
        self.turn_id += 1
        self.history.append({"role": "user", "content": text})
        self.transcript.append({"role": "user", "text": text, "t": time.time()})
        await self.event({"type": "transcript", "role": "user", "text": text, "final": True})
        self.task = asyncio.create_task(self._respond(self.turn_id, text, time.perf_counter(), {}))

    async def resume(self, prior):
        """Take over a live call from another engine (hot failover) without the caller noticing."""
        self.id, self.started, self.lang = prior.id, prior.started, prior.lang
        self.transcript, self.turns = list(prior.transcript), list(prior.turns)
        self.history = [{"role": "user" if t["role"] == "user" else "assistant", "content": t["text"]}
                        for t in prior.transcript][-16:]
        self.providers["engine"] = "cascade (failover)"
        HUB.calls[self.id] = self
        await self.event({"type": "engine", "engine": "cascade", "failover": True})
        self.set_state("listening")

    async def say(self, text: str):
        """Operator 'whisper-to-agent' / inject: speak text right now."""
        await self.interrupt()
        self.turn_id += 1
        self.task = asyncio.create_task(self._respond(self.turn_id, None, time.perf_counter(), {}, forced=text))

    # ---------------- one conversational turn ----------------
    async def _turn(self, tid: int, audio: np.ndarray, t_end: float):
        m = {"turn": tid}
        self.set_state("thinking")
        t0 = time.perf_counter()
        self._pending_audio = audio
        text, prov = await stt.transcribe(audio, SETTINGS["stt_chain"],
                                          self.lang_mode if self.lang_mode != "auto" else "auto")
        self._pending_audio = None
        m["stt_ms"] = round((time.perf_counter() - t0) * 1000)
        m["stt"] = self.providers["stt"] = prov
        if text.lower().strip() in JUNK or tid != self.turn_id:
            self.set_state("listening")
            return
        if self.lang_mode == "auto":
            self.lang = stt.detect_lang(text)
        self.transcript.append({"role": "user", "text": text, "t": time.time()})
        await self.event({"type": "transcript", "role": "user", "text": text, "final": True, "lang": self.lang})
        self.history.append({"role": "user", "content": text})
        await self._respond(tid, text, t_end, m)

    async def _respond(self, tid: int, text: str | None, t_end: float, m: dict, forced: str | None = None):
        lang = self.lang
        spoken: list[str] = []
        first_audio = {"t": None}
        queues: asyncio.Queue = asyncio.Queue()   # ordered queue of per-chunk audio queues
        tts_stats: dict = {}

        async def synth_chunk(chunk: str, q: asyncio.Queue):
            try:
                async for a in tts.render(chunk, lang, chain=self.tts_chain, stats=tts_stats):
                    await q.put(a)
            finally:
                await q.put(None)

        async def player():
            while True:
                q = await queues.get()
                if q is None:
                    return
                while True:
                    a = await q.get()
                    if a is None:
                        break
                    if first_audio["t"] is None:
                        first_audio["t"] = time.perf_counter()
                        self.set_state("speaking")
                    await self.play(a)

        async def filler():
            await asyncio.sleep(0.9)
            clips = _FILLER_CACHE.get(f"{lang}:{tts.first_available(self.tts_chain)}") or []
            if first_audio["t"] is None and clips and forced is None:
                # masks a slow LLM/TTS vendor with a natural backchannel; e2e metric stays honest
                m["filler_ms"] = round((time.perf_counter() - t_end) * 1000)
                self.set_state("speaking")
                await self.play(clips[tid % len(clips)])

        player_task = asyncio.create_task(player())
        filler_task = asyncio.create_task(filler())
        synth_tasks = []
        try:
            def emit(chunk: str):
                spoken.append(chunk)
                q: asyncio.Queue = asyncio.Queue()
                synth_tasks.append(asyncio.create_task(synth_chunk(chunk, q)))
                queues.put_nowait(q)

            t_llm = time.perf_counter()
            if forced:
                emit(forced)
                m["llm"] = "operator"
            else:
                msgs = [{"role": "system", "content": system_prompt(self.lang_mode if self.lang_mode != "auto" else lang)}]
                msgs += self.history[-16:]
                chunker = Chunker()
                got_first = False

                def on_llm(p):
                    m["llm"] = self.providers["llm"] = p

                async for tok in guarded_reply(msgs, lang, on_llm):
                    if not got_first:
                        m["llm_ttft_ms"] = round((time.perf_counter() - t_llm) * 1000)
                        got_first = True
                    for c in chunker.push(tok):
                        emit(c)
                    await self.event({"type": "agent_partial", "text": ex.strip_tags(" ".join(spoken) + " " + chunker.buf)})
                for c in chunker.flush():
                    emit(c)
                m["llm_ms"] = round((time.perf_counter() - t_llm) * 1000)
            queues.put_nowait(None)
            reply = ex.strip_tags(" ".join(spoken))
            await self.event({"type": "transcript", "role": "agent", "text": reply, "final": True, "raw": " ".join(spoken)})
            self.history.append({"role": "assistant", "content": " ".join(spoken)})
            self.transcript.append({"role": "agent", "text": reply, "t": time.time()})
            await player_task
            m["tts"] = self.providers["tts"] = tts_stats.get("tts_provider", "-")
            if first_audio["t"]:
                m["e2e_ms"] = round((first_audio["t"] - t_end) * 1000)
                if "stt_ms" in m:  # only real spoken turns count toward voice-to-voice KPIs
                    HUB.turn_latencies.append(m["e2e_ms"])
                else:
                    m["typed"] = True
            self.turns.append(m)
            await self.event({"type": "metrics", **m})
            await self._wait_playback()
            if tid == self.turn_id:
                self.set_state("listening")
        except asyncio.CancelledError:
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken) + " —(cut off by caller)"})
            m["interrupted"] = True
            self.turns.append(m)
            raise
        finally:
            filler_task.cancel()
            player_task.cancel()
            for t in synth_tasks:
                t.cancel()
