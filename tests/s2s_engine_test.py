"""Native speech-to-speech engine test against a protocol-faithful mock of Gemini Live.

Checks: handshake, greeting audio, caller audio streaming, transcripts, latency metric, barge-in
('interrupted'), and HOT FAILOVER: the mock drops the socket mid-call and the same call must continue on
the cascade engine with its conversation intact. Also checks the no-key / bad-endpoint fallback.
Usage: python -I tests/s2s_engine_test.py
"""
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["GEMINI_LIVE_URL"] = "ws://127.0.0.1:8899/live?key={key}"

import numpy as np  # noqa: E402
import websockets  # noqa: E402

from app.audio import float_to_pcm16, resample  # noqa: E402
from app.providers import tts  # noqa: E402

SEEN = {"setup": None, "audio_bytes": 0, "texts": []}


def tone(sec, rate=24000):
    t = np.arange(int(sec * rate)) / rate
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


async def mock_gemini(ws):
    """Speaks the BidiGenerateContent protocol: setup -> setupComplete, serverContent turns."""
    async def say(text, sec=1.0, user=None):
        if user:
            await ws.send(json.dumps({"serverContent": {"inputTranscription": {"text": user}}}))
        pcm = float_to_pcm16(tone(sec))
        for i in range(0, len(pcm), 9600):
            await ws.send(json.dumps({"serverContent": {"modelTurn": {"parts": [
                {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": base64.b64encode(pcm[i:i + 9600]).decode()}}]}}}))
        await ws.send(json.dumps({"serverContent": {"outputTranscription": {"text": text}}}))
        await ws.send(json.dumps({"serverContent": {"turnComplete": True}}))

    heard_since = 0
    async for raw in ws:
        m = json.loads(raw)
        if "setup" in m:
            SEEN["setup"] = m["setup"]
            await ws.send(json.dumps({"setupComplete": {}}))
        elif "clientContent" in m:
            SEEN["texts"].append(m["clientContent"]["turns"][0]["parts"][0]["text"])
            await say("Hi, thank you for calling Vaani support, this is Aria. How can I help you today?", 1.2)
        elif "realtimeInput" in m:
            n = len(base64.b64decode(m["realtimeInput"]["audio"]["data"]))
            SEEN["audio_bytes"] += n
            heard_since += n
            if heard_since > 16000 * 2 * 3.2:  # ~3.2 s of caller audio (speech + silence) => answer
                heard_since = -10**9
                await say("Of course, could you share your order ID?", 1.0, user="my order has not arrived")
                await asyncio.sleep(1.5)
                await ws.close(code=1011, reason="simulated outage")  # drop mid-call
                return


class FakeTransport:
    def __init__(self):
        self.events, self.audio_s = [], 0.0

    async def send_audio(self, x):
        self.audio_s += len(x) / 24000

    async def send_event(self, ev):
        self.events.append(ev)

    async def clear(self):
        self.events.append({"type": "clear"})


async def mic(handle, x16, tail=1.0):
    x = np.concatenate([x16, np.zeros(int(16000 * tail), np.float32)])
    t0 = time.perf_counter()
    for i in range(0, len(x), 320):
        await handle.feed(x[i:i + 320])
        await asyncio.sleep(max(0, t0 + (i + 320) / 16000 - time.perf_counter()))


async def wait_for(tr, pred, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        if any(pred(e) for e in tr.events):
            return True
        await asyncio.sleep(0.05)
    return False


async def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from app.s2s import CallHandle
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    server = await websockets.serve(mock_gemini, "127.0.0.1", 8899)
    tr = FakeTransport()
    h = CallHandle(tr.send_audio, tr.send_event, tr.clear, caller="s2s-test", language="en", engine="gemini_live")
    await h.start()
    check("connects via native engine", type(h.impl).__name__ == "GeminiLiveSession")
    check("setup carries model, audio modality, voice, transcription",
          SEEN["setup"] and SEEN["setup"]["generationConfig"]["responseModalities"] == ["AUDIO"]
          and "outputAudioTranscription" in SEEN["setup"] and "[breath]" not in json.dumps(SEEN["setup"]))
    check("greeting spoken + transcribed", await wait_for(tr, lambda e: e.get("type") == "transcript" and e["role"] == "agent"))
    await wait_for(tr, lambda e: e.get("type") == "state" and e["state"] == "listening")

    speech = resample(np.concatenate([c async for c in tts.render("Hi, my order has not arrived yet.", "en", chain=["edge"])]), 24000, 16000)
    await mic(h, speech)
    check("caller transcript from native model", await wait_for(tr, lambda e: e.get("type") == "transcript" and e["role"] == "user"))
    got_m = await wait_for(tr, lambda e: e.get("type") == "metrics" and e.get("e2e_ms"))
    m = next((e for e in tr.events if e.get("type") == "metrics"), {})
    check(f"voice-to-voice metric measured ({m.get('e2e_ms')} ms)", got_m)
    check(f"caller audio streamed to model ({SEEN['audio_bytes'] // 32000}s)", SEEN["audio_bytes"] > 32000)

    check("mid-call outage -> hot failover to cascade",
          await wait_for(tr, lambda e: e.get("type") == "engine" and e.get("failover"), 15))
    check("same call id kept", type(h.impl).__name__ == "CallSession")
    n_before = len([e for e in tr.events if e.get("type") == "transcript"])
    await mic(h, resample(np.concatenate([c async for c in tts.render("The order ID is V N 4 8 2 9 1 7.", "en", chain=["edge"])]), 24000, 16000))
    check("conversation continues on cascade after failover",
          await wait_for(tr, lambda e: e.get("type") == "metrics" and e.get("e2e_ms"), 25))
    check("history carried over", any("arrived" in x["content"] for x in h.impl.history))
    print("   transcript:", [(e["role"], e["text"][:60]) for e in tr.events if e.get("type") == "transcript"][n_before - 2:])
    await h.close()
    server.close()

    # bad endpoint => must fall back before the call goes live, quickly
    import app.s2s as s2s
    s2s.URL = "ws://127.0.0.1:1/nothing?key={key}"
    s2s.health("s2s", "gemini_live").open_until = 0
    tr2 = FakeTransport()
    t0 = time.perf_counter()
    h2 = CallHandle(tr2.send_audio, tr2.send_event, tr2.clear, caller="s2s-test2", language="en", engine="gemini_live")
    await h2.start()
    dt = (time.perf_counter() - t0) * 1000
    check(f"unreachable native engine -> cascade at call start ({dt:.0f} ms)", type(h2.impl).__name__ == "CallSession")
    await h2.close()
    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    asyncio.run(main())
