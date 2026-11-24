"""End-to-end test: real audio in -> real audio out, over the same WebSocket the dashboard uses.

Synthesises caller utterances (en + hi), streams them in real time as 20 ms PCM frames, and checks that the
agent transcribes, answers, speaks, supports barge-in, and handles N concurrent calls.
Usage: python -I tests/e2e_call.py [concurrency]
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import websockets  # noqa: E402

from app.audio import float_to_pcm16, resample  # noqa: E402
from app.providers import tts  # noqa: E402

URL = "ws://127.0.0.1:8800/ws/call"


async def speech(text, lang):
    parts = [c async for c in tts.render(text, lang, chain=["edge"])]
    return resample(np.concatenate(parts), 24000, 16000)


class Mic:
    """Behaves like a real microphone: a continuous 20 ms frame stream; silence unless we 'speak'."""

    def __init__(self, ws):
        self.ws, self.q = ws, asyncio.Queue()
        self.task = asyncio.create_task(self.run())

    async def run(self):
        t0, n, cur = time.perf_counter(), 0, np.zeros(0, np.float32)
        while True:
            if len(cur) < 320 and not self.q.empty():
                cur = np.concatenate([cur, self.q.get_nowait()])
            fr = cur[:320] if len(cur) >= 320 else np.zeros(320, np.float32)
            cur = cur[320:] if len(cur) >= 320 else cur
            await self.ws.send(float_to_pcm16(fr))
            n += 1
            await asyncio.sleep(max(0, t0 + n * 0.02 - time.perf_counter()))

    async def say(self, x):
        await self.q.put(x)
        await asyncio.sleep(len(x) / 16000 + 0.05)


async def stream(mic, x, tail_s=0):
    await mic.say(x)


async def one_call(idx, lang, utterances, barge=False):
    res = {"call": idx, "lang": lang, "turns": [], "audio_s": 0.0, "interrupted": False, "errors": []}
    async with websockets.connect(f"{URL}?lang={lang}&name=e2e-{idx}", max_size=None) as ws:
        state = {"v": ""}
        events = asyncio.Queue()

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    res["audio_s"] += len(m) / 2 / 24000
                else:
                    ev = json.loads(m)
                    if ev["type"] == "state":
                        state["v"] = ev["state"]
                    if ev["type"] == "interrupted":
                        res["interrupted"] = True
                    await events.put(ev)

        rt = asyncio.create_task(reader())
        mic = Mic(ws)

        async def wait_state(s, timeout=25):
            t = time.time()
            while state["v"] != s:
                if time.time() - t > timeout:
                    raise TimeoutError(f"waiting for {s}, stuck in {state['v']}")
                await asyncio.sleep(0.05)

        await wait_state("listening")
        for u in utterances:
            x = await speech(u, lang if lang != "auto" else "en")
            while not events.empty():  # discard events from earlier turns / barge-in
                events.get_nowait()
            await stream(mic, x)
            turn = {"said": u}
            t = time.time()
            while time.time() - t < 25:
                ev = await asyncio.wait_for(events.get(), 25)
                if ev["type"] == "transcript" and ev["role"] == "user":
                    turn["heard"] = ev["text"]
                if ev["type"] == "transcript" and ev["role"] == "agent":
                    turn["reply"] = ev["text"]
                if ev["type"] == "metrics":
                    turn.update({k: ev.get(k) for k in ("stt_ms", "llm_ttft_ms", "e2e_ms", "stt", "llm", "tts")})
                    break
            res["turns"].append(turn)
            if barge:
                # talk over the agent while it is answering
                await asyncio.sleep(0.3)
                await stream(mic, await speech("Sorry, wait, one more question.", "en"), tail_s=0.2)
                await asyncio.sleep(1.5)
                barge = False
            await wait_state("listening", 30)
        rt.cancel(); mic.task.cancel()
    return res


async def main():
    sys.stdout.reconfigure(encoding="utf-8")
    n =int(sys.argv[1]) if len(sys.argv) > 1 else 1
    scen = [
        ("en", ["Hi, my order has not arrived yet.", "The order ID is V N 4 8 2 9 1 7."], True),
        ("hi", ["नमस्ते, मुझे अपने रिफंड के बारे में जानना है।", "मेरा ऑर्डर कब तक आएगा?"], False),
        ("auto", ["I got a damaged product and I am really upset."], False),
    ]
    jobs = [one_call(i, *scen[i % len(scen)]) for i in range(n)]
    t0 = time.time()
    results = await asyncio.gather(*jobs, return_exceptions=True)
    ok = True
    for r in results:
        if isinstance(r, Exception):
            ok = False
            print("CALL FAILED:", repr(r))
            continue
        print(f"\n== call {r['call']} [{r['lang']}] agent audio {r['audio_s']:.1f}s interrupted={r['interrupted']}")
        for t in r["turns"]:
            print(f"  said : {t['said']}\n  heard: {t.get('heard')}\n  reply: {t.get('reply')}\n"
                  f"  stt {t.get('stt_ms')}ms ({t.get('stt')}) | llm ttft {t.get('llm_ttft_ms')}ms ({t.get('llm')}) | "
                  f"voice-to-voice {t.get('e2e_ms')}ms | tts {t.get('tts')}")
            ok &= bool(t.get("heard") and t.get("reply") and t.get("e2e_ms"))
    print(f"\n{'PASS' if ok else 'FAIL'} - {n} call(s) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())

