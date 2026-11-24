"""HTTP + WebSocket entrypoint.

  /                 dashboard
  /ws/call          browser call (binary PCM16 16k in, PCM16 24k out, JSON events)
  /ws/monitor       live dashboard feed
  /twilio/voice     TwiML webhook  -> /twilio/stream (G.711 mu-law 8k media stream)
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import health
from .audio import float_to_pcm16, mulaw_decode, mulaw_encode, pcm16_to_float, resample, to_wav
from .config import KEYS, MAX_CONCURRENT_CALLS, OUT_RATE, PUBLIC_HOST, ROOT, SETTINGS
from .hub import HUB
from .providers import llm, stt, tts
from .s2s import CallHandle
from .session import warm_caches

log = logging.getLogger("voice")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
WEB = ROOT / "web"

app = FastAPI(title="Vaani Voice AI")
app.mount("/static", StaticFiles(directory=WEB), name="static")


def register_providers():
    for n in SETTINGS["llm_chain"]:
        health.health("llm", n, llm.configured(n))
    for n in SETTINGS["stt_chain"]:
        health.health("stt", n, stt.configured(n))
    for n in SETTINGS["tts_chain"]:
        health.health("tts", n, tts.configured(n))
    health.health("s2s", "gemini_live", bool(KEYS["gemini"]))


@app.on_event("startup")
async def startup():
    register_providers()
    asyncio.create_task(_warm())


async def _warm():
    t = time.perf_counter()
    try:
        await warm_caches()
        log.info("caches warm in %.0f ms", (time.perf_counter() - t) * 1000)
    except Exception as e:  # noqa: BLE001
        log.warning("cache warm failed: %s", e)


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/state")
async def state():
    return {"settings": SETTINGS, "providers": health.snapshot(), "stats": HUB.stats(),
            "calls": [c.summary() for c in HUB.calls.values()], "history": list(HUB.history)[-50:][::-1],
            "keys": {k: bool(v) for k, v in KEYS.items()}, "max_calls": MAX_CONCURRENT_CALLS}


@app.post("/api/settings")
async def update_settings(req: Request):
    body = await req.json()
    for k, v in body.items():
        if k not in SETTINGS:
            continue
        cur = SETTINGS[k]
        try:
            if isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            elif isinstance(cur, list) and isinstance(v, str):
                v = [s.strip() for s in v.split(",") if s.strip()]
        except (TypeError, ValueError):
            continue
        SETTINGS[k] = v
    register_providers()
    for c in HUB.calls.values():
        c.vad.set_endpoint(SETTINGS["endpoint_ms"])
    asyncio.create_task(_warm())
    return {"ok": True, "settings": SETTINGS}


@app.post("/api/calls/{cid}/say")
async def op_say(cid: str, req: Request):
    c = HUB.calls.get(cid)
    if not c:
        return JSONResponse({"error": "no such call"}, 404)
    await c.say((await req.json())["text"])
    return {"ok": True}


@app.post("/api/calls/{cid}/hangup")
async def op_hangup(cid: str):
    c = HUB.calls.get(cid)
    if c:
        await c.event({"type": "hangup"})
        await c.close()
    return {"ok": True}


@app.get("/api/calls/log")
async def call_log():
    from .hub import LOG_DIR
    out = []
    for f in sorted(LOG_DIR.glob("*.json"))[-40:][::-1]:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            pass
    return out


@app.get("/api/preview")
async def preview(text: str, lang: str = "en"):
    chunks = [c async for c in tts.render(text, lang)]
    x = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    return Response(to_wav(x, OUT_RATE), media_type="audio/wav")


@app.get("/api/voices")
async def voices():
    import edge_tts
    vs = await edge_tts.list_voices()
    keep = [v for v in vs if v["Locale"] in ("en-IN", "hi-IN") or "Multilingual" in v["ShortName"]]
    out = [{"id": v["ShortName"], "locale": v["Locale"], "gender": v["Gender"], "engine": "edge"} for v in keep]
    try:
        for v in tts._kokoro_load().get_voices():
            if v[:2] in ("af", "am", "bf", "bm", "hf", "hm"):
                out.append({"id": v, "locale": "hi-IN" if v[0] == "h" else "en", "engine": "kokoro",
                            "gender": "Female" if v[1] == "f" else "Male"})
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------- browser calls ----------------
@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    await ws.accept()
    if len(HUB.calls) >= MAX_CONCURRENT_CALLS:
        await ws.send_json({"type": "error", "message": "All lines busy"})
        await ws.close()
        return
    lang = ws.query_params.get("lang") or None
    send_lock = asyncio.Lock()

    async def send_audio(x: np.ndarray):
        step = OUT_RATE // 5  # 200ms frames keep the socket responsive to interrupts
        for i in range(0, len(x), step):
            async with send_lock:
                await ws.send_bytes(float_to_pcm16(x[i:i + step]))

    async def send_event(ev: dict):
        async with send_lock:
            await ws.send_text(json.dumps(ev, ensure_ascii=False))

    async def clear():
        await send_event({"type": "clear"})

    call = CallHandle(send_audio, send_event, clear, channel="web", caller=ws.query_params.get("name", "Browser"),
                      language=lang, engine=ws.query_params.get("engine") or None)
    await call.start()
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await call.feed(pcm16_to_float(msg["bytes"]))
            elif msg.get("text"):
                d = json.loads(msg["text"])
                if d.get("type") == "hangup":
                    break
                if d.get("type") == "say":
                    await call.say(d["text"])
                if d.get("type") == "text":   # typed user turn (handy for silent demos)
                    await call.user_text(d["text"])
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("call %s crashed: %s", call.id, e)
    finally:
        await call.close()


# ---------------- dashboard monitor ----------------
@app.websocket("/ws/monitor")
async def ws_monitor(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue()
    HUB.monitors.add(q)

    async def ticker():
        while True:
            await asyncio.sleep(1.0)
            q.put_nowait({"type": "tick", "stats": HUB.stats(), "providers": health.snapshot(),
                          "calls": [c.summary() for c in HUB.calls.values()]})

    t = asyncio.create_task(ticker())
    try:
        while True:
            ev = await q.get()
            await ws.send_text(json.dumps(ev, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        pass
    finally:
        t.cancel()
        HUB.monitors.discard(q)


# ---------------- telephony: Twilio Media Streams ----------------
@app.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(req: Request):
    host = PUBLIC_HOST or req.headers.get("host")
    lang = "".join(c for c in req.query_params.get("lang", "") if c.isalpha())[:8]
    engine = "".join(c for c in req.query_params.get("engine", "") if c.isalnum() or c == "_")[:16]
    twiml = (f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
             f'<Stream url="wss://{host}/twilio/stream"><Parameter name="lang" value="{lang}"/>'
             f'<Parameter name="engine" value="{engine}"/></Stream>'
             f'</Connect></Response>')
    return Response(twiml, media_type="application/xml")


@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    sid = {"v": None}
    call: CallHandle | None = None
    lock = asyncio.Lock()

    async def send_audio(x: np.ndarray):
        if not sid["v"]:
            return
        payload = base64.b64encode(mulaw_encode(resample(x, OUT_RATE, 8000))).decode()
        async with lock:
            await ws.send_text(json.dumps({"event": "media", "streamSid": sid["v"], "media": {"payload": payload}}))

    async def send_event(ev: dict):
        pass

    async def clear():
        if sid["v"]:
            async with lock:
                await ws.send_text(json.dumps({"event": "clear", "streamSid": sid["v"]}))

    try:
        while True:
            d = json.loads(await ws.receive_text())
            ev = d.get("event")
            if ev == "start":
                sid["v"] = d["start"]["streamSid"]
                params = d["start"].get("customParameters", {})
                call = CallHandle(send_audio, send_event, clear, channel="phone",
                                  caller=d["start"].get("callSid", "twilio")[:12], language=params.get("lang") or None,
                                  engine=params.get("engine") or None)
                await call.start()
            elif ev == "media" and call:
                x = mulaw_decode(base64.b64decode(d["media"]["payload"]))
                await call.feed(resample(x, 8000, 16000))
            elif ev == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if call:
            await call.close()
