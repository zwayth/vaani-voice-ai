"""Streaming LLM with an ordered fallback chain.

A provider only "wins" once it delivers its first token inside FIRST_TOKEN_TIMEOUT; otherwise we fail over
to the next one, so a slow or dead vendor can never hang a live call.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import AsyncIterator

import httpx

from ..config import KEYS, env
from ..health import health

FIRST_TOKEN_TIMEOUT = float(env("LLM_FIRST_TOKEN_TIMEOUT", "4.0"))

OPENAI_COMPAT = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "groq", env("GROQ_MODEL", "qwen/qwen3.8-27b")),
    "openai": ("https://api.openai.com/v1/chat/completions", "openai", env("OPENAI_MODEL", "gpt-4.1-mini")),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "gemini",
               env("GEMINI_MODEL", "gemini-2.5-flash-lite")),
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "openrouter",
                   env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")),
    "pollinations": ("https://text.pollinations.ai/openai", None, "openai-fast"),
}

_client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), http2=False,
                            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))


def configured(name: str) -> bool:
    if name in ("pollinations", "scripted"):
        return True
    if name == "anthropic":
        return bool(KEYS["anthropic"])
    spec = OPENAI_COMPAT.get(name)
    return bool(spec and KEYS.get(spec[1] or "", ""))


async def _openai_compat(name: str, messages: list, temperature: float) -> AsyncIterator[str]:
    url, keyname, model = OPENAI_COMPAT[name]
    headers = {"Content-Type": "application/json"}
    if keyname:
        headers["Authorization"] = f"Bearer {KEYS[keyname]}"
    body = {"model": model, "messages": messages, "stream": True, "temperature": temperature, "max_tokens": 160}
    if name == "pollinations":  # keyless reasoning model: token caps starve the visible answer
        body = {"model": model, "messages": messages, "stream": True}
    elif "qwen3" in model:  # thinking off: a phone call can't wait for hidden reasoning
        body["reasoning_effort"] = "none"
    elif "gpt-oss" in model:
        body.update({"reasoning_effort": "low", "include_reasoning": False, "max_tokens": 400})
    async with _client.stream("POST", url, json=body, headers=headers) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{name} HTTP {r.status_code}: {(await r.aread())[:200]!r}")
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                continue
            ch = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if ch:
                yield ch


async def _anthropic(messages: list, temperature: float) -> AsyncIterator[str]:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    msgs = [m for m in messages if m["role"] != "system"]
    body = {"model": env("ANTHROPIC_MODEL", "claude-haiku-5-5"), "system": system, "messages": msgs,
            "max_tokens": 160, "temperature": temperature, "stream": True}
    headers = {"x-api-key": KEYS["anthropic"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
    async with _client.stream("POST", "https://api.anthropic.com/v1/messages", json=body, headers=headers) as r:
        if r.status_code != 200:
            raise RuntimeError(f"anthropic HTTP {r.status_code}: {(await r.aread())[:200]!r}")
        async for line in r.aiter_lines():
            if line.startswith("data:"):
                d = json.loads(line[5:])
                if d.get("type") == "content_block_delta":
                    t = d.get("delta", {}).get("text")
                    if t:
                        yield t


# ---------------- deterministic last-resort brain (works fully offline) ----------------

def _scripted(messages: list, lang: str) -> str:
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "").lower()
    hi = lang == "hi"
    if re.search(r"\d{4,}", last):
        return ("[breath] शुक्रिया, मैंने चेक किया - आपका ऑर्डर आउट फॉर डिलीवरी है और आज रात 8 बजे तक पहुँच जाएगा। और कुछ मदद कर सकती हूँ?"
                if hi else "[breath] Thanks, I've checked - your order is out for delivery and will reach you by 8 PM today. "
                "Anything else I can help with?")
    rules = [
        (("angry", "worst", "frustrat", "bakwas", "बकवास", "गुस्सा", "pareshan", "परेशान", "terrible", "useless"),
         "[sigh] मुझे सच में खेद है कि आपको ये परेशानी हुई। मैं अभी इसे ठीक करती हूँ - क्या आप अपना ऑर्डर आईडी बता सकते हैं?",
         "[sigh] I'm really sorry you've had to deal with this. Let me fix it right now - could you share your order ID?"),
        (("order", "track", "delivery", "deliver", "ऑर्डर", "डिलीवरी", "kab aayega", "कब आएगा", "where is"),
         "बिलकुल, मैं चेक करती हूँ। [breath] क्या आप अपना ऑर्डर आईडी या रजिस्टर्ड मोबाइल नंबर बता सकते हैं?",
         "Of course, let me check that for you. [breath] Could you share your order ID or registered mobile number?"),
        (("refund", "money", "paisa", "पैसा", "रिफंड", "payment", "charged", "पेमेंट"),
         "समझ गई। रिफंड पिकअप के बाद 5 से 7 वर्किंग दिनों में आपके ओरिजिनल पेमेंट मेथड में आ जाता है। क्या मैं आपका रिफंड स्टेटस चेक करूँ?",
         "Understood. Refunds reach your original payment method within 5 to 7 working days after pickup. Shall I check yours?"),
        (("return", "replace", "wapas", "वापस", "रिटर्न", "damaged", "broken", "खराब", "wrong"),
         "[sigh] ओह, ये तो ठीक नहीं हुआ। आप 10 दिनों के अंदर रिटर्न कर सकते हैं - मैं अभी पिकअप बुक कर दूँ?",
         "[sigh] Oh, that's not okay at all. You can return it within 10 days - shall I book a pickup right now?"),
        (("human", "agent", "manager", "person", "senior", "insaan", "इंसान", "मैनेजर"),
         "बिलकुल। मैं एक सीनियर स्पेशलिस्ट से आपको कॉलबैक अरेंज करती हूँ - आपके लिए कौन सा समय ठीक रहेगा?",
         "Absolutely. I'll arrange a callback from a senior specialist - what time works best for you?"),
        (("thank", "thanks", "dhanyavad", "धन्यवाद", "shukriya", "शुक्रिया", "bye", "that's all", "bas"),
         "आपका बहुत-बहुत धन्यवाद! [laugh] आपका दिन शुभ हो।", "My pleasure! [laugh] Have a lovely day."),
    ]
    for keys, h, e in rules:
        if any(k in last for k in keys):
            return h if hi else e
    return ("[breath] अच्छा, समझ गई। क्या आप थोड़ा और बता सकते हैं कि दिक्कत क्या आ रही है?" if hi
            else "[breath] Got it. Could you tell me a little more about what's going wrong?")



_POLL = {"busy": False}


async def _try(name: str, h, messages: list, temperature: float):
    """Return (first_token, generator) or None if the provider failed / was too slow."""
    t0 = time.perf_counter()
    gen = _anthropic(messages, temperature) if name == "anthropic" else _openai_compat(name, messages, temperature)
    try:
        first = await asyncio.wait_for(gen.__anext__(),
                                       timeout=2.5 if name == "pollinations" else FIRST_TOKEN_TIMEOUT)
    except StopAsyncIteration:
        h.fail("empty response")
        return None
    except Exception as e:  # noqa: BLE001 - any vendor failure => next provider
        h.fail(f"{type(e).__name__}: {e}")
        try:
            await gen.aclose()
        except Exception:  # noqa: BLE001
            pass
        return None
    h.ok((time.perf_counter() - t0) * 1000)
    return first, gen


async def stream_reply(messages: list, chain: list[str], temperature: float, lang: str,
                       on_provider=None) -> AsyncIterator[str]:
    """Yield text tokens from the first provider in `chain` that responds in time."""
    for name in chain:
        h = health("llm", name, configured(name))
        if not h.available:
            continue
        if name == "scripted":
            if on_provider:
                on_provider(name)
            h.ok(1)
            for word in _scripted(messages, lang).split(" "):
                yield word + " "
            return
        if name == "pollinations":
            if _POLL["busy"]:  # keyless tier serves one request per IP: never queue a live call behind it
                continue
            _POLL["busy"] = True
        try:
            got = await _try(name, h, messages, temperature)
            if got is None:
                continue
            first, gen = got
            if on_provider:
                on_provider(name)
            yield first
            try:
                async for tok in gen:
                    yield tok
            except Exception as e:  # noqa: BLE001 - mid-stream failure: keep what was said
                h.fail(f"mid-stream {type(e).__name__}")
            return
        finally:
            if name == "pollinations":
                _POLL["busy"] = False
    # everything failed: never leave the caller in silence
    yield "Sorry, could you say that once more?" if lang != "hi" else "माफ़ कीजिए, क्या आप एक बार फिर बोल सकते हैं?"