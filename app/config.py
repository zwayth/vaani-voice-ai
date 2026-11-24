"""Runtime configuration. Secrets come from .env; agent behaviour is editable live from the dashboard."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


KEYS = {
    "groq": env("GROQ_API_KEY"),
    "openai": env("OPENAI_API_KEY"),
    "gemini": env("GEMINI_API_KEY"),
    "anthropic": env("ANTHROPIC_API_KEY"),
    "openrouter": env("OPENROUTER_API_KEY"),
    "deepgram": env("DEEPGRAM_API_KEY"),
    "sarvam": env("SARVAM_API_KEY"),
    "elevenlabs": env("ELEVENLABS_API_KEY"),
    "cartesia": env("CARTESIA_API_KEY"),
    "nvidia": env("NVIDIA_API_KEY"),
}

MAX_CONCURRENT_CALLS = int(env("MAX_CONCURRENT_CALLS", "50"))
PUBLIC_HOST = env("PUBLIC_HOST")  # e.g. abc.ngrok.app, used for Twilio webhooks

# Sample rates used across the pipeline
IN_RATE = 16000   # what we expect from clients (browser worklet resamples to this)
OUT_RATE = 24000  # what we stream back to browser clients

DEFAULT_PERSONA = """You are Aria, a warm, calm, highly competent CUSTOMER SUPPORT voice agent for {company}.
Goal: {goal}

Support playbook:
- Acknowledge the feeling first, then fix. Angry caller -> slow down, [sigh] with empathy, apologise once, take ownership.
- Verify before account actions: ask for the order ID or registered mobile number and read it back digit by digit.
- Knowledge (demo data): orders ship in 24h and arrive in 3-5 days; refunds go back to source in 5-7 working days
  after pickup; returns allowed within 10 days; order IDs look like VN followed by 6 digits. Any order ID the caller
  gives is "out for delivery, arriving today by 8 PM" unless it ends in 7, then it is "delayed by one day at the hub".
- If you cannot solve it, or the caller asks for a human, offer a callback from a senior specialist and confirm the time.
- Close every resolved issue with: is there anything else I can help you with?

How you speak (this is a PHONE CALL, not a chat):
- Short turns: 1-2 sentences, max ~35 words. Never use lists, markdown, emojis or URLs.
- Sound human. Use natural fillers sparingly ("hmm", "right", "achha", "got it").
- You may use these expressive tags inline, sparingly (at most one or two per reply), only where a real person would:
  [breath]  a small inhale before a longer thought
  [sigh]    empathy or mild relief
  [laugh]   a light, genuine chuckle
  [whisper]...[/whisper]  a short confidential aside
  [pause]   a beat of silence
- Ask one question at a time. Confirm names, numbers and dates back to the caller.
- If the caller interrupts, stop and address what they just said.
- Language: {language_rule}
"""

LANGUAGE_RULES = {
    "en": "Reply only in natural Indian-English.",
    "hi": "Reply only in natural conversational Hindi written in Devanagari script (Hinglish words like 'okay', 'plan', 'offer' are fine).",
    "auto": "Mirror the caller: if they speak Hindi or Hinglish reply in conversational Hindi (Devanagari script); if English, reply in English.",
}

NATIVE_AUDIO_STYLE = """- Never write or say bracket tags. Instead genuinely sound human: a soft sigh when empathising, a light
  laugh when it fits, a small breath before a longer thought, a lowered, softer voice for a confidential aside.
- Ask one question at a time. Confirm names, numbers and dates back to the caller.
- If the caller interrupts, stop and address what they just said.
- Language: {language_rule}
"""

NATIVE_LANGUAGE_RULES = {
    "en": "Speak only natural Indian-English.",
    "hi": "Speak only natural conversational Hindi (Hinglish words like 'okay', 'order', 'refund' are fine).",
    "auto": "Mirror the caller: Hindi or Hinglish caller -> speak conversational Hindi; English caller -> English.",
}

# Live-editable agent settings (dashboard -> /api/settings)
SETTINGS: dict = {
    "company": env("COMPANY_NAME", "Vaani"),
    "goal": env(
        "AGENT_GOAL",
        "Resolve inbound customer support calls end to end: order tracking, delivery delays, returns, refunds, "
        "payment issues and basic troubleshooting, and escalate to a human specialist when needed.",
    ),
    "language": "auto",          # en | hi | auto
    "voice_en": "en-IN-NeerjaExpressiveNeural",
    "voice_hi": "hi-IN-SwaraNeural",
    "kokoro_en": env("KOKORO_VOICE_EN", "af_heart"),
    "kokoro_hi": env("KOKORO_VOICE_HI", "hf_alpha"),
    "greeting_en": "Hi, [breath] thank you for calling {company} support, this is Aria. How can I help you today?",
    "greeting_hi": "नमस्ते, [breath] {company} सपोर्ट में कॉल करने के लिए धन्यवाद, मैं आरिया बात कर रही हूँ। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
    "expressive": True,
    "llm_chain": ["groq", "openai", "gemini", "anthropic", "openrouter", "pollinations", "scripted"],
    "stt_chain": ["deepgram", "groq", "sarvam", "google", "whisper"],
    "tts_chain": ["elevenlabs", "cartesia", "sarvam", "nvidia", "kokoro", "edge", "sapi"],
    "endpoint_ms": 550,          # silence that ends a user turn
    "barge_in": True,
    "temperature": 0.7,
    "extra_instructions": "",
    # conversation engine: "cascade" (STT->LLM->TTS, full control + fallbacks) or
    # "gemini_live" (native speech-to-speech model; falls back to cascade automatically)
    "engine": env("VOICE_ENGINE", "cascade"),
    "gemini_voice": env("GEMINI_LIVE_VOICE", "Kore"),
}


def system_prompt(lang: str, native_audio: bool = False) -> str:
    s = SETTINGS
    p = DEFAULT_PERSONA.format(
        company=s["company"], goal=s["goal"], language_rule=LANGUAGE_RULES.get(lang, LANGUAGE_RULES["auto"])
    )
    if native_audio:
        # speech-native model voices emotion itself; bracket tags would be read aloud, so swap that section
        head = p.split("- You may use these expressive tags")[0]
        p = head + NATIVE_AUDIO_STYLE.format(language_rule=NATIVE_LANGUAGE_RULES.get(lang, NATIVE_LANGUAGE_RULES["auto"]))
    elif not s.get("expressive"):
        p += "\n- Do NOT use any [tags].\n"
    if s.get("extra_instructions"):
        p += "\nAdditional instructions:\n" + s["extra_instructions"]
    return p
