<p align="center">
  <img src="assets/logo-dark.svg#gh-dark-mode-only" alt="Vaani" width="340">
  <img src="assets/logo-light.svg#gh-light-mode-only" alt="Vaani" width="340">
</p>

<p align="center"><b>Real-time speech-to-speech AI support agent · English + हिंदी</b></p>

---

A customer calls → speaks naturally (English, Hindi or Hinglish) → the AI agent **listens, understands and talks
back in a human voice** — with breaths, sighs, soft laughs and whispers — and stops instantly if the caller
interrupts. Many calls run in parallel; an operator dashboard shows every call live.

## Run it

```powershell
cd D:\voice
.\run.ps1
```
Open **http://localhost:8800** → *Start call* → talk. (Headphones give the cleanest barge-in.)

Test everything without a microphone (real synthesized audio, 3 parallel calls, EN + HI + auto, barge-in):
```powershell
D:\voice\.venv\Scripts\python.exe -I D:\voice\tests\e2e_call.py 3
```

## Make it fast for the pitch (2 minutes)
Everything works with **zero keys**, but the free keyless LLM is the slowest link (0.4–2.5s, one request at a time).
Put keys in `D:\voice\.env` (copy of `.env.example`) and restart:

| Key | What it fixes | Expected effect |
|---|---|---|
| `GROQ_API_KEY` (free, console.groq.com) | LLM **and** STT (Whisper-large-v3-turbo) | voice-to-voice drops to roughly **0.7–1.1s** |
| `DEEPGRAM_API_KEY` | nova-3 `language=multi` STT, best for Hinglish code-switching | STT ~200–300ms |
| `SARVAM_API_KEY` | Indian-language STT + Bulbul Hindi voice | most natural Hindi |
| `CARTESIA_API_KEY` / `ELEVENLABS_API_KEY` | premium voices, ~100–300ms first audio | `ELEVENLABS_MODEL=eleven_v3` = native [sighs]/[whispers] |

## Two conversation engines (switch on the Live Console, or per phone call with `?engine=`)

| | **Cascade** (default) | **Native S2S — Gemini Live** |
|---|---|---|
| How | STT → LLM → TTS, all streamed | one speech-native model hears audio, answers in audio |
| Strengths | every stage swappable with failover, best Hindi voices, explicit sigh/breath/whisper control, exact policy control | hears tone of voice, very natural prosody and turn-taking, fewest hops |
| Needs | nothing (free stack) — keys make it faster | `GEMINI_API_KEY` (free tier, aistudio.google.com) |

The native engine can't take the call down: no key, or Gemini unreachable within 2s, starts the call on the cascade.
If Gemini drops mid-call, the **same call** moves to the cascade engine with its conversation history, so the
caller just keeps talking to Aria. `tests/s2s_engine_test.py` covers all of this against a protocol-faithful
mock of Gemini Live.

Latency on the dashboard is measured from **when the caller stopped talking** to the agent's first sound, including
the end-of-turn silence wait, so a stopwatch on the call will agree with it.

## Architecture

```
 Browser (WebRTC mic, AEC/NS/AGC) ─┐                      ┌─ /ws/monitor → dashboard (live calls, KPIs)
 Twilio / Exotel (μ-law 8k)  ──────┤                      │
                                   ▼                      │
                    ┌──────────── CallSession (one per call, asyncio) ─────────────┐
   20ms PCM ─▶ VAD (WebRTC + adaptive noise floor) ─▶ endpoint (550ms) ─▶ STT chain
                    │        └─ barge-in: caller speech >240ms while agent talks → cancel + flush
                    ▼
               LLM chain (streaming) ─▶ chunker (first chunk ≥8 chars / 6 words) ─▶ TTS chain
                                                         per chunk, in parallel, played in order
                    ▼
               Expressive engine: [breath] [sigh] [laugh] [hmm] [pause] [whisper]…[/whisper]
               native tags where the vendor supports them, otherwise synthesized locally
                    ▼
               PCM 24k ─▶ caller            (filler "hmm/achha" masks a slow vendor at 900ms)
```

**Fallback chains** (reorder live in *Providers*; each provider has a circuit breaker with exponential cool-down,
so a dead or slow vendor is skipped instead of adding latency):

- **STT**: Deepgram nova-3 → Groq Whisper → Sarvam Saarika → Google (free, EN/HI auto-pick) → faster-whisper (offline)
- **LLM**: Groq → OpenAI → Gemini → Anthropic → OpenRouter → Pollinations (keyless) → scripted offline brain
- **TTS**: ElevenLabs → Cartesia → Sarvam → NVIDIA Magpie → **Kokoro-82M (local)** → Microsoft Edge neural → Windows SAPI

**Capacity-aware routing**: the local Kokoro voice is CPU-bound, so each call is assigned it only while
`KOKORO_MAX_CALLS` (default 2) isn't exceeded; extra calls get the cloud voice. The choice is fixed per call, so a
caller never hears the voice change. Greetings, fillers and repeated phrases are pre-rendered/cached.

**Telephony**: point a Twilio number's voice webhook to `https://<PUBLIC_HOST>/twilio/voice` (expose with ngrok or
Cloudflare Tunnel and set `PUBLIC_HOST`). Media Streams are bridged at `/twilio/stream`, including `clear` for barge-in.

## Files
- `app/session.py` — the per-call orchestrator (turn-taking, barge-in, streaming pipeline, metrics)
- `app/providers/{stt,llm,tts}.py` — vendor adapters + failover
- `app/expressive.py` — tag parser, procedural breath/sigh/laugh/mouth sounds, whisper converter
- `app/vad.py`, `app/audio.py`, `app/health.py`, `app/hub.py`, `app/main.py`
- `app/s2s.py` — native speech-to-speech engine (Gemini Live) + `CallHandle` engine router / hot failover
- `web/` — dashboard (Geist + Geist Mono, Noto Sans Devanagari for Hindi; ink + acid-lime + electric-violet)
- `data/calls/*.json` — every finished call: transcript + per-turn latency
- `samples/` — rendered expressive voice samples

## Research notes (Nov 2026)
- NVIDIA's current open TTS is **Magpie TTS Multilingual** (open weights, NIM on build.nvidia.com); NVIDIA also
  released NemotronLabs VoiceChat (full-duplex). Wired as an optional TTS adapter (`NVIDIA_API_KEY`, `pip install nvidia-riva-client`); Hindi is routed elsewhere.
- Expressive tags: ElevenLabs v3 supports `[sighs] [whispers] [laughs]` but has no real-time WebSocket; Cartesia
  Sonic-3 supports `[laughter]` plus emotion controls. Hence the local expressive engine works with *every* vendor.
- Hindi: Sarvam (Saarika STT / Bulbul TTS) is the Indian-language specialist; Deepgram nova-3 `language=multi`
  handles Hinglish code-switching.
- Frameworks reviewed: Pipecat, LiveKit Agents. This build is framework-free (FastAPI + asyncio) to keep latency
  and failure modes fully under our control; Pipecat/LiveKit can be adopted later for WebRTC/SIP scale-out.
