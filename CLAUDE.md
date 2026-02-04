# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered phone agent using Deepgram Voice Agent API (STT + LLM + TTS) and Telnyx for telephony. Single-file Python application that makes outbound calls with an AI voice agent.

## Commands

```bash
# Make outbound call (exits cleanly after call ends)
python telnyx_voice_agent.py --to "+1234567890" --ngrok

# With custom agent persona
python telnyx_voice_agent.py --to "+1234567890" --ngrok --prompt "You are..." --greeting "Hello!"

# With different LLM model (default: claude-3-5-haiku-latest)
python telnyx_voice_agent.py --to "+1234567890" --ngrok --model "gpt-4o-mini"

# With different TTS voice (default: deepgram/aura-2-thalia-en)
python telnyx_voice_agent.py --to "+1234567890" --ngrok --voice "elevenlabs/rachel"
python telnyx_voice_agent.py --to "+1234567890" --ngrok --voice "deepgram/aura-2-orion-en"

# Debug mode (verbose logging)
python telnyx_voice_agent.py --to "+1234567890" --ngrok --debug

# Server-only mode (stays running for inbound calls)
python telnyx_voice_agent.py --server-only --ngrok

# Custom ngrok domain (paid plan)
python telnyx_voice_agent.py --to "+1234567890" --ngrok --ngrok-domain your-domain.ngrok-free.dev
```

## Architecture

The system bridges async FastAPI with the synchronous Deepgram SDK using thread-safe queues:

```
Phone ←→ Telnyx (mulaw 8kHz) ←→ FastAPI ←→ Deepgram Thread (linear16 16kHz) ←→ Deepgram Voice Agent
```

**Threading model**: FastAPI runs async; Deepgram SDK is sync and runs in a dedicated thread. Communication happens via `queue.Queue`:
- `input_queue`: Telnyx audio → Deepgram
- `output_queue`: Deepgram audio → Telnyx

**Audio conversion**: Telnyx uses mulaw 8kHz (PCMU), Deepgram uses linear16 16kHz. Conversion via `audioop` (or `audioop-lts` for Python 3.13+).

**Cross-thread signaling**: Uses `asyncio.Event` + `call_soon_threadsafe()` pattern for immediate notification from sync Deepgram thread to async FastAPI context. Used for:
- **Barge-in**: `barge_in_async_notify` - clears output queue and sends `{"event": "clear"}` to Telnyx
- **Hangup**: `hangup_async_notify` - executes hangup immediately when tool is called

## Key Components in telnyx_voice_agent.py

- `CallSession`: Per-call state including thread-safe queues, async events, and output sample rate
- `deepgram_worker()`: Runs Deepgram connection in dedicated thread, handles message routing
- `SessionManager`: Creates/tracks/cleanup call sessions
- `CallManager`: Telnyx API wrapper for outbound calls and hangup
- `@app.websocket("/telnyx")`: Main WebSocket handler bridging Telnyx ↔ Deepgram
- `TOOL_HANDLERS`: Dict mapping function names to handlers (e.g., `get_secret`, `hangup`)
- `VALID_MODELS`: Dict mapping provider names to supported model IDs
- `VALID_VOICES`: Dict mapping TTS providers to voice configurations
- `create_think_provider()`: Factory function to create the appropriate LLM provider
- `create_speak_provider()`: Factory function to create the appropriate TTS provider
- `shutdown_event`: Signals clean exit after outbound call completes

## Supported LLM Models

Deepgram Voice Agent supports multiple LLM providers (managed by Deepgram, no API keys needed):

- **Anthropic**: claude-sonnet-4-5, claude-4-5-haiku-latest, claude-3-5-haiku-latest, claude-sonnet-4-20250514
- **OpenAI**: gpt-5.1-chat-latest, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini

Default: `claude-3-5-haiku-latest`

## Supported TTS Voices

Use the `--voice` flag with format `provider/voice-id`:

- **Deepgram** (16kHz output): aura-2-thalia-en, aura-2-luna-en, aura-2-stella-en, aura-2-athena-en, aura-2-hera-en, aura-2-orion-en, aura-2-arcas-en, aura-2-perseus-en, aura-2-angus-en, aura-2-orpheus-en, aura-2-helios-en, aura-2-zeus-en
- **ElevenLabs** (24kHz output): rachel, adam, bella, josh, elli, sam

Default: `deepgram/aura-2-thalia-en`

Audio conversion handles sample rate differences automatically (16kHz or 24kHz → 8kHz mulaw for Telnyx).

## Adding Custom Tools

1. Add handler to `TOOL_HANDLERS` dict
2. Add `AgentV1Function` definition in `create_agent_settings()`

## Environment Variables

Required: `TELNYX_API_KEY`, `TELNYX_CONNECTION_ID`, `TELNYX_PHONE_NUMBER`, `DEEPGRAM_API_KEY`

Optional: `NGROK_AUTH_TOKEN` (for `--ngrok`), `PUBLIC_WS_URL` (if not using ngrok), `SERVER_HOST`, `SERVER_PORT`
