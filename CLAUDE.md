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

- `CallSession`: Per-call state including thread-safe queues and async events
- `deepgram_worker()`: Runs Deepgram connection in dedicated thread, handles message routing
- `SessionManager`: Creates/tracks/cleanup call sessions
- `CallManager`: Telnyx API wrapper for outbound calls and hangup
- `@app.websocket("/telnyx")`: Main WebSocket handler bridging Telnyx ↔ Deepgram
- `TOOL_HANDLERS`: Dict mapping function names to handlers (e.g., `get_secret`, `hangup`)
- `shutdown_event`: Signals clean exit after outbound call completes

## Adding Custom Tools

1. Add handler to `TOOL_HANDLERS` dict
2. Add `AgentV1Function` definition in `create_agent_settings()`

## Environment Variables

Required: `TELNYX_API_KEY`, `TELNYX_CONNECTION_ID`, `TELNYX_PHONE_NUMBER`, `DEEPGRAM_API_KEY`

Optional: `NGROK_AUTH_TOKEN` (for `--ngrok`), `PUBLIC_WS_URL` (if not using ngrok), `SERVER_HOST`, `SERVER_PORT`
