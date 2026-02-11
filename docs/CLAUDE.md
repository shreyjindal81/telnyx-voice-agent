# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered phone agent using Deepgram Voice Agent API (STT + LLM + TTS) and Telnyx for telephony. Single-file Node.js application that makes outbound calls with an AI voice agent.

The script manages tunnels with `@ngrok/ngrok` when `--ngrok` is used.
Call recordings are enabled by default, saved locally, then deleted from Telnyx after successful download.

## Commands

```bash
# Install dependencies
npm --prefix skill install

# Make outbound call (exits cleanly after call ends)
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok

# With personality and task (recommended)
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok \
  --personality "Sarah, a friendly receptionist at Smile Dental" \
  --task "Confirm John's appointment for Tuesday at 3pm"

# With custom greeting
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok \
  --personality "Sales rep at Acme Corp" \
  --task "Follow up on quote #12345" \
  --greeting "Hi, is this John? This is Sarah from Acme Corp."

# With different LLM model (default: gpt-4o-mini)
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok --model "gpt-4o-mini"

# With different TTS voice (default: elevenlabs/rachel)
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok --voice "elevenlabs/adam"
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok --voice "deepgram/aura-2-orion-en"

# Debug mode (verbose logging)
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok --debug

# Server-only mode (stays running for inbound calls)
node skill/telnyx_voice_agent.js --server-only --ngrok

# Custom ngrok domain (paid plan)
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok --ngrok-domain your-domain.ngrok-free.dev
```

## Architecture

The system uses an event-driven Node.js WebSocket bridge:

```
Phone ←→ Telnyx (mulaw 8kHz) ←→ Node.js/ws bridge ←→ Deepgram Voice Agent (linear16 16kHz/24kHz)
```

- `CallSession`: Per-call state including output queue, barge-in flags, and Deepgram socket
- `SessionManager`: Creates/tracks/cleans up sessions
- `CallManager`: Telnyx REST wrapper for outbound calls, hangup, recording retrieval, local persistence, and Telnyx cleanup
- `/telnyx` WebSocket: Bidirectional Telnyx media stream handler
- `/webhook` HTTP endpoint: Telnyx webhook receiver
- `TOOL_HANDLERS`: Function-call handlers (`get_secret`, `hangup`)
- `createAgentSettings()`: Deepgram agent settings payload builder

Audio conversion is done in-process:
- Telnyx inbound: mulaw 8kHz → linear16 16kHz
- Deepgram outbound: linear16 16kHz/24kHz → mulaw 8kHz

## Supported LLM Models

- **Anthropic**: `claude-3-5-haiku-latest`, `claude-sonnet-4-20250514`
- **OpenAI**: `gpt-5.1-chat-latest`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`

Default: `gpt-4o-mini`

## Supported TTS Voices

Use `--voice` with format `provider/voice-id`:

- **Deepgram** (16kHz output): `aura-2-thalia-en`, `aura-2-luna-en`, `aura-2-stella-en`, `aura-2-athena-en`, `aura-2-hera-en`, `aura-2-orion-en`, `aura-2-arcas-en`, `aura-2-perseus-en`, `aura-2-angus-en`, `aura-2-orpheus-en`, `aura-2-helios-en`, `aura-2-zeus-en`
- **ElevenLabs** (24kHz output): `rachel`, `adam`, `bella`, `josh`, `elli`, `sam`

Default: `elevenlabs/rachel`

## CLI Arguments

- `--to`: Phone number to call (E.164 format)
- `--personality`: Agent personality description (who they are)
- `--task`: Task for the call (what to accomplish)
- `--greeting`: Custom opening greeting
- `--voice`: TTS voice (`provider/voice-id` format)
- `--model`: LLM model to use
- `--ngrok`: Auto-start ngrok tunnel
- `--ngrok-domain`: Custom ngrok domain
- `--debug`: Enable verbose logging
- `--server-only`: Run server only (no outbound call)

## Adding Custom Tools

1. Add a handler function to `TOOL_HANDLERS`
2. Add the function definition in `createAgentSettings()`

## Environment Variables

Required: `TELNYX_API_KEY`, `TELNYX_CONNECTION_ID`, `TELNYX_PHONE_NUMBER`, `DEEPGRAM_API_KEY`

Optional: `NGROK_AUTH_TOKEN` (for `--ngrok`), `PUBLIC_WS_URL` (if not using ngrok), `SERVER_HOST`, `SERVER_PORT`, `RECORDINGS_DIR` (defaults to `./recordings`)

## Operational Notes

- `--ngrok` requires a valid `NGROK_AUTH_TOKEN` and verified ngrok account.
- If calls connect but there is no audio, check Deepgram auth first (`401` means bad key or missing access).
- If port binding fails, set a different port (for example `SERVER_PORT=8788`).
- On each completed call, expect recording logs in this order: URL discovered, local file saved, Telnyx recording deleted.

## ClawHub Release Notes

- Do not auto-publish from agent flows.
- Before user-driven publish:
  - Run `npm --prefix skill run check`
  - Confirm `skill/SKILL.md` frontmatter and examples are current
  - Use explicit CLI publish flags (`--slug`, `--name`, `--version`, `--tags`, `--changelog`)
