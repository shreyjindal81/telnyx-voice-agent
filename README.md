# Telnyx Voice Agent

Production-style AI phone calls with real PSTN delivery.

This project bridges Telnyx media streams to Deepgram Voice Agent and gives you a CLI for outbound and server-only calling workflows. It is designed for practical outreach use cases: confirmations, follow-ups, reminders, callbacks, and scripted operations calls.

## Skill-First Structure

- `skill/`: publishable skill package (`SKILL.md`, runtime, package files, env example)
- `docs/`: maintainer docs and deep technical references
- `scripts/`: operational helpers (including installer)

## Why This Is Useful

- Real phone calls over Telnyx, not browser-only demos
- Deepgram handles STT + LLM + TTS in one realtime pipeline
- Interruptible conversations (barge-in) that feel natural
- Full-call recording enabled by default
- Recording is downloaded locally, then deleted from Telnyx automatically
- Clear terminal trace of the full interaction (`User: ...`, `Agent: ...`, lifecycle logs)

## What Happens On Each Call

1. Call is created through Telnyx with recording from answer enabled.
2. Telnyx streams call audio to `/telnyx` WebSocket on this server.
3. Audio is bridged to Deepgram Voice Agent and responses are streamed back.
4. On call end, recording URL is resolved.
5. Recording is saved to `RECORDINGS_DIR` (default `./recordings`).
6. Remote recording is deleted from Telnyx after successful local save.

## Quick Start (5 Minutes)

### 1. Install

```bash
npm --prefix skill install
cp skill/.env.example .env
```

### 2. Fill required env vars in `.env`

- `TELNYX_API_KEY`
- `TELNYX_CONNECTION_ID`
- `TELNYX_PHONE_NUMBER`
- `DEEPGRAM_API_KEY`

Optional:
- `NGROK_AUTH_TOKEN` (if using `--ngrok`)
- `PUBLIC_WS_URL` (if not using `--ngrok`)
- `RECORDINGS_DIR` (defaults to `./recordings`)

### 3. Configure Telnyx TeXML application

- Set webhook URL to `https://<your-domain>/webhook`
- Enable bidirectional streaming
- Use this app connection as `TELNYX_CONNECTION_ID`

If using local development, easiest path is `--ngrok` and let this script create the tunnel.

### 4. Place a test call

```bash
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok \
  --task "Quick test call. Confirm audio is clear, then end politely."
```

## Cost Analysis (As of February 11, 2026)

### 1) Deepgram

- Signup includes **$200 free credit**.
- After free credit, this project is billed under Deepgram usage pricing.
- Voice Agent API PAYG list pricing includes tiers that start at **$0.0500/min** and Standard at **$0.0800/min** (depends on configuration/tier).
- Deepgram states Voice Agent pricing is based on websocket connection time.

Practical note for this repo:
- Budget for roughly **$0.05-$0.08 per call minute** for Deepgram usage depending on your selected tier/config.

### 2) Telnyx

- Voice API pricing shows a **$0.002/min Voice API fee** for outbound/inbound, plus SIP trunking charges.
- Telnyx notes TeXML uses the same **$0.002/min** Voice API fee.
- SIP trunking PAYG reference rates start around:
  - **$0.005/min outbound local**
  - **$0.0035/min inbound local**
  - **$0.015/min inbound toll-free**
- This repo also uses priced Telnyx optional features:
  - **Media Streaming over WebSockets: $0.0035/min**
  - **Call recording: $0.002/min**
- Phone number rental starts at about **$1/month** for local/toll-free numbers.

Practical note for this repo:
- A common US local outbound Telnyx subtotal for this exact setup is around **$0.0125/min** (`$0.002 Voice API + $0.005 SIP + $0.0035 media streaming + $0.002 recording`), before taxes/fees and destination-specific adjustments.
- Combined with Deepgram, total blended cost is often around **$0.0625-$0.0925/min** depending on Deepgram Voice Agent tier.

Pricing references:
- Deepgram pricing: <https://deepgram.com/pricing>
- Telnyx Voice API pricing: <https://telnyx.com/pricing/voice-api/>
- Telnyx SIP trunking pricing: <https://telnyx.com/pricing/elastic-sip/mc>
- Telnyx numbers pricing: <https://telnyx.com/pricing/numbers>

## Usage Recipes

### Basic outbound call

```bash
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok
```

### High-context production call

```bash
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok \
  --personality "Maya, a calm and professional clinic coordinator at Lakeside Health." \
  --task "Confirm tomorrow's appointment for Jordan Lee at two thirty PM, verify callback number, and offer reschedule slots if needed." \
  --greeting "Hi, this is Maya from Lakeside Health. Is now a good time for a quick appointment confirmation?"
```

### Choose model and voice

```bash
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok \
  --model "claude-sonnet-4-20250514" \
  --voice "elevenlabs/rachel"
```

### Debug mode

```bash
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok --debug
```

### Server-only mode (inbound or multi-call runtime)

```bash
node skill/telnyx_voice_agent.js --server-only --ngrok
```

### Custom ngrok domain

```bash
node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok --ngrok-domain your-domain.ngrok-free.dev
```

## Recording Behavior (Default)

- Recording starts from answer (`record-from-answer`)
- Terminal prints recording URL when available
- Recording file is persisted locally
- Remote recording is deleted from Telnyx after successful local download

Expected end-of-call logs:

- `[Recording] Download URL: ...`
- `[Recording] Saved locally: ...`
- `[Recording] Deleted from Telnyx: ...`

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TELNYX_API_KEY` | Yes | Telnyx API key |
| `TELNYX_CONNECTION_ID` | Yes | TeXML app connection ID |
| `TELNYX_PHONE_NUMBER` | Yes | Caller number |
| `DEEPGRAM_API_KEY` | Yes | Deepgram API key with Voice Agent access |
| `PUBLIC_WS_URL` | No | Public WSS URL to `/telnyx` if not using `--ngrok` |
| `SERVER_HOST` | No | Bind host (default `0.0.0.0`) |
| `SERVER_PORT` | No | Bind port (default `8765`) |
| `RECORDINGS_DIR` | No | Local recording output dir (default `./recordings`) |
| `NGROK_AUTH_TOKEN` | No | Needed for `--ngrok` |

## Supported Models

Use `--model`:

- Anthropic: `claude-3-5-haiku-latest`, `claude-sonnet-4-20250514`
- OpenAI: `gpt-5.1-chat-latest`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini` (default)

Run `node skill/telnyx_voice_agent.js --help` for current model and voice lists.

## Architecture

```text
Phone <-> Telnyx (mulaw 8kHz) <-> Node.js/ws bridge <-> Deepgram Voice Agent (linear16)
```

Core components:

- `CallSession`: per-call state
- `SessionManager`: session lifecycle
- `CallManager`: dial/hangup/recording retrieval/local save/Telnyx deletion
- `/telnyx`: media WebSocket endpoint
- `/webhook`: Telnyx event receiver

## Troubleshooting

### Call connects but no audio

- Use `--debug` and check for Deepgram auth errors.
- Verify `DEEPGRAM_API_KEY` has Voice Agent access.
- Ensure `PUBLIC_WS_URL` or ngrok URL points to a live instance of this process.

### `EADDRINUSE` or bind failures

- Set a different port:

```bash
SERVER_PORT=8788 node skill/telnyx_voice_agent.js --to "+1234567890" --ngrok
```

### Recording URL appears but no local file

- Check `RECORDINGS_DIR` is writable.
- Verify network access from runtime to recording URL.

### Local save works but Telnyx deletion fails

- Confirm `TELNYX_API_KEY` permissions allow deleting recordings.
- Check post-call warning/error logs for the recording ID and API response.

## Publish To ClawHub

1. Validate:

```bash
npm --prefix skill run check
```

2. Login:

```bash
npx clawhub login
```

3. Publish:

```bash
npx clawhub publish ./skill \
  --slug telnyx-voice-agent \
  --name "Telnyx Voice Agent" \
  --version 1.0.1 \
  --tags latest,voice,phone,telnyx,deepgram \
  --changelog "Default full-call recording, local persistence, post-download Telnyx cleanup, and docs refresh."
```

## Extending With Custom Tools

Add handlers in `TOOL_HANDLERS` and define matching function schemas in `createAgentSettings()` inside `skill/telnyx_voice_agent.js`.
