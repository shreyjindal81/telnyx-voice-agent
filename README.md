# Telnyx Voice Agent

AI-powered phone agent using Deepgram Voice Agent API and Telnyx for telephony.

## Features

- **Outbound calls**: Make AI-powered calls that exit cleanly when complete
- **Voice AI**: Deepgram Voice Agent API (STT + LLM + TTS in one)
- **Multiple LLMs**: Choose from Anthropic (Claude) or OpenAI (GPT) models
- **Barge-in**: Interrupt the agent mid-speech (instant response)
- **Instant hangup**: Call ends immediately when hangup tool is triggered
- **Tool support**: Client-side function execution (e.g., hangup, custom tools)
- **Customizable**: Pass personality, task, greeting, voice, and LLM model via CLI
- **Built-in ngrok**: Automatic tunnel setup with `--ngrok` flag

## Requirements

- Node.js 18+
- Telnyx account with a phone number and TeXML application
- Deepgram account with Voice Agent API access
- Verified ngrok account + `NGROK_AUTH_TOKEN` if using `--ngrok`

## Setup

1. Install dependencies:
```bash
npm install
```

2. Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

3. Configure Telnyx:
   - Create a TeXML application in the Telnyx portal
   - Set the webhook URL to your ngrok endpoint (e.g., `https://your-domain.ngrok-free.dev/webhook`)
   - Enable bidirectional streaming
   - Note your Connection ID

4. (Optional) Set up ngrok auth token for `--ngrok` flag:
```bash
# Add to .env or export directly
export NGROK_AUTH_TOKEN=your_ngrok_auth_token
```

If you are not using `--ngrok`, set `PUBLIC_WS_URL` in `.env` to a reachable WSS URL for `/telnyx`.

## Usage

### Make an outbound call (exits when call ends):
```bash
node telnyx_voice_agent.js --to "+1234567890" --ngrok
```

### Custom agent persona:
```bash
node telnyx_voice_agent.js --to "+1234567890" --ngrok \
  --personality "You are a helpful assistant for Acme Corp..." \
  --task "Follow up with Morgan about order three seven one nine and confirm delivery window."
```

### Use a different LLM model:
```bash
# Use OpenAI GPT-4o-mini (default)
node telnyx_voice_agent.js --to "+1234567890" --ngrok --model "gpt-4o-mini"

# Use Claude Sonnet 4
node telnyx_voice_agent.js --to "+1234567890" --ngrok --model "claude-sonnet-4-20250514"
```

### Debug mode:
```bash
node telnyx_voice_agent.js --to "+1234567890" --ngrok --debug
```

### Server-only mode (stays running for multiple/inbound calls):
```bash
node telnyx_voice_agent.js --server-only --ngrok
```

### With custom ngrok domain (paid plan):
```bash
node telnyx_voice_agent.js --to "+1234567890" --ngrok --ngrok-domain your-domain.ngrok-free.dev
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Phone     │◄───►│    Telnyx    │◄───►│  This Server │
│   (mulaw)    │     │   (WebSocket)│     │ (Node.js/ws) │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                           Event-driven bridge
                                                  │
                                           ┌──────▼───────┐
                                           │   Deepgram   │
                                           │ Voice Agent  │
                                           │  (linear16)  │
                                           └──────────────┘
```

- **Telnyx**: Handles phone connectivity, sends/receives mulaw 8kHz audio
- **Node.js + ws**: WebSocket server bridging Telnyx and Deepgram
- **Deepgram Voice Agent**: All-in-one STT + LLM + TTS (configurable model)
- **@ngrok/ngrok**: Built-in tunnel management for `--ngrok`

## Supported LLM Models

Use `--model` to select the LLM (managed by Deepgram):

| Provider | Models |
|----------|--------|
| **Anthropic** | claude-3-5-haiku-latest, claude-sonnet-4-20250514 |
| **OpenAI** | gpt-4o-mini (default), gpt-5.1-chat-latest, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o |

Run `node telnyx_voice_agent.js --help` to see all available models.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TELNYX_API_KEY` | Telnyx API key |
| `TELNYX_CONNECTION_ID` | TeXML application connection ID |
| `TELNYX_PHONE_NUMBER` | Your Telnyx phone number |
| `DEEPGRAM_API_KEY` | Deepgram API key |
| `PUBLIC_WS_URL` | Public WebSocket URL (used if not using --ngrok) |
| `SERVER_HOST` | Server host (default: 0.0.0.0) |
| `SERVER_PORT` | Server port (default: 8765) |
| `NGROK_AUTH_TOKEN` | ngrok auth token (optional, for --ngrok flag) |

## Troubleshooting

- **Call rings but no audio**:
  - Verify `DEEPGRAM_API_KEY` is valid and has Voice Agent access.
  - Run with `--debug` and check for Deepgram `401` errors.
- **ngrok fails to start**:
  - Ensure `NGROK_AUTH_TOKEN` is set.
  - Ensure your ngrok account is verified.
- **Port bind error (`EADDRINUSE`/`EPERM`)**:
  - Change `SERVER_PORT` (for example `SERVER_PORT=8788`).

## ClawHub Prep

Manual publish workflow (recommended):

1. Validate package locally:
```bash
npm run check
```
2. Ensure `SKILL.md` metadata and commands match current behavior.
3. Login once:
```bash
npx clawhub login
```
4. Publish manually with explicit version/tags:
```bash
npx clawhub publish . \
  --slug telnyx-voice-agent \
  --name "Telnyx Voice Agent" \
  --version 1.0.0 \
  --tags latest,voice,phone,telnyx,deepgram \
  --changelog "Initial JavaScript release with Telnyx + Deepgram voice calling."
```

Notes:
- This project is prepared for manual publish; it does not auto-publish.
- Bump `--version` and changelog text for each new release.

## Adding Custom Tools

Edit `TOOL_HANDLERS` and `createAgentSettings()` in `telnyx_voice_agent.js`:

```js
async function myToolHandler(parameters) {
  // Your logic here
  return "Result";
}

TOOL_HANDLERS.my_tool = myToolHandler;
```

Then add the function definition in `createAgentSettings()`.
