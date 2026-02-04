# Telnyx Voice Agent

AI-powered phone agent using Deepgram Voice Agent API and Telnyx for telephony.

## Features

- **Outbound calls**: Make AI-powered calls that exit cleanly when complete
- **Voice AI**: Deepgram Voice Agent API (STT + LLM + TTS in one)
- **Multiple LLMs**: Choose from Anthropic (Claude) or OpenAI (GPT) models
- **Barge-in**: Interrupt the agent mid-speech (instant response)
- **Instant hangup**: Call ends immediately when hangup tool is triggered
- **Tool support**: Client-side function execution (e.g., hangup, custom tools)
- **Customizable**: Pass custom prompts, greetings, and LLM model via CLI
- **Built-in ngrok**: Automatic tunnel setup with `--ngrok` flag

## Requirements

- Python 3.10+
- Telnyx account with a phone number and TeXML application
- Deepgram account with Voice Agent API access
- ngrok (or similar) for exposing local server

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
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

## Usage

### Make an outbound call (exits when call ends):
```bash
python telnyx_voice_agent.py --to "+1234567890" --ngrok
```

### Custom agent persona:
```bash
python telnyx_voice_agent.py --to "+1234567890" --ngrok \
  --prompt "You are a helpful assistant for Acme Corp..." \
  --greeting "Hello! Thanks for calling Acme Corp. How can I help?"
```

### Use a different LLM model:
```bash
# Use OpenAI GPT-4o-mini instead of default Claude
python telnyx_voice_agent.py --to "+1234567890" --ngrok --model "gpt-4o-mini"

# Use Claude Sonnet 4.5
python telnyx_voice_agent.py --to "+1234567890" --ngrok --model "claude-sonnet-4-5"
```

### Debug mode:
```bash
python telnyx_voice_agent.py --to "+1234567890" --ngrok --debug
```

### Server-only mode (stays running for multiple/inbound calls):
```bash
python telnyx_voice_agent.py --server-only --ngrok
```

### With custom ngrok domain (paid plan):
```bash
python telnyx_voice_agent.py --to "+1234567890" --ngrok --ngrok-domain your-domain.ngrok-free.dev
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Phone     │◄───►│    Telnyx    │◄───►│  This Server │
│   (mulaw)    │     │   (WebSocket)│     │  (FastAPI)   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                          Thread-safe queues
                                                 │
                                          ┌──────▼───────┐
                                          │   Deepgram   │
                                          │ Voice Agent  │
                                          │  (linear16)  │
                                          └──────────────┘
```

- **Telnyx**: Handles phone connectivity, sends/receives mulaw 8kHz audio
- **FastAPI**: WebSocket server bridging Telnyx and Deepgram
- **Deepgram Voice Agent**: All-in-one STT + LLM + TTS (configurable model)

## Supported LLM Models

Use `--model` to select the LLM (managed by Deepgram):

| Provider | Models |
|----------|--------|
| **Anthropic** | claude-sonnet-4-5, claude-4-5-haiku-latest, claude-3-5-haiku-latest (default), claude-sonnet-4-20250514 |
| **OpenAI** | gpt-5.1-chat-latest, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini |

Run `python telnyx_voice_agent.py --help` to see all available models.

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

## Adding Custom Tools

Edit the `TOOL_HANDLERS` dict and `create_agent_settings()` in `telnyx_voice_agent.py`:

```python
def my_tool_handler(parameters: dict) -> str:
    # Your logic here
    return "Result"

TOOL_HANDLERS = {
    "my_tool": my_tool_handler,
    # ...
}
```

Then add the function definition in `create_agent_settings()`.
