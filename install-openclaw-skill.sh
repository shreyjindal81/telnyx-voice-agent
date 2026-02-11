#!/bin/bash
# Install Telnyx Voice Agent skill for OpenClaw
# Usage: ./install-openclaw-skill.sh

set -e

SKILL_NAME="telnyx-voice-agent"
SKILL_DIR="${HOME}/.openclaw/workspace/skills/${SKILL_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing ${SKILL_NAME} skill for OpenClaw..."

# Create skill directory
mkdir -p "$SKILL_DIR"

# Copy skill files
cp "${SCRIPT_DIR}/telnyx_voice_agent.js" "$SKILL_DIR/"
cp "${SCRIPT_DIR}/package.json" "$SKILL_DIR/"
cp "${SCRIPT_DIR}/.env.example" "$SKILL_DIR/"

# Create SKILL.md
cat > "$SKILL_DIR/SKILL.md" << 'EOM'
---
name: telnyx-voice-agent
description: Run AI-powered outbound phone calls with Telnyx + Deepgram Voice Agent. Use when the user wants real phone outreach (follow-ups, confirmations, reminders, callbacks) with configurable personality, task context, model, and voice.
metadata: {"openclaw": {"emoji": "📞", "requires": {"bins": ["node", "npm"], "env": ["TELNYX_API_KEY", "TELNYX_CONNECTION_ID", "TELNYX_PHONE_NUMBER", "DEEPGRAM_API_KEY"]}, "primaryEnv": "TELNYX_API_KEY", "os": ["darwin", "linux"]}}
---

# Telnyx Voice Agent

Make AI-powered outbound phone calls with customizable personas, voices, and behaviors.

## Prerequisites

Install JavaScript dependencies (one-time):
```bash
npm --prefix {baseDir} install
```

If using `--ngrok`, `NGROK_AUTH_TOKEN` must be configured and the ngrok account must be verified.
If not using `--ngrok`, set `PUBLIC_WS_URL` to a reachable `wss://.../telnyx` endpoint.

## Usage

When the user wants to make a phone call, collect:
- **phone_number** (required): E.164 format (e.g., +15551234567)
- **personality** (optional): Agent identity/persona for the call
- **task** (optional): Detailed objective for the call
- **greeting** (optional): What the agent says when the call connects
- **voice** (optional): TTS voice - format: `provider/voice-id`
  - ElevenLabs: rachel, adam, bella, josh, elli, sam
  - Deepgram: aura-2-thalia-en, aura-2-orion-en, etc.
- **model** (optional): LLM model (default: gpt-4o-mini)

## Commands

### Make an outbound call:
```bash
node {baseDir}/telnyx_voice_agent.js --to "<phone_number>" --ngrok
```

### With personality and task:
```bash
node {baseDir}/telnyx_voice_agent.js --to "<phone_number>" --ngrok \
  --personality "<persona>" \
  --task "<task>" \
  --greeting "<greeting>"
```

### With specific voice:
```bash
node {baseDir}/telnyx_voice_agent.js --to "<phone_number>" --ngrok --voice "elevenlabs/rachel"
```

### Full example:
```bash
node {baseDir}/telnyx_voice_agent.js \
  --to "+15551234567" \
  --ngrok \
  --personality "You are a friendly sales representative from Acme Corp" \
  --task "Follow up with Morgan on quote three seven one nine and confirm best callback time" \
  --greeting "Hi! This is Sarah from Acme Corp, is this a good time to talk?" \
  --voice "elevenlabs/rachel" \
  --model "claude-3-5-haiku-latest"
```

## Available Voices

**ElevenLabs** (high quality):
- elevenlabs/rachel - Female, American (default)
- elevenlabs/adam - Male, American
- elevenlabs/bella - Female, American
- elevenlabs/josh - Male, American

**Deepgram** (low latency):
- deepgram/aura-2-thalia-en - Female, American
- deepgram/aura-2-orion-en - Male, American
- deepgram/aura-2-athena-en - Female, British

## Notes

- The call runs until the other party hangs up or the agent triggers hangup
- ngrok tunnel is automatically managed (no manual setup needed)
- Environment variables must be configured in OpenClaw settings
EOM

echo ""
echo "✅ ${SKILL_NAME} skill installed to: $SKILL_DIR"
echo ""
echo "📦 Installed files:"
ls -la "$SKILL_DIR"
echo ""
echo "⚠️  Before using, configure these environment variables in OpenClaw:"
echo "   - TELNYX_API_KEY"
echo "   - TELNYX_CONNECTION_ID"
echo "   - TELNYX_PHONE_NUMBER"
echo "   - DEEPGRAM_API_KEY"
echo "   - NGROK_AUTH_TOKEN (if using --ngrok)"
echo ""
echo "📚 Install JavaScript dependencies with:"
echo "   npm --prefix $SKILL_DIR install"
echo ""
echo "🎉 Done! Try asking OpenClaw: \"Call +15551234567\""
