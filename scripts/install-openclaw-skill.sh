#!/bin/bash
# Install ClawCall AI skill for OpenClaw
# Usage: ./scripts/install-openclaw-skill.sh

set -e

SKILL_NAME="clawcall-ai"
SKILL_DIR="${HOME}/.openclaw/workspace/skills/${SKILL_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_SKILL_DIR="${REPO_ROOT}/skill"

echo "Installing ${SKILL_NAME} skill for OpenClaw..."

# Create skill directory
mkdir -p "$SKILL_DIR"

# Copy skill package files
cp "${SOURCE_SKILL_DIR}/SKILL.md" "$SKILL_DIR/"
cp "${SOURCE_SKILL_DIR}/telnyx_voice_agent.js" "$SKILL_DIR/"
cp "${SOURCE_SKILL_DIR}/package.json" "$SKILL_DIR/"
cp "${SOURCE_SKILL_DIR}/package-lock.json" "$SKILL_DIR/"
cp "${SOURCE_SKILL_DIR}/.env.example" "$SKILL_DIR/"

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
echo "   - RECORDINGS_DIR (optional, defaults to ./recordings)"
echo ""
echo "📚 Install JavaScript dependencies with:"
echo "   npm --prefix $SKILL_DIR install"
echo ""
echo "🎉 Done! Try asking OpenClaw: \"Call +15551234567\""
