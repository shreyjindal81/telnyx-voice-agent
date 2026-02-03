#!/usr/bin/env python3
"""
Voice Agent using ElevenLabs Conversational AI
With client-side tool support
"""

import signal
import sys
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# Your API key
API_KEY = "sk_5ca4f5345276cb1928f505004798fac4a1655890d37b53d0"

# Apex Mobile Customer Support agent
AGENT_ID = "agent_1401kggv1z3sex5aprg6r9nj3ycx"


# ============================================
# CLIENT TOOL HANDLERS
# ============================================

def get_secret_handler(parameters: dict) -> str:
    """
    Mock tool that returns a static secret value.
    In a real app, this could fetch from a database, API, etc.
    """
    print("\n🔧 [TOOL CALLED] get_secret")
    print(f"   Parameters: {parameters}")

    # Return the secret value
    secret = "ALPHA-BRAVO-7749"
    print(f"   Returning: {secret}")

    return secret


# ============================================
# MAIN
# ============================================

def main():
    print("🎙️  Voice Agent - ElevenLabs Conversational AI")
    print("=" * 50)

    # Initialize client
    client = ElevenLabs(api_key=API_KEY)

    # Verify API connection
    print("📡 Verifying API connection...")
    try:
        voices = client.voices.get_all()
        print(f"✅ API connected! Found {len(voices.voices)} voices available.")
    except Exception as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)

    # Get agent info
    print(f"\n🤖 Using agent: {AGENT_ID}")
    try:
        agent = client.conversational_ai.agents.get(agent_id=AGENT_ID)
        print(f"   Agent name: {agent.name}")
        print(f"   Language: {agent.conversation_config.agent.language}")

        # Show configured tools
        tools = agent.conversation_config.agent.prompt.tools
        if tools:
            print(f"   Tools: {', '.join(t.name for t in tools)}")
    except Exception as e:
        print(f"❌ Could not get agent info: {e}")
        sys.exit(1)

    # Set up client tools
    print("\n🔧 Registering client tools...")
    client_tools = ClientTools()
    client_tools.register("get_secret", get_secret_handler)
    print("   ✓ get_secret registered")

    # Set up audio interface
    print("\n🎧 Setting up audio interface...")
    try:
        audio_interface = DefaultAudioInterface()
    except Exception as e:
        print(f"❌ Audio interface error: {e}")
        print("Make sure your microphone and speakers are properly connected.")
        sys.exit(1)

    # Create conversation
    print("\n" + "=" * 50)
    print("🚀 Starting conversation...")
    print("=" * 50)
    print("💡 Say 'bye' or 'thank you, that's all' to end the call")
    print("💡 Ask for your 'secret code' to test the tool")
    print("💡 Press Ctrl+C to force quit")
    print("⚠️  USE HEADPHONES to avoid audio feedback/echo!")
    print("=" * 50 + "\n")

    conversation = Conversation(
        client=client,
        agent_id=AGENT_ID,
        requires_auth=True,
        audio_interface=audio_interface,
        client_tools=client_tools,
        callback_agent_response=lambda response: print(f"\n🤖 Agent: {response}"),
        callback_user_transcript=lambda transcript: print(f"\n👤 You: {transcript}"),
    )

    # Handle Ctrl+C gracefully
    def signal_handler(_sig, _frame):
        print("\n\n👋 Ending conversation...")
        conversation.end_session()
        print("Thanks for using the voice agent!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start the conversation
    conversation.start_session()

    # Keep running until interrupted
    try:
        conversation.wait_for_session_end()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
