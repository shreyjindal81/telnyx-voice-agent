#!/usr/bin/env python3
"""
Voice Agent using Deepgram Voice Agent API
With client-side tool support (mirroring ElevenLabs implementation)
"""

import signal
import sys
import json
import threading
import time
import pyaudio

from deepgram import DeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    AgentV1SettingsMessage,
    AgentV1AudioConfig,
    AgentV1AudioInput,
    AgentV1AudioOutput,
    AgentV1Agent,
    AgentV1Listen,
    AgentV1ListenProvider,
    AgentV1Think,
    AgentV1AnthropicThinkProvider,
    AgentV1SpeakProviderConfig,
    AgentV1DeepgramSpeakProvider,
    AgentV1Function,
    AgentV1FunctionCallRequestEvent,
    AgentV1FunctionCallResponseMessage,
    AgentV1ConversationTextEvent,
)


# Your API key
API_KEY = "5ce590600eaa6e945dc3a17daf11e32a485c3706"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# Agent configuration (mirroring ElevenLabs Apex Mobile Customer Support agent)
AGENT_PROMPT = """You are a helpful customer support agent for Apex Mobile.
You can help users with their account, billing questions, and technical support.
When a user asks for their secret code, use the get_secret function to retrieve it.
When the user says goodbye, wants to end the call, or indicates they're done, use the hangup function to end the call.
Be friendly, concise, and helpful."""

GREETING = "Hello! Welcome to Apex Mobile support. How can I help you today?"


# ============================================
# CLIENT TOOL HANDLERS
# ============================================

# Global flag for hangup
should_hangup = False


def get_secret_handler(parameters: dict) -> str:
    """
    Mock tool that returns a static secret value.
    In a real app, this could fetch from a database, API, etc.
    """
    print("\n[TOOL CALLED] get_secret")
    print(f"   Parameters: {parameters}")

    # Return the secret value
    secret = "ALPHA-BRAVO-7749"
    print(f"   Returning: {secret}")

    return secret


def hangup_handler(parameters: dict) -> str:
    """
    Ends the call when the user wants to hang up.
    """
    global should_hangup
    print("\n[TOOL CALLED] hangup")
    print("   Ending call...")
    should_hangup = True
    return "Call ended. Goodbye!"


# Tool registry for dispatching function calls
TOOL_HANDLERS = {
    "get_secret": get_secret_handler,
    "hangup": hangup_handler,
}


# ============================================
# AUDIO INTERFACE CLASS
# ============================================

class AudioInterface:
    """Handles microphone input and speaker output using PyAudio with barge-in support."""

    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.pyaudio_instance = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.is_running = False

        # Barge-in support: buffered audio playback
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.playback_thread = None

    def start(self):
        """Initialize and start audio streams."""
        # Input stream (microphone)
        self.input_stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Output stream (speaker)
        self.output_stream = self.pyaudio_instance.open(
            format=FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        self.is_running = True

        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def read_microphone(self) -> bytes:
        """Read a chunk of audio from the microphone."""
        if self.input_stream and self.is_running:
            try:
                return self.input_stream.read(self.chunk_size, exception_on_overflow=False)
            except Exception:
                return b''
        return b''

    def queue_audio(self, audio_data: bytes):
        """Queue audio data for playback (allows barge-in interruption)."""
        with self.buffer_lock:
            self.audio_buffer.append(audio_data)

    def clear_buffer(self):
        """Clear audio buffer for barge-in - stops agent speech immediately."""
        with self.buffer_lock:
            self.audio_buffer.clear()

    def _playback_loop(self):
        """Background thread that plays audio from buffer."""
        while self.is_running:
            audio_chunk = None
            with self.buffer_lock:
                if self.audio_buffer:
                    audio_chunk = self.audio_buffer.pop(0)

            if audio_chunk and self.output_stream:
                try:
                    self.output_stream.write(audio_chunk)
                except Exception:
                    pass
            else:
                time.sleep(0.01)  # Small sleep when buffer is empty

    def stop(self):
        """Stop and close audio streams."""
        self.is_running = False

        # Clear any remaining audio
        self.clear_buffer()

        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception:
                pass

        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception:
                pass

        try:
            self.pyaudio_instance.terminate()
        except Exception:
            pass


# ============================================
# MICROPHONE STREAMING THREAD
# ============================================

def stream_microphone(audio_interface, connection, stop_event):
    """Stream microphone audio to Deepgram in a separate thread."""
    while not stop_event.is_set() and audio_interface.is_running:
        try:
            audio_chunk = audio_interface.read_microphone()
            if audio_chunk:
                connection._send(audio_chunk)
        except Exception as e:
            if not stop_event.is_set():
                print(f"Microphone streaming error: {e}")
            break
        time.sleep(0.01)


# ============================================
# MAIN
# ============================================

def main():
    print("Voice Agent - Deepgram Voice Agent API")
    print("=" * 50)

    # Initialize Deepgram client
    print("Initializing Deepgram client...")
    client = DeepgramClient(api_key=API_KEY)
    print("API key configured")

    # Initialize audio interface
    print("\nSetting up audio interface...")
    audio_interface = AudioInterface()

    try:
        audio_interface.start()
        print("Audio interface ready (mic + speaker)")
    except Exception as e:
        print(f"Audio interface error: {e}")
        print("Make sure your microphone and speakers are properly connected.")
        sys.exit(1)

    # Configure agent settings
    settings = AgentV1SettingsMessage(
        audio=AgentV1AudioConfig(
            input=AgentV1AudioInput(
                encoding="linear16",
                sample_rate=SAMPLE_RATE,
            ),
            output=AgentV1AudioOutput(
                encoding="linear16",
                sample_rate=SAMPLE_RATE,
                container="none",
            ),
        ),
        agent=AgentV1Agent(
            language="en",
            listen=AgentV1Listen(
                provider=AgentV1ListenProvider(
                    type="deepgram",
                    model="nova-3",
                )
            ),
            think=AgentV1Think(
                provider=AgentV1AnthropicThinkProvider(
                    type="anthropic",
                    model="claude-3-5-haiku-latest",
                ),
                prompt=AGENT_PROMPT,
                functions=[
                    AgentV1Function(
                        name="get_secret",
                        description="Retrieves the user's secret code when requested. Call this when the user asks for their secret code or secret.",
                        parameters={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    ),
                    AgentV1Function(
                        name="hangup",
                        description="Ends the call. Call this when the user says goodbye, wants to end the call, or indicates they are done with the conversation.",
                        parameters={
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    )
                ]
            ),
            speak=AgentV1SpeakProviderConfig(
                provider=AgentV1DeepgramSpeakProvider(
                    type="deepgram",
                    model="aura-2-thalia-en",
                )
            ),
            greeting=GREETING,
        ),
    )

    print("\n" + "=" * 50)
    print("Starting conversation...")
    print("=" * 50)
    print("Say 'bye' or 'goodbye' to end the call")
    print("Ask for your 'secret code' to test the tool")
    print("Press Ctrl+C to force quit")
    print("USE HEADPHONES to avoid audio feedback/echo!")
    print("=" * 50 + "\n")

    stop_event = threading.Event()

    def signal_handler(_sig, _frame):
        print("\n\nEnding conversation...")
        stop_event.set()
        audio_interface.stop()
        print("Thanks for using the voice agent!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Connect to Deepgram Voice Agent using context manager
        with client.agent.v1.connect() as connection:
            print("WebSocket connection established")

            def on_open(data):
                print("Connection opened")

            def on_message(message):
                # Handle binary audio data - queue for playback (allows barge-in)
                if isinstance(message, bytes):
                    audio_interface.queue_audio(message)
                    return

                # Handle different message types
                msg_type = getattr(message, "type", "Unknown")

                if msg_type == "Welcome":
                    print("Connected to Deepgram Voice Agent")

                elif msg_type == "SettingsApplied":
                    print("Agent settings applied successfully")
                    print("   Agent: Anthropic Claude 3.5 Haiku")
                    print("   STT: Deepgram Nova-3")
                    print("   TTS: Deepgram Aura-2")
                    print("   Tools: get_secret, hangup registered")
                    print("   Barge-in: enabled")

                elif msg_type == "ConversationText":
                    role = getattr(message, "role", "")
                    content = getattr(message, "content", "")
                    if role == "user":
                        print(f"\nYou: {content}")
                    elif role == "assistant":
                        print(f"\nAgent: {content}")

                elif msg_type == "UserStartedSpeaking":
                    # BARGE-IN: Clear audio buffer to stop agent speech immediately
                    audio_interface.clear_buffer()
                    print("\n[Listening...]")

                elif msg_type == "AgentThinking":
                    print("[Agent thinking...]")

                elif msg_type == "FunctionCallRequest":
                    # Handle function/tool calls from the agent
                    # FunctionCallRequest has a 'functions' array
                    functions = getattr(message, "functions", [])

                    for func in functions:
                        function_name = getattr(func, "name", "")
                        function_id = getattr(func, "id", "")
                        arguments_str = getattr(func, "arguments", "{}")

                        print(f"\n[Function call: {function_name}]")

                        if function_name in TOOL_HANDLERS:
                            try:
                                params = json.loads(arguments_str) if arguments_str else {}
                                result = TOOL_HANDLERS[function_name](params)

                                # Send function result back to agent
                                response = AgentV1FunctionCallResponseMessage(
                                    name=function_name,
                                    content=result,
                                    id=function_id
                                )
                                connection.send_function_call_response(response)
                            except Exception as e:
                                print(f"Error executing tool {function_name}: {e}")

                elif msg_type == "Error":
                    print(f"Error: {message}")

            def on_error(error):
                print(f"Error: {error}")

            def on_close(data):
                print("Connection closed")

            # Register event handlers
            connection.on(EventType.OPEN, on_open)
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.ERROR, on_error)
            connection.on(EventType.CLOSE, on_close)

            # Send settings to configure the agent
            print("Sending agent configuration...")
            connection.send_settings(settings)

            # Start listening for events in a background thread
            listener_thread = threading.Thread(
                target=connection.start_listening,
                daemon=True
            )
            listener_thread.start()

            # Wait a moment for connection to establish
            time.sleep(0.5)

            # Start microphone streaming in a separate thread
            mic_thread = threading.Thread(
                target=stream_microphone,
                args=(audio_interface, connection, stop_event),
                daemon=True
            )
            mic_thread.start()

            # Keep running until interrupted or hangup
            global should_hangup
            while not stop_event.is_set() and not should_hangup:
                time.sleep(0.1)

            if should_hangup:
                print("\n\nCall ended by agent.")
                print("Thanks for using the voice agent!")

    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()
        audio_interface.stop()


if __name__ == "__main__":
    main()
