#!/usr/bin/env python3
"""
Voice Agent using Deepgram Voice Agent API with Telnyx Phone Integration.
Uses thread-safe queues to bridge async FastAPI with sync Deepgram SDK.
"""

import argparse
import asyncio
import atexit
import base64
import json
import logging
import os
import queue
import sys
import threading
import time

# audioop was removed in Python 3.13, use audioop-lts as fallback
try:
    import audioop
except ImportError:
    import audioop_lts as audioop

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from telnyx import Telnyx

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
    AgentV1OpenAiThinkProvider,
    AgentV1SpeakProviderConfig,
    AgentV1DeepgramSpeakProvider,
    AgentV1ElevenLabsSpeakProvider,
    AgentV1Function,
    AgentV1FunctionCallResponseMessage,
)

# Load environment variables
load_dotenv()

# Optional ngrok import
try:
    from pyngrok import ngrok, conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================
# LLM MODEL CONFIGURATION
# ============================================

# Valid models per provider (managed by Deepgram - no custom endpoint required)
VALID_MODELS = {
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-4-5-haiku-latest",
        "claude-3-5-haiku-latest",
        "claude-sonnet-4-20250514",
    ],
    "open_ai": [
        "gpt-5.1-chat-latest",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
    ],
}

DEFAULT_MODEL = "claude-3-5-haiku-latest"


# ============================================
# TTS VOICE CONFIGURATION
# ============================================

VALID_VOICES = {
    "deepgram": {
        "aura-2-thalia-en": {"description": "Thalia (Female, American)"},
        "aura-2-luna-en": {"description": "Luna (Female, American)"},
        "aura-2-stella-en": {"description": "Stella (Female, American)"},
        "aura-2-athena-en": {"description": "Athena (Female, British)"},
        "aura-2-hera-en": {"description": "Hera (Female, American)"},
        "aura-2-orion-en": {"description": "Orion (Male, American)"},
        "aura-2-arcas-en": {"description": "Arcas (Male, American)"},
        "aura-2-perseus-en": {"description": "Perseus (Male, American)"},
        "aura-2-angus-en": {"description": "Angus (Male, Irish)"},
        "aura-2-orpheus-en": {"description": "Orpheus (Male, American)"},
        "aura-2-helios-en": {"description": "Helios (Male, British)"},
        "aura-2-zeus-en": {"description": "Zeus (Male, American)"},
    },
    "elevenlabs": {
        "rachel": {"voice_id": "21m00Tcm4TlvDq8ikWAM", "description": "Rachel (Female, American)"},
        "adam": {"voice_id": "pNInz6obpgDQGcFmaJgB", "description": "Adam (Male, American)"},
        "bella": {"voice_id": "EXAVITQu4vr4xnSDxMaL", "description": "Bella (Female, American)"},
        "josh": {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "description": "Josh (Male, American)"},
        "elli": {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "description": "Elli (Female, American)"},
        "sam": {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "description": "Sam (Male, American)"},
    },
}

DEFAULT_VOICE = "elevenlabs/rachel"


def get_voice_config(voice_str: str) -> tuple[str, str, dict]:
    """Parse voice string and return (provider, voice_name, config)."""
    if "/" not in voice_str:
        raise ValueError(f"Invalid voice format '{voice_str}'. Use 'provider/voice-id' format.")

    provider, voice_name = voice_str.split("/", 1)

    if provider not in VALID_VOICES:
        raise ValueError(f"Unknown voice provider '{provider}'. Valid providers: {', '.join(VALID_VOICES.keys())}")

    if voice_name not in VALID_VOICES[provider]:
        valid_voices = list(VALID_VOICES[provider].keys())
        raise ValueError(f"Unknown voice '{voice_name}' for provider '{provider}'. Valid voices: {', '.join(valid_voices)}")

    return provider, voice_name, VALID_VOICES[provider][voice_name]


def validate_voice(voice_str: str) -> bool:
    """Check if a voice string is valid."""
    try:
        get_voice_config(voice_str)
        return True
    except ValueError:
        return False


def get_all_valid_voices() -> list[str]:
    """Get a list of all valid voice strings."""
    voices = []
    for provider, voice_dict in VALID_VOICES.items():
        for voice_name in voice_dict.keys():
            voices.append(f"{provider}/{voice_name}")
    return voices


def get_voice_sample_rate(voice_str: str) -> int:
    """Get the output sample rate for a voice provider."""
    provider, _, _ = get_voice_config(voice_str)
    if provider == "elevenlabs":
        return 24000
    return 16000  # Deepgram default


def get_provider_for_model(model: str) -> str:
    """Get the provider type for a given model."""
    for provider, models in VALID_MODELS.items():
        if model in models:
            return provider
    return None


def validate_model(model: str) -> bool:
    """Check if a model is valid."""
    return get_provider_for_model(model) is not None


def get_all_valid_models() -> list[str]:
    """Get a flat list of all valid models."""
    return [model for models in VALID_MODELS.values() for model in models]


# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Application configuration."""
    def __init__(self, prompt: str = None, greeting: str = None, model: str = None, voice: str = None):
        self.telnyx_api_key = os.environ.get("TELNYX_API_KEY", "")
        self.telnyx_connection_id = os.environ.get("TELNYX_CONNECTION_ID", "")
        self.telnyx_phone_number = os.environ.get("TELNYX_PHONE_NUMBER", "")
        self.deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        self.public_ws_url = os.environ.get("PUBLIC_WS_URL", "")
        self.server_host = os.environ.get("SERVER_HOST", "0.0.0.0")
        self.server_port = int(os.environ.get("SERVER_PORT", "8765"))
        # Agent customization
        self.agent_prompt = prompt
        self.agent_greeting = greeting
        self.agent_model = model or DEFAULT_MODEL
        self.agent_voice = voice or DEFAULT_VOICE


# Agent configuration
AGENT_PROMPT = """You are a test voice agent. This is a demo environment for testing voice AI capabilities.

Available tools you can use:
- get_secret: Returns a test secret code when the user asks for it
- hangup: Ends the call when the user says goodbye or wants to hang up

Be friendly and concise. Let users know they can test the available tools."""

GREETING = "Hi! This is a test voice agent. You can ask me for a secret code or just chat. Say goodbye when you're done."


# ============================================
# AUDIO CONVERSION UTILITIES
# ============================================

def mulaw_8k_to_linear16_16k(mulaw_data: bytes) -> bytes:
    """Convert mulaw 8kHz (Telnyx) to linear16 16kHz (Deepgram)."""
    # Decode mulaw to linear PCM (16-bit)
    linear_8k = audioop.ulaw2lin(mulaw_data, 2)
    # Resample from 8kHz to 16kHz
    linear_16k, _ = audioop.ratecv(linear_8k, 2, 1, 8000, 16000, None)
    return linear_16k


def linear16_16k_to_mulaw_8k(linear_data: bytes) -> bytes:
    """Convert linear16 16kHz (Deepgram) to mulaw 8kHz (Telnyx)."""
    # Resample from 16kHz to 8kHz
    linear_8k, _ = audioop.ratecv(linear_data, 2, 1, 16000, 8000, None)
    # Encode to mulaw
    mulaw_data = audioop.lin2ulaw(linear_8k, 2)
    return mulaw_data


def linear16_24k_to_mulaw_8k(linear_data: bytes) -> bytes:
    """Convert linear16 24kHz (ElevenLabs) to mulaw 8kHz (Telnyx)."""
    # Resample from 24kHz to 8kHz
    linear_8k, _ = audioop.ratecv(linear_data, 2, 1, 24000, 8000, None)
    # Encode to mulaw
    mulaw_data = audioop.lin2ulaw(linear_8k, 2)
    return mulaw_data


# ============================================
# TOOL HANDLERS
# ============================================

def get_secret_handler(_parameters: dict) -> str:
    """Returns the user's secret code."""
    logger.info("[TOOL] get_secret called")
    return "ALPHA-BRAVO-7749"


def hangup_handler(_parameters: dict) -> str:
    """Signals that the call should end."""
    logger.info("[TOOL] hangup called")
    return "Call ended. Goodbye!"


TOOL_HANDLERS = {
    "get_secret": get_secret_handler,
    "hangup": hangup_handler,
}


# ============================================
# CALL SESSION (Thread-Safe Queues)
# ============================================

class CallSession:
    """Manages a single phone call session with thread-safe queues."""

    def __init__(self, stream_id: str, call_control_id: str, output_sample_rate: int = 16000):
        self.stream_id = stream_id
        self.call_control_id = call_control_id

        # Thread-safe queues for audio bridging
        self.input_queue = queue.Queue()   # Telnyx audio → Deepgram
        self.output_queue = queue.Queue()  # Deepgram audio → Telnyx

        # Audio configuration
        self.output_sample_rate = output_sample_rate  # Sample rate from TTS provider

        # Threading events
        self.barge_in_event = threading.Event()
        self.stop_event = threading.Event()
        self.should_hangup = threading.Event()

        # Async event for faster barge-in notification (set from sync thread)
        self.barge_in_async_notify = None  # Will be set to asyncio.Event in async context
        self.hangup_async_notify = None  # Will be set to asyncio.Event for immediate hangup
        self.event_loop = None  # Reference to the async event loop

        # Deepgram thread
        self.deepgram_thread = None

        # Stats
        self.audio_in_count = 0
        self.audio_out_count = 0


# ============================================
# DEEPGRAM WORKER THREAD
# ============================================

def create_think_provider(model: str):
    """Create the appropriate think provider for the given model."""
    provider_type = get_provider_for_model(model)

    if provider_type == "anthropic":
        return AgentV1AnthropicThinkProvider(
            type="anthropic",
            model=model,
        )
    elif provider_type == "open_ai":
        return AgentV1OpenAiThinkProvider(
            type="open_ai",
            model=model,
        )
    else:
        # Fallback to anthropic with default model
        logger.warning(f"Unknown model {model}, falling back to {DEFAULT_MODEL}")
        return AgentV1AnthropicThinkProvider(
            type="anthropic",
            model=DEFAULT_MODEL,
        )


def create_speak_provider(voice: str):
    """Create the appropriate speak provider for the given voice."""
    provider, voice_name, voice_config = get_voice_config(voice)

    if provider == "deepgram":
        return AgentV1DeepgramSpeakProvider(
            type="deepgram",
            model=voice_name,
        )
    elif provider == "elevenlabs":
        return AgentV1ElevenLabsSpeakProvider(
            type="eleven_labs",
            model_id="eleven_turbo_v2_5",
            voice_id=voice_config["voice_id"],
        )
    else:
        # Fallback to Deepgram default
        logger.warning(f"Unknown voice provider {provider}, falling back to Deepgram")
        return AgentV1DeepgramSpeakProvider(
            type="deepgram",
            model="aura-2-thalia-en",
        )


def create_agent_settings(prompt: str = None, greeting: str = None, model: str = None, voice: str = None) -> AgentV1SettingsMessage:
    """Create Deepgram Voice Agent settings."""
    agent_prompt = prompt or AGENT_PROMPT
    agent_greeting = greeting or GREETING
    agent_model = model or DEFAULT_MODEL
    agent_voice = voice or DEFAULT_VOICE

    # Determine output sample rate based on voice provider
    output_sample_rate = get_voice_sample_rate(agent_voice)

    return AgentV1SettingsMessage(
        audio=AgentV1AudioConfig(
            input=AgentV1AudioInput(
                encoding="linear16",
                sample_rate=16000,
            ),
            output=AgentV1AudioOutput(
                encoding="linear16",
                sample_rate=output_sample_rate,
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
                provider=create_think_provider(agent_model),
                prompt=agent_prompt,
                functions=[
                    AgentV1Function(
                        name="get_secret",
                        description="Retrieves the user's secret code when requested.",
                        parameters={"type": "object", "properties": {}, "required": []}
                    ),
                    AgentV1Function(
                        name="hangup",
                        description="Ends the call when the user says goodbye or wants to end.",
                        parameters={"type": "object", "properties": {}, "required": []}
                    )
                ]
            ),
            speak=AgentV1SpeakProviderConfig(
                provider=create_speak_provider(agent_voice)
            ),
            greeting=agent_greeting,
        ),
    )


def deepgram_worker(session: CallSession, config: Config):
    """
    Runs the Deepgram Voice Agent in a dedicated thread.
    Mirrors the pattern from the working local agent.
    """
    logger.info(f"[Deepgram] Starting worker for session {session.stream_id}")

    try:
        client = DeepgramClient(api_key=config.deepgram_api_key)

        with client.agent.v1.connect() as connection:
            logger.info("[Deepgram] WebSocket connection established")

            def on_message(message):
                """Handle messages from Deepgram Voice Agent."""
                # Binary audio data → queue for Telnyx
                if isinstance(message, bytes):
                    session.output_queue.put(message)
                    return

                msg_type = getattr(message, "type", "Unknown")

                if msg_type == "Welcome":
                    logger.info("[Deepgram] Connected to Voice Agent API")

                elif msg_type == "SettingsApplied":
                    logger.info("[Deepgram] Settings applied - Agent ready")

                elif msg_type == "ConversationText":
                    role = getattr(message, "role", "")
                    content = getattr(message, "content", "")
                    if role == "user":
                        logger.info(f"User: {content}")
                    elif role == "assistant":
                        logger.info(f"Agent: {content}")

                elif msg_type == "UserStartedSpeaking":
                    # Signal barge-in to clear output queue IMMEDIATELY
                    session.barge_in_event.set()
                    # Also trigger async notification if available (for faster response)
                    if session.barge_in_async_notify and session.event_loop:
                        try:
                            # Thread-safe way to set asyncio.Event from sync context
                            session.event_loop.call_soon_threadsafe(session.barge_in_async_notify.set)
                        except Exception:
                            pass  # Event loop might be closed
                    logger.info("[Deepgram] User started speaking (barge-in)")

                elif msg_type == "AgentThinking":
                    logger.debug("[Deepgram] Agent thinking...")

                elif msg_type == "FunctionCallRequest":
                    handle_function_call(message, connection, session)

                elif msg_type == "Error":
                    logger.error(f"[Deepgram] Error: {message}")

            def on_error(error):
                logger.error(f"[Deepgram] Error: {error}")

            def on_close(_data):
                logger.info("[Deepgram] Connection closed")
                session.stop_event.set()

            # Register event handlers
            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.ERROR, on_error)
            connection.on(EventType.CLOSE, on_close)

            # Send settings
            logger.info("[Deepgram] Sending agent settings...")
            connection.send_settings(create_agent_settings(
                prompt=config.agent_prompt,
                greeting=config.agent_greeting,
                model=config.agent_model,
                voice=config.agent_voice
            ))

            # Start listener in sub-thread (like local agent)
            listener_thread = threading.Thread(
                target=connection.start_listening,
                daemon=True
            )
            listener_thread.start()
            logger.info("[Deepgram] Listener thread started")

            # Give connection time to establish
            time.sleep(0.3)

            # Input loop: read from queue, send to Deepgram
            logger.info("[Deepgram] Starting audio input loop")
            while not session.stop_event.is_set():
                try:
                    # Get audio from queue with timeout
                    audio = session.input_queue.get(timeout=0.05)
                    connection._send(audio)
                    session.audio_in_count += 1

                    if session.audio_in_count % 100 == 0:
                        logger.debug(f"[Deepgram] Sent {session.audio_in_count} audio chunks")

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"[Deepgram] Error sending audio: {e}")
                    break

            logger.info("[Deepgram] Input loop ended")

    except Exception as e:
        logger.error(f"[Deepgram] Worker error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.stop_event.set()
        logger.info(f"[Deepgram] Worker ended for session {session.stream_id}")


def handle_function_call(message, connection, session: CallSession):
    """Handle function calls from the agent."""
    functions = getattr(message, "functions", [])

    for func in functions:
        func_name = getattr(func, "name", "")
        func_id = getattr(func, "id", "")
        args_str = getattr(func, "arguments", "{}")

        logger.info(f"[Function] {func_name} called")

        if func_name in TOOL_HANDLERS:
            try:
                params = json.loads(args_str) if args_str else {}
                result = TOOL_HANDLERS[func_name](params)

                # Signal hangup if needed
                if func_name == "hangup":
                    session.should_hangup.set()
                    # Notify async context immediately for fast hangup
                    if session.hangup_async_notify and session.event_loop:
                        try:
                            session.event_loop.call_soon_threadsafe(
                                session.hangup_async_notify.set
                            )
                        except Exception:
                            pass  # Event loop might be closed

                # Send response back to agent
                response = AgentV1FunctionCallResponseMessage(
                    name=func_name,
                    content=result,
                    id=func_id
                )
                connection.send_function_call_response(response)

            except Exception as e:
                logger.error(f"[Function] Error executing {func_name}: {e}")


# ============================================
# SESSION MANAGER
# ============================================

class SessionManager:
    """Manages active call sessions."""

    def __init__(self, config: Config):
        self.config = config
        self.sessions: dict[str, CallSession] = {}
        # Store call_control_id from webhook for later use
        self.pending_call_control_ids: dict[str, str] = {}

    def create_session(self, stream_id: str, call_control_id: str) -> CallSession:
        """Create a new call session and start Deepgram worker."""
        # Get output sample rate based on configured voice
        output_sample_rate = get_voice_sample_rate(self.config.agent_voice)
        session = CallSession(stream_id, call_control_id, output_sample_rate=output_sample_rate)

        # Start Deepgram in dedicated thread
        session.deepgram_thread = threading.Thread(
            target=deepgram_worker,
            args=(session, self.config),
            daemon=True
        )
        session.deepgram_thread.start()

        self.sessions[stream_id] = session
        logger.info(f"[Session] Created session {stream_id}")
        return session

    def get_session(self, stream_id: str) -> CallSession:
        return self.sessions.get(stream_id)

    def close_session(self, stream_id: str):
        """Close and cleanup a session."""
        session = self.sessions.get(stream_id)
        if session:
            session.stop_event.set()
            del self.sessions[stream_id]
            logger.info(f"[Session] Closed session {stream_id}")


# ============================================
# CALL MANAGER (TELNYX)
# ============================================

class CallManager:
    """Manages Telnyx outbound calls."""

    def __init__(self, config: Config):
        self.config = config
        self.client = Telnyx(api_key=config.telnyx_api_key)

    async def initiate_call(self, to_number: str, from_number: str = None) -> dict:
        """Initiate an outbound call."""
        from_number = from_number or self.config.telnyx_phone_number

        logger.info(f"[Telnyx] Calling {to_number} from {from_number}")

        response = self.client.calls.dial(
            connection_id=self.config.telnyx_connection_id,
            to=to_number,
            from_=from_number,
            stream_url=self.config.public_ws_url,
            stream_track="both_tracks",
            stream_bidirectional_mode="rtp",
            stream_bidirectional_codec="PCMU",
            webhook_url_method="POST",
        )

        call_data = response.data
        logger.info(f"[Telnyx] Call initiated: {call_data.call_control_id}")

        return {
            "call_control_id": call_data.call_control_id,
            "status": "initiated",
        }

    async def hangup(self, call_control_id: str):
        """Hang up a call."""
        logger.info(f"[Telnyx] Hanging up {call_control_id}")
        try:
            self.client.calls.actions.hangup(call_control_id=call_control_id)
        except Exception as e:
            logger.error(f"[Telnyx] Hangup error: {e}")


# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(title="Telnyx Voice Agent")
config: Config = None
session_manager: SessionManager = None
call_manager: CallManager = None
shutdown_event: asyncio.Event = None  # Signal to shutdown after call ends
server_only_mode: bool = True  # Whether to stay up after call ends


def init_app(prompt: str = None, greeting: str = None, model: str = None, voice: str = None, server_only: bool = True):
    """Initialize application components."""
    global config, session_manager, call_manager, shutdown_event, server_only_mode
    config = Config(prompt=prompt, greeting=greeting, model=model, voice=voice)
    session_manager = SessionManager(config)
    call_manager = CallManager(config)
    shutdown_event = asyncio.Event()
    server_only_mode = server_only


@app.websocket("/telnyx")
async def telnyx_websocket(websocket: WebSocket):
    """WebSocket endpoint for Telnyx media streaming."""
    await websocket.accept()
    logger.info("[WebSocket] Telnyx connected")

    session: CallSession = None
    output_task = None
    hangup_task = None

    async def handle_barge_in():
        """Handle barge-in: clear queue AND tell Telnyx to stop playing."""
        nonlocal session
        cleared = 0
        while not session.output_queue.empty():
            try:
                session.output_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break

        # CRITICAL: Send "clear" event to Telnyx to flush their audio buffer
        try:
            clear_message = {
                "event": "clear",
                "stream_id": session.stream_id
            }
            await websocket.send_json(clear_message)
            logger.info(f"[Barge-in] Cleared {cleared} chunks + sent clear to Telnyx")
        except Exception as e:
            logger.error(f"[Barge-in] Error sending clear: {e}")

        session.barge_in_event.clear()
        session.barge_in_async_notify.clear()

    async def send_output_audio():
        """Async task to send Deepgram audio back to Telnyx."""
        nonlocal session
        while session and not session.stop_event.is_set():
            try:
                # Check for barge-in FIRST (highest priority)
                if session.barge_in_event.is_set():
                    await handle_barge_in()
                    continue

                # Get audio from output queue (non-blocking)
                try:
                    audio = session.output_queue.get_nowait()
                except queue.Empty:
                    # Wait for either: new audio, barge-in signal, or timeout
                    try:
                        await asyncio.wait_for(
                            session.barge_in_async_notify.wait(),
                            timeout=0.01  # 10ms max wait
                        )
                        # Barge-in triggered while waiting
                        await handle_barge_in()
                        continue
                    except asyncio.TimeoutError:
                        continue

                # Check barge-in AGAIN before sending (in case it was set while getting audio)
                if session.barge_in_event.is_set():
                    await handle_barge_in()
                    continue

                # Convert and send to Telnyx (use appropriate converter based on sample rate)
                if session.output_sample_rate == 24000:
                    mulaw_audio = linear16_24k_to_mulaw_8k(audio)
                else:
                    mulaw_audio = linear16_16k_to_mulaw_8k(audio)
                audio_base64 = base64.b64encode(mulaw_audio).decode("utf-8")

                message = {
                    "event": "media",
                    "stream_id": session.stream_id,
                    "media": {"payload": audio_base64}
                }
                await websocket.send_json(message)

                session.audio_out_count += 1
                if session.audio_out_count % 50 == 0:
                    logger.debug(f"[WebSocket] Sent {session.audio_out_count} audio chunks to Telnyx")

            except Exception as e:
                if not session.stop_event.is_set():
                    logger.error(f"[WebSocket] Output error: {e}")
                break

    try:
        while True:
            # Receive message from Telnyx
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                # Check if stop was signaled (e.g., by hangup watcher)
                if session and session.stop_event.is_set():
                    break
                continue

            message = json.loads(data)
            event = message.get("event")

            if event == "connected":
                logger.info("[WebSocket] Telnyx stream connected")

            elif event == "start":
                stream_id = message.get("stream_id")
                # call_control_id is nested inside "start" object
                start_data = message.get("start", {})
                call_control_id = start_data.get("call_control_id", "")

                logger.info(f"[WebSocket] Stream started: {stream_id}, call_control_id: {call_control_id}")

                # Create session (starts Deepgram worker)
                session = session_manager.create_session(stream_id, call_control_id)

                # Initialize async events for fast notification
                session.barge_in_async_notify = asyncio.Event()
                session.hangup_async_notify = asyncio.Event()
                session.event_loop = asyncio.get_running_loop()

                # Hangup watcher task - executes hangup immediately when signaled
                async def hangup_watcher():
                    await session.hangup_async_notify.wait()
                    logger.info("[Hangup] Executing hangup immediately")
                    if call_manager and session.call_control_id:
                        await call_manager.hangup(session.call_control_id)
                    session.stop_event.set()

                hangup_task = asyncio.create_task(hangup_watcher())

                # Start output task
                output_task = asyncio.create_task(send_output_audio())

            elif event == "media":
                if not session:
                    continue

                # Extract audio from Telnyx message
                media = message.get("media", {})
                audio_base64 = media.get("payload")
                track = media.get("track", "inbound")

                if track == "inbound" and audio_base64:
                    # Decode and convert audio
                    mulaw_audio = base64.b64decode(audio_base64)
                    linear_audio = mulaw_8k_to_linear16_16k(mulaw_audio)

                    # Put in input queue for Deepgram worker
                    session.input_queue.put(linear_audio)

            elif event == "stop":
                logger.info("[WebSocket] Stream stopped")
                break

    except WebSocketDisconnect:
        logger.info("[WebSocket] Telnyx disconnected")
    except Exception as e:
        logger.error(f"[WebSocket] Error: {e}")
    finally:
        if output_task:
            output_task.cancel()
        if hangup_task:
            hangup_task.cancel()
        if session:
            session_manager.close_session(session.stream_id)
        # Signal shutdown if not in server-only mode
        if not server_only_mode and shutdown_event:
            logger.info("[Server] Call ended, shutting down...")
            shutdown_event.set()


@app.post("/webhook")
async def telnyx_webhook(request: Request):
    """HTTP endpoint for Telnyx webhooks."""
    try:
        payload = await request.json()
        data = payload.get("data", {})
        event_type = data.get("event_type", "")
        event_payload = data.get("payload", {})

        logger.info(f"[Webhook] {event_type}")

        # Capture call_control_id from streaming.started event
        if event_type == "streaming.started":
            call_control_id = event_payload.get("call_control_id", "")
            stream_id = event_payload.get("stream_id", "")
            if call_control_id and stream_id and session_manager:
                # Update the session with the call_control_id
                session = session_manager.get_session(stream_id)
                if session and not session.call_control_id:
                    session.call_control_id = call_control_id
                    logger.info(f"[Webhook] Updated session {stream_id} with call_control_id: {call_control_id}")

        return JSONResponse({"status": "ok"})

    except Exception as e:
        logger.error(f"[Webhook] Error: {e}")
        return JSONResponse({"status": "error"}, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================
# NGROK TUNNEL MANAGEMENT
# ============================================

ngrok_tunnel = None  # Global reference for cleanup


def start_ngrok(port: int, domain: str = None) -> str:
    """Start ngrok tunnel and return the public URL."""
    global ngrok_tunnel

    if not NGROK_AVAILABLE:
        raise RuntimeError("pyngrok not installed. Run: pip install pyngrok")

    # Check for ngrok auth token
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)

    # Configure ngrok
    if domain:
        # Use custom domain (requires ngrok paid plan)
        ngrok_tunnel = ngrok.connect(port, "http", hostname=domain)
    else:
        # Use random ngrok URL
        ngrok_tunnel = ngrok.connect(port, "http")

    # Extract URL and convert to wss
    public_url = ngrok_tunnel.public_url
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/telnyx"

    logger.info(f"[ngrok] Tunnel established: {public_url}")
    logger.info(f"[ngrok] WebSocket URL: {ws_url}")

    # Register cleanup
    atexit.register(stop_ngrok)

    return ws_url


def stop_ngrok():
    """Stop ngrok tunnel."""
    global ngrok_tunnel
    if ngrok_tunnel:
        try:
            ngrok.disconnect(ngrok_tunnel.public_url)
            logger.info("[ngrok] Tunnel closed")
        except Exception:
            pass
        ngrok_tunnel = None


# ============================================
# CLI
# ============================================

def parse_args():
    """Parse command line arguments."""
    # Build voice help text
    deepgram_voices = [f"deepgram/{v}" for v in VALID_VOICES["deepgram"].keys()]
    elevenlabs_voices = [f"elevenlabs/{v}" for v in VALID_VOICES["elevenlabs"].keys()]

    parser = argparse.ArgumentParser(
        description="Telnyx Voice Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  Anthropic: {', '.join(VALID_MODELS['anthropic'])}
  OpenAI:    {', '.join(VALID_MODELS['open_ai'])}

Available voices (format: provider/voice-id):
  Deepgram:   {', '.join(deepgram_voices[:4])}
              {', '.join(deepgram_voices[4:8])}
              {', '.join(deepgram_voices[8:])}
  ElevenLabs: {', '.join(elevenlabs_voices)}

Default model: {DEFAULT_MODEL}
Default voice: {DEFAULT_VOICE}
"""
    )
    parser.add_argument("--to", type=str, help="Phone number to call")
    parser.add_argument("--prompt", type=str, help="System prompt for the agent")
    parser.add_argument("--greeting", type=str, help="Initial greeting message")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--voice", type=str, default=DEFAULT_VOICE,
        help=f"TTS voice as provider/voice-id (default: {DEFAULT_VOICE})"
    )
    parser.add_argument("--server-only", action="store_true", help="Only run server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--ngrok", action="store_true", help="Start ngrok tunnel automatically")
    parser.add_argument("--ngrok-domain", type=str, help="Custom ngrok domain (requires paid plan)")
    args = parser.parse_args()

    if not args.server_only and not args.to:
        parser.error("--to is required unless using --server-only")

    # Validate model
    if not validate_model(args.model):
        valid_models = get_all_valid_models()
        parser.error(f"Invalid model '{args.model}'. Valid models: {', '.join(valid_models)}")

    # Validate voice
    if not validate_voice(args.voice):
        valid_voices = get_all_valid_voices()
        parser.error(f"Invalid voice '{args.voice}'. Valid voices: {', '.join(valid_voices)}")

    return args


async def run_server_and_call(args, ngrok_url: str = None):
    """Run the server and optionally make a call."""
    init_app(prompt=args.prompt, greeting=args.greeting, model=args.model, voice=args.voice, server_only=args.server_only)

    # Override public URL if ngrok provided one
    if ngrok_url:
        config.public_ws_url = ngrok_url

    logger.info(f"Starting server on {config.server_host}:{config.server_port}")
    logger.info(f"Public URL: {config.public_ws_url}")
    if config.agent_prompt:
        logger.info(f"Custom prompt: {config.agent_prompt[:50]}...")
    if config.agent_greeting:
        logger.info(f"Custom greeting: {config.agent_greeting}")
    logger.info(f"Using model: {config.agent_model} ({get_provider_for_model(config.agent_model)})")
    logger.info(f"Using voice: {config.agent_voice}")

    server_config = uvicorn.Config(
        app,
        host=config.server_host,
        port=config.server_port,
        log_level="debug" if args.debug else "info",
    )
    server = uvicorn.Server(server_config)
    server_task = asyncio.create_task(server.serve())

    await asyncio.sleep(1)

    if not args.server_only and args.to:
        try:
            await call_manager.initiate_call(args.to)
            logger.info("Call initiated - waiting for call to complete...")
        except Exception as e:
            logger.error(f"Call failed: {e}")
            return
    else:
        logger.info("Server running in server-only mode (press Ctrl+C to exit)")

    try:
        if args.server_only:
            # Server-only mode: run until cancelled
            await server_task
        else:
            # Outbound call mode: wait for shutdown signal, then exit
            await shutdown_event.wait()
            logger.info("[Server] Shutdown signal received")
            server.should_exit = True
            await asyncio.sleep(0.5)  # Give server time to cleanup
    except asyncio.CancelledError:
        pass


def main():
    """Main entry point."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    ngrok_url = None

    # Start ngrok if requested
    if args.ngrok:
        port = int(os.environ.get("SERVER_PORT", "8765"))
        try:
            ngrok_url = start_ngrok(port, domain=args.ngrok_domain)
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")
            sys.exit(1)

    try:
        asyncio.run(run_server_and_call(args, ngrok_url=ngrok_url))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        stop_ngrok()


if __name__ == "__main__":
    main()
