# Telnyx Voice Agent - Developer Guide

This guide explains the infrastructure, architecture, and internals of the Telnyx Voice Agent.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Threading Model](#threading-model)
- [Audio Pipeline](#audio-pipeline)
- [Network Flow](#network-flow)
- [Session Management](#session-management)
- [Barge-in Handling](#barge-in-handling)
- [Tool/Function Calling](#toolfunction-calling)
- [Ngrok Integration](#ngrok-integration)
- [Configuration](#configuration)
- [Debugging](#debugging)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              YOUR SERVER                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Main Process                                    │ │
│  │                                                                         │ │
│  │   ┌─────────────────────┐              ┌─────────────────────────┐     │ │
│  │   │   FastAPI/Uvicorn   │              │   Deepgram Thread       │     │ │
│  │   │   (async event      │              │   (synchronous SDK)     │     │ │
│  │   │    loop)            │              │                         │     │ │
│  │   │                     │              │   - connection          │     │ │
│  │   │   /telnyx WebSocket │──────────────│   - _send()             │     │ │
│  │   │   (receives audio)  │ input_queue  │   - start_listening()   │     │ │
│  │   │                     │ (thread-safe)│                         │     │ │
│  │   │   /telnyx WebSocket │◄─────────────│   - on_message()        │     │ │
│  │   │   (sends audio)     │ output_queue │     (queues audio)      │     │ │
│  │   └─────────────────────┘              └─────────────────────────┘     │ │
│  │            │                                      │                    │ │
│  │            │ mulaw 8kHz                           │ linear16 16kHz     │ │
│  │            ▼                                      ▼                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
        │                                                      │
        │ WebSocket                                            │ WebSocket
        │ (mulaw 8kHz PCMU)                                    │ (linear16 16kHz)
        ▼                                                      ▼
┌───────────────┐                                      ┌───────────────┐
│    Telnyx     │                                      │   Deepgram    │
│   Cloud API   │                                      │  Voice Agent  │
│               │                                      │     API       │
│  - SIP/PSTN   │                                      │  - STT (nova) │
│  - WebSocket  │                                      │  - LLM (select)│
│    streaming  │                                      │  - TTS (aura) │
└───────────────┘                                      └───────────────┘
        │
        │ PSTN/SIP
        ▼
┌───────────────┐
│    Phone      │
│   (Caller)    │
└───────────────┘
```

### Component Summary

| Component | Role | Protocol |
|-----------|------|----------|
| **Phone** | End user making/receiving calls | PSTN/SIP |
| **Telnyx** | Telephony provider, media streaming | WebSocket (mulaw 8kHz) |
| **FastAPI** | WebSocket server, audio bridge | HTTP + WebSocket |
| **Deepgram** | Voice AI (STT + LLM + TTS) | WebSocket (linear16 16kHz) |

---

## Threading Model

The Deepgram Voice Agent SDK is **synchronous** and blocks on I/O operations. FastAPI/Uvicorn runs an **async** event loop. These cannot run directly together.

### Solution: Thread-Safe Queues

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Main Process                                 │
│                                                                      │
│  ┌──────────────────────┐        ┌──────────────────────┐          │
│  │  Async Context       │        │  Sync Context        │          │
│  │  (FastAPI event loop)│        │  (Deepgram thread)   │          │
│  │                      │        │                      │          │
│  │  await websocket.    │        │  connection._send()  │          │
│  │    receive_text()    │        │  connection.         │          │
│  │                      │        │    start_listening() │          │
│  │  await websocket.    │        │                      │          │
│  │    send_json()       │        │                      │          │
│  └──────────┬───────────┘        └──────────┬───────────┘          │
│             │                               │                       │
│             │    queue.Queue (thread-safe)  │                       │
│             │                               │                       │
│             ▼                               ▼                       │
│       ┌──────────┐                   ┌──────────┐                  │
│       │ input_   │  Telnyx→Deepgram  │ output_  │ Deepgram→Telnyx  │
│       │ queue    │──────────────────►│ queue    │◄─────────────────│
│       └──────────┘                   └──────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Thread Breakdown

| Thread | Function | Blocking? |
|--------|----------|-----------|
| **Main (uvicorn)** | FastAPI async event loop | No (async) |
| **Deepgram worker** | Runs Deepgram connection + input loop | Yes (sync) |
| **Deepgram listener** | `connection.start_listening()` | Yes (sync) |
| **Output sender** | Async task sending audio to Telnyx | No (async) |

### Code Flow

```python
# FastAPI WebSocket handler (async)
@app.websocket("/telnyx")
async def telnyx_websocket(websocket: WebSocket):
    session = session_manager.create_session(...)  # Starts Deepgram thread

    # Start async output sender
    output_task = asyncio.create_task(send_output_audio())

    while True:
        message = await websocket.receive_text()  # Non-blocking
        # Convert audio and put in input_queue
        session.input_queue.put(linear_audio)     # Thread-safe

# Deepgram worker (sync, runs in dedicated thread)
def deepgram_worker(session, config):
    with client.agent.v1.connect() as connection:
        # Input loop - reads from queue, sends to Deepgram
        while not session.stop_event.is_set():
            audio = session.input_queue.get(timeout=0.05)  # Blocking
            connection._send(audio)                         # Blocking
```

---

## Audio Pipeline

### Format Conversion

Telnyx and Deepgram use different audio formats:

| System | Format | Sample Rate | Encoding |
|--------|--------|-------------|----------|
| **Telnyx** | PCMU (mulaw) | 8 kHz | 8-bit companded |
| **Deepgram** | Linear PCM | 16 kHz | 16-bit signed |

### Conversion Functions

```
Telnyx → Deepgram:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  mulaw 8kHz │───►│ linear16    │───►│ linear16    │
│  (decode)   │    │ 8kHz        │    │ 16kHz       │
└─────────────┘    └─────────────┘    └─────────────┘
                    ulaw2lin()         ratecv() 8k→16k

Deepgram → Telnyx:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ linear16    │───►│ linear16    │───►│  mulaw 8kHz │
│ 16kHz       │    │ 8kHz        │    │  (encode)   │
└─────────────┘    └─────────────┘    └─────────────┘
                   ratecv() 16k→8k     lin2ulaw()
```

### Implementation

```python
def mulaw_8k_to_linear16_16k(mulaw_data: bytes) -> bytes:
    """Telnyx → Deepgram"""
    linear_8k = audioop.ulaw2lin(mulaw_data, 2)           # Decode mulaw
    linear_16k, _ = audioop.ratecv(linear_8k, 2, 1, 8000, 16000, None)  # Resample
    return linear_16k

def linear16_16k_to_mulaw_8k(linear_data: bytes) -> bytes:
    """Deepgram → Telnyx"""
    linear_8k, _ = audioop.ratecv(linear_data, 2, 1, 16000, 8000, None)  # Resample
    mulaw_data = audioop.lin2ulaw(linear_8k, 2)           # Encode mulaw
    return mulaw_data
```

### Python 3.13+ Compatibility

The `audioop` module was removed in Python 3.13. Use `audioop-lts` as a drop-in replacement:

```python
try:
    import audioop
except ImportError:
    import audioop_lts as audioop  # pip install audioop-lts
```

---

## Network Flow

### Inbound Call Flow

```
1. Phone calls Telnyx number
         │
         ▼
2. Telnyx answers, connects WebSocket to your server
         │
         ▼
3. Server creates CallSession, starts Deepgram thread
         │
         ▼
4. Deepgram sends greeting audio
         │
         ▼
5. Audio loop begins:

   ┌─────────────────────────────────────────────────────────┐
   │                                                          │
   │  Phone speaks → Telnyx → Server → Deepgram               │
   │       │                              │                   │
   │       │                              ▼                   │
   │       │                         STT → LLM → TTS          │
   │       │                              │                   │
   │       │                              ▼                   │
   │  Phone hears ← Telnyx ← Server ← Deepgram (audio)        │
   │                                                          │
   └─────────────────────────────────────────────────────────┘
```

### Outbound Call Flow

```
1. Server calls Telnyx API: calls.dial()
         │
         ▼
2. Telnyx dials phone number
         │
         ▼
3. Phone answers
         │
         ▼
4. Telnyx connects WebSocket to your server
         │
         ▼
5. Same as inbound from step 3
```

### WebSocket Message Types

#### Telnyx → Server

| Event | Description |
|-------|-------------|
| `connected` | WebSocket established |
| `start` | Stream started, contains `stream_id` and `call_control_id` |
| `media` | Audio chunk (base64-encoded mulaw) |
| `stop` | Stream ended |

```json
// Example "start" message
{
  "event": "start",
  "stream_id": "abc123",
  "start": {
    "call_control_id": "xyz789",
    "stream_url": "wss://..."
  }
}

// Example "media" message
{
  "event": "media",
  "stream_id": "abc123",
  "media": {
    "track": "inbound",
    "payload": "base64_audio_data..."
  }
}
```

#### Server → Telnyx

| Event | Description |
|-------|-------------|
| `media` | Audio chunk to play (base64-encoded mulaw) |
| `clear` | Stop playing audio immediately (barge-in) |

```json
// Send audio
{
  "event": "media",
  "stream_id": "abc123",
  "media": {
    "payload": "base64_audio_data..."
  }
}

// Clear audio buffer (barge-in)
{
  "event": "clear",
  "stream_id": "abc123"
}
```

---

## Session Management

### CallSession Class

Each phone call gets a `CallSession` instance that manages state:

```python
class CallSession:
    def __init__(self, stream_id: str, call_control_id: str):
        self.stream_id = stream_id              # Telnyx stream identifier
        self.call_control_id = call_control_id  # For hangup/control

        # Thread-safe queues
        self.input_queue = queue.Queue()        # Telnyx → Deepgram
        self.output_queue = queue.Queue()       # Deepgram → Telnyx

        # Threading events
        self.barge_in_event = threading.Event() # Signal barge-in (sync)
        self.stop_event = threading.Event()     # Signal shutdown
        self.should_hangup = threading.Event()  # Signal hangup request

        # Async/sync bridge for fast barge-in
        self.barge_in_async_notify = None       # asyncio.Event (set in async context)
        self.event_loop = None                  # Reference to async loop

        # Deepgram thread reference
        self.deepgram_thread = None
```

### Session Lifecycle

```
┌─────────────────┐
│ WebSocket opens │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ "start" event   │──► Create CallSession
│ received        │    Start Deepgram thread
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ "media" events  │──► Audio processing loop
│ (bidirectional) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ "stop" event or │──► Stop Deepgram thread
│ hangup          │    Cleanup session
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Session closed  │
└─────────────────┘
```

---

## Barge-in Handling

Barge-in allows the user to interrupt the agent mid-speech. This requires:

1. **Detection**: Deepgram sends `UserStartedSpeaking` event
2. **Queue clearing**: Discard pending audio in output queue
3. **Telnyx notification**: Send `clear` event to stop playback

### Barge-in Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Agent speaking ─────────────────────────────────────────────────► │
│                                                                      │
│                          User interrupts                             │
│                               │                                      │
│                               ▼                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │ 1. Deepgram detects speech start                              │ │
│   │ 2. Deepgram sends "UserStartedSpeaking"                       │ │
│   │ 3. on_message() sets barge_in_event                           │ │
│   │ 4. on_message() calls loop.call_soon_threadsafe() to set      │ │
│   │    barge_in_async_notify                                       │ │
│   │ 5. send_output_audio() detects event                          │ │
│   │ 6. Clears output_queue                                        │ │
│   │ 7. Sends {"event": "clear"} to Telnyx                         │ │
│   │ 8. Telnyx stops playing buffered audio                        │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                      │
│   Agent stops speaking ◄───────────────────────────────────────────  │
│   User continues speaking ─────────────────────────────────────────► │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Triple-Check Pattern

The output sender checks for barge-in at three points to minimize latency:

```python
async def send_output_audio():
    while not session.stop_event.is_set():
        # CHECK 1: Before getting audio
        if session.barge_in_event.is_set():
            await handle_barge_in()
            continue

        try:
            audio = session.output_queue.get_nowait()
        except queue.Empty:
            # Wait for either audio or barge-in
            await asyncio.wait_for(
                session.barge_in_async_notify.wait(),
                timeout=0.01
            )
            continue

        # CHECK 2: After getting audio but before sending
        if session.barge_in_event.is_set():
            await handle_barge_in()
            continue

        # Send audio...
```

### Cross-Thread Notification

The barge-in event must notify the async context from the sync Deepgram thread:

```python
# In Deepgram thread (sync context)
def on_message(message):
    if msg_type == "UserStartedSpeaking":
        session.barge_in_event.set()  # Threading.Event (sync)

        # Notify async context immediately
        if session.barge_in_async_notify and session.event_loop:
            session.event_loop.call_soon_threadsafe(
                session.barge_in_async_notify.set
            )
```

---

## Tool/Function Calling

The Deepgram Voice Agent can call client-side functions (tools) during conversation.

### Defining Tools

Tools are defined in the agent settings:

```python
AgentV1Think(
    functions=[
        AgentV1Function(
            name="get_secret",
            description="Retrieves the user's secret code when requested.",
            parameters={"type": "object", "properties": {}, "required": []}
        ),
        AgentV1Function(
            name="hangup",
            description="Ends the call when the user says goodbye.",
            parameters={"type": "object", "properties": {}, "required": []}
        )
    ]
)
```

### Handling Tool Calls

```python
TOOL_HANDLERS = {
    "get_secret": lambda params: "ALPHA-BRAVO-7749",
    "hangup": lambda params: "Call ended. Goodbye!",
}

def handle_function_call(message, connection, session):
    functions = getattr(message, "functions", [])

    for func in functions:
        func_name = getattr(func, "name", "")
        func_id = getattr(func, "id", "")

        if func_name in TOOL_HANDLERS:
            result = TOOL_HANDLERS[func_name](params)

            # Special handling for hangup - notify async context immediately
            if func_name == "hangup":
                session.should_hangup.set()
                if session.hangup_async_notify and session.event_loop:
                    session.event_loop.call_soon_threadsafe(
                        session.hangup_async_notify.set
                    )

            # Send response back to agent
            response = AgentV1FunctionCallResponseMessage(
                name=func_name,
                content=result,
                id=func_id
            )
            connection.send_function_call_response(response)
```

### Immediate Hangup Pattern

The hangup tool uses the same cross-thread notification pattern as barge-in:

```
Deepgram Thread (sync)              Async Context (FastAPI)
        │                                   │
        │ hangup tool called                │
        │         │                         │
        ▼         ▼                         │
  session.should_hangup.set()               │
        │                                   │
        └─► call_soon_threadsafe( ──────────┼──► hangup_async_notify.set()
              hangup_async_notify.set       │         │
            )                               │         ▼
                                            │   hangup_watcher wakes up
                                            │         │
                                            │         ▼
                                            │   call_manager.hangup()
                                            │   (call ends immediately)
```

This ensures the call ends within milliseconds of the tool being triggered, rather than waiting for a timeout.

### Adding Custom Tools

1. Add handler function:
```python
def weather_handler(params: dict) -> str:
    city = params.get("city", "Unknown")
    # Call weather API...
    return f"Weather in {city}: 72°F, sunny"

TOOL_HANDLERS["get_weather"] = weather_handler
```

2. Add function definition in `create_agent_settings()`:
```python
AgentV1Function(
    name="get_weather",
    description="Gets current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    }
)
```

---

## Ngrok Integration

Ngrok provides a public URL for local development. It's integrated directly into the script.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Local Machine                               │
│                                                                      │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐  │
│   │   Python    │◄───────►│   ngrok     │◄───────►│  Internet   │  │
│   │   :8765     │ local   │   agent     │ tunnel  │             │  │
│   └─────────────┘         └─────────────┘         └─────────────┘  │
│                                   │                      │          │
│                                   │                      │          │
│                                   ▼                      ▼          │
│                           https://xyz.ngrok-free.dev     │          │
│                           wss://xyz.ngrok-free.dev/telnyx│          │
│                                                          │          │
└──────────────────────────────────────────────────────────┼──────────┘
                                                           │
                                                           ▼
                                                   ┌─────────────┐
                                                   │   Telnyx    │
                                                   │   Cloud     │
                                                   └─────────────┘
```

### Usage

```bash
# Random ngrok URL (free)
python telnyx_voice_agent.py --server-only --ngrok

# Custom domain (paid plan)
python telnyx_voice_agent.py --server-only --ngrok --ngrok-domain your-domain.ngrok-free.dev
```

### Implementation

```python
def start_ngrok(port: int, domain: str = None) -> str:
    """Start ngrok tunnel and return the public WebSocket URL."""
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)

    if domain:
        tunnel = ngrok.connect(port, "http", hostname=domain)
    else:
        tunnel = ngrok.connect(port, "http")

    # Convert https:// to wss://
    public_url = tunnel.public_url
    ws_url = public_url.replace("https://", "wss://") + "/telnyx"

    atexit.register(stop_ngrok)  # Cleanup on exit
    return ws_url
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `TELNYX_API_KEY` | Yes | Telnyx API key |
| `TELNYX_CONNECTION_ID` | Yes | TeXML application connection ID |
| `TELNYX_PHONE_NUMBER` | Yes | Your Telnyx phone number |
| `DEEPGRAM_API_KEY` | Yes | Deepgram API key |
| `PUBLIC_WS_URL` | No* | WebSocket URL (not needed with --ngrok) |
| `SERVER_HOST` | No | Server host (default: 0.0.0.0) |
| `SERVER_PORT` | No | Server port (default: 8765) |
| `RECORDINGS_DIR` | No | Local recordings folder (default: `./recordings`) |
| `NGROK_AUTH_TOKEN` | No | ngrok auth token for --ngrok flag |

### Config Class

```python
class Config:
    def __init__(self, prompt: str = None, greeting: str = None, model: str = None):
        self.telnyx_api_key = os.environ.get("TELNYX_API_KEY", "")
        self.telnyx_connection_id = os.environ.get("TELNYX_CONNECTION_ID", "")
        self.telnyx_phone_number = os.environ.get("TELNYX_PHONE_NUMBER", "")
        self.deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY", "")
        self.public_ws_url = os.environ.get("PUBLIC_WS_URL", "")
        self.server_host = os.environ.get("SERVER_HOST", "0.0.0.0")
        self.server_port = int(os.environ.get("SERVER_PORT", "8765"))
        self.agent_prompt = prompt      # CLI override
        self.agent_greeting = greeting  # CLI override
        self.agent_model = model or DEFAULT_MODEL  # CLI override (default: claude-3-5-haiku-latest)
```

### CLI Arguments

```bash
python telnyx_voice_agent.py [OPTIONS]

Options:
  --to NUMBER           Phone number to call (required unless --server-only)
  --server-only         Only run server, don't make outbound call
  --prompt TEXT         Custom system prompt for the agent
  --greeting TEXT       Custom greeting message
  --model MODEL         LLM model to use (default: claude-3-5-haiku-latest)
  --ngrok               Start ngrok tunnel automatically
  --ngrok-domain DOMAIN Custom ngrok domain (paid plan)
  --debug               Enable debug logging
```

### Supported LLM Models

The `--model` parameter accepts models from these providers (managed by Deepgram):

| Provider | Models |
|----------|--------|
| **Anthropic** | claude-sonnet-4-5, claude-4-5-haiku-latest, claude-3-5-haiku-latest, claude-sonnet-4-20250514 |
| **OpenAI** | gpt-5.1-chat-latest, gpt-5.1, gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini |

The provider is automatically determined from the model name. Invalid models are rejected at startup.

---

## Clean Shutdown

When making outbound calls (without `--server-only`), the script exits cleanly after the call ends.

### How It Works

```
Call ends (hangup or disconnect)
        │
        ▼
Session closed in cleanup
        │
        ▼
Wait for recording URL / metadata
        │
        ▼
Download recording to local disk
        │
        ▼
Delete recording from Telnyx
        │
        ▼
shutdown signal set  ──────────►  run_server_and_call() wakes up
                                           │
                                           ▼
                                   server shutdown starts
                                           │
                                           ▼
                                   ngrok tunnel closed
                                           │
                                           ▼
                                   Process exits (code 0)
```

### Server-Only Mode

With `--server-only`, the server stays running after calls end to handle more calls. Use Ctrl+C to exit.

---

## Debugging

### Log Levels

```bash
# Normal logging
python telnyx_voice_agent.py --server-only --ngrok

# Debug logging (verbose)
python telnyx_voice_agent.py --server-only --ngrok --debug
```

### Key Log Messages

| Log | Meaning |
|-----|---------|
| `[WebSocket] Telnyx connected` | Telnyx established WebSocket |
| `[Deepgram] WebSocket connection established` | Connected to Deepgram API |
| `[Deepgram] Settings applied - Agent ready` | Deepgram ready to process |
| `User: ...` | User speech transcription |
| `Agent: ...` | Agent response transcription |
| `[Barge-in] Cleared N chunks + sent clear to Telnyx` | Barge-in processed |
| `[TOOL] get_secret called` | Tool was invoked |
| `[TOOL] hangup called` | Hangup triggered |
| `[Recording] Download URL: ...` | Recording URL resolved |
| `[Recording] Saved locally: ...` | Local recording persisted |
| `[Recording] Deleted from Telnyx: ...` | Remote recording cleanup complete |
| `[Server] Call ended, shutting down...` | Clean shutdown initiated |
| `[Server] Shutdown signal received` | Server stopping |

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `CLIENT_MESSAGE_TIMEOUT` | Audio not sent fast enough | Check thread is running, queue not blocked |
| No audio to Telnyx | Conversion error or queue empty | Check audio format conversion |
| Barge-in slow | Missing `clear` event | Ensure `handle_barge_in()` sends clear |
| Call doesn't connect | Wrong WebSocket URL | Verify `PUBLIC_WS_URL` or use `--ngrok` |
| ngrok error | Missing auth token | Set `NGROK_AUTH_TOKEN` in .env |
| Recording not saved locally | Folder or permissions issue | Set writable `RECORDINGS_DIR` |
| Recording not deleted on Telnyx | Missing/late recording ID | Check post-call logs for cleanup warnings |

### Testing Locally

```bash
# Terminal 1: Start server with ngrok
python telnyx_voice_agent.py --server-only --ngrok --debug

# Terminal 2: Make outbound call (after copying ngrok URL to Telnyx)
python telnyx_voice_agent.py --to "+1234567890" --ngrok

# Or call your Telnyx number from a phone
```

---

## Code Structure

```
telnyx_voice_agent.py
├── LLM Model Configuration
│   ├── VALID_MODELS dict
│   ├── get_provider_for_model()
│   ├── validate_model()
│   └── get_all_valid_models()
├── Configuration
│   └── class Config
├── Audio Conversion
│   ├── mulaw_8k_to_linear16_16k()
│   └── linear16_16k_to_mulaw_8k()
├── Tool Handlers
│   ├── get_secret_handler()
│   ├── hangup_handler()
│   └── TOOL_HANDLERS dict
├── Session Management
│   ├── class CallSession
│   └── class SessionManager
├── Deepgram Worker
│   ├── create_think_provider()
│   ├── create_agent_settings()
│   ├── deepgram_worker()
│   └── handle_function_call()
├── Call Manager
│   └── class CallManager (Telnyx)
├── FastAPI App
│   ├── @app.websocket("/telnyx")
│   ├── @app.post("/webhook")
│   └── @app.get("/health")
├── Ngrok
│   ├── start_ngrok()
│   └── stop_ngrok()
└── CLI
    ├── parse_args()
    ├── run_server_and_call()
    └── main()
```

---

## Further Reading

- [Telnyx TeXML Streaming](https://developers.telnyx.com/docs/voice/programmable-voice/texml)
- [Deepgram Voice Agent API](https://developers.deepgram.com/docs/voice-agent)
- [Deepgram Python SDK](https://github.com/deepgram/deepgram-python-sdk)
- [pyngrok Documentation](https://pyngrok.readthedocs.io/)
