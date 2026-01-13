"""
LiveKit Voice Agent for Neymar AI Clone.

Real-time voice conversation using:
- Groq Whisper for STT (Speech-to-Text)
- Groq LLM for conversation (Llama 3.3 70B)
- ElevenLabs for TTS (Text-to-Speech)
- Silero VAD for voice activity detection

Usage:
    python -m voice_call_neymar.agent start
"""

import json
import logging
import os
import sys

# Fix Windows console encoding for non-ASCII characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

from livekit.agents import AgentSession, AutoSubscribe, JobContext, JobRequest, WorkerOptions, cli
from livekit.agents.voice import Agent
from livekit.plugins import elevenlabs, groq, silero
from livekit.rtc import DataPacketKind

from .config import config
from .prompts import NEYMAR_SYSTEM_PROMPT, NEYMAR_GREETING

logger = logging.getLogger("neymar-agent")

# #region agent log - debug helper
import time as _time
_LOG_PATH = r"c:\Users\PC\Desktop\fish-speech\.cursor\debug.log"
def _dbg(hyp: str, loc: str, msg: str, data: dict = None):
    import json as _j
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(_j.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data or {}, "timestamp": int(_time.time() * 1000)}) + "\n")
# #endregion


async def request_fnc(req: JobRequest) -> None:
    """Accept all incoming job requests."""
    # #region agent log
    _dbg("A", "agent.py:request_fnc", "Job request received", {"room": req.room.name})
    # #endregion
    logger.info(f"Job request received for room: {req.room.name}")
    await req.accept()
    # #region agent log
    _dbg("A", "agent.py:request_fnc", "Job request accepted", {"room": req.room.name})
    # #endregion


async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the voice agent."""
    # #region agent log
    _dbg("B", "agent.py:entrypoint", "Entrypoint called", {"room": ctx.room.name, "job_id": ctx.job.id if ctx.job else "unknown"})
    # #endregion
    
    # Validate configuration
    missing = config.validate()
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Please set them in your .env file or environment")
        return

    logger.info(f"Connecting to room: {ctx.room.name}")
    
    # Wait for a participant to join
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    # #region agent log
    _dbg("D", "agent.py:entrypoint", "Connected to room", {"room": ctx.room.name})
    # #endregion

    # Wait for the first participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")
    # #region agent log
    _dbg("C", "agent.py:entrypoint", "Participant joined", {"identity": participant.identity, "room": ctx.room.name})
    # #endregion

    # Read response delay from participant metadata
    response_delay = 0.8  # default
    if participant.metadata:
        try:
            metadata = json.loads(participant.metadata)
            response_delay = float(metadata.get("response_delay", 0.8))
            response_delay = max(0.3, min(2.0, response_delay))  # clamp to valid range
            logger.info(f"Response delay set to: {response_delay}s")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Could not parse participant metadata: {e}")

    # Create the agent session with STT/LLM/TTS configuration
    # detect_language=True tells Whisper to auto-detect and transcribe in original language
    session = AgentSession(
        stt=groq.STT(
            model=config.groq.stt_model,
            api_key=config.groq.api_key,
            detect_language=True,
        ),
        llm=groq.LLM(
            model=config.groq.llm_model,
            api_key=config.groq.api_key,
        ),
        tts=elevenlabs.TTS(
            voice_id=config.elevenlabs.voice_id,
            model=config.elevenlabs.model,
            api_key=config.elevenlabs.api_key,
        ),
        vad=silero.VAD.load(),
        allow_interruptions=True,
        min_endpointing_delay=response_delay,
    )

    # Create the agent with Neymar's persona instructions
    agent = Agent(instructions=NEYMAR_SYSTEM_PROMPT)

    # Helper to send transcript data to UI
    async def send_transcript(text: str, role: str) -> None:
        """Send transcript message via data channel."""
        try:
            data = json.dumps({"type": "transcript", "role": role, "text": text})
            await ctx.room.local_participant.publish_data(
                data.encode("utf-8"),
                kind=DataPacketKind.KIND_RELIABLE,
            )
        except Exception as e:
            logger.debug(f"Failed to send transcript: {e}")

    # Register event handlers for transcripts
    import asyncio
    
    @session.on("user_input_transcribed")
    def on_user_input(event):
        if event.is_final and event.transcript:
            logger.info(f"[USER] {event.transcript}")
            asyncio.create_task(send_transcript(event.transcript, "user"))

    @session.on("agent_speech_committed") 
    def on_agent_speech(event):
        if hasattr(event, "text") and event.text:
            logger.info(f"[AGENT] {event.text}")
            asyncio.create_task(send_transcript(event.text, "agent"))

    # Start the agent session
    await session.start(
        agent=agent,
        room=ctx.room,
    )

    logger.info("Neymar agent started - ready for conversation")
    # #region agent log
    _dbg("C", "agent.py:entrypoint", "Agent session started", {"room": ctx.room.name})
    # #endregion

    # Let the SDK handle session lifecycle - it auto-closes on participant disconnect
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info("Session ended - job complete")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
        ),
    )
