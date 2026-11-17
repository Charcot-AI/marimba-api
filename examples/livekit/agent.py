"""Run with: uv run --with livekit-plugins-silero examples/livekit/agent.py dev."""

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import silero

from marimba.integrations.livekit import MarimbaTurnDetector

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=silero.VAD.load(min_silence_duration=0.25),
        stt="deepgram/nova-3:multi",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3",
        turn_handling={"turn_detection": MarimbaTurnDetector()},
    )
    await session.start(room=ctx.room, agent=Agent(instructions="Be a helpful voice assistant."))


if __name__ == "__main__":
    cli.run_app(server)
