import asyncio
from dataclasses import replace

import numpy as np
import pytest
from marimba import Question
from marimba.testing import FakeMarimba

from marimba_demo.cards import available_cards, questions_for
from marimba_demo.protocol import HOP_SAMPLES, Frame
from marimba_demo.replay import REPLAY_SILENCE_SECONDS, pcm16, replay
from marimba_demo.scheduler import Scheduler


class SpeechModel(FakeMarimba):
    def decide_encoded(self, encoded, questions):
        return [
            replace(
                decision,
                value=0.5 <= encoded.snapshot.window_end_s <= 3.0,
                probability=0.9 if encoded.snapshot.window_end_s <= 3.0 else 0.1,
            )
            if decision.task == "voice_activity"
            else decision
            for decision in super().decide_encoded(encoded, questions)
        ]


def test_replay_emits_every_hop_and_drains_closed_utterance():
    model = SpeechModel(slow_decoder_seconds=0.01)
    questions = questions_for(available_cards(model), Question)
    events = asyncio.run(replay(model, questions, np.zeros(64000, np.float32)))
    rolling = [
        event
        for event in events
        if any(d["task"] == "voice_activity" for d in event.get("decisions", []))
    ]
    hops = 16 + round(REPLAY_SILENCE_SECONDS * 4)
    assert [event["sequence"] for event in rolling] == list(range(hops))
    assert [event["window_end_at_ms"] for event in rolling] == [
        (i + 1) * 250.0 for i in range(hops)
    ]
    assert all(event.get("coalesced_hops", 0) == 0 for event in events)
    for task in ("turn_end", "language", "term"):
        decisions = [
            d
            for event in events
            for d in event.get("decisions", [])
            if d["task"] == task
        ]
        assert decisions
        assert decisions[-1]["window_end_s"] == 1.5
    assert any(
        event.get("tasks") == ["term"] and event["running"] is False for event in events
    )


def without_timing(payload):
    return {
        key: [without_timing(d) for d in value] if key == "decisions" else value
        for key, value in payload.items()
        if key not in {"window_latency_ms", "latency_ms", "replay"}
    }


def test_replay_matches_nonlagging_live_scheduler_payloads():
    async def run():
        questions = [Question.noul("Is someone speaking?", seconds=2)]
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        recorded = await replay(SpeechModel(), questions, audio)
        audio = np.pad(audio, (0, round(REPLAY_SILENCE_SECONDS * 16000)))
        events, ready = [], asyncio.Queue()

        async def send(payload):
            events.append(payload)
            await ready.put(payload)

        scheduler = Scheduler(SpeechModel(), questions, send)
        workers = [
            asyncio.create_task(fn())
            for fn in (scheduler.live, scheduler.turn, scheduler.review)
        ]
        try:
            for i, start in enumerate(range(0, len(audio), HOP_SAMPLES)):
                scheduler.ingest(
                    Frame(i, i * 250.0, pcm16(audio[start : start + HOP_SAMPLES]))
                )
                await asyncio.wait_for(ready.get(), 2)
            scheduler.finish()
            await asyncio.gather(*workers)
        finally:
            scheduler.finish(aborted=True)
            await asyncio.gather(*workers, return_exceptions=True)
        assert [without_timing(event) for event in recorded] == [
            without_timing(event) for event in events
        ]

    asyncio.run(run())


def test_replay_pads_only_the_final_frame_and_clips_pcm():
    class RecordingModel(SpeechModel):
        def __init__(self):
            super().__init__()
            self.snapshots = []

        def _encode_snapshot(self, snapshot):
            self.snapshots.append(snapshot)
            return super()._encode_snapshot(snapshot)

    model = RecordingModel()
    events = asyncio.run(
        replay(
            model,
            [Question.noul("Is someone speaking?")],
            np.full(4001, 2.0, np.float32),
        )
    )
    assert len(events) == 2 + round(REPLAY_SILENCE_SECONDS * 4)
    np.testing.assert_array_equal(
        model.snapshots[1].audio[-4000:], np.r_[32767 / 32768, np.zeros(3999)]
    )
    assert events[1]["captured_at_ms"] == 250.0
    assert all(
        event["replay"]["trailing_silence_s"] == REPLAY_SILENCE_SECONDS
        for event in events
    )
    assert not model.snapshots[-1].audio[-round(REPLAY_SILENCE_SECONDS * 16000) :].any()


@pytest.mark.parametrize("audio", [np.array([]), np.zeros((2, 4)), np.array([np.nan])])
def test_replay_rejects_invalid_audio(audio):
    with pytest.raises(ValueError, match="finite mono"):
        asyncio.run(replay(SpeechModel(), [], audio))


def test_replay_propagates_encoder_errors_instead_of_waiting_forever():
    class BrokenModel(SpeechModel):
        def _encode_snapshot(self, snapshot):
            raise RuntimeError("encoder failed")

    async def run():
        with pytest.raises(RuntimeError, match="Model inference failed"):
            await asyncio.wait_for(
                replay(
                    BrokenModel(),
                    [Question.noul("Is someone speaking?")],
                    np.zeros(4000, np.float32),
                ),
                2,
            )

    asyncio.run(run())
