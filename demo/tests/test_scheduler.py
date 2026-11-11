import asyncio
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from marimba import Question, WindowSnapshot
from marimba.testing import FakeMarimba

from marimba_demo.protocol import Frame
from marimba_demo.scheduler import Scheduler, utterance_chunks
from marimba_demo.speech import SpeechSegment, SpeechTimeline, speaker_change_eligible


def timeline():
    return SpeechTimeline(iterator=lambda *args, **kwargs: None)


def packet(sequence):
    return Frame(
        sequence, sequence * 250.0, np.full(4000, sequence + 1, dtype=np.int16)
    )


@pytest.mark.parametrize(
    ("segments", "full", "expected"),
    [
        ([(0, 0.5), (6, 6.5)], True, True),
        ([(0, 0.49), (6, 8)], True, False),
        ([(0, 6), (6, 6.49)], True, False),
        ([(0, 8)], False, False),
        ([], True, False),
        ([(0, 0.25), (2, 2.25), (6, 6.5)], True, True),
    ],
)
def test_speaker_change_eligibility(segments, full, expected):
    speech = timeline()
    speech.segments = [
        SpeechSegment(round(s * 16000), round(e * 16000)) for s, e in segments
    ]
    mask = np.ones(128000, dtype=bool)
    mask[0] = full
    snapshot = WindowSnapshot(np.zeros(128000, np.float32), mask, 0, 8)
    assert speaker_change_eligible(snapshot, speech) is expected


def test_coverage_includes_current_speech_without_removing_real_silence():
    speech = timeline()
    speech.onset, speech.processed = 6 * 16000, 7 * 16000
    assert speech.coverage(6, 8) == 1
    assert speech.coverage(0, 6) == 0


def test_iterator_receives_contiguous_512_sample_blocks_and_preserves_exact_endpoints():
    blocks = []

    def iterator(audio, **kwargs):
        blocks.append(audio.copy())
        return {1: {"start": 0}, 9: {"end": 3584}}.get(len(blocks))

    speech = SpeechTimeline(iterator)
    source = np.arange(8000, dtype=np.int16)
    assert speech.ingest(source[:4000]) == []
    assert speech.ingest(source[4000:]) == [SpeechSegment(0, 3584)]
    np.testing.assert_array_equal(np.concatenate(blocks), source[:7680] / 32768.0)
    assert speech.segments[0].seconds == (0, 0.224)


def test_silero_uses_200ms_silence_and_preserves_the_cpu_thread_budget(monkeypatch):
    settings = {}
    restored = []

    def iterator(model, **kwargs):
        assert model == "onnx-model"
        settings.update(kwargs)
        return lambda *args, **kwargs: None

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            get_num_threads=lambda: 4,
            set_num_threads=restored.append,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(
            load_silero_vad=lambda *, onnx: "onnx-model" if onnx else None,
            VADIterator=iterator,
        ),
    )
    SpeechTimeline()
    assert settings == {
        "threshold": 0.5,
        "sampling_rate": 16000,
        "min_silence_duration_ms": 200,
        "speech_pad_ms": 0,
    }
    assert restored == [4]


def test_utterance_splitting_is_consecutive_bounded_and_sample_exact():
    assert utterance_chunks(123, 123) == []
    assert utterance_chunks(123, 123 + 128001) == [(123, 128123), (128123, 128124)]
    assert utterance_chunks(16000, 288000) == [
        (16000, 144000),
        (144000, 272000),
        (272000, 288000),
    ]


class Model(FakeMarimba):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()

    def decide_encoded(self, encoded, questions):
        self.calls.append((encoded.snapshot, [q.task for q in questions]))
        return super().decide_encoded(encoded, questions)

    def decide_snapshot(self, snapshot, questions):
        self.started.set()
        assert self.release.wait(5)
        return super().decide_snapshot(snapshot, questions)


def test_turn_end_runs_only_on_offsets_and_ends_exactly_there():
    async def run():
        updates = asyncio.Queue()
        speech, model = timeline(), Model()
        questions = [
            Question.noul("Has the speaker finished?"),
            Question.noul("Is someone speaking?"),
        ]
        scheduler = Scheduler(model, questions, updates.put, speech=speech)
        live, turn = (
            asyncio.create_task(scheduler.live()),
            asyncio.create_task(scheduler.turn()),
        )
        try:
            scheduler.ingest(packet(0))
            await asyncio.wait_for(updates.get(), 2)
            assert not any("turn_end" in tasks for _, tasks in model.calls)
            scheduler.close_utterance(SpeechSegment(200, 3500), packet(0))
            scheduler.changed["turn"].set()
            result = await asyncio.wait_for(updates.get(), 2)
            decision = result["decisions"][0]
            assert decision["task"] == "turn_end"
            assert decision["kind"] == "noul"
            assert isinstance(decision["value"], bool)
            assert decision["threshold"] == 0.5
            assert decision["window_end_s"] == 3500 / 16000
            snapshot = model.calls[-1][0]
            assert snapshot.mask.sum() == 3500
            assert not snapshot.mask[:-3500].any()
            np.testing.assert_array_equal(
                snapshot.audio[-3500:], np.full(3500, 1 / 32768)
            )
            scheduler.ingest(packet(1))
            await asyncio.wait_for(updates.get(), 2)
            assert sum("turn_end" in tasks for _, tasks in model.calls) == 1
        finally:
            scheduler.finish(aborted=True)
            await asyncio.gather(live, turn)

    asyncio.run(run())


def test_live_neutral_change_skips_model_but_does_not_override_people():
    async def run():
        updates = asyncio.Queue()
        model = Model()
        questions = [
            Question.score("Has the speaker changed?"),
            Question.choice("How many people?", ["0", "1", "2", "3+"]),
        ]
        scheduler = Scheduler(model, questions, updates.put, speech=timeline())
        for i in range(32):
            scheduler.ingest(packet(i))
        worker = asyncio.create_task(scheduler.live())
        try:
            result = await asyncio.wait_for(updates.get(), 2)
            neutral = next(
                d for d in result["decisions"] if d["task"] == "speaker_change"
            )
            assert neutral["status"] == "not_enough_speech"
            assert neutral["score"] is neutral["value"] is None
            assert model.calls[0][1] == ["speaker_count"]
        finally:
            scheduler.finish(aborted=True)
            await worker

    asyncio.run(run())


def test_slow_decoder_keeps_newest_pending_utterance_and_capture_provenance():
    async def run():
        updates = asyncio.Queue()
        model = Model()
        model.release.clear()
        scheduler = Scheduler(
            model,
            [Question.noul('Was "marimba" spoken?')],
            updates.put,
            speech=timeline(),
        )
        for i in range(4):
            scheduler.ingest(packet(i))
        scheduler.close_utterance(SpeechSegment(0, 2000), packet(3))
        worker = asyncio.create_task(scheduler.review())
        try:
            assert await asyncio.to_thread(model.started.wait, 2)
            scheduler.close_utterance(SpeechSegment(4000, 6000), packet(3))
            scheduler.close_utterance(SpeechSegment(8000, 10000), packet(3))
            assert scheduler.dropped_utterances == 1
            for i in range(4, 44):
                scheduler.ingest(packet(i))
            assert scheduler.audio_start > 10000
            model.release.set()
            scheduler.finish()
            await asyncio.wait_for(worker, 3)
            assert [s.window_end_s for s, _ in model.calls] == [0.125, 0.625]
            messages = []
            while not updates.empty():
                messages.append(updates.get_nowait())
            decisions = [m for m in messages if m["type"] == "update"]
            assert decisions[0]["captured_at_ms"] == 750
            assert decisions[0]["window_end_at_ms"] == 125
            assert decisions[-1]["dropped_utterances"] == 1
        finally:
            model.release.set()
            scheduler.finish(aborted=True)
            await worker

    asyncio.run(run())


def test_long_closed_utterance_preserves_all_chunks_until_worker_claims_it():
    scheduler = Scheduler(
        Model(), [Question.noul('Was "marimba" spoken?')], None, speech=timeline()
    )
    scheduler.speech.onset = 1000
    for i in range(80):
        scheduler.ingest(packet(i))
    scheduler.close_utterance(SpeechSegment(1000, 300000), packet(79))
    chunks = scheduler.pending_utterance
    assert [w.snapshot.mask.sum() for w in chunks] == [128000, 128000, 43000]
    joined = np.concatenate([w.snapshot.audio[w.snapshot.mask] for w in chunks])
    expected = np.repeat(np.arange(1, 81, dtype=np.float32), 4000)[1000:300000] / 32768
    np.testing.assert_array_equal(joined, expected)
