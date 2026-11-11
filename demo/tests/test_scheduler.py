import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from marimba_demo.protocol import Frame
from marimba_demo.scheduler import Scheduler, cadence_settings, review_pause


class Stream:
    def __init__(self):
        self.count = 0

    def ingest(self, pcm):
        self.count += len(pcm)

    def snapshot(self, seconds=8):
        if not self.count:
            return None
        end = self.count / 16000
        return SimpleNamespace(window_start_s=max(0, end - seconds), window_end_s=end)


class Model:
    def __init__(self):
        self.decoder_started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def lane(self, question):
        return "encoder" if question.task == "voice_activity" else "decoder"

    def stream(self, questions, **kwargs):
        self.settings = kwargs
        return Stream()

    def encode(self, snapshot):
        return SimpleNamespace(snapshot=snapshot, encoder_latency_ms=0.0)

    def decide_encoded(self, encoded, questions):
        return []

    def decide_snapshot(self, snapshot, questions):
        self.calls.append((snapshot, [q.task for q in questions]))
        self.decoder_started.set()
        assert self.release.wait(5)
        return []


def packet(sequence):
    return Frame(sequence, sequence * 250.0, np.zeros(4000, dtype=np.int16))


def test_default_completion_pauses_are_per_card_and_overridable():
    assert review_pause("language", {}) == 2
    assert review_pause("term", {}) == 4
    assert review_pause("term", {"term": 1}) == 1


@pytest.mark.parametrize(("tail_dbfs", "gated"), [(-70, True), (-40, False)])
def test_speaker_change_is_gated_only_for_a_silent_tail(tail_dbfs, gated):
    async def run():
        updates = []
        question = SimpleNamespace(task="speaker_change")
        decision = SimpleNamespace(
            task="speaker_change",
            to_dict=lambda: {
                "task": "speaker_change",
                "value": 0.8,
                "probability": 0.8,
                "score": 0.8,
                "threshold": 0.5,
            },
        )
        model = Model()
        model.lane = lambda _: "encoder"
        model.decide_encoded = lambda encoded, questions: [decision]
        audio = np.full(8 * 16000, 0.1, dtype=np.float32)
        audio[-2 * 16000 :] = 10 ** (tail_dbfs / 20)

        async def send(data):
            updates.append(data)

        scheduler = Scheduler(model, [question], send)
        await scheduler.emit(
            SimpleNamespace(audio=audio, window_end_s=8.0),
            (31, 7750.0),
            [question],
            "encoder",
        )
        return updates[0]["decisions"][0]

    result = asyncio.run(run())
    if gated:
        assert result["gated"] == "silence"
        assert result["value"] == result["probability"] == result["score"] == 0.0
    else:
        assert "gated" not in result
        assert result["value"] == result["probability"] == result["score"] == 0.8


def test_slow_review_does_not_block_live_or_lose_capture_provenance():
    async def run():
        model = Model()
        updates = []
        live_advanced, review_returned = asyncio.Event(), asyncio.Event()

        async def send(data):
            updates.append(data)
            if data["type"] == "update":
                if data["lane"] == "encoder" and data["sequence"] >= 32:
                    live_advanced.set()
                if data["lane"] == "decoder":
                    review_returned.set()

        scheduler = Scheduler(
            model,
            [SimpleNamespace(task=t) for t in ("voice_activity", "language")],
            send,
        )
        workers = [
            asyncio.create_task(scheduler.live()),
            asyncio.create_task(scheduler.review()),
        ]
        try:
            for i in range(32):
                scheduler.ingest(packet(i))
            assert await asyncio.to_thread(model.decoder_started.wait, 2)
            scheduler.ingest(packet(32))
            await asyncio.wait_for(live_advanced.wait(), 2)
            assert not model.release.is_set()
            for i in range(33, 233):
                scheduler.ingest(packet(i))
            assert 8.0 not in scheduler.captures
            model.release.set()
            await asyncio.wait_for(review_returned.wait(), 2)
            review = next(
                d for d in updates if d["type"] == "update" and d["lane"] == "decoder"
            )
            assert review["sequence"] == 31
            assert review["captured_at_ms"] == 31 * 250
            assert len(model.calls) == 1
        finally:
            model.release.set()
            scheduler.finish(aborted=True)
            await asyncio.gather(*workers)

    asyncio.run(run())


def test_long_review_waits_for_its_entire_audio_context():
    async def run():
        model = Model()
        observed = asyncio.Event()

        async def send(data):
            if data["type"] == "update":
                observed.set()

        scheduler = Scheduler(
            model, [SimpleNamespace(task="language")], send, review_seconds=30
        )
        worker = asyncio.create_task(scheduler.review())
        try:
            for i in range(119):
                scheduler.ingest(packet(i))
            await asyncio.get_running_loop().run_in_executor(None, lambda: None)
            assert not model.decoder_started.is_set()
            scheduler.ingest(packet(119))
            assert await asyncio.to_thread(model.decoder_started.wait, 2)
            snapshot, _ = model.calls[0]
            assert (snapshot.window_start_s, snapshot.window_end_s) == (0, 30)
            assert model.settings["context_seconds"] == 30
            model.release.set()
            await asyncio.wait_for(observed.wait(), 2)
        finally:
            model.release.set()
            scheduler.finish(aborted=True)
            await worker

    asyncio.run(run())


def test_per_card_pause_does_not_enqueue_old_review_windows():
    async def run():
        model = Model()
        model.release.set()
        updates = asyncio.Queue()

        async def send(data):
            if data["type"] == "update":
                updates.put_nowait(data)

        scheduler = Scheduler(
            model,
            [SimpleNamespace(task=t) for t in ("language", "term")],
            send,
            intervals={"language": 0.25, "term": 10},
        )
        worker = asyncio.create_task(scheduler.review())
        try:
            for i in range(32):
                scheduler.ingest(packet(i))
            await asyncio.wait_for(updates.get(), 2)
            for i in range(32, 40):
                scheduler.ingest(packet(i))
            await asyncio.wait_for(updates.get(), 2)
            assert model.calls[0][1] == ["language", "term"]
            assert model.calls[1][1] == ["language"]
            assert model.calls[1][0].window_end_s == 10
            assert len(model.calls) == 2
        finally:
            scheduler.finish(aborted=True)
            await worker

    asyncio.run(run())


@pytest.mark.parametrize("value", [0, -1, 0.1, float("nan"), float("inf")])
def test_invalid_review_interval(value):
    with pytest.raises(ValueError):
        cadence_settings({"term": value})


@pytest.mark.parametrize("seconds", [0, 7, 31, float("nan"), float("inf")])
def test_invalid_review_context(seconds):
    with pytest.raises(ValueError):
        Scheduler(Model(), [], None, review_seconds=seconds)
