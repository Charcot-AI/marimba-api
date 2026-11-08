from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from marimba import Question
from marimba.testing import FakeMarimba


def test_ingest_and_decide():
    model = FakeMarimba()
    questions = [Question.noul("Is someone speaking?")]
    stream = model.stream(questions, window_seconds=0.5)
    assert stream.decide() is None
    assert stream.ingest(np.array([], dtype=np.float32)) is None
    assert stream.decide() is None
    assert stream.flush() == []
    audio = np.full(10000, 8192, dtype=np.int16)
    assert stream.ingest(audio) is None
    update = stream.decide()
    assert update.window_start_s == 0.125
    assert update.window_end_s == 0.625
    expected = model(audio[-8000:], questions, 16000)[0]
    assert update.decisions[0].probability == expected.probability
    assert update.decisions[0].window_end_s == update.window_end_s
    assert stream.flush() == []
    assert stream.decide().window_end_s == update.window_end_s
    with pytest.raises(ValueError):
        stream.ingest(np.array([np.nan], dtype=np.float32))
    assert stream.decide().window_end_s == update.window_end_s


def test_coalesce_latest_due_hop_retains_context():
    model = FakeMarimba()
    questions = [Question.noul("Is someone speaking?")]
    audio = np.linspace(0, 0.5, 19000, dtype=np.float32)
    regular = model.stream(questions, window_seconds=0.5)
    coalesced = model.stream(questions, window_seconds=0.5, coalesce=True)
    regular_updates, coalesced_updates = [], []
    for chunk in (audio[:10000], audio[10000:]):
        regular_updates.extend(regular.push(chunk))
        coalesced_updates.extend(coalesced.push(chunk))
    assert [u.window_end_s for u in regular_updates] == [0.25, 0.5, 0.75, 1.0]
    assert [u.window_end_s for u in coalesced_updates] == [0.5, 1.0]
    for actual, expected in zip(coalesced_updates, regular_updates[1::2], strict=True):
        assert actual.window_start_s == expected.window_start_s
        assert actual.decisions[0].probability == expected.decisions[0].probability
    assert coalesced.push(np.array([], dtype=np.float32)) == []
    final = coalesced.flush()[0]
    assert final.window_end_s == 19000 / 16000
    assert final.decisions[0].probability == regular.flush()[0].decisions[0].probability
    assert coalesced.flush() == []
    assert coalesced.buffer.size == 128000


def test_coalesce_waits_for_hop():
    stream = FakeMarimba().stream([], coalesce=True)
    assert stream.push(np.zeros(1000, dtype=np.float32)) == []
    updates = stream.push(np.zeros(3000, dtype=np.float32))
    assert len(updates) == 1 and updates[0].window_end_s == 0.25
    assert stream.flush() == []


@pytest.mark.parametrize("coalesce", [False, True])
def test_decide_snapshot_during_concurrent_ingest(coalesce):
    started, ingested = Event(), Event()
    observed = []

    class SlowFake(FakeMarimba):
        def decide_window(self, window, mask, questions, start_s, end_s):
            before, before_mask = window.copy(), mask.copy()
            if not observed:
                started.set()
                assert ingested.wait(timeout=5), "ingest blocked behind inference"
            np.testing.assert_array_equal(window, before)
            np.testing.assert_array_equal(mask, before_mask)
            observed.append((window.copy(), mask.copy(), start_s, end_s))
            return super().decide_window(window, mask, questions, start_s, end_s)

    stream = SlowFake().stream(
        [Question.noul("Is someone speaking?")], window_seconds=0.5, coalesce=coalesce
    )
    stream.ingest(np.full(4000, 0.2, dtype=np.float32))

    def ingest_during_inference():
        assert started.wait(timeout=5), "inference did not start"
        stream.ingest(np.full(16000, 0.9, dtype=np.float32))
        ingested.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        producer = executor.submit(ingest_during_inference)
        update = stream.decide()
        producer.result(timeout=5)
    assert update.window_start_s == 0.0
    assert update.window_end_s == 0.25
    assert update.decisions[0].window_start_s == 0.0
    assert update.decisions[0].window_end_s == 0.25
    window, mask, start, end = observed[0]
    assert (start, end) == (0.0, 0.25)
    np.testing.assert_array_equal(window[:4000], 0.0)
    np.testing.assert_array_equal(window[4000:], np.float32(0.2))
    assert mask.sum() == 4000
    latest = stream.decide()
    assert latest.window_start_s == 0.75
    assert latest.window_end_s == 1.25
    np.testing.assert_array_equal(observed[1][0], np.float32(0.9))


def test_failed_decide_does_not_corrupt_ring():
    class FailingFake(FakeMarimba):
        def decide_window(self, *args):
            raise RuntimeError("synthetic inference failure")

    stream = FailingFake().stream([])
    stream.ingest(np.ones(4000, dtype=np.float32))
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        stream.decide()
    stream.ingest(np.ones(4000, dtype=np.float32))
    assert stream.total == 8000
    with pytest.raises(ValueError):
        stream.push(np.array([np.inf], dtype=np.float32))
    assert stream.total == 8000
