from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from marimba import MAX_DECODER_SECONDS, Question, WindowSnapshot
from marimba.testing import FakeMarimba

VOICE = Question.noul("Is someone speaking?")
TERM = Question.noul('Did someone say "hello"?')


def test_snapshot_retention_is_immutable_and_does_not_reset_hops():
    stream = FakeMarimba().stream([VOICE], context_seconds=30)
    assert stream.snapshot() is None
    audio = np.arange(32 * 16000, dtype=np.float32)
    stream.ingest(audio)
    short, long = stream.snapshot(), stream.snapshot(30)
    assert (short.window_start_s, short.window_end_s) == (24, 32)
    assert (long.window_start_s, long.window_end_s) == (2, 32)
    np.testing.assert_array_equal(short.audio, audio[-128000:])
    np.testing.assert_array_equal(long.audio, audio[-480000:])
    with pytest.raises(ValueError):
        short.audio.setflags(write=True)
    with pytest.raises(ValueError):
        long.mask[0] = False
    stream.ingest(np.zeros(1000, dtype=np.float32))
    np.testing.assert_array_equal(long.audio, audio[-480000:])
    partial = FakeMarimba().stream([VOICE])
    partial.ingest(np.ones(2000, dtype=np.float32))
    assert partial.snapshot().mask.sum() == 2000
    assert len(partial.push(np.ones(2000, dtype=np.float32))) == 1


def test_encoded_ownership_and_lane_times():
    model = FakeMarimba()
    stream = model.stream([VOICE, TERM])
    stream.ingest(np.ones(16000, dtype=np.float32))
    snap = stream.snapshot()
    enc = model.encode(snap)
    saved = enc.l9.copy()
    assert enc.encoder_latency_ms >= 0
    assert not enc.l9.flags.writeable and not enc.l18.flags.writeable
    with pytest.raises(TypeError):
        enc.features[9] = np.zeros(1)
    stream.ingest(np.zeros(128000, dtype=np.float32))
    model.encode(stream.snapshot())
    np.testing.assert_array_equal(enc.l9, saved)
    decisions = model.decide_encoded(enc, [VOICE, TERM])
    assert all((d.window_start_s, d.window_end_s) == (0, 1) for d in decisions)
    assert all(d.latency_ms >= 0 for d in decisions)
    assert model.lane(VOICE) == "encoder" and model.lane(TERM) == "decoder"
    assert model.decide_encoded(enc, []) == []
    assert stream.decide([VOICE]).decisions[0].question == VOICE


@pytest.mark.parametrize("seconds", [0, -1, 30.1, float("nan"), float("inf")])
def test_snapshot_length_guards(seconds):
    stream = FakeMarimba().stream([], context_seconds=30)
    with pytest.raises(ValueError):
        stream.snapshot(seconds)


def test_encoder_and_decoder_length_guards():
    model = FakeMarimba()
    stream = model.stream([], context_seconds=30)
    stream.ingest(np.ones(16000, dtype=np.float32))
    for seconds in (1, 30):
        snap = stream.snapshot(seconds)
        with pytest.raises(ValueError, match="eight-second"):
            model.encode(snap)
        with pytest.raises(ValueError, match="decoder questions only"):
            model.decide_snapshot(snap, [VOICE])
        assert model.decide_snapshot(snap, [TERM])[0].window_end_s == 1
    huge = WindowSnapshot(np.ones(480001, dtype=np.float32), np.ones(480001, dtype=bool), 0, 30)
    with pytest.raises(ValueError, match="30"):
        model.decide_snapshot(huge, [TERM])
    assert MAX_DECODER_SECONDS == 30
    with pytest.raises(ValueError):
        model.stream([], context_seconds=31)
    with pytest.raises(ValueError):
        model.stream([], decoder_hop_seconds=0)
    with pytest.raises(ValueError):
        model.stream([], decoder_context_seconds=30)


def test_decoder_does_not_block_encoder_lane():
    entered, release, head_done = Event(), Event(), Event()

    class BlockingFake(FakeMarimba):
        def decide_window(self, window, mask, questions, start_s, end_s):
            if any(self.lane(q) == "decoder" for q in questions):
                entered.set()
                assert release.wait(5)
            else:
                head_done.set()
            return super().decide_window(window, mask, questions, start_s, end_s)

    model = BlockingFake(slow_decoder_seconds=0.01)
    stream = model.stream([], context_seconds=30)
    stream.ingest(np.ones(160000, dtype=np.float32))
    decoder_snapshot = stream.snapshot(30)
    with ThreadPoolExecutor(max_workers=2) as pool:
        decoder = pool.submit(model.decide_snapshot, decoder_snapshot, [TERM])
        try:
            assert entered.wait(5)
            stream.ingest(np.zeros(16000, dtype=np.float32))
            encoded = model.encode(stream.snapshot())
            encoder = pool.submit(model.decide_encoded, encoded, [VOICE])
            assert head_done.wait(1), "encoder waited for the decoder lock"
            assert encoder.result(1)[0].window_end_s == 11
            assert not decoder.done()
        finally:
            release.set()
        assert decoder.result(5)[0].window_end_s == 10


@pytest.mark.parametrize("coalesce", [False, True])
def test_cadence_and_long_context_timestamps(coalesce):
    stream = FakeMarimba().stream(
        [VOICE, TERM],
        context_seconds=30,
        decoder_context_seconds=30,
        decoder_hop_seconds=0.75,
        coalesce=coalesce,
    )
    updates = []
    for _ in range(8):
        updates.extend(stream.push(np.ones(4000, dtype=np.float32)))
    assert len(updates) == 8
    assert [u.window_end_s for u in updates if len(u.decisions) == 2] == [0.25, 1, 1.75]
    assert all(u.decisions[0].question == VOICE for u in updates)
    assert all(d.window_end_s == u.window_end_s for u in updates for d in u.decisions)
    assert all(d.latency_ms == u.latency_ms for u in updates for d in u.decisions)
    stream.ingest(np.ones(14 * 16000, dtype=np.float32))
    update = stream.push(np.ones(4000, dtype=np.float32))[-1]
    assert update.decisions[0].window_start_s == 8.25
    assert update.decisions[1].window_start_s == 0


def test_decoder_only_cadence_and_coalesced_context():
    stream = FakeMarimba().stream([TERM], context_seconds=30, decoder_hop_seconds=1, coalesce=True)
    assert len(stream.push(np.ones(16000, dtype=np.float32))[0].decisions) == 1
    assert stream.push(np.ones(4000, dtype=np.float32))[0].decisions == []
    stream.ingest(np.ones(10 * 16000, dtype=np.float32))
    assert stream.snapshot(30).mask.sum() == 180000
    assert stream.flush()[0].decisions[0].window_end_s == 11.25
