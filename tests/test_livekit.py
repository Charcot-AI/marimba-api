import asyncio
import importlib
import threading

import numpy as np
import pytest
import soxr

from marimba.decision import Decision


@pytest.fixture
def integration():
    pytest.importorskip("livekit.agents.voice.turn")
    return importlib.import_module("marimba.integrations.livekit")


class FakeModel:
    def __init__(self, probability=0.75):
        self.probability = probability
        self.calls = []

    def __call__(self, audio, questions, sampling_rate=None):
        self.calls.append((audio.copy(), questions, sampling_rate))
        q = questions[0]
        return [
            Decision(
                question=q,
                task=q.task,
                kind=q.kind,
                value=self.probability,
                probability=self.probability,
            )
        ]


def frame(samples, rate=48000, channels=1):
    from livekit import rtc

    return rtc.AudioFrame(
        data=np.asarray(samples, dtype=np.int16).tobytes(),
        sample_rate=rate,
        num_channels=channels,
        samples_per_channel=len(samples) // channels,
    )


def test_protocols_and_mapping(integration):
    from livekit.agents.voice.turn import _StreamingTurnDetector, _StreamingTurnDetectorStream

    async def run():
        model = FakeModel()
        detector = integration.MarimbaTurnDetector(model=model, unlikely_threshold=0.8)
        stream = detector.stream()
        assert isinstance(detector, _StreamingTurnDetector)
        assert isinstance(stream, _StreamingTurnDetectorStream)
        assert stream.provider == "marimba"
        assert stream.model == "charcot-ai/marimba"
        assert not stream.is_fallback
        assert stream.prediction_timeout > 0
        assert await stream.unlikely_threshold("en") == 0.8
        assert await stream.supports_language(None)
        assert await stream.backchannel_threshold(None) is None
        stream.push_audio(frame([8192, -16384], 16000))
        event = await stream.predict()
        assert event.type == "eot_prediction"
        assert event.end_of_turn_probability == 0.75
        assert event.inference_duration >= 0
        audio, questions, rate = model.calls[0]
        np.testing.assert_array_equal(audio, [0.25, -0.5])
        assert rate == 16000
        assert questions[0].text == "Has the speaker finished their turn?"
        assert questions[0].task == "turn_end"
        await stream.aclose()

    asyncio.run(run())


def test_resampling_and_window(integration):
    async def run():
        model = FakeModel()
        stream = integration.MarimbaTurnDetector(model=model).stream()
        pcm = np.arange(9 * 48000, dtype=np.int16)
        for chunk in np.array_split(pcm, 90):
            stream.push_audio(frame(chunk))
        await stream.predict()
        expected = soxr.resample(
            pcm[-8 * 48000 :].astype(np.float32) / 32768, 48000, 16000, quality="HQ"
        )
        np.testing.assert_allclose(model.calls[0][0], expected)
        assert len(expected) == 128000
        await stream.aclose()

    asyncio.run(run())


def test_lazy_load(integration, monkeypatch):
    model = FakeModel()
    loads = []

    def load():
        loads.append(True)
        return model

    monkeypatch.setattr(integration.Marimba, "from_pretrained", load)

    async def run():
        stream = integration.MarimbaTurnDetector().stream()
        assert not loads
        stream.push_audio(frame([0], 16000))
        await stream.predict()
        await stream.predict()
        assert loads == [True]
        await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["raise", "empty", "missing"])
def test_failure_defaults_to_complete(integration, failure):
    def broken(*args, **kwargs):
        if failure == "raise":
            raise RuntimeError("synthetic failure")
        if failure == "empty":
            return []
        return FakeModel(None)(*args, **kwargs)

    async def run():
        stream = integration.MarimbaTurnDetector(model=broken).stream()
        stream.push_audio(frame([0], 16000))
        assert (await stream.predict()).end_of_turn_probability == 1.0
        await stream.aclose()

    asyncio.run(run())


def test_flush_empty_and_end_input(integration):
    async def run():
        model = FakeModel()
        stream = integration.MarimbaTurnDetector(model=model).stream()
        assert (await stream.predict()).end_of_turn_probability == 1.0
        stream.push_audio(frame([1], 16000))
        stream.flush()
        assert (await stream.predict()).end_of_turn_probability == 1.0
        assert not model.calls
        stream.push_audio(frame([2], 48000))
        stream.end_input()
        stream.push_audio(frame([3], 16000))
        assert (await stream.predict()).end_of_turn_probability == 1.0
        await stream.aclose()
        await stream.aclose()

    asyncio.run(run())


def test_cancellation_ignores_stale_result(integration):
    entered, release = threading.Event(), threading.Event()

    def blocking(*args, **kwargs):
        entered.set()
        release.wait()
        return FakeModel()(*args, **kwargs)

    async def run():
        stream = integration.MarimbaTurnDetector(model=blocking).stream()
        stream.push_audio(frame([0], 16000))
        future = stream.predict()
        try:
            await asyncio.to_thread(entered.wait)
            stream.cancel_inference(timed_out=True)
            assert (await future).end_of_turn_probability == 0.0
        finally:
            release.set()
            await stream.aclose()
        assert future.result().end_of_turn_probability == 0.0

    asyncio.run(run())


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_threshold_validation(integration, threshold):
    with pytest.raises(ValueError, match="unlikely_threshold"):
        integration.MarimbaTurnDetector(unlikely_threshold=threshold)


def test_stereo_and_rate_change(integration):
    async def run():
        model = FakeModel()
        stream = integration.MarimbaTurnDetector(model=model).stream()
        stream.push_audio(frame([8192, 24576], 16000, 2))
        with pytest.raises(ValueError, match="sample rate"):
            stream.push_audio(frame([0], 48000))
        await stream.predict()
        np.testing.assert_array_equal(model.calls[0][0], [0.5])
        await stream.aclose()

    asyncio.run(run())
