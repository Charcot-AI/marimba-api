import asyncio
import importlib

import numpy as np
import pytest
import soxr

from marimba.decision import Decision


@pytest.fixture
def analyzer_class():
    pytest.importorskip("pipecat.audio.turn.smart_turn.base_smart_turn")
    return importlib.import_module("marimba.integrations.pipecat").MarimbaTurnAnalyzer


class FakeModel:
    def __init__(self, probability: float = 0.75):
        self.probability = probability
        self.audio = None
        self.questions = None
        self.sampling_rate = None

    def __call__(self, audio, questions, sampling_rate=None):
        self.audio = np.asarray(audio)
        self.questions = list(questions)
        self.sampling_rate = sampling_rate
        question = questions[0]
        return [
            Decision(
                question=question,
                task=question.task or "turn_end",
                kind=question.kind,
                value=self.probability,
                probability=self.probability,
            )
        ]


@pytest.mark.parametrize(
    ("probability", "threshold", "prediction"),
    [(0.5, 0.5, 1), (0.49, 0.5, 0), (0.75, 0.8, 0)],
)
def test_prediction_and_probability_mapping(analyzer_class, probability, threshold, prediction):
    model = FakeModel(probability)
    analyzer = analyzer_class(model=model, threshold=threshold)

    result = analyzer._predict_endpoint(np.ones(16000, dtype=np.float32))

    assert result == {"prediction": prediction, "probability": probability}
    assert model.questions[0].task == "turn_end"
    assert model.questions[0].text == "Has the speaker finished their turn?"
    assert model.sampling_rate == 16000


def test_short_audio_is_passed_unpadded(analyzer_class):
    model = FakeModel()
    audio = np.array([0.25, -0.5], dtype=np.float32)
    analyzer = analyzer_class(model=model)

    analyzer._predict_endpoint(audio)

    assert model.audio.shape == audio.shape
    np.testing.assert_array_equal(model.audio, audio)


def test_audio_is_truncated_to_last_eight_seconds(analyzer_class):
    model = FakeModel()
    audio = np.arange(128002, dtype=np.float32)
    analyzer = analyzer_class(model=model)

    analyzer._predict_endpoint(audio)

    np.testing.assert_array_equal(model.audio, audio[-128000:])


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 44100, 48000])
def test_audio_is_resampled_to_model_rate(analyzer_class, rate):
    model = FakeModel()
    analyzer = analyzer_class(model=model, sample_rate=rate)
    analyzer.set_sample_rate(rate)
    audio = np.linspace(-1, 1, rate, dtype=np.float32)
    resampled = soxr.resample(audio, rate, 16000, quality="HQ") if rate != 16000 else audio

    analyzer._predict_endpoint(audio)

    np.testing.assert_allclose(model.audio, resampled)
    assert model.sampling_rate == 16000


def test_default_model_is_loaded_lazily(analyzer_class, monkeypatch):
    integration = importlib.import_module("marimba.integrations.pipecat")
    model = FakeModel()
    loads = []

    class ModelFactory:
        @staticmethod
        def from_pretrained():
            loads.append(True)
            return model

    monkeypatch.setattr(integration, "Marimba", ModelFactory)
    analyzer = analyzer_class()
    assert loads == []

    analyzer._predict_endpoint(np.zeros(1, dtype=np.float32))
    analyzer._predict_endpoint(np.zeros(1, dtype=np.float32))

    assert loads == [True]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_threshold_must_be_a_probability(analyzer_class, threshold):
    with pytest.raises(ValueError, match="threshold"):
        analyzer_class(model=FakeModel(), threshold=threshold)


@pytest.mark.parametrize("probability", [0.25, 0.75])
def test_public_api_returns_turn_metrics(analyzer_class, probability):
    analyzer = analyzer_class(model=FakeModel(probability))
    analyzer.set_sample_rate(16000)

    async def run():
        try:
            analyzer.append_audio(np.ones(320, dtype=np.int16).tobytes(), is_speech=True)
            state, metrics = await analyzer.analyze_end_of_turn()
            assert state.name == ("COMPLETE" if probability >= 0.5 else "INCOMPLETE")
            assert metrics.probability == probability
            assert metrics.e2e_processing_time_ms >= 0
        finally:
            await analyzer.cleanup()

    asyncio.run(run())


def test_public_api_empty_audio_has_no_metrics(analyzer_class):
    analyzer = analyzer_class(model=FakeModel())
    analyzer.set_sample_rate(16000)

    async def run():
        try:
            state, metrics = await analyzer.analyze_end_of_turn()
            assert state.name == "INCOMPLETE"
            assert metrics is None
        finally:
            await analyzer.cleanup()

    asyncio.run(run())


def test_public_api_inference_error_allows_silence_fallback(analyzer_class, caplog):
    class BrokenModel:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("Synthetic inference failure")

    analyzer = analyzer_class(model=BrokenModel())
    analyzer.set_sample_rate(16000)

    async def run():
        try:
            analyzer.append_audio(np.ones(320, dtype=np.int16).tobytes(), is_speech=True)
            state, metrics = await analyzer.analyze_end_of_turn()
            assert state.name == "INCOMPLETE"
            assert metrics is None
            assert "Synthetic inference failure" in caplog.text
            assert analyzer.speech_triggered
            for _ in range(150):
                state = analyzer.append_audio(bytes(640), is_speech=False)
            assert state.name == "COMPLETE"
            assert not analyzer.speech_triggered
            state, metrics = await analyzer.analyze_end_of_turn()
            assert state.name == "INCOMPLETE"
            assert metrics is None
        finally:
            await analyzer.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("num_channels", [0, -1, 1.5])
def test_invalid_channels(analyzer_class, num_channels):
    with pytest.raises(ValueError, match="num_channels"):
        analyzer_class(model=FakeModel(), num_channels=num_channels)


def test_stereo_is_downmixed_before_buffering_and_timeout(analyzer_class):
    model = FakeModel(0.25)
    analyzer = analyzer_class(model=model, num_channels=2)
    analyzer.set_sample_rate(16000)
    stereo = np.tile(np.array([[1000, 3000], [1000, 3001]], dtype=np.int16), (160, 1))

    async def run():
        try:
            analyzer.append_audio(stereo.tobytes(), is_speech=True)
            _, metrics = await analyzer.analyze_end_of_turn()
            expected = np.rint(stereo.astype(np.float32).mean(axis=1)) / 32768
            np.testing.assert_array_equal(model.audio, expected)
            assert len(model.audio) == 320
            assert metrics.probability == 0.25
            for _ in range(149):
                state = analyzer.append_audio(bytes(1280), is_speech=False)
            assert state.name == "INCOMPLETE"
            assert analyzer.append_audio(bytes(1280), is_speech=False).name == "COMPLETE"
        finally:
            await analyzer.cleanup()

    asyncio.run(run())


def test_public_api_load_error_allows_retry(analyzer_class, monkeypatch, caplog):
    integration = importlib.import_module("marimba.integrations.pipecat")
    model = FakeModel()
    loads = []

    def load():
        loads.append(True)
        if len(loads) == 1:
            raise OSError("Synthetic missing cached model")
        return model

    monkeypatch.setattr(integration.Marimba, "from_pretrained", load)
    analyzer = analyzer_class()
    analyzer.set_sample_rate(16000)

    async def run():
        try:
            analyzer.append_audio(bytes(640), is_speech=True)
            state, metrics = await analyzer.analyze_end_of_turn()
            assert state.name == "INCOMPLETE"
            assert metrics is None
            assert "Synthetic missing cached model" in caplog.text
            state, metrics = await analyzer.analyze_end_of_turn()
            assert state.name == "COMPLETE"
            assert metrics.probability == 0.75
            assert len(loads) == 2
        finally:
            await analyzer.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("samples", [1, 160, 320, 1600])
def test_public_api_short_audio(analyzer_class, samples):
    model = FakeModel()
    analyzer = analyzer_class(model=model)
    analyzer.set_sample_rate(16000)
    audio = np.arange(samples, dtype=np.int16)

    async def run():
        try:
            analyzer.append_audio(audio.tobytes(), is_speech=True)
            _, metrics = await analyzer.analyze_end_of_turn()
            np.testing.assert_array_equal(model.audio, audio.astype(np.float32) / 32768)
            assert metrics.probability == 0.75
        finally:
            await analyzer.cleanup()

    asyncio.run(run())


def test_concurrent_analyzers_keep_independent_state(analyzer_class):
    model = FakeModel(0.25)
    analyzers = [analyzer_class(model=model), analyzer_class(model=model)]

    async def turn(analyzer, samples):
        analyzer.set_sample_rate(16000)
        try:
            for _ in range(20):
                analyzer.append_audio(np.ones(samples, dtype=np.int16).tobytes(), is_speech=True)
                state, metrics = await analyzer.analyze_end_of_turn()
                assert state.name == "INCOMPLETE"
                assert metrics.probability == 0.25
                analyzer.clear()
                state, metrics = await analyzer.analyze_end_of_turn()
                assert state.name == "INCOMPLETE"
                assert metrics is None
        finally:
            await analyzer.cleanup()

    async def run():
        await asyncio.gather(turn(analyzers[0], 320), turn(analyzers[1], 1600))

    asyncio.run(run())
