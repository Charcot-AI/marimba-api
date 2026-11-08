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


def test_audio_is_resampled_to_model_rate(analyzer_class):
    model = FakeModel()
    analyzer = analyzer_class(model=model, sample_rate=8000)
    analyzer.set_sample_rate(8000)
    audio = np.linspace(-1, 1, 8000, dtype=np.float32)
    resampled = soxr.resample(audio, 8000, 16000, quality="HQ")

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
