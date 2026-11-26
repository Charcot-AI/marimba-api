"""The fake model's signal rules, on synthetic audio only."""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from marimba import Question, WindowSnapshot, assemble_pair
from marimba.testing import FakeMarimba, frame_pitch, speaker_groups, trailing_drop_db

RATE = 16000
ROOT = Path(__file__).parents[1]
SPEECH = Question.score("Is someone speaking?")
TAIL = Question.noul("Is someone speaking?", seconds=1.0, threshold=0.1)
TURN_END = Question.score("Has the speaker finished?")
CHANGE = Question.score("Did the speaker change?")
SAME = Question.score("Is this the same speaker?")
COUNT = Question.choice("How many speakers?", ["0", "1", "2", "3+"])


def voice(f0: float, seconds: float, level: float = 0.1) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    wave = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 6))
    return (level * wave / np.sqrt(np.mean(wave**2))).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * RATE), dtype=np.float32)


def end_turn(utterance: np.ndarray, gain: float = 0.25) -> np.ndarray:
    out = utterance.copy()
    out[-round(0.3 * RATE) :] *= gain
    return out


def window(*parts: np.ndarray) -> np.ndarray:
    audio = np.concatenate(parts)
    assert len(audio) == 8 * RATE
    return audio


def ask(audio: np.ndarray, question: Question) -> float:
    return FakeMarimba().decide_windows([audio], [question])[0][0].probability


def rolling(stream: np.ndarray, question: Question, hop_s: float = 0.5) -> dict[float, float]:
    model = FakeMarimba()
    hop = round(hop_s * RATE)
    answers = {}
    for end in range(hop, len(stream) + 1, hop):
        heard = stream[max(0, end - 8 * RATE) : end]
        padded = np.pad(heard, (8 * RATE - len(heard), 0))
        answers[end / RATE] = model.decide_windows([padded], [question])[0][0].probability
    return answers


@pytest.mark.parametrize("rms", [0.04, 0.05, 0.1, 0.35])
def test_speech_level_audio_reads_as_speech_at_any_volume(rms: float) -> None:
    audio = voice(120.0, 8.0, level=rms)
    assert ask(audio, SPEECH) == pytest.approx(1.0, abs=1e-3)
    decision = FakeMarimba().decide_windows([audio], [TAIL])[0][0]
    assert decision.value is True and decision.threshold == 0.1


def test_silence_and_low_noise_do_not_read_as_speech() -> None:
    noise = np.random.default_rng(0).standard_normal(8 * RATE).astype(np.float32)
    assert ask(silence(8.0), SPEECH) == 0.0
    assert ask(noise * 10 ** (-70 / 20), SPEECH) == 0.0
    # Halfway between the two level bounds, in dB.
    assert ask(noise * 10 ** (-45 / 20), SPEECH) == pytest.approx(0.5, abs=0.02)


def test_voice_activity_follows_the_tail_the_question_asks_about() -> None:
    early = window(voice(120.0, 6.0), silence(2.0))
    late = window(silence(7.5), voice(120.0, 0.5))
    two = Question.score("Is someone speaking?", seconds=2.0)
    one = Question.score("Is someone speaking?", seconds=1.0)
    assert [ask(early, q) for q in (SPEECH, two, one)] == pytest.approx([0.75, 0.0, 0.0])
    assert [ask(late, q) for q in (SPEECH, two, one)] == pytest.approx([0.0625, 0.25, 0.5])


def test_voice_activity_ignores_frames_outside_the_mask() -> None:
    audio = window(silence(6.0), voice(120.0, 2.0))
    mask = np.zeros(8 * RATE, dtype=bool)
    mask[-2 * RATE :] = True
    model = FakeMarimba()
    snapshot = WindowSnapshot(audio, mask, 0.0, 2.0)
    decision = model.decide_encoded(model.encode(snapshot), [SPEECH])[0]
    assert decision.probability == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("f0", [70.0, 100.0, 160.0, 250.0, 390.0])
def test_pitch_of_a_harmonic_tone(f0: float) -> None:
    pitch = frame_pitch(voice(f0, 2.0))
    assert not np.isnan(pitch).any()
    np.testing.assert_allclose(2**pitch, f0, rtol=0.02)
    assert np.isnan(frame_pitch(silence(2.0))).all()
    noise = np.random.default_rng(0).standard_normal(2 * RATE).astype(np.float32)
    assert np.isnan(frame_pitch(0.1 * noise)).all()


def test_pair_questions_separate_two_voices() -> None:
    a, b = voice(100.0, 8.0), voice(250.0, 8.0)
    octaves = np.log2(2.5)
    different = octaves**2 / (octaves**2 + 0.25)
    for question, layout in ((SAME, 4), (CHANGE, 6)):
        split = layout * RATE
        one_voice = ask(a, question)
        two_voices = ask(np.concatenate((a[:split], b[split:])), question)
        swapped = ask(np.concatenate((b[:split], a[split:])), question)
        if question is SAME:
            assert one_voice == pytest.approx(1.0)
            assert [two_voices, swapped] == pytest.approx([1 - different] * 2, abs=1e-6)
        else:
            assert one_voice == pytest.approx(0.0)
            assert [two_voices, swapped] == pytest.approx([different] * 2, abs=1e-6)


def test_pair_scale_is_half_an_octave() -> None:
    a = voice(100.0, 4.0)
    for f0, expected in ((100.0, 1.0), (16000 / 113, 0.5), (200.0, 0.2)):
        answer = FakeMarimba().decide_pairs(SAME, [(a, voice(f0, 4.0))])[0]
        assert answer.probability == pytest.approx(expected, abs=0.01)
    # A noul answer uses the question threshold on the same probability.
    yes_no = Question.noul("Is this the same speaker?")
    answers = FakeMarimba().decide_pairs(yes_no, [(a, a), (a, voice(250.0, 4.0))])
    assert [d.value for d in answers] == [True, False]


def test_pairs_of_any_length_agree_with_assembled_windows() -> None:
    a, b = voice(100.0, 9.0), voice(250.0, 1.5)
    model = FakeMarimba()
    pairs = [(a, a[: 3 * RATE]), (a, b), (b, a), (b, b)]
    same = [d.probability for d in model.decide_pairs(SAME, pairs)]
    change = [d.probability for d in model.decide_pairs(CHANGE, pairs)]
    assert same == pytest.approx([1.0, 0.125, 0.125, 1.0], abs=0.01)
    assert change == pytest.approx([0.0, 0.875, 0.875, 0.0], abs=0.01)
    assembled = assemble_pair("speaker_change", a, b)
    assert ask(assembled, CHANGE) == change[1]


def test_pair_questions_need_voice_in_both_parts() -> None:
    a = voice(100.0, 8.0)
    noise = 0.1 * np.random.default_rng(0).standard_normal(8 * RATE).astype(np.float32)
    for other in (silence(8.0), noise):
        for question, split in ((SAME, 4 * RATE), (CHANGE, 6 * RATE)):
            assert ask(np.concatenate((a[:split], other[split:])), question) == 0.0
            assert ask(np.concatenate((other[:split], a[split:])), question) == 0.0


def test_rolling_speaker_change_peaks_on_the_voice_boundary() -> None:
    stream = np.concatenate((voice(100.0, 9.0), voice(250.0, 9.0)))
    answers = rolling(stream, CHANGE)
    peak = max(answers, key=answers.__getitem__)
    # The boundary at 9 s sits on the 6 s / 2 s split of the window ending at 11 s.
    assert peak == 11.0 and answers[peak] == pytest.approx(0.875, abs=0.01)
    assert sorted(answers.values())[-2] < answers[peak]
    assert all(
        value == pytest.approx(0.0, abs=1e-6) for end, value in answers.items() if end <= 9.0
    )
    assert answers[17.0] == pytest.approx(0.0, abs=1e-6)
    rising = [answers[end] for end in (9.5, 10.0, 10.5, 11.0)]
    falling = [answers[end] for end in (11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0)]
    assert rising == sorted(rising) and falling == sorted(falling, reverse=True)


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ([silence(8.0)], "0"),
        ([voice(100.0, 8.0)], "1"),
        ([silence(7.0), voice(100.0, 1.0)], "1"),
        ([silence(7.5), voice(100.0, 0.5)], "0"),
        ([voice(100.0, 4.0), voice(250.0, 4.0)], "2"),
        ([voice(100.0, 7.5), voice(250.0, 0.5)], "1"),
        ([voice(100.0, 2.0), silence(1.0), voice(250.0, 2.0), voice(100.0, 3.0)], "2"),
        ([voice(80.0, 3.0), voice(160.0, 3.0), voice(320.0, 2.0)], "3+"),
        ([voice(70.0, 2.0), voice(120.0, 2.0), voice(200.0, 2.0), voice(340.0, 2.0)], "3+"),
    ],
)
def test_speaker_count_follows_the_voices_in_the_window(
    parts: list[np.ndarray], expected: str
) -> None:
    audio = window(*parts)
    decision = FakeMarimba().decide_windows([audio], [COUNT])[0][0]
    assert decision.value == expected
    assert decision.probability == pytest.approx(0.7)
    assert decision.probabilities is not None
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)
    assert set(decision.probabilities) == {"0", "1", "2", "3+"}
    # Two voices in the count are two voices for the pair questions.
    if expected == "1":
        assert speaker_groups(audio) == 1


def test_speaker_count_agrees_with_the_pair_questions() -> None:
    a, b = voice(100.0, 4.0), voice(250.0, 4.0)
    model = FakeMarimba()
    for left, right, count in ((a, a, "1"), (a, b, "2")):
        audio = np.concatenate((left, right))
        same, counted = model.decide_windows([audio], [SAME, COUNT])[0]
        assert counted.value == count
        assert (same.probability >= 0.5) == (count == "1")


def test_turn_end_needs_a_trailing_drop_in_level() -> None:
    abrupt = window(silence(3.0), voice(120.0, 3.0), silence(2.0))
    trailing = window(silence(3.0), end_turn(voice(120.0, 3.0)), silence(2.0))
    assert ask(abrupt, TURN_END) == pytest.approx(0.0, abs=1e-3)
    assert trailing_drop_db(trailing, np.ones(8 * RATE, dtype=bool)) == pytest.approx(
        12.04, abs=0.3
    )
    assert ask(trailing, TURN_END) == pytest.approx(0.80, abs=0.01)
    # 6 dB is the midpoint; a larger drop reads higher.
    assert ask(window(silence(5.0), end_turn(voice(120.0, 3.0), 0.5)), TURN_END) == pytest.approx(
        0.5, abs=0.02
    )
    assert ask(window(silence(5.0), end_turn(voice(120.0, 3.0), 0.1)), TURN_END) > 0.9
    assert (
        FakeMarimba()
        .decide_windows([trailing], [Question.noul("Has the speaker finished?")])[0][0]
        .value
        is True
    )


def test_turn_end_is_no_without_speech_or_while_speech_continues() -> None:
    assert ask(silence(8.0), TURN_END) == 0.0
    assert ask(voice(120.0, 8.0), TURN_END) == pytest.approx(0.0, abs=1e-3)
    assert ask(window(silence(7.8), voice(120.0, 0.2)), TURN_END) == 0.0
    resumed = window(end_turn(voice(120.0, 3.0)), silence(1.0), voice(120.0, 4.0))
    assert ask(resumed, TURN_END) == pytest.approx(0.0, abs=1e-3)


def test_turn_end_holds_while_the_trail_off_is_the_last_sound() -> None:
    stream = np.concatenate((silence(1.0), end_turn(voice(120.0, 3.0)), silence(6.0)))
    answers = rolling(stream, TURN_END)
    assert all(answers[end] < 0.05 for end in (1.5, 2.0, 3.0, 3.5))
    assert all(answers[end] == pytest.approx(0.80, abs=0.01) for end in (4.0, 4.5, 6.0, 10.0))


def test_answers_are_deterministic_and_do_not_depend_on_the_question_text() -> None:
    audio = window(voice(100.0, 4.0), end_turn(voice(250.0, 3.0)), silence(1.0))
    questions = [SPEECH, TAIL, TURN_END, CHANGE, SAME, COUNT]
    first = FakeMarimba().decide_windows([audio], questions)[0]
    second = FakeMarimba().decide_windows([audio], questions[::-1])[0][::-1]
    assert [d.probability for d in first] == [d.probability for d in second]
    reworded = Question.score("Is the turn over?")
    assert ask(audio, reworded) == ask(audio, TURN_END)
    pcm = (audio * 32767).astype("<i2")
    assert ask(pcm.astype(np.float32) / 32768.0, CHANGE) == pytest.approx(
        ask(audio, CHANGE), abs=1e-3
    )


def test_decoder_questions_keep_the_placeholder_rule() -> None:
    model = FakeMarimba()
    language = Question.choice("What language?", ["English", "Portuguese"])
    loud, quiet = voice(120.0, 8.0, level=0.3), voice(120.0, 8.0, level=0.01)
    answers = [model(audio, [language], sampling_rate=RATE)[0] for audio in (loud, quiet)]
    assert all(set(d.probabilities) == {"English", "Portuguese"} for d in answers)
    assert answers[0].probability != answers[1].probability


def documented_recipe() -> dict[str, object]:
    doc = (ROOT / "docs" / "window-server.md").read_text()
    section = doc[doc.index("### Fixture recipe") :]
    code = re.search(r"```python\n(.*?)```", section, re.S)
    assert code is not None
    scope: dict[str, object] = {}
    exec(code.group(1), scope)
    return scope


def test_documented_fixture_triggers_every_decision() -> None:
    doc = (ROOT / "docs" / "window-server.md").read_text()
    scope = documented_recipe()
    stream = scope["stream"]
    assert isinstance(stream, np.ndarray) and len(stream) == 17 * RATE
    assert len(scope["pcm"]) == 2 * len(stream)
    model = FakeMarimba()
    questions = [Question.score("Is someone speaking?", seconds=1.0), TURN_END, CHANGE, COUNT]
    rows = re.findall(
        r"^\| ([\d.]+) s \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| `([^`]+)` \|", doc, re.M
    )
    assert len(rows) == 10
    for end_s, speech, turn_end, change, count in rows:
        end = round(float(end_s) * RATE)
        heard = stream[max(0, end - 8 * RATE) : end]
        padded = np.pad(heard, (8 * RATE - len(heard), 0))
        answers = model.decide_windows([padded], questions)[0]
        assert [d.probability for d in answers[:3]] == pytest.approx(
            [float(speech), float(turn_end), float(change)], abs=0.01
        ), end_s
        assert answers[3].value == count, end_s
    # Each decision is reached in both directions somewhere in the table.
    assert {row[4] for row in rows} == {"0", "1", "2"}
    assert {float(row[2]) >= 0.5 for row in rows} == {True, False}
    change_peaks = [float(row[0]) for row in rows if float(row[3]) == 0.87]
    assert change_peaks == [7.0, 13.0]
    a, b = scope["voice"](scope["A"], 4.0), scope["voice"](scope["B"], 4.0)
    same = [d.probability for d in model.decide_pairs(SAME, [(a, a), (a, b)])]
    change_pairs = [d.probability for d in model.decide_pairs(CHANGE, [(a, a), (a, b)])]
    assert same == pytest.approx([1.0, 0.125], abs=1e-3)
    assert change_pairs == pytest.approx([0.0, 0.875], abs=1e-3)
    for constant in ("0.125", "0.875", "`(a, a)`", "`(a, b)`"):
        assert constant in doc


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng is not installed")
def test_two_synthetic_speech_voices_are_separated(tmp_path: Path) -> None:
    from marimba.audio import decode_audio

    def speak(name: str, voice_name: str, pitch: int, text: str) -> np.ndarray:
        path = tmp_path / f"{name}.wav"
        subprocess.run(
            ["espeak-ng", "-v", voice_name, "-p", str(pitch), "-s", "150", "-w", str(path), text],
            check=True,
        )
        audio = decode_audio(str(path))
        loud = np.flatnonzero(np.abs(audio) > 0.01)
        return np.asarray(audio[loud[0] : loud[-1] + 1])

    low = [
        speak("a1", "en-us", 50, "I would like to book a table for two, please."),
        speak("a2", "en-us", 50, "Seven would be perfect, thank you. Can we sit outside?"),
    ]
    high = [
        speak("b1", "en-us+f3", 99, "Sure, what time would you like to come in?"),
        speak("b2", "en-us+f3", 99, "Of course. I have put you down for seven."),
    ]
    model = FakeMarimba()
    within = model.decide_pairs(SAME, [(low[0], low[1]), (high[0], high[1])])
    across = model.decide_pairs(SAME, [(low[0], high[0]), (high[1], low[1])])
    assert all(d.probability > 0.8 for d in within)
    assert all(d.probability < 0.2 for d in across)
    stream = np.concatenate((silence(1.0), low[0], high[0], silence(2.0)))
    speech = rolling(stream, Question.score("Is someone speaking?", seconds=1.0))
    start, end = 1.0, (len(stream) / RATE) - 2.0
    assert all(value >= 0.3 for at, value in speech.items() if start + 1.0 <= at <= end)
    assert speech[1.0] == 0.0 and speech[len(stream) // (RATE // 2) / 2] < 0.1
