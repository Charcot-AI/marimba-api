import numpy as np
import pytest
from marimba import Question
from marimba.decision import Decision
from marimba.testing import FakeMarimba

from marimba_demo.pipeline import PipelineConfig, run_pipeline, window_ending_at
from marimba_demo.speakers import (
    CutConfig,
    PrimerBank,
    Word,
    change_peaks,
    cut_candidates,
    snap_cuts,
    speaker_cuts,
)
from marimba_demo.units import UnitConfig, snap_units

RATE = 16000


def tone(seconds, frequency, level=0.3):
    t = np.arange(round(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def share(audio, frequency):
    """Fraction of the signal power within 20 Hz of ``frequency``."""
    power = np.abs(np.fft.rfft(audio)) ** 2
    bins = np.fft.rfftfreq(len(audio), 1 / RATE)
    return float(power[np.abs(bins - frequency) < 20].sum() / max(power.sum(), 1e-12))


def quiet(seconds):
    return np.zeros(round(seconds * RATE), dtype=np.float32)


def pitch(audio):
    if not np.abs(audio).max() > 0.01:
        return 0.0
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return float(np.argmax(spectrum) * RATE / len(audio))


class ToneModel(FakeMarimba):
    """Synthetic readouts with the lags the demo corrects for; each pitch is a speaker."""

    def __init__(self, turn_end=0.0):
        super().__init__()
        self.turn_end = turn_end
        self.calls = {
            "voice_activity": 0,
            "turn_end": 0,
            "same_speaker": 0,
            "speaker_change": 0,
        }

    def decide_window(self, window, mask, questions, start_s, end_s):
        decisions = []
        for question in questions:
            self.calls[question.task] += 1
            if question.task == "voice_activity":
                # Voiced about 0.5 s after an onset and silent about 0.75 s after an offset.
                probed = window[-round(0.8 * RATE) : -round(0.45 * RATE)]
                probability = float(np.mean(np.abs(probed) > 0.01))
            elif question.task == "turn_end":
                probability = self.turn_end
            else:
                split = 4 * RATE if question.task == "same_speaker" else 6 * RATE
                left, right = pitch(window[split - RATE : split]), pitch(window[split:])
                same = abs(left - right) < 0.1 * max(left, right, 1.0)
                probability = float(same == (question.task == "same_speaker"))
            decisions.append(
                Decision(
                    question,
                    question.task,
                    question.kind,
                    probability >= question.resolved_threshold
                    if question.kind == "noul"
                    else probability,
                    probability,
                    threshold=question.resolved_threshold,
                    window_start_s=start_s,
                    window_end_s=end_s,
                    status="experimental",
                )
            )
        return decisions


def words_every(step_s, sentence_ends=()):
    def transcribe(audio, start_s):
        count = int(len(audio) / RATE / step_s)
        words = []
        for index in range(count):
            end_s = start_s + (index + 1) * step_s
            spoken = audio[
                round(index * step_s * RATE) : round((index + 1) * step_s * RATE)
            ]
            if np.mean(np.abs(spoken) > 0.01) < 0.5:
                continue
            final = any(abs(end_s - mark) < step_s / 2 for mark in sentence_ends)
            words.append(Word("word." if final else "word", end_s - step_s, end_s))
        return words

    return transcribe


def test_window_ending_at_pads_on_the_left():
    audio = np.arange(3 * RATE, dtype=np.float32)
    window = window_ending_at(audio, 2 * RATE)
    assert window.shape == (8 * RATE,)
    assert not window[: 6 * RATE].any() and window[-1] == 2 * RATE - 1
    np.testing.assert_array_equal(
        window_ending_at(np.arange(10 * RATE), 9 * RATE)[0], RATE
    )


def test_candidates_and_peaks_are_spaced():
    words = [Word("w", index * 0.2, index * 0.2 + 0.2) for index in range(11)]
    assert cut_candidates(words) == [0, 3, 6, 9]
    assert cut_candidates(words, CutConfig(candidate_step_s=0.0)) == list(range(10))
    scores = [(1.0, 0.6), (2.0, 0.9), (3.9, 0.8), (4.0, 0.7), (6.0, 0.49), (9.0, 0.5)]
    # 3.9 is 1.9 s from the strongest peak, 4.0 is exactly 2 s away.
    assert change_peaks(scores) == [(2.0, 0.9), (4.0, 0.7), (9.0, 0.5)]
    assert change_peaks(scores, CutConfig(threshold=0.85, min_peak_gap_s=0.5)) == [
        (2.0, 0.9)
    ]


def test_cuts_snap_to_a_sentence_end_in_reach_or_the_nearest_word():
    texts = ["a", "b.", "c", "d", "e", "f", "g", "h?", "i", "j", "k", "l"]
    words = [
        Word(text, index * 0.5, index * 0.5 + 0.5) for index, text in enumerate(texts)
    ]
    cuts = snap_cuts(words, [(2.5, 0.9), (4.5, 0.8), (5.4, 0.7), (5.6, 0.6)])
    # 2.5 s is exactly 1.5 s from the sentence end at 1.0 s; 5.6 s is out of reach of 4.0 s.
    assert [(c.word_index, c.time_s, c.at_sentence_end) for c in cuts] == [
        (1, 1.0, True),
        (7, 4.0, True),
        (10, 5.5, False),
    ]
    assert cuts[1].peak_s == 4.5  # two peaks on one word keep the stronger one
    far = snap_cuts(words, [(2.6, 0.9)], CutConfig(sentence_snap_s=0.4))
    assert [(c.word_index, c.at_sentence_end) for c in far] == [(4, False)]
    assert snap_cuts(words[:1], [(0.2, 0.9)]) == []


def test_speaker_cuts_score_six_seconds_before_and_two_after_each_candidate():
    audio = np.concatenate((tone(7.0, 200), tone(5.0, 400)))
    words = [Word("w.", index * 0.5, index * 0.5 + 0.5) for index in range(24)]
    seen = []

    def score(pairs):
        seen.extend((len(reference), len(segment)) for reference, segment in pairs)
        return [share(s, 400) * (1 - share(r[-2 * RATE :], 400)) for r, s in pairs]

    def audio_at(start_s, end_s):
        return audio[max(0, round(start_s * RATE)) : round(end_s * RATE)]

    cuts = speaker_cuts(words, score, audio_at)
    assert [(c.time_s, round(c.probability, 2)) for c in cuts] == [(7.0, 1.0)]
    assert seen[0] == (8000, 32000) and seen[13] == (96000, 32000)
    # The last candidates have less than two seconds left in the unit.
    assert seen[-1] == (96000, 8000)
    assert speaker_cuts(words[:1], score, audio_at) == []


def test_primer_bank_starts_empty_and_is_capped():
    asked = []

    def score(pairs):
        asked.append(len(pairs))
        return [float(abs(pitch(a) - pitch(b)) < 20) for a, b in pairs]

    bank = PrimerBank()
    assert bank.primers == []
    order = [200, 300, 200, 200, 200, 400, 500, 600, 300]
    speakers = [bank.assign(tone(3.0, f), score)[0] for f in order]
    # The fifth distinct voice finds the bank full and takes its closest speaker.
    assert speakers[:7] == [0, 1, 0, 0, 0, 2, 3] and speakers[8] == 1
    assert 0 <= speakers[7] < 4
    assert len(bank.primers) == 4
    assert [len(group) for group in bank.primers] == [3, 2, 1, 1]
    # The first span has nothing to be compared with; later ones meet every primer.
    assert asked == [1, 2, 3, 4, 4, 5, 6, 6]
    bank.assign(tone(9.0, 300), score)
    assert [len(primer) / RATE for primer in bank.primers[1]] == [3.0, 3.0, 4.0]


def test_primer_bank_keeps_short_spans_out_unless_they_open_a_speaker():
    def score(pairs):
        return [float(abs(pitch(a) - pitch(b)) < 20) for a, b in pairs]

    bank = PrimerBank()
    assert bank.assign(tone(1.0, 200), score) == (0, None)
    assert bank.assign(tone(1.0, 200), score) == (0, 1.0)
    assert [len(group) for group in bank.primers] == [1]
    bank.assign(tone(2.0, 200), score)
    assert [len(group) for group in bank.primers] == [2]


@pytest.fixture
def conversation():
    # Speaker A, a 1 s pause, A again, then B without a pause, a 7 s pause, then C.
    parts = [
        (quiet(1.0), None),
        (tone(3.0, 200), 0),
        (quiet(1.0), None),
        (tone(4.0, 200), 0),
        (tone(4.0, 400), 1),
        (quiet(7.0), None),
        (tone(3.0, 600), 2),
        (quiet(3.0), None),
    ]
    return np.concatenate([audio for audio, _ in parts])


def test_pipeline_builds_units_cuts_and_speakers_at_a_500_ms_hop(conversation):
    model = ToneModel()
    result = run_pipeline(
        model,
        Question,
        conversation,
        transcribe=words_every(0.25, sentence_ends=(4.0, 9.0)),
    )
    assert len(result.hops) == 52 and result.hops[0][0] == 0.5
    assert [s.seconds for s in result.segments] == [
        (1.0, 4.25),
        (5.0, 13.25),
        (20.0, 23.25),
    ]
    # The 1 s pause is joined, the 7 s pause is not; edges snap into the silence.
    assert len(result.units) == 2
    first, second = result.units
    assert 0.5 <= first.start / RATE <= 1.0 and 13.0 <= first.end / RATE <= 13.5
    assert 19.5 <= second.start / RATE <= 20.0 and 23.0 <= second.end / RATE <= 23.5
    assert first.audio_start == max(0, first.start - 4000)
    assert first.audio_end == first.end + 4000
    # A unit is emitted once the pause can no longer be bridged, from audio heard so far.
    assert result.emitted_at_s == [19.0, 26.0]
    # Words follow the padded unit audio, so the cut is one 10 ms frame off the true change.
    assert [c.time_s for c in result.cuts[0]] == [pytest.approx(9.0, abs=0.05)]
    assert result.cuts[1] == []
    assert result.cuts[0][0].at_sentence_end
    spans = [(round(s.start_s, 2), round(s.end_s, 2), s.speaker) for s in result.spans]
    assert [speaker for *_, speaker in spans] == [0, 1, 2]
    assert spans[0][1] == spans[1][0] == pytest.approx(9.0, abs=0.05)
    assert model.calls["turn_end"] == 3


def test_pipeline_matches_offline_snapping_and_splits_at_turn_ends(conversation):
    model = ToneModel(turn_end=1.0)
    result = run_pipeline(model, Question, conversation)
    assert [u.turn_ended for u in result.units] == [True, True, True]
    assert result.words == [[], [], []] and result.cuts == [[], [], []]
    # Each unit is emitted on the hop that closes its segment.
    assert result.emitted_at_s == [5.0, 14.0, 24.0]
    raw = [
        type(result.units[0])(s.start, s.end, turn_ended=True) for s in result.segments
    ]
    offline = snap_units(conversation, raw)
    assert [(u.start, u.end) for u in result.units] == [
        (u.start, u.end) for u in offline
    ]
    for earlier, later in zip(result.units, result.units[1:]):
        assert earlier.audio_end <= later.audio_start
    assert [s.speaker for s in result.spans] == [0, 0, 1]


def test_pipeline_parameters_are_configurable(conversation):
    config = PipelineConfig(
        hop_s=0.25,
        units=UnitConfig(max_pause_s=0.5, margin_s=0.1),
        max_speakers=1,
    )
    result = run_pipeline(ToneModel(), Question, conversation, config=config)
    assert len(result.hops) == 104
    assert len(result.units) == 3
    assert result.units[0].audio_start == result.units[0].start - 1600
    assert {s.speaker for s in result.spans} == {0}


def test_pipeline_closes_speech_that_runs_to_the_end():
    audio = np.concatenate((quiet(1.0), tone(3.0, 200)))
    result = run_pipeline(FakeMarimba(), Question, audio)
    assert len(result.units) == 1
    assert result.units[0].end <= len(audio) and result.units[0].audio_end <= len(audio)
