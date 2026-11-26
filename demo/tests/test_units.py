import numpy as np
import pytest

from marimba_demo.speech import SpeechSegment
from marimba_demo.units import (
    SegmentDetector,
    SpeechUnit,
    UnitConfig,
    UnitJoiner,
    frame_energy_db,
    snap_edge,
    snap_units,
)

RATE = 16000


def seg(start_s, end_s):
    return SpeechSegment(round(start_s * RATE), round(end_s * RATE))


def observe(detector, readings, hop_s=0.5):
    segments = []
    for index, probability in enumerate(readings):
        segments += detector.observe((index + 1) * hop_s, probability)
    return segments


def test_detector_opens_and_closes_on_one_threshold_with_lag_correction():
    detector = SegmentDetector()
    # 0.1 opens, 0.0999 closes: one threshold, no hangover.
    readings = [0.0, 0.1, 0.9, 0.0999, 0.5, 0.2, 0.0, 0.0]
    segments = observe(detector, readings)
    # Opening window ends at 1.0 s; first window below the threshold ends at 2.0 s.
    assert [s.seconds for s in segments] == [(0.5, 1.25), (2.0, 2.75)]
    assert detector.onset is None


def test_detector_drops_short_segments_and_slices_long_runs():
    config = UnitConfig(onset_lag_s=0.5, offset_lag_s=0.75, min_segment_s=0.25)
    # Opens at 0.5 - 0.5 = 0.0 and closes at 1.0 - 0.75 = 0.25: exactly the minimum.
    assert [s.seconds for s in observe(SegmentDetector(config), [0.5, 0.0])] == [
        (0.0, 0.25)
    ]
    shorter = UnitConfig(min_segment_s=0.3)
    assert observe(SegmentDetector(shorter), [0.5, 0.0]) == []
    detector = SegmentDetector(UnitConfig(max_unit_s=2.0))
    segments = observe(detector, [1.0] * 12 + [0.0])
    assert [s.seconds for s in segments] == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.75)]


def test_joiner_groups_short_pauses_until_a_turn_end_or_the_cap():
    joiner = UnitJoiner()
    assert joiner.add(seg(0, 2)) == []
    assert joiner.add(seg(7, 9)) == []  # a 5 s pause still joins
    closed = joiner.add(seg(14.01, 15))  # 5.01 s does not
    assert [u.seconds for u in closed] == [(0.0, 9.0)]
    closed = joiner.add(seg(16, 17), turn_ended=True)
    assert [(u.seconds, u.turn_ended) for u in closed] == [((14.01, 17.0), True)]
    assert joiner.pending is None
    assert joiner.add(seg(17.5, 40)) == []
    # Joining would exceed 30 s, so the pending unit closes.
    assert [u.seconds for u in joiner.add(seg(41, 48))] == [(17.5, 40.0)]
    assert [u.seconds for u in joiner.finish()] == [(41.0, 48.0)]
    assert joiner.finish() == []


def test_joiner_slices_segments_longer_than_the_cap():
    joiner = UnitJoiner()
    closed = joiner.add(seg(0, 70), turn_ended=True)
    assert [(u.seconds, u.turn_ended) for u in closed] == [
        ((0.0, 30.0), False),
        ((30.0, 60.0), False),
        ((60.0, 70.0), True),
    ]


def test_joiner_releases_a_full_unit_without_waiting_for_more_speech():
    joiner = UnitJoiner()
    # A 30 s slice of continuous speech cannot grow, so it is ready at once.
    assert [u.seconds for u in joiner.add(seg(0, 30))] == [(0.0, 30.0)]
    assert joiner.pending is None
    assert joiner.add(seg(30, 40)) == []
    # Joined speech that reaches the cap exactly is released too.
    assert [u.seconds for u in joiner.add(seg(41, 60))] == [(30.0, 60.0)]


def test_joiner_closes_a_unit_once_the_pause_cannot_be_bridged():
    joiner = UnitJoiner()
    joiner.add(seg(0, 2))
    # A segment opening on the next hop would start at 7.5 - 0.5 = 7.0 s: still 5 s.
    assert joiner.advance(7.5) == []
    assert [u.seconds for u in joiner.advance(8.0)] == [(0.0, 2.0)]
    joiner.add(seg(10, 12))
    # An open segment that started within the pause keeps the unit pending.
    assert joiner.advance(30.0, onset=round(16.5 * RATE)) == []
    assert len(joiner.advance(30.0, onset=round(17.5 * RATE))) == 1


def tone(seconds, level=0.3):
    t = np.arange(round(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds, level=1e-4):
    return np.full(round(seconds * RATE), level, dtype=np.float32)


def test_energy_is_smoothed_over_five_frames_without_darker_edges():
    energy = frame_energy_db(np.ones(RATE, dtype=np.float32))
    assert energy.shape == (100,)
    np.testing.assert_allclose(energy, 0.0, atol=1e-9)
    step = frame_energy_db(
        np.concatenate((quiet(0.5, 0.0), np.ones(RATE // 2, np.float32)))
    )
    # A centered five-frame mean sees 1, 2, 3 loud frames just before the step.
    np.testing.assert_allclose(
        step[47:52], 10 * np.log10(np.array([0, 1, 2, 3, 4]) / 5 + 1e-12), atol=1e-6
    )


def test_snap_picks_the_quietest_frame_and_breaks_ties_towards_the_edge():
    audio = np.concatenate((tone(1.0), quiet(0.2), tone(1.0)))
    energy = frame_energy_db(audio)
    total = len(audio)
    # Raw edge 0.3 s early: the quiet gap at 1.0 to 1.2 s is within 0.5 s.
    edge = snap_edge(energy, round(0.7 * RATE), 0, total)
    assert 1.02 * RATE <= edge <= 1.18 * RATE
    # All frames of a flat signal tie at 1 dB resolution, so the edge stays in its frame.
    flat = frame_energy_db(tone(3.0))
    assert (
        abs(snap_edge(flat, round(1.503 * RATE), 0, 3 * RATE) - round(1.503 * RATE))
        < 160
    )
    # Out of reach: the search is limited to 0.5 s and to the neighbor bounds.
    assert snap_edge(energy, round(0.4 * RATE), 0, total) < 0.9 * RATE + 80
    assert snap_edge(energy, round(0.7 * RATE), 0, round(0.9 * RATE)) <= 0.9 * RATE
    assert snap_edge(energy, 100, 200, 100) == 100


def test_snap_units_moves_unit_edges_and_pads_within_the_gap():
    audio = np.concatenate(
        (
            quiet(1.0),
            tone(2.0),
            quiet(0.3),
            tone(2.0),
            quiet(2.0),
            tone(1.0),
            quiet(1.0),
        )
    )
    units = [
        SpeechUnit(round(1.2 * RATE), round(2.8 * RATE)),
        SpeechUnit(round(3.5 * RATE), round(5.6 * RATE)),
        SpeechUnit(round(7.1 * RATE), round(8.5 * RATE), turn_ended=True),
    ]
    first, second, third = snap_units(audio, units)
    # Unit edges land in the quiet audio next to each true boundary.
    assert 0.5 <= first.start / RATE <= 1.0 and 3.0 <= first.end / RATE <= 3.3
    assert first.end <= second.start <= 3.3 * RATE
    assert 5.3 <= second.end / RATE <= 6.1 and 6.6 <= third.start / RATE <= 7.3
    assert third.turn_ended and third.end <= len(audio)
    # The 0.3 s gap is shared at its midpoint; open sides get the full 0.25 s margin.
    assert first.audio_end == (first.end + second.start) // 2
    assert second.audio_start == (first.end + second.start + 1) // 2
    assert first.audio_start == first.start - 4000
    assert second.audio_end == second.end + 4000
    assert third.audio_start == third.start - 4000
    assert third.audio_end == min(len(audio), third.end + 4000)
    for unit in (first, second, third):
        assert unit.audio_start <= unit.start < unit.end <= unit.audio_end


def test_snap_units_respects_the_audio_offset_and_bounds():
    audio = np.concatenate((quiet(0.1), tone(1.0), quiet(0.1)))
    offset = 5 * RATE
    (unit,) = snap_units(
        audio, [SpeechUnit(offset + 1600, offset + 17600)], audio_start=offset
    )
    assert offset <= unit.audio_start <= unit.start <= offset + 1600
    assert offset + 17600 <= unit.end <= unit.audio_end <= offset + len(audio)


@pytest.mark.parametrize("field", ["threshold", "onset_lag_s", "offset_lag_s"])
def test_detector_parameters_are_configurable(field):
    values = {"threshold": 0.6, "onset_lag_s": 0.25, "offset_lag_s": 0.5}
    detector = SegmentDetector(UnitConfig(**{field: values[field]}))
    default = observe(SegmentDetector(), [0.0, 0.5, 0.9, 0.9, 0.0])
    assert observe(detector, [0.0, 0.5, 0.9, 0.9, 0.0]) != default
