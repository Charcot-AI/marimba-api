import asyncio
import threading

import numpy as np
import pytest
from marimba import Question, WindowSnapshot
from marimba.decision import Decision
from marimba.testing import FakeMarimba

from marimba_demo.protocol import Frame
from marimba_demo.scheduler import Scheduler, Work, utterance_chunks
from marimba_demo.speech import SpeechSegment, SpeechTimeline, speaker_change_eligible


def timeline():
    return SpeechTimeline()


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


def test_open_readout_run_can_make_speaker_change_eligible():
    speech = timeline()
    speech.observe(1, True)
    speech.observe(8, True)
    snapshot = WindowSnapshot(np.zeros(128000, np.float32), np.ones(128000, bool), 0, 8)
    assert speaker_change_eligible(snapshot, speech)


def test_coverage_includes_current_speech_without_removing_real_silence():
    speech = timeline()
    speech.observe(7, True)
    assert speech.onset == round(6.5 * 16000)
    assert speech.coverage(6, 8) == 0.5
    assert speech.coverage(0, 6) == 0


def test_timeline_applies_measured_onset_and_offset_lags():
    speech = timeline()
    assert speech.observe(1, False) == []
    assert speech.observe(1, True) == []
    assert speech.onset == 8000
    assert speech.observe(1.25, True) == []
    assert speech.coverage(0, 2) == 0.75

    (segment,) = speech.observe(3, False)
    assert segment == SpeechSegment(8000, 20000)
    assert segment.seconds == (0.5, 1.25)
    assert speech.onset is None
    assert speech.coverage(0, 2) == 0.75


def test_open_run_coverage_stops_at_last_observed_window_end():
    speech = timeline()
    speech.observe(1, True)
    assert speech.coverage(0, 2) == 0.5
    speech.observe(1.5, True)
    assert speech.coverage(0, 2) == 1


def test_short_run_never_closes_before_its_onset():
    speech = timeline()
    speech.observe(1, True)
    assert speech.observe(1.5, False) == [SpeechSegment(8000, 8000)]


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


def test_internal_tail_question_updates_timeline_without_sending_its_decision():
    async def run():
        updates = asyncio.Queue()
        model = Model()
        card_question = Question.noul("Is someone speaking?")
        scheduler = Scheduler(model, [card_question], updates.put, speech=timeline())
        internal = scheduler.internal_speech_question
        assert internal == Question.noul(
            "Is someone speaking?", task="voice_activity", seconds=2.0
        )

        snapshot = WindowSnapshot(
            np.zeros(128000, np.float32), np.ones(128000, bool), 0, 8
        )
        await scheduler.emit(
            Work(snapshot, (0, 0)), [card_question, internal], "encoder"
        )

        update = await updates.get()
        assert [decision["question"]["text"] for decision in update["decisions"]] == [
            "Is someone speaking?"
        ]
        assert scheduler.speech.last_observed_end == 128000

    asyncio.run(run())


def test_no_flip_closes_turn_window_at_the_lag_adjusted_sample():
    async def run():
        updates = asyncio.Queue()
        model = Model()
        voice = Question.noul(
            "Is someone speaking?", task="voice_activity", seconds=2.0
        )
        turn = Question.noul("Has the speaker finished?")
        scheduler = Scheduler(model, [voice, turn], updates.put, speech=timeline())
        scheduler.audio = np.ones(64000, np.float32)
        scheduler.total = 64000
        scheduler.latest_capture = (7, 3750.0)
        scheduler.latest_capture_start_sample = 60000
        speaking = iter((True, False))

        def compute(snapshot, questions, lane):
            return (
                [
                    Decision(
                        question,
                        question.task,
                        question.kind,
                        next(speaking),
                        0.8,
                        score=0.8,
                        threshold=question.resolved_threshold,
                        window_start_s=snapshot.window_start_s,
                        window_end_s=snapshot.window_end_s,
                        status="validated",
                    )
                    for question in questions
                ],
                0.0,
            )

        scheduler.compute = compute
        for end_s in (2.0, 4.0):
            snapshot = WindowSnapshot(
                np.zeros(128000, np.float32), np.ones(128000, bool), 0, end_s
            )
            await scheduler.emit(
                Work(snapshot, scheduler.latest_capture), [voice], "encoder"
            )

        (work,) = scheduler.turn_queue
        assert work.snapshot.window_end_s == 2.25
        assert work.window_end_at_ms == 2250
        assert scheduler.changed["turn"].is_set()

    asyncio.run(run())


def test_turn_end_emits_speaking_state_only_when_speech_opens():
    async def run():
        updates = asyncio.Queue()
        voice = Question.noul(
            "Is someone speaking?", task="voice_activity", seconds=2.0
        )
        turn = Question.noul("Has the speaker finished?", task="turn_end")
        scheduler = Scheduler(Model(), [voice, turn], updates.put, speech=timeline())
        speaking = iter((True, True))

        def compute(snapshot, questions, lane):
            return (
                [
                    Decision(
                        question,
                        question.task or "",
                        question.kind,
                        next(speaking),
                        probability=0.8,
                        score=0.8,
                        threshold=question.resolved_threshold,
                        window_start_s=snapshot.window_start_s,
                        window_end_s=snapshot.window_end_s,
                        status="validated",
                    )
                    for question in questions
                ],
                0.0,
            )

        scheduler.compute = compute
        for end_s in (2.0, 2.25):
            snapshot = WindowSnapshot(
                np.zeros(128000, np.float32), np.ones(128000, bool), 0, end_s
            )
            await scheduler.emit(Work(snapshot, (0, 0)), [voice], "encoder")

        first, second = await updates.get(), await updates.get()
        speaking_payloads = [
            decision
            for update in (first, second)
            for decision in update["decisions"]
            if decision["task"] == "turn_end"
        ]
        assert len(speaking_payloads) == 1
        payload = speaking_payloads[0]
        assert first["lane"] == "encoder"
        assert payload["question"]["text"] == turn.text
        assert payload["value"] is False
        assert payload["probability"] is None
        assert payload["probabilities"] is None
        assert payload["score"] is None
        assert payload["threshold"] is None
        assert payload["status"] == "speaking"
        assert payload["window_start_s"] == payload["window_end_s"] == 1.5
        assert payload["latency_ms"] == 0

    asyncio.run(run())


def test_review_emits_language_before_term_and_finishes_language_status_first():
    async def run():
        events = []

        class OrderedModel(FakeMarimba):
            def decide_snapshot(self, snapshot, questions):
                tasks = [question.task for question in questions]
                events.append(("infer", tuple(tasks)))
                decisions = []
                for question in questions:
                    language = question.task == "language"
                    decisions.append(
                        Decision(
                            question,
                            question.task or "",
                            question.kind,
                            "en" if language else True,
                            probability=0.9,
                            probabilities={"en": 0.9, "pt": 0.05, "es": 0.05}
                            if language
                            else None,
                            score=0.9,
                            threshold=question.resolved_threshold,
                            window_start_s=snapshot.window_start_s,
                            window_end_s=snapshot.window_end_s,
                            status="validated",
                        )
                    )
                return decisions

        async def send(message):
            if message["type"] == "review_status":
                events.append(("status", tuple(message["tasks"]), message["running"]))
            else:
                events.append(("update", message["decisions"][0]["task"]))

        language = Question.choice(
            "Which language is spoken?", ["en", "pt", "es"], task="language"
        )
        term = Question.noul('Was "marimba" spoken?', task="term")
        scheduler = Scheduler(OrderedModel(), [language, term], send, speech=timeline())
        scheduler.pending_utterance = [
            Work(
                WindowSnapshot(
                    np.zeros(128000, np.float32), np.ones(128000, bool), 0, 8
                ),
                (0, 0),
            ),
            Work(
                WindowSnapshot(
                    np.zeros(128000, np.float32), np.ones(128000, bool), 8, 16
                ),
                (1, 250),
            ),
        ]
        scheduler.finish()
        await asyncio.wait_for(scheduler.review(), 3)

        inference = [event[1] for event in events if event[0] == "infer"]
        emitted = [event[1] for event in events if event[0] == "update"]
        assert inference == [
            ("language",),
            ("language",),
            ("term",),
            ("term",),
        ]
        assert emitted == ["language", "language", "term", "term"]
        language_finished = events.index(("status", ("language",), False))
        first_term = events.index(("infer", ("term",)))
        assert language_finished < first_term

    asyncio.run(run())


def test_lagging_encoder_retains_complete_turn_and_utterance_windows():
    async def run():
        class LaggingModel(Model):
            def encode(self, snapshot):
                self.started.set()
                assert self.release.wait(5)
                return super().encode(snapshot)

            def decide_encoded(self, encoded, questions):
                return [
                    Decision(
                        question,
                        question.task or "",
                        question.kind,
                        False,
                        probability=0.1,
                        score=0.1,
                        threshold=question.resolved_threshold,
                        window_start_s=encoded.snapshot.window_start_s,
                        window_end_s=encoded.snapshot.window_end_s,
                        status="validated",
                    )
                    for question in questions
                ]

        updates = asyncio.Queue()
        model = LaggingModel()
        model.release.clear()
        speech = timeline()
        scheduler = Scheduler(
            model,
            [
                Question.noul(
                    "Is someone speaking?", task="voice_activity", seconds=2.0
                ),
                Question.noul("Has the speaker finished?"),
                Question.noul('Was "marimba" spoken?'),
            ],
            updates.put,
            speech=speech,
        )
        for i in range(52):
            scheduler.ingest(packet(i))
        speech.observe(6.5, True)
        snapshot = scheduler.snapshot(2 * 16000, 10 * 16000)
        emitting = asyncio.create_task(
            scheduler.emit(
                Work(snapshot, scheduler.latest_capture),
                [scheduler.speech_question],
                "encoder",
            )
        )
        try:
            assert await asyncio.to_thread(model.started.wait, 2)
            for i in range(52, 64):
                scheduler.ingest(packet(i))
            assert scheduler.total == 16 * 16000
            model.release.set()
            await asyncio.wait_for(emitting, 3)

            (turn_work,) = scheduler.turn_queue
            turn_snapshot = turn_work.snapshot
            assert turn_snapshot.window_start_s == 0.25
            assert turn_snapshot.window_end_s == 8.25
            assert turn_snapshot.mask.all()
            expected_turn = np.repeat(np.arange(2, 34, dtype=np.float32), 4000) / 32768
            np.testing.assert_array_equal(turn_snapshot.audio, expected_turn)

            (utterance_work,) = scheduler.pending_utterance
            utterance_snapshot = utterance_work.snapshot
            assert utterance_snapshot.window_start_s == 6.0
            assert utterance_snapshot.window_end_s == 8.25
            assert utterance_snapshot.mask.sum() == 36000
            assert not utterance_snapshot.mask[:-36000].any()
            expected_utterance = (
                np.repeat(np.arange(25, 34, dtype=np.float32), 4000) / 32768
            )
            np.testing.assert_array_equal(
                utterance_snapshot.audio[-36000:], expected_utterance
            )
        finally:
            model.release.set()
            scheduler.finish(aborted=True)
            await asyncio.gather(emitting, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(("start", "end"), [(0, 8000), (4000, 12001)])
def test_snapshot_rejects_ranges_outside_retained_audio(start, end):
    scheduler = Scheduler(Model(), [], None, speech=timeline())
    scheduler.audio = np.zeros(8000, np.float32)
    scheduler.audio_start = 4000
    scheduler.total = 12000

    with pytest.raises(ValueError, match="outside retained audio"):
        scheduler.snapshot(start, end)


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
            scheduler.close_utterance(SpeechSegment(200, 3500))
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
            assert result["window_end_at_ms"] == 218.75
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
            assert model.calls[0][1] == ["speaker_count", "voice_activity"]
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
            scheduler.speech.observe((i + 1) * 0.25, False)
        scheduler.close_utterance(SpeechSegment(0, 2000))
        worker = asyncio.create_task(scheduler.review())
        try:
            assert await asyncio.to_thread(model.started.wait, 2)
            scheduler.close_utterance(SpeechSegment(4000, 6000))
            scheduler.close_utterance(SpeechSegment(8000, 10000))
            assert scheduler.dropped_utterances == 1
            for i in range(4, 44):
                scheduler.ingest(packet(i))
                scheduler.speech.observe((i + 1) * 0.25, False)
            assert scheduler.audio_start == 4000
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
    scheduler.close_utterance(SpeechSegment(1000, 300000))
    chunks = scheduler.pending_utterance
    assert [w.snapshot.mask.sum() for w in chunks] == [128000, 128000, 43000]
    joined = np.concatenate([w.snapshot.audio[w.snapshot.mask] for w in chunks])
    expected = np.repeat(np.arange(1, 81, dtype=np.float32), 4000)[1000:300000] / 32768
    np.testing.assert_array_equal(joined, expected)
