import { describe, expect, it } from 'vitest';
import { instrument } from './instruments';
import { bottomLine, reading, timing } from './presentation';
import {
  initialState,
  reduceCard,
  shownDecision,
  shownState,
  type Card,
  type Decision,
} from './state';

const card: Card = {
  task: 'speaker_count', title: 'People', text: '', kind: 'choice',
  options: ['1', '2'], status: 'validated', lane: 'encoder',
};
const decision: Decision = {
  task: card.task, question: '', kind: 'choice', value: '2',
  probability: null, probabilities: null, score: null, threshold: null,
  window_end_s: 72.75, window_start_s: 64.75, latency_ms: 12,
};
describe('card presentation', () => {
  it('shows turn-end yes/no and a 0.5 probability marker', () => {
    const turn = { ...card, task: 'turn_end', kind: 'noul' as const, options: [] };
    const no = { ...decision, task: 'turn_end', kind: 'noul' as const, value: false, probability: 0.49, threshold: 0.5 };
    const state = reduceCard(initialState(), no, 1000);
    expect(reading(turn, state.decision)).toBe('no');
    expect(reading(turn, { ...no, value: true, probability: 0.5 })).toBe('yes');
    expect(instrument(turn, state, 1000)).toContain('class="threshold" x1="150"');
  });
  it('renders speech in progress without a gauge needle or probability line', () => {
    const turn: Card = {
      ...card, task: 'turn_end', kind: 'noul', options: [], cadence: 'at end of speech',
    };
    const speaking: Decision = {
      ...decision,
      task: 'turn_end',
      kind: 'noul',
      value: false,
      probability: null,
      threshold: null,
      status: 'speaking',
      window_start_s: 2,
      window_end_s: 2,
      latency_ms: 0,
    };
    const state = reduceCard(initialState(), speaking, 1000);
    expect(reading(turn, state.decision)).toBe('no');
    expect(instrument(turn, state, 1000)).not.toContain('class="needle"');
    expect(bottomLine(turn, state.decision)).toBe('');
    expect(timing(turn, state, 1000).interval).toBe('SPEECH IN PROGRESS');
  });
  it('uses a neutral collecting state before any decision', () => {
    expect(reading(card)).toBe('—');
    expect(reading({ ...card, kind: 'noul' })).toBe('no');
    expect(reading({ ...card, kind: 'score' })).toBe('—');
    expect(timing(card, initialState(), 0).window).toBe('COLLECTING 8 s OF AUDIO');
    expect(timing({ ...card, context_seconds: 16 }, initialState(), 0).window)
      .toBe('COLLECTING 16 s OF AUDIO');
  });
  it('renders null noul values as no while other kinds stay neutral', () => {
    for (const value of [null, undefined]) {
      expect(reading({ ...card, kind: 'noul' }, { ...decision, value } as unknown as Decision))
        .toBe('no');
      for (const kind of ['choice', 'score'] as const) {
        expect(reading({ ...card, kind }, { ...decision, value } as unknown as Decision)).toBe('—');
      }
    }
  });
  it('returns to no after a held Term yes expires', () => {
    const term: Card = {
      ...card, task: 'term', title: 'Marimba spoken', kind: 'noul', options: [], lane: 'decoder',
    };
    const yes: Decision = { ...decision, task: 'term', kind: 'noul', value: true };
    const state = reduceCard(initialState(), yes, 0);
    const visible = shownDecision(term, state, 2001);
    expect(visible).toBeUndefined();
    expect(reading(term, visible)).toBe('no');
  });
  it('shows the held Turn yes with its original receipt age', () => {
    const turn: Card = {
      ...card, task: 'turn_end', kind: 'noul', options: [], cadence: 'at end of speech',
    };
    const yes: Decision = {
      ...decision,
      task: 'turn_end',
      kind: 'noul',
      value: true,
      window_end_s: 1,
      window_start_s: 0,
    };
    const speaking: Decision = {
      ...yes,
      value: false,
      probability: null,
      threshold: null,
      status: 'speaking',
      window_end_s: 2,
    };
    const withYes = reduceCard(initialState(), yes, 0, -100);
    const state = reduceCard(withYes, speaking, 500);
    const shown = shownState(turn, state, 1000);

    expect(timing(turn, shown, 1000).interval).toContain('RECEIVED 1.0 s AGO');
  });
  it('renders readings and the same timing fields for either lane', () => {
    expect(reading(card, decision)).toBe('2');
    expect(reading({ ...card, kind: 'noul' }, { ...decision, value: false })).toBe('no');
    expect(reading({ ...card, kind: 'score' }, { ...decision, kind: 'score', score: 0.7 })).toBe('0.70');
    const state = reduceCard(initialState(), decision, 1000);
    for (const lane of ['encoder', 'decoder'] as const) {
      const result = timing({ ...card, lane }, state, 1500);
      expect(result.window).toBe('AUDIO 64.75–72.75 s / AGE 0.5 s');
      expect(result.interval).toContain('UPDATE — / RECEIVED 0.5 s AGO');
      expect(result.latency).toBe('12 ms / model');
      expect(result.interval).toContain(lane === 'encoder' ? 'EVERY 250 MS' : 'PER UTTERANCE');
    }
  });
  it('shows offset cadence and preserves the reading and audio footer while pending', () => {
    const turn = { ...card, task: 'turn_end', cadence: 'at end of speech' as const };
    expect(timing(turn, initialState(), 0).window).toContain('WAITING FOR END OF SPEECH');
    expect(timing(turn, initialState(), 0).interval).toContain('AT END OF SPEECH');
    const previous = reduceCard(initialState(), decision, 1000, 800);
    const pending = reduceCard(previous, {
      ...decision, window_end_s: 73, value: null, status: 'not_enough_speech',
    }, 2000, 1800);
    expect(reading(card, pending.decision)).toBe('2');
    expect(timing(card, pending, 2000)).toEqual(timing(card, previous, 2000));
    const first = reduceCard(initialState(), {
      ...decision, value: null, status: 'not_enough_speech',
    }, 2000);
    expect(reading(card, first.decision)).toBe('—');
    expect(timing(card, first, 2000).latency).toBe('awaiting audio');
  });
});
