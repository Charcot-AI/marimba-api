import { describe, expect, it } from 'vitest';
import { reading, timing } from './presentation';
import { initialState, reduceCard, type Card, type Decision } from './state';

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
  it('uses a neutral collecting state before any decision', () => {
    expect(reading(card)).toBe('—');
    expect(timing(card, initialState(), 0).window).toBe('COLLECTING 8 s OF AUDIO');
    expect(timing({ ...card, context_seconds: 16 }, initialState(), 0).window)
      .toBe('COLLECTING 16 s OF AUDIO');
  });
  it('does not render missing runtime values as null, undefined, or no', () => {
    for (const value of [null, undefined]) {
      for (const kind of ['choice', 'score', 'noul'] as const) {
        expect(reading({ ...card, kind }, { ...decision, value } as unknown as Decision)).toBe('—');
      }
    }
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
      expect(result.interval).toContain(lane === 'encoder' ? 'EVERY ENCODER TICK' : 'MIN PAUSE 2 s AFTER COMPLETION');
    }
  });
});
