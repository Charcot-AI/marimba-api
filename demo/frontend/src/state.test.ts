import { describe, expect, it } from 'vitest';
import {
  encodeFrame,
  defaultThreshold,
  initialState,
  isRollingUpdate,
  percentile,
  reduceCard,
  renderLatency,
  shownDecision,
  type Card,
  type Decision,
} from './state';

const decision = (p: number, end = 1): Decision => ({
  task: 'voice_activity',
  question: 'Speech?',
  kind: 'noul',
  value: p >= 0.5,
  probability: p,
  probabilities: null,
  score: null,
  threshold: 0.5,
  window_end_s: end,
  latency_ms: 12,
});
const termCard: Card = {
  task: 'term',
  title: 'Marimba spoken',
  text: 'Was "marimba" spoken?',
  kind: 'noul',
  options: [],
  status: 'unvalidated',
  lane: 'decoder',
};
describe('card state', () => {
  it('holds a positive term decision for two seconds', () => {
    const yes = { ...decision(1), task: 'term', value: true };
    const state = reduceCard(initialState(), yes, 0);
    expect(shownDecision(termCard, state, 1999)).toBe(yes);
    expect(shownDecision(termCard, state, 2001)).toBeUndefined();
  });
  it('keeps a negative term decision visible', () => {
    const no = { ...decision(0), task: 'term', value: false };
    const state = reduceCard(initialState(), no, 0);
    expect(shownDecision(termCard, state, 5000)).toBe(no);
  });
  it('keeps positive non-term decisions visible', () => {
    const turnEndCard = { ...termCard, task: 'turn_end' };
    const yes = { ...decision(1), task: 'turn_end', value: true };
    const state = reduceCard(initialState(), yes, 0);
    expect(shownDecision(turnEndCard, state, 5000)).toBe(yes);
  });
  it('retains the last real decision, trail and timing while an update is ineligible', () => {
    const first = reduceCard(initialState(), decision(0.2, 8), 1000, 800);
    const pending = reduceCard(first, {
      ...decision(0, 9), value: null, status: 'not_enough_speech',
    }, 2000, 1800);
    expect(pending).toEqual({ ...first, pending: true });
    expect(pending.decision).toBe(first.decision);
    expect(pending.trail).toBe(first.trail);
    const next = reduceCard(pending, decision(0.4, 10), 3000, 2800);
    expect(next.pending).toBe(false);
    expect(next.decision?.window_end_s).toBe(10);
    expect(next.trail).toEqual([{ time: 1000, value: 0.2 }, { time: 3000, value: 0.4 }]);
  });
  it('keeps an empty reading pending until the first real decision', () => {
    const pending = reduceCard(initialState(), {
      ...decision(0), value: null, status: 'not_enough_speech',
    }, 1000);
    expect(pending).toEqual({ ...initialState(), pending: true });
    const next = reduceCard(pending, decision(0.2, 2), 2000);
    expect(next.pending).toBe(false);
    expect(next.trail).toEqual([{ time: 2000, value: 0.2 }]);
    expect(next.crossedAt).toBe(-Infinity);
  });
  it('ignores an older ineligible update after a newer real decision', () => {
    const first = reduceCard(initialState(), decision(0.2, 8), 1000);
    expect(reduceCard(first, {
      ...decision(0, 7), value: null, status: 'not_enough_speech',
    }, 2000)).toBe(first);
    expect(first.pending).toBe(false);
  });
  it('does not replace rolling counters or watchdog timing with offset-triggered turns', () => {
    expect(isRollingUpdate('encoder', [{ ...decision(0.9), task: 'turn_end' }])).toBe(false);
    expect(isRollingUpdate('encoder', [decision(0.9)])).toBe(true);
    expect(isRollingUpdate('decoder', [{ ...decision(0.9), task: 'language' }])).toBe(false);
    expect(isRollingUpdate('encoder', [])).toBe(false);
  });
  it('uses task-specific defaults when a decision has no threshold', () => {
    expect(defaultThreshold('voice_activity')).toBe(0.225);
    expect(defaultThreshold('turn_end')).toBe(0.5);
    const first = reduceCard(initialState(), { ...decision(0.2), threshold: null }, 100);
    expect(
      reduceCard(first, { ...decision(0.23, 2), threshold: null }, 300).crossedAt,
    ).toBe(300);
  });
  it('does not flash on the first result; flashes only at threshold crossings', () => {
    const first = reduceCard(initialState(), decision(0.1), 100);
    expect(first.crossedAt).toBe(-Infinity);
    expect(reduceCard(first, decision(0.8, 2), 300).crossedAt).toBe(300);
    expect(reduceCard(first, decision(0.2, 2), 300).crossedAt).toBe(-Infinity);
  });
  it('retains only the five-second score trail', () => {
    const first = reduceCard(initialState(), { ...decision(0.1), kind: 'score', score: 0.1 }, 100);
    const second = reduceCard(first, { ...decision(0.8, 2), kind: 'score', score: 0.8 }, 5200);
    expect(second.trail).toEqual([{ time: 5200, value: 0.8 }]);
  });
  it('ignores out-of-order results', () => {
    const first = reduceCard(initialState(), decision(0.1, 5), 100);
    expect(reduceCard(first, decision(0.8, 2), 200)).toBe(first);
  });
  it('ignores duplicate results from the same window', () => {
    const first = reduceCard(initialState(), decision(0.1, 20), 100);
    expect(reduceCard(first, decision(0.8, 20), 200)).toBe(first);
  });
  it('orders speaking updates after real turn decisions', () => {
    const turn = { ...decision(0.8, 8), task: 'turn_end' };
    const first = reduceCard(initialState(), turn, 100);
    const speaking = {
      ...turn,
      value: false,
      probability: null,
      score: null,
      threshold: null,
      status: 'speaking',
      window_end_s: 9,
      window_start_s: 9,
      latency_ms: 0,
    };
    const next = reduceCard(first, speaking, 200);
    expect(next.decision?.status).toBe('speaking');
    expect(reduceCard(next, { ...speaking, window_end_s: 9 }, 300)).toBe(next);
    expect(reduceCard(next, { ...turn, window_end_s: 8 }, 300)).toBe(next);
    expect(next.pending).toBe(false);
  });
  it('accepts a real turn decision tied with the speaking onset', () => {
    const speaking = {
      ...decision(0, 9),
      task: 'turn_end',
      value: false,
      probability: null,
      threshold: null,
      status: 'speaking',
    };
    const state = reduceCard(initialState(), speaking, 100);
    const real = { ...decision(0.8, 9), task: 'turn_end' };
    expect(reduceCard(state, real, 200).decision).toBe(real);
  });
  it('measures each card update interval without treating another lane as stale', () => {
    const live = reduceCard(initialState(), decision(0.1, 20), 100);
    const review = reduceCard(initialState(), { ...decision(0.8, 8), task: 'term' }, 250);
    expect(review.decision?.window_end_s).toBe(8);
    expect(live.intervalMs).toBeUndefined();
    expect(reduceCard(live, decision(0.2, 21), 700).intervalMs).toBe(600);
  });
  it('retains the audio endpoint separately from receipt time', () => {
    const review = reduceCard(initialState(), decision(0.8, 30), 10000, 4000);
    expect(review.receivedAt).toBe(10000);
    expect(review.windowEndAt).toBe(4000);
  });
  it('does not interpret choice probabilities as binary crossings', () => {
    const first = reduceCard(initialState(), { ...decision(0.1), kind: 'choice' }, 100);
    expect(reduceCard(first, { ...decision(0.8, 2), kind: 'choice' }, 200).crossedAt).toBe(
      -Infinity,
    );
  });
});
describe('latency and wire format', () => {
  it('measures live latency from capture start and reviews from the packet endpoint', () => {
    expect(renderLatency(1000, 36000, 'encoder')).toBe(35000);
    expect(renderLatency(1000, 36000, 'decoder')).toBe(34750);
  });
  it('does not display a negative duration if estimated capture time is ahead', () => {
    expect(renderLatency(1000, 1240, 'decoder')).toBe(0);
  });
  it('uses nearest rank p95 without inventing empty measurements', () => {
    expect(percentile([])).toBeNull();
    expect(percentile(Array.from({ length: 20 }, (_, i) => i + 1))).toBe(19);
  });
  it('encodes header and signed PCM as little endian', () => {
    const frame = encodeFrame(new Int16Array([-32768, 1234]), 7, 900.25);
    const view = new DataView(frame);
    expect(view.byteLength).toBe(16);
    expect(view.getUint32(0, true)).toBe(7);
    expect(view.getFloat64(4, true)).toBe(900.25);
    expect(view.getInt16(12, true)).toBe(-32768);
    expect(view.getInt16(14, true)).toBe(1234);
  });
});
