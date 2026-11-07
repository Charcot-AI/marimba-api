import { describe, expect, it } from 'vitest';
import { encodeFrame, initialState, percentile, reduceCard, type Decision } from './state';

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
describe('card state', () => {
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
  it('does not interpret choice probabilities as binary crossings', () => {
    const first = reduceCard(initialState(), { ...decision(0.1), kind: 'choice' }, 100);
    expect(reduceCard(first, { ...decision(0.8, 2), kind: 'choice' }, 200).crossedAt).toBe(
      -Infinity,
    );
  });
});
describe('latency and wire format', () => {
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
