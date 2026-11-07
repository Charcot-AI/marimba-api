export type Kind = 'noul' | 'score' | 'choice';
export interface Card {
  task: string;
  title: string;
  text: string;
  kind: Kind;
  options: string[];
  status: 'validated' | 'experimental';
}
export interface Decision {
  task: string;
  question: string;
  kind: Kind;
  value: boolean | number | string;
  probability: number | null;
  probabilities: Record<string, number> | null;
  score: number | null;
  threshold: number | null;
  window_end_s: number;
  latency_ms: number;
}
export interface CardState {
  decision?: Decision;
  trail: { time: number; value: number }[];
  crossedAt: number;
  receivedAt: number;
}
export const initialState = (): CardState => ({
  trail: [],
  crossedAt: -Infinity,
  receivedAt: -Infinity,
});
export const scalar = (d: Decision): number =>
  d.kind === 'score' ? (d.score ?? 0) : (d.probability ?? 0);
export function reduceCard(state: CardState, decision: Decision, now: number): CardState {
  if (state.decision && decision.window_end_s <= state.decision.window_end_s) return state;
  const previous = state.decision;
  const crossed =
    previous &&
    decision.kind !== 'choice' &&
    scalar(previous) >= (previous.threshold ?? 0.5) !==
      scalar(decision) >= (decision.threshold ?? 0.5);
  return {
    decision,
    receivedAt: now,
    trail: [
      ...state.trail.filter((p) => p.time >= now - 5000),
      { time: now, value: scalar(decision) },
    ].slice(-100),
    crossedAt: crossed ? now : state.crossedAt,
  };
}
export function percentile(values: number[], fraction = 0.95): number | null {
  if (!values.length) return null;
  return [...values].sort((a, b) => a - b)[Math.max(0, Math.ceil(values.length * fraction) - 1)];
}
export function encodeFrame(pcm: Int16Array, sequence: number, capturedAt: number): ArrayBuffer {
  const buffer = new ArrayBuffer(12 + pcm.length * 2);
  const view = new DataView(buffer);
  view.setUint32(0, sequence, true);
  view.setFloat64(4, capturedAt, true);
  pcm.forEach((sample, i) => view.setInt16(12 + i * 2, sample, true));
  return buffer;
}
