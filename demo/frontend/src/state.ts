export type Kind = 'noul' | 'score' | 'choice';
export const PACKET_DURATION_MS = 250;
export const defaultThreshold = (task: string): number =>
  task === 'voice_activity' ? 0.225 : 0.5;
export interface Card {
  task: string;
  title: string;
  text: string;
  kind: Kind;
  options: string[];
  status: 'validated' | 'experimental' | 'unvalidated';
  evaluation?: string;
  uncalibrated?: boolean;
  lane?: 'encoder' | 'decoder';
  context_seconds?: number;
  cadence_seconds?: number;
  cadence?: 'every 250 ms' | 'at end of speech' | 'per utterance';
}
export interface Decision {
  task: string;
  question: string;
  kind: Kind;
  value: boolean | number | string | null;
  probability: number | null;
  probabilities: Record<string, number> | null;
  score: number | null;
  threshold: number | null;
  status?: string;
  window_end_s: number;
  window_start_s?: number;
  latency_ms: number;
  window_latency_ms?: number;
}
export interface CardState {
  decision?: Decision;
  pending: boolean;
  trail: { time: number; value: number }[];
  crossedAt: number;
  receivedAt: number;
  intervalMs?: number;
  windowEndAt?: number;
}
export const isRollingUpdate = (lane: 'encoder' | 'decoder', decisions: Decision[]): boolean =>
  lane === 'encoder' && decisions.some((decision) => decision.task !== 'turn_end');
export const initialState = (): CardState => ({
  pending: false,
  trail: [],
  crossedAt: -Infinity,
  receivedAt: -Infinity,
});
export const scalar = (d: Decision): number =>
  d.kind === 'score' ? (d.score ?? 0) : (d.probability ?? 0);
export function reduceCard(
  state: CardState,
  decision: Decision,
  now: number,
  windowEndAt = now,
): CardState {
  if (state.decision && decision.window_end_s <= state.decision.window_end_s) return state;
  if (decision.status === 'not_enough_speech') return { ...state, pending: true };
  const previous = state.decision;
  const crossed =
    previous && previous.value !== null && decision.value !== null &&
    decision.kind !== 'choice' &&
    scalar(previous) >= (previous.threshold ?? defaultThreshold(previous.task)) !==
      scalar(decision) >= (decision.threshold ?? defaultThreshold(decision.task));
  return {
    decision,
    pending: false,
    receivedAt: now,
    intervalMs: state.decision ? now - state.receivedAt : undefined,
    windowEndAt,
    trail: [
      ...state.trail.filter((p) => p.time >= now - 5000),
      ...(decision.value === null ? [] : [{ time: now, value: scalar(decision) }]),
    ].slice(-100),
    crossedAt: crossed ? now : state.crossedAt,
  };
}
export function percentile(values: number[], fraction = 0.95): number | null {
  if (!values.length) return null;
  return [...values].sort((a, b) => a - b)[Math.max(0, Math.ceil(values.length * fraction) - 1)];
}
export function renderLatency(
  capturedAt: number,
  renderedAt: number,
  lane: 'encoder' | 'decoder',
): number {
  const origin = capturedAt + (lane === 'decoder' ? PACKET_DURATION_MS : 0);
  return Math.max(0, renderedAt - origin);
}
export function encodeFrame(pcm: Int16Array, sequence: number, capturedAt: number): ArrayBuffer {
  const buffer = new ArrayBuffer(12 + pcm.length * 2);
  const view = new DataView(buffer);
  view.setUint32(0, sequence, true);
  view.setFloat64(4, capturedAt, true);
  pcm.forEach((sample, i) => view.setInt16(12 + i * 2, sample, true));
  return buffer;
}
