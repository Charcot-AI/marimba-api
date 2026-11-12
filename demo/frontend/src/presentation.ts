import {
  defaultThreshold,
  scalar,
  type Card,
  type CardState,
  type Decision,
} from './state';

export function reading(card: Card, decision?: Decision): string {
  if (decision?.value == null) return '—';
  if (card.kind === 'noul') return decision.value ? 'yes' : 'no';
  if (card.kind === 'score') return scalar(decision).toFixed(2);
  return String(decision.value);
}

export function timing(card: Card, state: CardState, now: number) {
  const d = state.decision;
  const seconds = card.context_seconds ?? 8;
  const cadence = (card.cadence ?? (card.lane === 'decoder' ? 'per utterance' : 'every 250 ms')).toUpperCase();
  return {
    latency: d ? `${(d.window_latency_ms ?? d.latency_ms).toFixed(0)} ms / model` : 'awaiting audio',
    interval: card.task === 'turn_end' && d?.status === 'speaking'
      ? 'SPEECH IN PROGRESS'
      : `${cadence} / UPDATE ${state.intervalMs === undefined ? '—' : `${state.intervalMs.toFixed(0)} ms`} / RECEIVED ${d ? `${((now - state.receivedAt) / 1000).toFixed(1)} s AGO` : '—'}`,
    window: d
      ? `AUDIO ${(d.window_start_s ?? Math.max(0, d.window_end_s - seconds)).toFixed(2)}–${d.window_end_s.toFixed(2)} s / AGE ${(Math.max(0, now - (state.windowEndAt ?? state.receivedAt)) / 1000).toFixed(1)} s`
      : card.cadence === 'at end of speech' || card.lane === 'decoder'
        ? 'WAITING FOR END OF SPEECH / UP TO 8 s'
        : `COLLECTING ${seconds} s OF AUDIO`,
  };
}

export function bottomLine(card: Card, decision?: Decision): string {
  if (!decision) {
    return card.kind === 'choice'
      ? 'OPTION PROBABILITIES'
      : `THRESHOLD / ${defaultThreshold(card.task)}`;
  }
  if (card.task === 'turn_end' && decision.status === 'speaking') return '';
  if (card.kind === 'score') return '5 SECOND TRAIL';
  if (card.kind === 'noul') {
    return `${card.uncalibrated ? 'RAW N-BEST' : 'P(YES)'} / ${(decision.probability ?? 0).toFixed(3)}`;
  }
  return card.task === 'language' ? 'EN/PT/ES CLOSED SET' : 'OPTION PROBABILITIES';
}
