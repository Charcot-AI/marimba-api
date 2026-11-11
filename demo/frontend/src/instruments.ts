import { defaultThreshold, type Card, type CardState, scalar } from './state';

const clamp = (n: number) => Math.max(0, Math.min(1, n));
export function instrument(card: Card, state: CardState, now: number): string {
  const decision = state.decision;
  const value = decision ? clamp(scalar(decision)) : 0;
  const threshold = decision?.threshold ?? defaultThreshold(card.task);
  if (card.kind === 'choice') {
    return card.options
      .map(
        (_, i) =>
          `<div class="vu-row"><span class="option" data-option="${i}"></span><div class="vu-track"><i style="width:${clamp(decision?.probabilities?.[card.options[i]] ?? 0) * 100}%"></i></div><span class="prob">${
            decision
              ? Math.round((decision.probabilities?.[card.options[i]] ?? 0) * 100)
                  .toString()
                  .padStart(2, '0')
              : '—'
          }</span></div>`,
      )
      .join('');
  }
  const marks = Array.from(
    { length: 21 },
    (_, i) =>
      `<line x1="${10 + i * 14}" y1="${i % 5 === 0 ? 35 : 43}" x2="${10 + i * 14}" y2="53"/>`,
  ).join('');
  const trail =
    card.kind === 'score'
      ? state.trail
          .filter((p) => p.time >= now - 5000)
          .map(
            (p) =>
              `<line class="trail" style="opacity:${Math.max(0, (1 - (now - p.time) / 5000) * 0.35)}" x1="${10 + clamp(p.value) * 280}" y1="18" x2="${10 + clamp(p.value) * 280}" y2="32"/>`,
          )
          .join('')
      : '';
  return `<svg class="ruler" viewBox="0 0 300 78" aria-hidden="true">${marks}${trail}<line class="threshold" x1="${10 + threshold * 280}" y1="9" x2="${10 + threshold * 280}" y2="58"/>${decision ? `<path class="needle" d="M ${10 + value * 280} 8 v 50 m -4 -50 l 4 6 l 4 -6"/>` : ''}<text x="10" y="75">0</text><text x="143" y="75">.5</text><text x="284" y="75">1</text></svg>`;
}
