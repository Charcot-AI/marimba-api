import { describe, expect, it } from 'vitest';
import { build } from 'vite';
import { fileURLToPath } from 'node:url';

async function productionAssets() {
  const result = await build({
    configFile: fileURLToPath(new URL('../vite.config.ts', import.meta.url)),
    build: { write: false },
    logLevel: 'silent',
  });
  if (Array.isArray(result) || !('output' in result)) {
    throw new Error('Expected one production bundle');
  }
  return result.output;
}

describe('production assets', () => {
  it('scopes the theme-aware pending shimmer to reading text and disables motion on request', async () => {
    const output = await productionAssets();
    const css = output
      .filter((asset) => asset.type === 'asset' && asset.fileName.endsWith('.css'))
      .map((asset) => (asset.type === 'asset' ? asset.source.toString() : ''))
      .join('');
    const pendingRules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
      .filter((rule) => rule[1].includes('.pending'));
    expect(pendingRules.map((rule) => rule[1])).toEqual(['.reading.pending', '.reading.pending']);
    expect(pendingRules[0][2]).toContain('linear-gradient(');
    expect(pendingRules[0][2]).toContain('var(--ink)');
    expect(pendingRules[0][2]).toContain('var(--muted)');
    expect(pendingRules[0][2]).toContain('background-clip:text');
    expect(pendingRules[0][2]).toContain('animation:reading-pending');
    expect(css).toContain('@keyframes reading-pending');
    expect(css).toMatch(/@media\s*\(prefers-reduced-motion:\s*reduce\)\{\.reading\.pending\{[^}]*animation:none/);
    expect(css).not.toMatch(/\.stale|\.neutral/);
    const entry = output.find((asset) => asset.type === 'chunk' && asset.isEntry);
    if (entry?.type !== 'chunk') throw new Error('Entry chunk missing');
    for (const text of ['not enough speech', 'NOT SCORED', 'REQUIRES ≥0.5', 'COMPUTING / PREVIOUS RESULT RETAINED'])
      expect(entry.code).not.toContain(text);
  });
  it('emits a same-origin asset compatible with the restrictive server CSP', async () => {
    const output = await productionAssets();
    const worklet = output.find((asset) => /audio-worklet-[\w-]+\.js$/.test(asset.fileName));
    expect(worklet?.type).toBe('asset');
    if (worklet?.type !== 'asset') throw new Error('Worklet asset missing');
    expect(worklet.source.toString()).toContain('registerProcessor');
    const entry = output.find((asset) => asset.type === 'chunk' && asset.isEntry);
    if (entry?.type !== 'chunk') throw new Error('Entry chunk missing');
    expect(entry.code).toContain(worklet.fileName);
    expect(entry.code).not.toContain('data:text/javascript');
  });

  it('balances all decision cards in one responsive grid', async () => {
    const output = await productionAssets();
    const css = output
      .filter((asset) => asset.type === 'asset' && asset.fileName.endsWith('.css'))
      .map((asset) => (asset.type === 'asset' ? asset.source.toString() : ''))
      .join('');
    expect(css).toMatch(
      /\.cards\s*\{[^}]*grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/,
    );
    expect(css).toMatch(/grid-template-columns:\s*repeat\(2,/);
    expect(css).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)/);
    expect(css).not.toContain('#decision-groups');
    const entry = output.find((asset) => asset.type === 'chunk' && asset.isEntry);
    if (entry?.type !== 'chunk') throw new Error('Entry chunk missing');
    expect(entry.code).not.toContain('BUFFERED REVIEWS');
    expect(entry.code).not.toContain('LIVE SIGNALS');
    expect(entry.code).not.toContain('review-p95');
  });
});
