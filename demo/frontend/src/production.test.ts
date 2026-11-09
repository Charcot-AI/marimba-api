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

  it('balances the live and review groups without an empty third column', async () => {
    const output = await productionAssets();
    const css = output
      .filter((asset) => asset.type === 'asset' && asset.fileName.endsWith('.css'))
      .map((asset) => (asset.type === 'asset' ? asset.source.toString() : ''))
      .join('');
    expect(css).toMatch(
      /#decision-groups\s*\{[^}]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/,
    );
    expect(css).not.toMatch(/grid-template-columns:\s*repeat\(3,/);
  });
});
