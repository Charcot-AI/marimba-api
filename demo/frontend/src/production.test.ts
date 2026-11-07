import { describe, expect, it } from 'vitest';
import { build } from 'vite';
import { fileURLToPath } from 'node:url';

describe('production audio worklet', () => {
  it('emits a same-origin asset compatible with the restrictive server CSP', async () => {
    const result = await build({
      configFile: fileURLToPath(new URL('../vite.config.ts', import.meta.url)),
      build: { write: false },
      logLevel: 'silent',
    });
    if (Array.isArray(result) || !('output' in result)) {
      throw new Error('Expected one production bundle');
    }
    const worklet = result.output.find((asset) => /audio-worklet-[\w-]+\.js$/.test(asset.fileName));
    expect(worklet?.type).toBe('asset');
    if (worklet?.type !== 'asset') throw new Error('Worklet asset missing');
    expect(worklet.source.toString()).toContain('registerProcessor');
    const entry = result.output.find((asset) => asset.type === 'chunk' && asset.isEntry);
    if (entry?.type !== 'chunk') throw new Error('Entry chunk missing');
    expect(entry.code).toContain(worklet.fileName);
    expect(entry.code).not.toContain('data:text/javascript');
  });
});
