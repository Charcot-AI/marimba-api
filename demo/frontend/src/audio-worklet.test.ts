import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { describe, expect, it } from 'vitest';

interface Frame {
  pcm: ArrayBuffer;
  endedAt: number;
}
interface Processor {
  process(inputs: Float32Array[][]): boolean;
}
const source = readFileSync(new URL('./audio-worklet.js', import.meta.url), 'utf8');

function capture(rate: number, samples: number, channels: number[]) {
  const frames: Frame[] = [];
  let Capture: { new (): Processor };
  class Worklet {
    port = {
      postMessage(frame: Frame) {
        frames.push(frame);
      },
    };
  }
  const scope = {
    AudioWorkletProcessor: Worklet,
    sampleRate: rate,
    currentTime: 0,
    registerProcessor(_name: string, processor: typeof Capture) {
      Capture = processor;
    },
  };
  runInNewContext(source, scope);
  const instance = new Capture!();
  for (let offset = 0; offset < samples; offset += 128) {
    scope.currentTime = offset / rate;
    const chunk = channels.map((value) =>
      new Float32Array(Math.min(128, samples - offset)).fill(value),
    );
    expect(instance.process([chunk])).toBe(true);
  }
  return frames;
}

describe('PCM capture worklet', () => {
  it.each([16000, 44100, 48000])(
    'emits four exact 16 kHz hops per second from %i Hz input',
    (rate) => {
      const frames = capture(rate, rate, [0.5]);
      expect(frames).toHaveLength(4);
      frames.forEach((frame, index) => {
        const pcm = new Int16Array(frame.pcm);
        expect(pcm).toHaveLength(4000);
        expect(pcm[2000]).toBe(16384);
        expect(frame.endedAt).toBeCloseTo((index + 1) / 4, 3);
      });
    },
  );
  it('downmixes channels and clamps PCM peaks', () => {
    expect([...new Int16Array(capture(16000, 4000, [0.5, -0.5])[0].pcm)]).toEqual(
      new Array(4000).fill(0),
    );
    expect(new Int16Array(capture(16000, 4000, [2])[0].pcm)[0]).toBe(32767);
    expect(new Int16Array(capture(16000, 4000, [-2])[0].pcm)[0]).toBe(-32768);
  });
});
