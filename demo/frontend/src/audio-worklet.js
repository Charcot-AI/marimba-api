class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.output = new Int16Array(4000);
    this.index = 0;
    this.position = 0;
    this.total = 0;
    this.previous = 0;
  }
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    const length = channels[0].length;
    for (let i = 0; i < length; i++) {
      let current = 0;
      for (const channel of channels) current += channel[i] / channels.length;
      const absolute = this.total + i;
      while (this.position <= absolute) {
        const fraction = this.position - (absolute - 1);
        const value = this.previous + (current - this.previous) * fraction;
        this.output[this.index++] = Math.max(-32768, Math.min(32767, Math.round(value * 32767)));
        this.position += sampleRate / 16000;
        if (this.index === 4000) {
          this.port.postMessage(
            { pcm: this.output.buffer, endedAt: currentTime + (i + 1) / sampleRate },
            [this.output.buffer],
          );
          this.output = new Int16Array(4000);
          this.index = 0;
        }
      }
      this.previous = current;
    }
    this.total += length;
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
