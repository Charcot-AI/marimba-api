import './style.css';
import workletUrl from './audio-worklet.js?url';
import { instrument } from './instruments';
import {
  encodeFrame,
  initialState,
  percentile,
  reduceCard,
  scalar,
  type Card,
  type CardState,
  type Decision,
} from './state';
import type { SttAdapter, SttFactory, Transcript } from './stt';

interface Config {
  cards: Card[];
  runtime: { backend: string; device: string; dtype: string };
  stt: boolean;
}
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `<div class="shell"><header><a class="brand" href="https://charcot.ai" aria-label="Charcot AI"><img src="/marimba.svg" alt="marimba"></a><span class="edition">charcot.ai / live instruments</span><button id="theme" aria-label="Switch color theme">Dark ↗</button></header>
<section class="intro"><div><p class="eyebrow">AUDIO → TYPED DECISIONS</p><h1>Listen. Decide. Repeat.</h1><p class="subtitle">One audio stream. Parallel questions. Continuous answers.</p></div><div class="latency"><span class="eyebrow">CAPTURE → RENDER · P95</span><div><span id="p95">—</span><span class="unit">ms</span></div><span id="target">target ≤ 300 ms / rolling 60 s</span></div></section>
<section class="source" aria-label="Audio input"><div class="source-label"><span id="live-dot"></span><span id="source-name">INPUT / IDLE</span></div><canvas id="wave" width="720" height="72" aria-label="Live audio amplitude"></canvas><span id="clock">00:00</span><label class="file-button">Open audio<input type="file" id="file" accept="audio/*"></label><button id="start" class="primary">Start microphone ↗</button><button id="stop" disabled>Stop ■</button></section>
<p id="status" role="status">Loading local model capabilities…</p><section id="cards" class="cards" aria-label="Live decision cards"></section>
<section id="transcript" class="transcript" hidden><div class="eyebrow">LIVE TRANSCRIPT / OPTIONAL ADAPTER</div><p id="words" aria-live="polite"></p></section>
<footer><span id="runtime">LOCAL INFERENCE</span><span>8 s window / 250 ms ingest</span><span id="dropped">0 coalesced decisions</span><span>Audio is not saved locally.</span></footer></div>`;
const $ = <T extends HTMLElement = HTMLElement>(selector: string) =>
  document.querySelector<T>(selector)!;
const status = (text: string) => {
  $('#status').textContent = text;
};
let dark = localStorage.getItem('theme') === 'dark';
function setTheme() {
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  $('#theme').textContent = dark ? 'Light ↗' : 'Dark ↗';
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}
$('#theme').onclick = () => {
  dark = !dark;
  setTheme();
};
setTheme();
let config: Config;
const states = new Map<string, CardState>();
let context: AudioContext | undefined,
  mic: MediaStream | undefined,
  source: AudioNode | undefined,
  worklet: AudioWorkletNode | undefined;
let socket: WebSocket | undefined,
  stt: SttAdapter | undefined,
  active = false,
  stopping = false;
let sequence = 0,
  started = 0,
  lastResult = 0,
  lastPaint = 0;
let latency: { time: number; ms: number }[] = [];
let amplitudes: number[] = [];
let transcriptEvents = new Map<string, Transcript>();
let localConfig: Record<string, unknown> = {};
let timeout: ReturnType<typeof setTimeout> | undefined;
let endedTimer: ReturnType<typeof setTimeout> | undefined;

function renderCards(now: number) {
  for (const card of config.cards) {
    const state = states.get(card.task)!;
    const element = document.getElementById(`card-${card.task}`)!;
    const d = state.decision;
    element.classList.toggle('crossed', now - state.crossedAt < 650);
    element.classList.toggle('stale', !!d && now - state.receivedAt > 2500);
    element.querySelector('.reading')!.textContent = !d
      ? '—'
      : card.kind === 'noul'
        ? d.value
          ? 'yes'
          : 'no'
        : card.kind === 'score'
          ? scalar(d).toFixed(2)
          : String(d.value);
    element.querySelector('.card-latency')!.textContent = d
      ? `${d.latency_ms.toFixed(0)} ms / model`
      : 'awaiting audio';
    element.querySelector('.instrument')!.innerHTML = instrument(card, state, now);
    element.querySelectorAll('[data-option]').forEach((el, i) => {
      el.textContent = card.options[i];
    });
    element.querySelector('.card-bottom')!.textContent = !d
      ? card.kind === 'choice'
        ? 'OPTION PROBABILITIES'
        : 'THRESHOLD / 0.50'
      : now - state.receivedAt > 2500
        ? 'STALE / WAITING FOR MODEL'
        : card.kind === 'score'
          ? '5 SECOND TRAIL'
          : card.kind === 'noul'
            ? `${card.uncalibrated ? 'RAW N-BEST' : 'P(YES)'} / ${(d.probability ?? 0).toFixed(3)}`
            : card.task === 'language'
              ? 'EN/PT/ES CLOSED SET'
              : 'OPTION PROBABILITIES';
  }
}
function makeCards() {
  $('#cards').replaceChildren();
  config.cards.forEach((card, index) => {
    states.set(card.task, initialState());
    const el = document.createElement('article');
    el.id = `card-${card.task}`;
    el.className = 'card';
    el.innerHTML = `<div class="card-top"><span class="number">${String(index + 1).padStart(2, '0')} / ${card.kind.toUpperCase()}</span><span class="badge">${card.uncalibrated ? 'UNCALIBRATED' : card.status === 'experimental' ? 'EXPERIMENTAL' : 'LIVE'}</span></div><h2></h2><div class="reading-row"><span class="reading">—</span><span class="card-latency">awaiting audio</span></div><div class="instrument"></div><div class="card-bottom"></div>`;
    el.title = card.evaluation ?? '';
    el.querySelector('h2')!.textContent = card.title;
    $('#cards').append(el);
  });
  renderCards(performance.now());
}
function onTranscript(event: Transcript) {
  for (const id of event.replaces ?? []) transcriptEvents.delete(id);
  transcriptEvents.set(event.id ?? crypto.randomUUID(), event);
  transcriptEvents = new Map([...transcriptEvents].slice(-200));
  $('#words').replaceChildren(
    ...[...transcriptEvents.values()].map((part) => {
      const span = document.createElement('span');
      span.textContent = `${part.text} `;
      span.className = part.final ? '' : 'partial';
      return span;
    }),
  );
}
async function prepareStt() {
  if (!config.stt) return;
  const url = '/local/stt_adapter.js';
  const module = (await import(/* @vite-ignore */ url)) as { createAdapter: SttFactory };
  stt = module.createAdapter(localConfig);
  await stt.start(onTranscript, (message) => status(`Transcript adapter: ${message}`));
}
function sendPcm(pcm: Int16Array, capturedAt: number) {
  if (!active || stopping || !socket || socket.readyState !== WebSocket.OPEN) return;
  const rms = Math.sqrt(pcm.reduce((sum, x) => sum + (x / 32768) ** 2, 0) / pcm.length);
  amplitudes.push(rms);
  amplitudes = amplitudes.slice(-100);
  stt?.push(pcm, Date.now() - (performance.now() - capturedAt));
  if (socket.bufferedAmount > 16384) {
    status('Audio intake overloaded. Stopped to preserve continuity.');
    void stop();
    return;
  }
  socket.send(encodeFrame(pcm, sequence++, capturedAt));
}
function armTimeout() {
  clearTimeout(timeout);
  timeout = setTimeout(() => {
    status('Model timed out after 30 seconds. Restart the input.');
    void stop();
  }, 30000);
}
async function start(file?: File) {
  if (active || stopping) return;
  $('#start').setAttribute('disabled', '');
  $('#file').setAttribute('disabled', '');
  try {
    active = true;
    sequence = 0;
    latency = [];
    amplitudes = [];
    transcriptEvents.clear();
    makeCards();
    $('#words').replaceChildren();
    $('#clock').textContent = '00:00';
    $('#p95').textContent = '—';
    $('#target').textContent = 'target ≤ 300 ms / rolling 60 s';
    $('#dropped').textContent = '0 coalesced decisions';
    status('Connecting to local inference…');
    try {
      context = new AudioContext({ sampleRate: 16000 });
    } catch (error) {
      if (!(error instanceof DOMException) || error.name !== 'NotSupportedError') throw error;
      context = new AudioContext();
    }
    await context.resume();
    await context.audioWorklet.addModule(workletUrl);
    socket = new WebSocket(
      `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/stream`,
    );
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('Local socket connection timed out.')),
        10000,
      );
      socket!.onopen = () => {
        clearTimeout(timer);
        resolve();
      };
      socket!.onerror = () => {
        clearTimeout(timer);
        reject(new Error('Cannot connect to the local server.'));
      };
    });
    const connection = socket;
    connection.onmessage = (event) => {
      if (socket !== connection) return;
      const data = JSON.parse(event.data) as {
        type: string;
        message?: string;
        captured_at_ms: number;
        decision_interval_ms: number;
        coalesced_hops: number;
        decisions: Decision[];
      };
      if (data.type === 'error') {
        status(data.message ?? 'Inference error');
        void stop();
        return;
      }
      if (data.type === 'stopped') {
        connection.close();
        return;
      }
      if (active && !stopping) armTimeout();
      const now = performance.now();
      lastResult = now;
      $('#dropped').textContent =
        `${data.decision_interval_ms.toFixed(0)} ms decision interval / ${data.coalesced_hops} coalesced`;
      for (const d of data.decisions)
        if (states.has(d.task)) states.set(d.task, reduceCard(states.get(d.task)!, d, now));
      if (data.decisions.length)
        requestAnimationFrame(() => {
          renderCards(performance.now());
          latency.push({ time: performance.now(), ms: performance.now() - data.captured_at_ms });
          latency = latency.filter((p) => p.time >= performance.now() - 60000);
          $('#p95').textContent = (percentile(latency.map((p) => p.ms)) ?? 0).toFixed(0);
          const miss = (percentile(latency.map((p) => p.ms)) ?? 0) > 300;
          $('#target').textContent = miss
            ? 'above 300 ms target / rolling 60 s'
            : 'target ≤ 300 ms / rolling 60 s';
        });
      status(
        data.coalesced_hops
          ? 'Listening / all audio retained; slower inference coalesces decision ticks, not input.'
          : 'Listening / decisions update as each window completes.',
      );
    };
    connection.onclose = () => {
      if (socket === connection && active && !stopping) {
        status('Connection closed. Start again to resume.');
        void stop();
      }
    };
    await prepareStt();
    worklet = new AudioWorkletNode(context, 'pcm-capture');
    const audioContext = context;
    worklet.port.onmessage = (event) => {
      const capturedAt =
        performance.now() - (audioContext.currentTime - event.data.endedAt) * 1000 - 250;
      sendPcm(new Int16Array(event.data.pcm), capturedAt);
    };
    if (file) {
      const buffer = await context.decodeAudioData(await file.arrayBuffer());
      const player = context.createBufferSource();
      player.buffer = buffer;
      source = player;
      player.connect(context.destination);
      player.connect(worklet);
      player.start();
      player.onended = () => {
        endedTimer = setTimeout(() => {
          if (active) void stop();
        }, 300);
      };
    } else {
      mic = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      source = context.createMediaStreamSource(mic);
      source.connect(worklet);
    }
    const silent = context.createGain();
    silent.gain.value = 0;
    worklet.connect(silent).connect(context.destination);
    started = performance.now();
    lastResult = started;
    $('#live-dot').classList.add('active');
    $('#source-name').textContent = file ? `FILE / ${file.name}` : 'MIC / LIVE';
    $('#stop').removeAttribute('disabled');
    status('Listening / audio stays in memory.');
    armTimeout();
  } catch (error) {
    status(error instanceof Error ? error.message : String(error));
    await stop();
  }
}
async function stop() {
  if (stopping) return;
  stopping = true;
  active = false;
  clearTimeout(endedTimer);
  if (source instanceof AudioBufferSourceNode) {
    source.onended = null;
    source.stop();
  }
  clearTimeout(timeout);
  source?.disconnect();
  worklet?.disconnect();
  mic?.getTracks().forEach((track) => track.stop());
  mic = undefined;
  await context?.close();
  context = undefined;
  if (socket?.readyState === WebSocket.OPEN) {
    const closing = socket;
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        closing.close();
        resolve();
      }, 2000);
      closing.addEventListener(
        'close',
        () => {
          clearTimeout(timer);
          resolve();
        },
        { once: true },
      );
      closing.send(JSON.stringify({ type: 'stop' }));
    });
  } else socket?.close();
  socket = undefined;
  try {
    await stt?.stop();
  } catch {
    status('Transcript adapter did not finish cleanly.');
  }
  stt = undefined;
  $('#live-dot').classList.remove('active');
  $('#source-name').textContent = 'INPUT / STOPPED';
  $('#start').removeAttribute('disabled');
  $('#file').removeAttribute('disabled');
  $('#stop').setAttribute('disabled', '');
  stopping = false;
}
$('#start').onclick = () => void start();
$('#stop').onclick = () => {
  status('Stopped / last decisions retained.');
  void stop();
};
$('#file').onchange = (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) void start(file);
  (event.target as HTMLInputElement).value = '';
};
window.addEventListener('pagehide', () => {
  mic?.getTracks().forEach((track) => track.stop());
  socket?.close();
  void context?.close();
});
function paint(now: number) {
  if (now - lastPaint > 100) {
    if (config) renderCards(now);
    lastPaint = now;
    if (active && started) {
      const seconds = Math.floor((now - started) / 1000);
      $('#clock').textContent = `${Math.floor(seconds / 60)
        .toString()
        .padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
      if (now - lastResult > 3000)
        status(
          'Waiting for the model / audio context is retained; inference runs on the latest window.',
        );
    }
    const canvas = $<HTMLCanvasElement>('#wave');
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--ink');
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 36);
    ctx.lineTo(720, 36);
    ctx.globalAlpha = 0.18;
    ctx.stroke();
    ctx.globalAlpha = 1;
    amplitudes.forEach((rms, i) => {
      const x = 720 - (amplitudes.length - i) * 7;
      const h = Math.min(32, rms * 150);
      ctx.beginPath();
      ctx.moveTo(x, 36 - h);
      ctx.lineTo(x, 36 + h);
      ctx.stroke();
    });
  }
  requestAnimationFrame(paint);
}
async function init() {
  config = (await (await fetch('/api/config')).json()) as Config;
  if (config.stt) localConfig = await (await fetch('/api/local-config')).json();
  $('#transcript').hidden = !config.stt;
  $('#runtime').textContent =
    `${config.runtime.device} / ${config.runtime.backend} / ${config.runtime.dtype}`.toUpperCase();
  makeCards();
  status(
    config.cards.length
      ? 'Ready / choose your microphone or a public audio file.'
      : 'No supported cards in this checkpoint. Install a compatible model.',
  );
  if (!config.cards.length) {
    $('#start').setAttribute('disabled', '');
    $('#file').setAttribute('disabled', '');
  }
  requestAnimationFrame(paint);
}
init().catch(() => {
  status('Cannot load model capabilities. Check the local server.');
  $('#start').setAttribute('disabled', '');
});
