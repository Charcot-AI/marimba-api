import './style.css';
import workletUrl from './audio-worklet.js?url';
import { instrument } from './instruments';
import {
  encodeFrame,
  initialState,
  isRollingUpdate,
  PACKET_DURATION_MS,
  reduceCard,
  shownDecision,
  type Card,
  type CardState,
  type Decision,
} from './state';
import type { SttAdapter, SttFactory, Transcript } from './stt';
import { drainSocket } from './socket';
import { bottomLine, reading, timing } from './presentation';

interface Config {
  cards: Card[];
  runtime: { backend: string; device: string; dtype: string };
  stt: boolean;
}
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `<div class="shell"><header><a class="brand" href="https://charcot.ai" aria-label="Charcot AI"><img src="/marimba.svg" alt="marimba"></a><span class="edition">charcot.ai / live instruments</span><button id="theme" aria-label="Switch color theme">Dark ↗</button></header>
<section class="intro"><div><p class="eyebrow">AUDIO → TYPED DECISIONS</p><h1>Listen. Decide. Repeat.</h1><p class="subtitle">One audio stream. Parallel questions. Continuous answers.</p></div></section>
<section class="source" aria-label="Audio input"><div class="source-label"><span id="live-dot"></span><span id="source-name">INPUT / IDLE</span></div><canvas id="wave" width="720" height="72" aria-label="Live audio amplitude"></canvas><span id="clock">00:00</span><label class="file-button">Open audio<input type="file" id="file" accept="audio/*"></label><button id="start" class="primary">Start microphone ↗</button><button id="stop" disabled>Stop ■</button></section>
<p id="status" role="status">Loading local model capabilities…</p>
<div id="cards" class="cards" aria-label="Audio decision cards"></div><section id="transcript" class="transcript" hidden><div class="eyebrow">LIVE TRANSCRIPT / OPTIONAL ADAPTER</div><p id="words" aria-live="polite"></p></section>
<footer><span id="runtime">LOCAL INFERENCE</span><span>8 s maximum / 250 ms ingest</span><span id="dropped">0 coalesced decisions</span><span id="utterance-queue">0 QUEUED UTTERANCES</span><span>Audio is not saved locally.</span></footer></div>`;
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
const reviewsRunning = new Set<string>();
let amplitudes: number[] = [];
let transcriptEvents = new Map<string, Transcript>();
let localConfig: Record<string, unknown> = {};
let timeout: ReturnType<typeof setTimeout> | undefined;
let endedTimer: ReturnType<typeof setTimeout> | undefined;

function renderCards(now: number) {
  for (const card of config.cards) {
    const state = states.get(card.task)!;
    const element = document.getElementById(`card-${card.task}`)!;
    const d = shownDecision(card, state, now);
    const shownState = { ...state, decision: d };
    const review = card.lane === 'decoder';
    const stale =
      !!d && !review && card.task !== 'turn_end' && now - state.receivedAt > 2500;
    const pending =
      !(card.task === 'turn_end' && d?.status === 'speaking') &&
      (state.pending || (review && reviewsRunning.has(card.task)) || stale);
    element.classList.toggle('crossed', now - state.crossedAt < 650);
    const value = element.querySelector<HTMLElement>('.reading')!;
    value.textContent = reading(card, d);
    value.classList.toggle('pending', pending);
    value.setAttribute('aria-busy', String(pending));
    const footer = timing(card, shownState, now);
    element.querySelector('.card-latency')!.textContent = footer.latency;
    element.querySelector('.card-interval')!.textContent = footer.interval;
    element.querySelector('.card-window')!.textContent = footer.window;
    element.querySelector('.instrument')!.innerHTML = instrument(card, shownState, now);
    element.querySelectorAll('[data-option]').forEach((el, i) => {
      el.textContent = card.options[i];
    });
    element.querySelector('.card-bottom')!.textContent = bottomLine(card, d);
  }
}
function makeCards() {
  $('#cards').replaceChildren();
  config.cards.forEach((card, index) => {
    states.set(card.task, initialState());
    const el = document.createElement('article');
    el.id = `card-${card.task}`;
    el.className = 'card';
    const badge = card.status === 'experimental'
        ? 'EXPERIMENTAL'
        : '';
    const tags = [badge, card.uncalibrated ? 'UNCALIBRATED' : ''].filter(Boolean).join(' / ');
    el.innerHTML = `<div class="card-top"><span class="number">${String(index + 1).padStart(2, '0')} / ${card.kind.toUpperCase()}</span><span class="badge">${tags}</span></div><h2></h2><div class="reading-row"><span class="reading">${reading(card)}</span></div><div class="instrument"></div><div class="card-bottom"></div><div class="card-window"></div><div class="card-interval"></div><div class="card-latency"></div>`;
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
  if (!config.cards.some((card) => card.lane !== 'decoder' && card.task !== 'turn_end')) return;
  const seconds = 30;
  timeout = setTimeout(() => {
    status(`Model timed out after ${seconds} seconds. Restart the input.`);
    void stop();
  }, seconds * 1000);
}
async function start(file?: File) {
  if (active || stopping) return;
  $('#start').setAttribute('disabled', '');
  $('#file').setAttribute('disabled', '');
  try {
    active = true;
    sequence = 0;
    reviewsRunning.clear();
    amplitudes = [];
    transcriptEvents.clear();
    makeCards();
    $('#words').replaceChildren();
    $('#clock').textContent = '00:00';
    $('#dropped').textContent = '0 coalesced decisions';
    $('#utterance-queue').textContent = '0 QUEUED UTTERANCES';
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
        window_end_at_ms?: number;
        decision_interval_ms: number;
        coalesced_hops: number;
        queued_utterances: number;
        decisions: Decision[];
        lane: 'encoder' | 'decoder';
        tasks: string[];
        running: boolean;
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
      if (data.type === 'review_status') {
        for (const task of data.tasks) {
          if (data.running) reviewsRunning.add(task);
          else reviewsRunning.delete(task);
        }
        return;
      }
      if (data.type !== 'update') return;
      $('#utterance-queue').textContent = `${data.queued_utterances} QUEUED UTTERANCES`;
      const live = isRollingUpdate(data.lane, data.decisions);
      if (active && !stopping && (live || !config.cards.some((card) => card.lane !== 'decoder')))
        armTimeout();
      const now = performance.now();
      if (live || !config.cards.some((card) => card.lane !== 'decoder')) lastResult = now;
      if (live) {
        $('#dropped').textContent =
          `${data.decision_interval_ms.toFixed(0)} ms live interval / ${data.coalesced_hops} coalesced`;
      }
      for (const d of data.decisions)
        if (states.has(d.task))
          states.set(
            d.task,
            reduceCard(states.get(d.task)!, d, now, data.window_end_at_ms ?? data.captured_at_ms + PACKET_DURATION_MS),
          );
      if (data.decisions.length)
        requestAnimationFrame(() => {
          renderCards(performance.now());
        });
      if (active && !stopping)
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
          if (active) void stop('Stopped / last decisions retained.');
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
async function stop(message?: string) {
  if (stopping) return;
  const finishing = 'Stopped / finishing current model work; audio capture has ended.';
  if (message) status(finishing);
  stopping = true;
  active = false;
  $('#stop').setAttribute('disabled', '');
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
  const transcript = stt;
  stt = undefined;
  const finishingTranscript = (async () => {
    try {
      await transcript?.stop();
    } catch {
      status('Transcript adapter did not finish cleanly.');
    }
  })();
  const drained = await drainSocket(socket, 90000);
  socket = undefined;
  await finishingTranscript;
  if (message && $('#status').textContent === finishing)
    status(
      drained ? message : 'Stopped / the model is still busy. Wait or restart the local server.',
    );
  reviewsRunning.clear();
  $('#live-dot').classList.remove('active');
  $('#source-name').textContent = 'INPUT / STOPPED';
  $('#start').removeAttribute('disabled');
  $('#file').removeAttribute('disabled');
  $('#stop').setAttribute('disabled', '');
  stopping = false;
}
$('#start').onclick = () => void start();
$('#stop').onclick = () => {
  void stop('Stopped / last decisions retained.');
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
      if (now - lastResult > 3000) {
        status(
          'Listening / rolling heads and end-of-speech decisions use their own audio windows.',
        );
      }
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
