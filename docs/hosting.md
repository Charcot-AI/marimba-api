# Hosting the OpenAI-compatible server

The [server](openai-server.md), which also serves the
[stateless window endpoints](window-server.md), is a single FastAPI process holding one
model in GPU memory. Decoder questions (language, term) share one lock per process, so scale out
with workers rather than threads.

## Image

```dockerfile
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
WORKDIR /app
COPY . /app/marimba
RUN pip install --no-cache-dir "/app/marimba[torch,server]"
ENV PORT=80
# Bake weights at build time with a read token passed as a build secret, or mount a
# volume with HF_HOME so cold starts do not download.
CMD ["sh", "-c", "marimba serve --host 0.0.0.0 --port ${PORT} --device cuda --backend torch --max-concurrency 4"]
```

Set `MARIMBA_API_KEY` as a secret environment variable when the endpoint is reachable
from the internet.

The published image, `.runpod/Dockerfile.server`, downloads the pinned model manifest,
encoder snapshot and tokenizer at build time and then sets `HF_HUB_OFFLINE=1`, so a
container starts without network access to the Hub. Its command passes
`--max-concurrency 64 --batched-encoder`, the settings for streaming traffic described
in [Sizing for streaming clients](#sizing-for-streaming-clients).

## Threads and connections

A GPU server does little CPU work per window, and unpinned BLAS and OpenMP pools on a
many-core host spend that time contending instead: one unpinned server measured 194 ms
per hop where a pinned one took about 12 ms. Pin the pools:

- `marimba serve --threads 1` caps the torch intra-op and inter-op pools and exports
  `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `NUMEXPR_NUM_THREADS`.
- Libraries that read those variables at import need them in the environment before
  the process starts. The image sets all four to `1` and passes `--threads 1`.
- On a CPU-only host the encoder itself runs on these threads, so use the number of
  cores reserved for the process instead of `1`. Without `--threads` nothing is pinned.

HTTP connections are persistent: the server keeps an idle connection open for
`--keep-alive-seconds` (default 75), so a streaming client that sends a request every
hop reuses one connection. Keep any proxy in front of the server at a longer idle
timeout than the client's hop.

## TLS

`marimba serve --tls-certfile cert.pem --tls-keyfile key.pem` terminates TLS in the
server process, so a port exposed directly to the internet needs no proxy in front of
it. Both flags default to the environment variables `MARIMBA_TLS_CERTFILE` and
`MARIMBA_TLS_KEYFILE`, the listener then serves HTTPS only, and startup fails when one
file is missing. Without them the server speaks plain HTTP.

The image also accepts the pair as base64-encoded PEM in `MARIMBA_TLS_CERT_B64` and
`MARIMBA_TLS_KEY_B64`, for platforms that inject environment variables but cannot mount
files; its entrypoint writes them to `/run/marimba` with owner-only permissions.

Two ways to give clients a certificate they can verify:

- **Public certificate.** Point a DNS name at the host and issue a certificate for it.
  A direct port is rarely 80 or 443, so use the ACME DNS-01 challenge.
- **Private certificate authority.** Issue the server certificate for a fixed name of
  your choosing and ship the CA certificate to clients. Clients connect to whatever
  address the host has and verify against the fixed name, so the certificate survives
  an address change.

```python
import http.client
import ssl

context = ssl.create_default_context(cafile="ca.pem")


class PinnedConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)
        self.sock = context.wrap_socket(self.sock, server_hostname="marimba.internal")


connection = PinnedConnection("203.0.113.7", 31234, timeout=5)
```

Set `MARIMBA_API_KEY` as well: TLS protects the bearer token and the audio in transit,
and the token is what keeps other callers out.

## Batched encoder for `/v1/pairs`

`marimba serve --batched-encoder` answers the windows of one
[`/v1/pairs`](window-server.md#post-v1pairs) request with one encoder forward per
`--encoder-batch-size` windows (default 16). It is off by default in the CLI and on in
the published image. Responses have the same schema either way; see
[batch.md](batch.md#batched-encoder-forward) for the measured difference against the
sequential path.

## Sizing for streaming clients

A streaming session keeps the server busy all the time: one
[`/v1/windows`](window-server.md#post-v1windows) request per hop, plus `/v1/pairs`
batches and extra `turn_end` questions. Every window is 256,000 bytes, so at a 500 ms
hop the hops alone upload 4.1 Mbit/s per session.

`--max-concurrency` (default 4) is how many requests one process works on at once. A
request that arrives while every slot is busy gets `429` immediately; nothing is queued.
Four slots suit request/response traffic. For streaming, raise the ceiling so a burst
waits a few milliseconds instead of being refused, and answer each `/v1/pairs` batch
with one encoder forward:

```dockerfile
CMD ["marimba", "serve", "--host", "0.0.0.0", "--port", "8000", "--backend", "torch", "--device", "cuda", "--model-id", "charcot-ai/marimba", "--prompt-price", "0.000002", "--completion-price", "0.000001", "--datacenters", "US", "--ready", "--threads", "1", "--keep-alive-seconds", "75", "--max-concurrency", "64", "--batched-encoder"]
```

This is the command of `.runpod/Dockerfile.server`. `marimba serve` without these flags
keeps four slots and the sequential path.

The ceiling counts every inference request, `/v1/chat/completions` included. Decoder
questions (language, term) run one at a time per process, so with 64 slots a burst of
them waits in line instead of getting `429`. Lower the ceiling on a deployment that
serves mostly decoder questions.

### Measured capacity

One NVIDIA L4 in bf16 with a container quota of about five CPU cores, `--threads 1`.
Each synthetic session sent, over HTTPS with the bearer token on persistent connections:

- one `/v1/windows` request every 500 ms with three questions (`voice_activity` at 1 s,
  `turn_end`, `speaker_change`);
- one `/v1/pairs` request with five windows every 4 s;
- one more `/v1/windows` request with `turn_end` every 5 s.

That is 2.45 requests and 3.45 windows per second, or 7.1 Mbit/s of upload (3.2 GB per
hour) per session. The audio is a public LibriSpeech sample. The load generator ran on
the server host, so the numbers leave out network time. Hop latency runs from the moment
a hop is due to its response; a hop is skipped when the previous one is still in flight
a full hop later. Each point ran for three minutes.

| Configuration | Points measured (sessions) | Largest point with no hop refused or skipped | Hop p95 there | First point that degrades |
|---|---|---:|---:|---|
| Default: 4 slots, sequential pairs | 8, 16, 32, 64, 128 | 8 | 35 ms | 16: 6.7% of hops refused with `429` |
| 256 slots, `--batched-encoder`, one process | 8, 16, 24, 32, 64, 128 | 24 | 252 ms | 32: 23.6% of hops skipped |
| Same flags, two processes on the GPU | 36, 48 | 36 | 160 ms | 48: 13.1% of hops skipped |
| Same flags, three processes on the GPU | 42, 72 | 42 | 152 ms | 72: 40.2% of hops skipped |

One process with 256 slots and `--batched-encoder`:

| Sessions | Hop p50 | Hop p95 | Hop p99 | Hops skipped | Server process CPU (one core = 100%) | GPU utilization (mean) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 18 ms | 43 ms | 64 ms | 0% | 37% | 10% |
| 16 | 27 ms | 81 ms | 121 ms | 0% | 75% | 21% |
| 24 | 64 ms | 252 ms | 390 ms | 0% | 113% | 32% |
| 32 | 900 ms | 1,167 ms | 1,286 ms | 23.6% | 125% | 35% |

The limit was the CPU of the Python process, not the GPU: at saturation the server
process used about 1.2 cores while the GPU averaged 35%. More processes on the same GPU
therefore raise the ceiling, and three of them brought the GPU to 86% at 42 sessions.
`marimba serve` starts one process, so run several on different ports and spread the
sessions across them; requests are stateless, so any session can use any process. Each
process holds its own copy of the model, about 3 GB of GPU memory in bf16.

Leave headroom below the largest clean point: one process at 24 sessions already had a
p99 of 390 ms, while 16 sessions kept p95 at 81 ms.

What a client sees past the limit depends on the ceiling:

- With four slots the server refuses instead of queueing. At 128 sessions 86.4% of hops
  got `429` and the accepted ones still had a p95 of 87 ms.
- With 256 slots nothing is refused. Requests wait inside the server, hop latency passes
  the hop interval and clients skip hops.
- With 64 slots no request was refused up to 32 sessions; `429` appeared only past
  saturation (40-second runs).

In both cases the server recovered on its own: after a minute at 64 sessions, hop p95
was back to its usual value within the first ten seconds at 16 sessions.

Other measurements on the same GPU:

- `--batched-encoder` cut the server time of a five-window `/v1/pairs` request from
  79 ms to 32 ms at 8 sessions, and raised the largest clean point of one process from
  16 to 24 sessions (64 slots, 40-second runs). In bf16 it moves probabilities by up to
  0.017 without flipping a decision; see [batch.md](batch.md#batched-encoder-forward).
- In float32, one process with the same flags held 16 sessions (hop p95 213 ms) and
  skipped 27.8% of hops at 24. GPU memory rose from about 3 GB to about 5 GB.
- With hops only (no `/v1/pairs`, no extra `turn_end`), one process held 32 sessions
  (hop p95 189 ms) and skipped 1.4% of hops at 36.

Not measured: other GPUs, runs longer than three minutes, the break point between the
listed points, clients across a network, and a load balancer in front of several
processes.

## Option A: Runpod Serverless load-balancing endpoint (recommended)

Load-balancing endpoints route HTTP directly to workers at
`https://ENDPOINT_ID.api.runpod.ai/<path>`, so `/v1/chat/completions`,
`/v1/models`, and `/ping` work unchanged; no queue handler is needed.

- GPU: 24 GB tier (L4, A5000, 3090). Weights and runtime fit comfortably; measured
  latencies below are on an L4.
- Workers: **minimum 0** while testing and under review; maximum 2 to 3.
  Idle timeout around 60 seconds.
- Health: Runpod polls `/ping`; it returns `200` once the model is loaded, because the
  app starts serving only after `Marimba.from_pretrained` returns.
- Auth: the Runpod gateway itself requires `Authorization: Bearer <Runpod API key>`.
  A router would then hold a Runpod key; scope it to this endpoint, or front the
  endpoint with your own gateway that checks `MARIMBA_API_KEY` instead.
- Cold start: container start plus weight load (seconds to tens of seconds) on the
  first request after idle. Routers measure time to first token and uptime, so keep
  one warm worker only once real traffic justifies it.

## Option B: GPU pod behind HTTPS

Run the same image on a pod and expose port 80 through the provider's HTTPS proxy or
your own TLS terminator. This is simpler to reason about (no gateway key, no cold
starts) but bills 24/7, so it fits only after traffic is steady.

## Cost estimate

Inputs: Runpod's published Serverless flex price for the 24 GB tier, **$0.75 per
GPU-hour** (about $0.00021 per second; check the current price before deploying), and
the [measured L4 bf16 end-to-end p50 latencies](benchmarks.md#end-to-end-per-path-latency-l4-bf16)
for one eight-second window.

| Question | L4 p50 | GPU cost per question at 100% utilization |
|---|---:|---:|
| Turn end, speaker readouts | 12 ms | $0.0000025 |
| Voice activity | 14 ms | $0.0000029 |
| Language | 154 ms | $0.000032 |
| Term | 1,399 ms | $0.00029 |

Scenarios, excluding HTTP and decode overhead:

| Scenario | Monthly cost |
|---|---:|
| Idle, minimum workers 0 | $0 compute, plus container disk (about $0.10 per GB-month) |
| 1,000 requests per day, one question each, 25% term, random arrivals, 60 s idle timeout: each request bills its compute (0.36 s on average) plus about 43 s of idle tail | about $280 |
| 10,000 requests per day, same mix: a request arrives every 8.6 s on average, so the worker almost never scales down | about $550, the same as one warm worker |
| One warm worker 24/7 (Serverless active or pod) | about $550 at flex price, before discounts |

Runpod bills the idle timeout, so at these rates idle tails, not compute, set the cost. A shorter idle timeout lowers it at the price of more cold starts.

Do not provision always-on workers or pods until traffic warrants it.
