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
container starts without network access to the Hub.

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

## Batched encoder for `/v1/pairs`

`marimba serve --batched-encoder` answers the windows of one
[`/v1/pairs`](window-server.md#post-v1pairs) request with one encoder forward per
`--encoder-batch-size` windows (default 16). It is off by default and responses have the
same schema either way; see [batch.md](batch.md#batched-encoder-forward) for the measured
difference against the sequential path.

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
