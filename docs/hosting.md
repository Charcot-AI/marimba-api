# Hosting the OpenAI-compatible server

The [server](openai-server.md) is a single FastAPI process holding one model in GPU
memory. Decoder questions (language, term) share one lock per process, so scale out
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
| 10,000 requests per day, one question each, 25% term: about 4,600 GPU-seconds per day, doubled for idle tails and cold starts | about $60 |
| One warm worker 24/7 (Serverless active or pod) | about $550 at flex price, before discounts |

Do not provision always-on workers or pods until traffic warrants it.
