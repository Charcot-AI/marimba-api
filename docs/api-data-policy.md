# Marimba API Data Policy

Effective date: 2026-11-13

This policy explains how Charcot AI ("we") handles data sent to the hosted Marimba API, including requests routed to us through OpenRouter. It covers the API only. Our website is covered by https://charcot.ai/privacy.

## Summary

- **Prompt logging:** We do not log or store request audio, question text or model outputs.
- **Retention:** Request content stays in memory only while we process it and is discarded once the response is sent. Operational logs, which contain no request content, are kept for up to 90 days.
- **Training:** We do not use API requests or outputs to train or improve any model.

## What we process

Each request contains one audio clip (base64; up to 30 seconds and 16 MiB) and one or more text questions about it. Marimba returns structured decisions, such as the spoken language, whether a turn has ended or whether speech is present, with probabilities.

We decode the audio and run inference on it in memory. We do not write request audio, question text or outputs to disk, to a database or to any other storage. Request content is discarded when processing ends.

Audio can contain voices and personal information. Use the API only for audio you have the right to submit. Do not send audio that you are not allowed to share with a service provider.

## What we log

Our server keeps operational logs only:

- **Access logs:** the time, HTTP method, path, response status and the network address that connected to the server. When requests arrive through OpenRouter or our hosting provider's gateway, that address is usually an intermediary address, not the end user's.
- **Error logs:** the processing stage and the type of error, for example `ValueError`. We do not log error messages or tracebacks because they could contain request content.

These logs contain no audio, question text or outputs. Our hosting provider stores them for up to 90 days and then deletes them automatically. We use them only to operate, secure and debug the service.

Billing usage, meaning token counts per request, is reported to OpenRouter in each response so it can bill and pay us. We do not keep per-request usage records of our own. We reconcile our monthly invoice against OpenRouter's usage reports.

## Training

We do not use API inputs or outputs to train, fine-tune, evaluate or improve Marimba or any other model, and we do not share them with anyone for that purpose.

## Zero data retention

We do not claim zero data retention at this time. Our own server does not store request content, but we have not yet received written confirmation that our hosting provider never stores request content in transit. We will update this policy if that changes.

## Subprocessors

| Subprocessor | Role | Data involved |
|---|---|---|
| Runpod, Inc. | GPU hosting (Runpod Serverless). Requests pass through Runpod's gateway to our workers. | Request and response content in transit and in worker memory; our operational logs (no request content), kept up to 90 days. |
| OpenRouter, Inc. | API router. It forwards developers' requests to us and returns our responses. | Requests and responses it routes to us, under OpenRouter's own privacy policy and settings: https://openrouter.ai/privacy |

We have not yet received written confirmation from Runpod that its gateway does not log or retain request or response bodies, and we have not yet signed Runpod's Data Processing Agreement.

Serving region: United States. Our endpoint is configured to run only in Runpod data centers in the United States.

## Security

- All traffic to the API uses HTTPS (TLS).
- The API accepts requests only with an API key issued by us. Runpod's gateway requires its own key as well.
- Request content exists only in the memory of the worker that processes it.
- Runpod reports SOC 2 Type II and ISO/IEC 27001:2022 certifications for its platform. Charcot AI itself is not SOC 2 certified.

## Legal requests

We do not store request content, so we cannot produce it later. We may disclose operational logs if the law requires it.

## Changes

We will post changes here and update the effective date. If a change weakens these commitments, we will notify OpenRouter before it takes effect.

## Contact

Charcot AI
solano@charcot.ai
