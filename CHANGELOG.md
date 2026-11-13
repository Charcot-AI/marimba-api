# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.2 - 2026-11-13

### Added

- OpenAI-compatible chat completions server.
- Vapi custom transcriber example.

### Changed

- Live demo term card turns off 2 s after a detection.

## 0.1.1 - 2026-11-12

### Added

- LiveKit Agents turn detector.
- Hugging Face Space app on ZeroGPU.

### Changed

- Shorter Runpod Hub description.

## 0.1.0 - 2026-11-12

### Added

- `marimba-audio` inference package with a Torch backend, batch decisions, and
  streaming over short audio windows.
- Typed questions (choice, yes/no, and score) that return a value, a
  probability, and the threshold used.
- Local live demo with parallel decision cards for microphone or file audio.
- Pipecat integration: `MarimbaTurnAnalyzer` for end-of-turn detection, with a
  runnable example in `examples/pipecat`.
- Runpod Serverless decoder worker.
