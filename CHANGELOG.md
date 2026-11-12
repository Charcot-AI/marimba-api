# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-11-12

### Added

- `marimba-audio` inference package with a Torch backend, batch decisions, and
  streaming over short audio windows.
- Typed questions (choice, yes/no, and score) that return calibrated decisions.
- Local live demo with parallel decision cards for microphone or file audio.
- Pipecat integration: `MarimbaTurnAnalyzer` for end-of-turn detection, with a
  runnable example in `examples/pipecat`.
- Runpod Serverless decoder worker.
