# Marimba turn detection for Pipecat

Marimba's `MarimbaTurnAnalyzer` implements Pipecat's Smart Turn interface for
local end-of-turn decisions. Maintained by Charcot AI, the authors of Marimba.
Tested with Pipecat v1.16.0 on Linux CPU, Python 3.13, and PyTorch 2.8.0.

## Install

Python 3.11 or newer is required for the Pipecat extra. Install from PyPI:

```sh
pip install "marimba-audio[pipecat]"
```

Core installation does not include an inference engine; add the `torch` extra
(`pip install "marimba-audio[pipecat,torch]"`) unless you already have one.

### From source

```sh
uv venv
uv pip install -e ".[pipecat,torch]" "pipecat-ai==1.16.0"
```

The `torch` extra provides the inference engine. Prepare the default model and
encoder weights once with network access (and `HF_TOKEN` if access requires
authentication):

```sh
uv run --no-sync python -c 'from marimba import Marimba; Marimba.from_pretrained(backend="torch", device="cpu", dtype="float32")'
```

Then the example runs fully offline on CPU, without API keys or audio downloads:

```sh
env -u HF_TOKEN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync python examples/pipecat/turn_analyzer.py
```

It generates two enveloped tones followed by silence, feeds 20 ms int16 PCM
chunks through Pipecat's public `set_sample_rate`, `append_audio`, and
`analyze_end_of_turn` API, and prints the state, probability, and model latency
for each decision. `COMPLETE` means the probability met the default 0.5
threshold; `INCOMPLETE` means it did not. Latency excludes model loading and
audio generation; the first inference can include warmup.

This deliberately uses the analyzer API rather than a VAD-driven pipeline:
synthetic tones are not speech, so a speech VAD may never start a turn. The
example explicitly labels the tone chunks as speech and silence chunks as
nonspeech. It exercises real model inference and Pipecat buffering/metrics,
not speech recognition or conversational quality. Missing cached weights or
inference errors are logged and return `INCOMPLETE` without probability metrics,
allowing Pipecat's normal silence timeout to end the turn without a fake prediction.
No real-time guarantee is claimed.

## Use in a Pipecat pipeline

For live speech, let a VAD supply speech boundaries and let Marimba decide
whether a pause ends the turn:

```python
from marimba.integrations.pipecat import MarimbaTurnAnalyzer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

aggregators = LLMContextAggregatorPair(
    LLMContext(),
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=MarimbaTurnAnalyzer())]
        ),
    ),
)
# Supply your transport and STT/LLM/TTS services.
pipeline = Pipeline(
    [
        transport.input(),
        stt,
        aggregators.user(),
        llm,
        tts,
        transport.output(),
        aggregators.assistant(),
    ]
)
```

The default strategy waits for a transcript as well as a complete turn. For an
audio-only pipeline without STT, set `wait_for_transcript=False` on the stop
strategy. Pipecat configures the input sample rate; Marimba resamples to 16 kHz
and retains up to eight seconds. Pass a preloaded CPU model as in the runnable
example to choose the runtime explicitly.

Incoming PCM16 is mono by default. For interleaved multichannel input, set
`MarimbaTurnAnalyzer(num_channels=2)` to match the transport. The analyzer
downmixes to mono PCM16 before buffering, with rounding to the nearest sample
(at most half a PCM16 step). Pipecat's analyzer API does not pass channel metadata,
so the count cannot be inferred automatically. A VAD must also receive the channel
format it supports; Silero expects mono, so downmix upstream of Silero when using it.

See the [main README](../../README.md), [streaming guide](../../docs/streaming.md),
and [model card](https://huggingface.co/charcot-ai/marimba) for capabilities and limitations.

## License

This example and the Marimba integration are licensed under
[Apache-2.0](../../LICENSE). Pipecat is a separate BSD-2-Clause dependency.
