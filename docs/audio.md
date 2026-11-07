# Audio inputs

The call, batch and map_fn interfaces use the same decoder:

| Input | Example / convention |
|---|---|
| File path | `"speech.wav"` or `Path("speech.wav")` |
| Encoded audio | WAV/FLAC bytes or `{"bytes": encoded, "path": None}` |
| File mapping | `{"path": "speech.wav"}` |
| Decoded mapping | `{"array": samples, "sampling_rate": 16000}` |
| Array | `model(samples, questions, sampling_rate=16000)` |
| Lazy datasets audio | A decoder with `get_all_samples()`, including compatible TorchCodec decoders |

File formats depend on the installed SoundFile/libsndfile support. No arbitrary
codec support is implied. When both bytes and path are present, bytes take precedence.
Arrays are int16 or floating point; int16 is scaled by 32768. Floating-point values
must be finite. Two-dimensional arrays are samples by channels and mixed to mono;
lazy decoder tensors are channels by samples.

SoundFile handles file decoding and soxr handles resampling to 16 kHz float32.
An array must carry its sampling rate. Streaming requires already resampled mono
16-kHz chunks; its sample clock must not be inferred from wall time.

Eight-second windows keep the tail of long input, rather than segmenting an entire
recording. Use [streaming](streaming.md) for a recording's successive windows.
Decoder context above eight seconds is **unvalidated**, even though the API accepts
up to 30 seconds. It is not a general long-recording transcription API.
