# Replicate (Cog) runner

`predict.py` answers typed questions about one audio file with
`Marimba.from_pretrained(backend="torch")`, using the same validated inference
as the Python package. Unlike the Runpod decoder worker, it accepts every
question type the package supports.

## Input

- `audio`: an audio file. Audio is resampled to 16 kHz mono and only the last
  eight seconds are scored.
- `questions`: a JSON list of questions with the `Question` fields `text`,
  `kind` (`noul`, `score`, or `choice`), and the optional `options`, `task`,
  `threshold`, and `seconds`.

```json
[
  {"text": "Is someone speaking?", "kind": "noul"},
  {"text": "Has the speaker finished?", "kind": "score"},
  {"text": "What language is being spoken?", "kind": "choice",
   "options": ["English", "Portuguese", "Spanish"]},
  {"text": "Did someone say “marimba”?", "kind": "noul"},
  {"text": "How many speakers?", "kind": "choice", "options": ["0", "1", "2", "3+"]}
]
```

The output is `{"decisions": [...], "model_revision": "..."}`, where each
decision is `Decision.to_dict()`. PII questions return `status: "unavailable"`
because the runner does not configure a PII detector.

## Build and push

Run from the repository root. The build downloads the pinned model and encoder
revisions into the image, so predictions do not need Hub access. Pass a Hub
token as a build secret only while the model repository requires one:

```sh
cog build -f cog/cog.yaml --secret id=hf_token,src=$HOME/.cache/huggingface/token
cog predict -f cog/cog.yaml -i audio=@speech.wav
cog push -f cog/cog.yaml r8.im/OWNER/marimba --secret id=hf_token,src=$HOME/.cache/huggingface/token
```
