---
title: Marimba scaffold
emoji: 🎵
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: space/app.py
python_version: "3.11"
private: true
---

# Private Space scaffold — not deployed

This directory is a local scaffold only. No Space or public model repository has
been created or pushed. The private marker is descriptive metadata, **not access
control**: any future Space must be explicitly created with private visibility.
Converted head/readout weights are Apache-2.0; see ../hf/README.md.

From the source checkout root:

```sh
pip install -r space/requirements.txt
python space/app.py
```

The requirements resolve the package from the checkout root, not an assumed PyPI
release. A future Space build needs the package source, pyproject.toml, licenses
and this directory in the same layout; installing only this directory will not
work. Root requirements should install space/requirements.txt for that layout.
Configure HF_TOKEN through a secret manager only if authenticated access is needed.
Never paste credentials into questions. Do not enable public share links.

The app is CPU float32 with one queued request at a time. Upload a file and edit
the JSON list of supported Question constructor arguments. Decisions are JSON.
Only the last eight seconds are scored, not an entire recording. Term decoding
can be slow; same-speaker is experimental. Context above eight seconds is
**unvalidated**; no real-time or measured Mac performance is claimed.

Do not upload sensitive recordings. Gradio caches uploads temporarily; this is
not a privacy, retention or deletion guarantee. A deployed service needs its own
authentication, consent, retention controls and operational review.
