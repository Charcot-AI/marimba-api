FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The base image's Debian cryptography installation has no pip RECORD.
RUN python -m pip install --no-cache-dir --ignore-installed cryptography

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir '.[torch]' 'runpod==1.8.1'

COPY .runpod/handler.py ./handler.py
CMD ["python", "-u", "/app/handler.py"]
