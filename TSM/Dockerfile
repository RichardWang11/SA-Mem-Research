FROM python:3.11.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tsm
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY . .

CMD ["python", "scripts/verify_offline_repro.py"]
