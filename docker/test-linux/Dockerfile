# docker compose run --rm pvtest

FROM python:3.12-slim

# Runtime deps (OpenMP runtime) + any tools you like
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN pip install --no-cache-dir -U pip setuptools wheel
