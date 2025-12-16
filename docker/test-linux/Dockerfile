# docker compose run --rm pvtest

FROM python:3.12-slim

# Build + runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    tk \
    tcl \
 && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN pip install --no-cache-dir -U pip setuptools wheel