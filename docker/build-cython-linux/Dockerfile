# docker compose run --rm ivbuild

FROM python:3.12-slim

# Compiler + OpenMP
RUN apt-get update && apt-get install -y \
    build-essential \
    g++

WORKDIR /work

RUN pip install -U pip setuptools wheel cython numpy
