# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CC=gcc \
    CXX=g++

WORKDIR /app

# Docling нужны системные библиотеки компьютерного зрения. Triton при первом
# GPU-запуске embeddings собирает launcher-модуль, поэтому компилятор нужен
# не только во время сборки образа, но и внутри запущенного контейнера.
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxcb1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-deps --editable .

ENTRYPOINT ["rag"]
CMD ["--help"]
