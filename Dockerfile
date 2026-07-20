# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Docling обрабатывает PDF через библиотеки компьютерного зрения. В базовом
# python:3.11-slim нет их системных зависимостей, из-за чего загрузка PDF падает
# с ошибками libxcb.so.1/libGL.so.1 ещё до создания чанков.
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
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
