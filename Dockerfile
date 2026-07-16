# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-deps --editable .

ENTRYPOINT ["rag"]
CMD ["--help"]