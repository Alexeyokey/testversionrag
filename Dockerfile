# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Этот слой изменится только при изменении зависимостей.
COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install --editable .
    
# Исходники копируются после зависимостей.
COPY pyproject.toml README.md ./
COPY src ./src

# Устанавливаем только сам проект — зависимости уже установлены.
RUN python -m pip install --no-deps .

ENTRYPOINT ["rag"]
CMD ["--help"]