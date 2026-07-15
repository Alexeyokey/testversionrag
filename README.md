# Hybrid RAG project

Самостоятельный Python-проект, выделенный из `exam (10).ipynb`. Он загружает документы,
разбивает их на фрагменты, сохраняет эмбеддинги в Qdrant и выполняет гибридный поиск
Qdrant + BM25 с необязательным CrossEncoder-переранжированием.

## Структура

```text
rag_project/
├── src/rag_app/
│   ├── cli.py           # команды index, search и ask
│   ├── config.py        # конфигурация из переменных окружения
│   ├── documents.py     # TXT/MD/RST/PDF/DOCX/XLS/XLSX/URL
│   ├── embeddings.py    # Hugging Face embedding model
│   ├── vector_store.py  # работа с Qdrant
│   ├── retrieval.py     # BM25 и reciprocal-rank fusion
│   ├── reranker.py      # CrossEncoder reranker
│   ├── generation.py    # локальная генеративная модель
│   └── service.py       # прикладной сценарий RAG
├── tests/
├── data/                # локальные документы, не коммитятся
├── pyproject.toml
├── Dockerfile
└── compose.yaml
```

## Локальный запуск

Требуется Python 3.11 или новее и запущенный Qdrant. Docker-образ использует Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d qdrant
```

Индексация файла, папки или URL:

```powershell
rag index "C:\path\to\documents" --recreate
```

Гибридный поиск:

```powershell
rag search "Дата заключения договора"
rag search "Дата заключения договора" --json
```

Для генерации ответа в `.env.example` выбрана небольшая официальная модель Qwen3.5:

```dotenv
RAG_GENERATION_MODEL=Qwen/Qwen3.5-0.8B
```

После этого:

```powershell
rag ask "Когда заключён договор?"
```

При первом вызове каждая Hugging Face модель будет загружена в локальный cache. На CPU
индексация, переранжирование и генерация могут занимать заметное время. Переранжирование
можно отключить через `RAG_ENABLE_RERANKER=false`.

AWQ-модели требуют дополнительный backend `gptqmodel`. На Windows с Python 3.14 его
зависимости могут собираться из исходников и требовать Visual Studio Build Tools, поэтому
для локального запуска рекомендуется неквантованный checkpoint без суффикса `AWQ`.

## Docker

Файлы уже подготовлены для контейнерного запуска. Сначала положите документы в `data/`,
затем соберите образ и запустите Qdrant:

```powershell
docker compose build app
docker compose up -d qdrant
docker compose run --rm app index /data --recreate
docker compose run --rm app search "Дата заключения договора"
```

Compose хранит Qdrant и cache моделей в именованных volumes. Каталог `data/` монтируется
в контейнер только для чтения. Текущий Dockerfile устанавливает обычную сборку PyTorch;
для GPU потребуется CUDA-совместимый базовый образ и NVIDIA Container Toolkit.

## Конфигурация

Основные параметры находятся в `.env.example`:

- `QDRANT_URL`, `QDRANT_API_KEY`, `RAG_COLLECTION` — подключение и коллекция;
- `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL`, `RAG_GENERATION_MODEL` — модели;
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` — разбиение документов;
- `RAG_TOP_K`, `RAG_CANDIDATE_K` — количество результатов;
- `RAG_VECTOR_WEIGHT`, `RAG_BM25_WEIGHT`, `RAG_RANK_CONSTANT` — объединение рангов.

Размер вектора определяется по реально полученному эмбеддингу при создании коллекции.
Если embedding-модель меняется, запустите индексацию с `--recreate`.

## Проверки

```powershell
pytest
ruff check .
```

Исходный ноутбук не изменяется и проект не зависит от состояния его ячеек.
