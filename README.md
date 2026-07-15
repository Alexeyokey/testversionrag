# Hybrid RAG project

Самостоятельный Python-проект, выделенный из `exam (10).ipynb`. Он загружает документы,
разбивает их на фрагменты, сохраняет эмбеддинги в Qdrant и выполняет гибридный поиск
Qdrant + BM25 с необязательным CrossEncoder-переранжированием. Ответ по найденному
контексту генерирует отдельный сервер vLLM через OpenAI-совместимый API.

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
│   ├── generation.py    # клиент OpenAI-совместимого API vLLM
│   └── service.py       # прикладной сценарий RAG
├── tests/
├── data/                # локальные документы, не коммитятся
├── pyproject.toml
├── Dockerfile
└── compose.yaml
```

## Быстрый запуск с vLLM

Нужны Docker Desktop с Linux-контейнерами, NVIDIA Container Toolkit и доступная NVIDIA GPU.
На Windows vLLM запускается в контейнере (либо в WSL), поскольку нативный Windows-запуск
официально не поддерживается.

Скопируйте конфигурацию и запустите Qdrant вместе с vLLM:

```powershell
Copy-Item .env.example .env
docker compose up -d qdrant vllm
docker compose logs -f vllm
```

Первый запуск скачает модель из `RAG_GENERATION_MODEL`. Когда в логах vLLM появится
сообщение о готовности сервера, проиндексируйте документы и задайте вопрос:

```powershell
docker compose build app
docker compose run --rm app index /data --recreate
docker compose run --rm app ask "Когда заключён договор?"
```

Для закрытой модели добавьте `HF_TOKEN` в `.env`. Чтобы использовать уже запущенный
локальный или удалённый vLLM, не запускайте сервис `vllm` и задайте адрес API:

```dotenv
RAG_GENERATION_MODEL=Qwen/Qwen3.5-0.8B
RAG_VLLM_BASE_URL=http://localhost:8000/v1
RAG_VLLM_API_KEY=
RAG_VLLM_TIMEOUT=120
```

Имя в `RAG_GENERATION_MODEL` должно совпадать с именем модели, опубликованным сервером.
Если сервер запущен с `--served-model-name`, укажите именно это имя.

## Локальный запуск CLI

Требуется Python 3.11 или новее. Qdrant и vLLM при этом могут работать в Docker:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d qdrant vllm
```

Основные команды:

```powershell
rag index "C:\path\to\documents" --recreate
rag search "Дата заключения договора"
rag search "Дата заключения договора" --json
rag ask "Когда заключён договор?"
```

Генеративная модель больше не загружается в процесс CLI. Эмбеддинг-модель и reranker
по-прежнему выполняются в приложении локально. Переранжирование можно отключить через
`RAG_ENABLE_RERANKER=false`.

## Конфигурация

Основные параметры находятся в `.env.example`:

- `QDRANT_URL`, `QDRANT_API_KEY`, `RAG_COLLECTION` — подключение к Qdrant;
- `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL` — модели retrieval-пайплайна;
- `RAG_GENERATION_MODEL` — имя модели, которую обслуживает vLLM;
- `RAG_VLLM_BASE_URL`, `RAG_VLLM_API_KEY`, `RAG_VLLM_TIMEOUT` — подключение к vLLM;
- `RAG_MAX_NEW_TOKENS` — предел длины ответа;
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` — разбиение документов;
- `RAG_TOP_K`, `RAG_CANDIDATE_K` — количество результатов;
- `RAG_VECTOR_WEIGHT`, `RAG_BM25_WEIGHT`, `RAG_RANK_CONSTANT` — объединение рангов.

Размер вектора определяется по реально полученному эмбеддингу при создании коллекции.
После смены embedding-модели заново выполните индексацию с `--recreate`.

Excel-листы обрабатываются как таблицы: загрузчик определяет строку заголовков и повторяет
названия колонок в каждом чанке. Это помогает сопоставлять запросы с конкретными строками,
датами и значениями.

## Проверки

```powershell
pytest
ruff check .
docker compose config
```

Исходный ноутбук не изменяется, проект не зависит от состояния его ячеек.
