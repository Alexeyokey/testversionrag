# Hybrid RAG project

Самостоятельный Python-проект, выделенный из `exam (10).ipynb`. Он загружает документы,
разбивает их на фрагменты, сохраняет эмбеддинги в Qdrant и выполняет гибридный поиск
Qdrant + BM25 с необязательным CrossEncoder-переранжированием. Ответ по найденному
контексту генерирует отдельный сервер vLLM через OpenAI-совместимый API.

## Структура

```text
rag_project/
├── src/rag_app/
│   ├── cli.py           # команды index, search, ask и chat
│   ├── chat.py          # интерактивная сессия и история диалога
│   ├── config.py        # конфигурация из переменных окружения
│   ├── documents.py     # загрузка, RTF-декодирование и чанкование
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
├── Dockerfile.vllm      # vLLM 0.19.0 + Transformers 5.5.4
└── compose.yaml
```

## Быстрый запуск с vLLM

Нужны Docker Desktop с Linux-контейнерами, NVIDIA Container Toolkit и доступная NVIDIA GPU.
На Windows vLLM запускается в контейнере (либо в WSL), поскольку нативный Windows-запуск
официально не поддерживается.

Скопируйте конфигурацию и запустите Qdrant вместе с vLLM:

```powershell
Copy-Item .env.example .env
docker compose build app vllm
docker compose up -d qdrant vllm
docker compose logs -f vllm
```

Тестовый профиль использует `QuantTrio/Qwen3.6-27B-AWQ`, vLLM 0.19.0 и
Transformers 5.5.4. Первый запуск скачает около 21 GiB файлов модели. Когда контейнер
станет healthy, проверьте версию, опубликованную модель и короткую генерацию:

```powershell
docker compose run --rm app check-vllm
```

После успешной проверки проиндексируйте документы и задайте вопрос:

```powershell
docker compose run --rm app index /data --recreate
docker compose run --rm app ask "Когда заключён договор?"
docker compose run --rm -it app chat --stream
```

Для закрытой модели добавьте `HF_TOKEN` в `.env`. Чтобы использовать уже запущенный
локальный или удалённый vLLM, не запускайте сервис `vllm` и задайте адрес API:

```dotenv
RAG_GENERATION_MODEL=QuantTrio/Qwen3.6-27B-AWQ
RAG_VLLM_BASE_URL=http://localhost:8000/v1
RAG_VLLM_API_KEY=
RAG_VLLM_TIMEOUT=120
RAG_VLLM_MAX_MODEL_LEN=8192
RAG_VLLM_MAX_NUM_SEQS=4
RAG_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE=4
RAG_VLLM_GPU_MEMORY_UTILIZATION=0.9
RAG_VLLM_TENSOR_PARALLEL_SIZE=1
```

Имя в `RAG_GENERATION_MODEL` должно совпадать с именем модели, опубликованным сервером.
Если сервер запущен с `--served-model-name`, укажите именно это имя.

### vLLM без Docker

GPU-сервер устанавливайте в отдельное чистое окружение Python 3.12, чтобы его
CUDA/PyTorch-зависимости не смешивались с приложением:

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install -r requirements-vllm.txt
```

`requirements-vllm.txt` фиксирует `vllm==0.19.0` и `transformers==5.5.4`.

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
rag chat --stream
```

Генеративная модель больше не загружается в процесс CLI. Эмбеддинг-модель и reranker
по-прежнему выполняются в приложении локально. Переранжирование можно отключить через
`RAG_ENABLE_RERANKER=false`.

### Интерактивный чат без повторной загрузки моделей

Команда `rag ask` запускает отдельный процесс для одного вопроса, поэтому локальные
embedding-модель и reranker загружаются заново при каждом новом запуске. Команда
`rag chat` создаёт один `RagService` и использует его до выхода из чата:

```bash
rag chat
rag chat --stream
rag chat --top-k 5 --history-turns 4
```

В Docker интерактивный терминал нужно передать контейнеру через `-it`:

```bash
docker compose up -d qdrant vllm
docker compose run --rm -it app chat --stream
```

Доступные команды внутри чата: `/stats`, `/clear`, `/quit`. `/clear` удаляет только
историю разговора; загруженные embedding-модель, reranker и индекс остаются в памяти.
Флаг `--no-warmup` откладывает их первую загрузку до первого вопроса. Qwen работает
в отдельном постоянно запущенном контейнере `vllm` и между вопросами не перезапускается.

`Dockerfile.vllm` основан на `vllm/vllm-openai:v0.19.0`, устанавливает
`transformers==5.5.4` и во время сборки проверяет обе версии. Модель передаётся
`vllm serve` позиционным аргументом. Так как приложение отправляет только текст,
`--language-model-only` не загружает визуальный encoder. Тестовый профиль ограничивает
контекст 8192 токенами и CUDA Graph четырьмя последовательностями.

Файлы AWQ-модели занимают около 21 GiB, а при работе дополнительно нужна память для
служебных буферов и KV cache. Поэтому одной 24-GiB GPU может не хватить даже при
квантизации. Для нескольких GPU укажите их число, например
`RAG_VLLM_TENSOR_PARALLEL_SIZE=2`; при нехватке памяти сначала уменьшайте
`RAG_VLLM_MAX_MODEL_LEN` и `RAG_VLLM_MAX_NUM_SEQS`.

## Конфигурация

Основные параметры находятся в `.env.example`:

- `QDRANT_URL`, `QDRANT_API_KEY`, `RAG_COLLECTION` — подключение к Qdrant;
- `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL` — модели retrieval-пайплайна;
- `RAG_GENERATION_MODEL` — имя модели, которую обслуживает vLLM;
- `RAG_VLLM_BASE_URL`, `RAG_VLLM_API_KEY`, `RAG_VLLM_TIMEOUT` — подключение к vLLM;
- `RAG_VLLM_MAX_MODEL_LEN` — размер контекста vLLM (по умолчанию 8192 токена);
- `RAG_VLLM_MAX_NUM_SEQS` — максимум одновременно обрабатываемых последовательностей;
- `RAG_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE` — максимальный batch для CUDA Graph capture;
- `RAG_VLLM_GPU_MEMORY_UTILIZATION` — доля памяти GPU для vLLM (по умолчанию 0.9);
- `RAG_VLLM_TENSOR_PARALLEL_SIZE` — число GPU для tensor parallel (по умолчанию 1);
- `RAG_MAX_NEW_TOKENS` — предел длины ответа;
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` — разбиение документов;
- `RAG_DOCLING_CHUNK_TOKENS` — токеновый лимит чанков PDF/DOCX/ODT/XLSX;
- `RAG_EMBEDDING_BATCH_SIZE` — число чанков в одной порции embedding и записи в Qdrant;
- `RAG_TOP_K`, `RAG_CANDIDATE_K` — количество результатов;
- `RAG_VECTOR_WEIGHT`, `RAG_BM25_WEIGHT`, `RAG_RANK_CONSTANT` — объединение рангов.

Размер вектора определяется по реально полученному эмбеддингу при создании коллекции.
После смены embedding-модели заново выполните индексацию с `--recreate`.

PDF, DOCX, ODT и XLSX преобразуются через Docling и делятся `HybridChunker`. Таблицы XLSX
сериализуются в Markdown с повторением заголовка при разделении, а данные встроенных
диаграмм не индексируются повторно. Содержимое ячеек сохраняется без эвристического
удаления повторяющихся значений, чтобы не потерять контекст таблицы.

RTF обрабатывается встроенным декодером без LibreOffice и дополнительных Python-пакетов.
Он учитывает `ansicpg`, кириллическую кодировку CP1251, Unicode-последовательности `\uN`,
скрытые служебные группы и видимый результат полей. Полученный текст делится
`RecursiveCharacterTextSplitter` с параметрами `RAG_CHUNK_SIZE` и `RAG_CHUNK_OVERLAP`.

## Проверки

```powershell
pytest
ruff check .
docker compose config
docker compose run --rm app check-vllm
```

### Простая проверка качества RAG

Сначала проиндексируйте тестовые документы, затем запустите вопросы из JSONL:

```powershell
rag index "C:\path\to\test-documents" --recreate
rag evaluate evaluation\testset.example.jsonl --output evaluation\results.json
```

Одна строка тестового набора выглядит так:

```json
{"question":"Когда заключён договор?","reference":"15 марта 2025 года","answer_terms":["15 марта 2025"],"context_terms":["договор"],"expected_sources":["contract.pdf"]}
```

- `answer_terms` проверяются в ответе генеративной модели;
- `context_terms` проверяются в найденных чанках и отдельно оценивают retrieval;
- `expected_sources` проверяет, что поиск вернул нужный файл;
- `reference` сохраняется в отчёте; если `answer_terms` не задан, он используется как
  ожидаемый фрагмент ответа.

Команда возвращает код `0`, если пройдены все проверки, и `1`, если хотя бы одна
проверка не прошла. Подробности записываются в JSON-отчёт.

### Семантическая оценка RAGAS через vLLM

RAGAS использует тот же OpenAI-compatible endpoint и по умолчанию ту же модель, что
команда `rag ask`. Обязательные метрики — `faithfulness`, `context_recall` и
`answer_accuracy`: только они определяют `passed` и основной `mean_score`.
`context_precision` и `answer_relevancy` остаются диагностическими и не проваливают
тест при низком значении.

```powershell
rag evaluate-ragas evaluation\testset.example.jsonl `
  --threshold 0.7 `
  --output evaluation\ragas-results.json
```

Чтобы не выполнять нестабильную и дополнительную по стоимости `answer_relevancy`,
добавьте флаг:

```powershell
rag evaluate-ragas evaluation\ragas_documents_testset.jsonl `
  --skip-answer-relevancy `
  --output evaluation\ragas-results.json
```

В таком отчёте `scores.answer_relevancy` и сводное значение метрики равны JSON `null`,
а `skipped_metrics` явно содержит `answer_relevancy`. Это означает «не измерялось»,
а не нулевое качество.

`Context Precision` по умолчанию проверяет найденные чанки параллельно. Число
одновременных запросов к judge задаётся через
`RAGAS_CONTEXT_PRECISION_CONCURRENCY` (по умолчанию `5`). Если эта диагностическая
метрика не нужна, её можно полностью отключить:

```powershell
rag evaluate-ragas evaluation\ragas_documents_testset.jsonl `
  --skip-context-precision `
  --output evaluation\ragas-results.json
```

Тогда `scores.context_precision` будет равен JSON `null`, а имя метрики появится в
`skipped_metrics`.

В Docker тот же запуск выглядит так:

```bash
docker compose run --rm app evaluate-ragas \
  /evaluation/testset.example.jsonl \
  --threshold 0.7 \
  --output /evaluation/ragas-results.json
```

Флаги `--skip-answer-relevancy` и `--skip-context-precision` доступны и при
Docker-запуске.

Compose подключает локальный каталог `./evaluation` как `/evaluation`, поэтому входной
набор доступен контейнеру, а JSON-отчёт сохраняется на хосте.

Итоговые значения judge-метрик не кэшируются: Faithfulness, Context Recall,
Answer Accuracy, Context Precision и Answer Relevancy рассчитываются заново при каждом
запуске. Сохраняются только дорогие промежуточные артефакты RAGAS:

- `faithfulness_statements` — атомарные факты, один раз выделенные из конкретного ответа;
- `answer_relevancy_questions` — три вопроса, один раз сгенерированные из ответа.

Faithfulness каждый раз заново проверяет сохранённые факты по текущему retrieval context.
Answer Relevancy каждый раз заново строит embeddings и считает similarity. DeepEval
полностью пересчитывает свои метрики и ничего не записывает в этот кэш.

В JSON-результате `artifacts` содержит использованную разметку,
`cached_artifacts` показывает, что было прочитано с диска, а
`summary.artifact_cache_hits` — количество таких чтений. Управление кэшем:

CLI отдельно показывает `создано` и `повторно использовано`. Поэтому при первом
запуске ожидается `создано: N; повторно использовано: 0`, а при идентичном повторном
запуске — `создано: 0; повторно использовано: N`.

```bash
# Заново сгенерировать и заменить совпавшие факты и вопросы.
rag evaluate-ragas evaluation/testset.example.jsonl --refresh-artifact-cache

# Использовать другой каталог или полностью отключить кэш на один запуск.
rag evaluate-ragas evaluation/testset.example.jsonl --artifact-cache-dir evaluation/my-cache
rag evaluate-ragas evaluation/testset.example.jsonl --no-artifact-cache
```

У каждой строки JSONL обязательно должно быть поле `reference`. Judge настраивается так:

```dotenv
# Пусто — использовать RAG_GENERATION_MODEL и RAG_VLLM_BASE_URL.
RAGAS_JUDGE_MODEL=
RAGAS_THRESHOLD=0.7
RAGAS_MAX_TOKENS=2048
RAGAS_CONTEXT_PRECISION_CONCURRENCY=5
EVALUATION_ARTIFACT_CACHE_ENABLED=true
EVALUATION_ARTIFACT_CACHE_DIR=evaluation/artifact-cache
```

Если `RAGAS_JUDGE_MODEL` задан, это имя должно обслуживаться сервером по адресу
`RAG_VLLM_BASE_URL`. Один vLLM-сервис обычно обслуживает одну модель, поэтому для работы
без второго GPU оставьте значение пустым: генерация и judging будут выполняться
последовательно одной моделью. RAGAS использует структурированные ответы OpenAI API;
выбранная модель и версия vLLM должны поддерживать JSON schema/structured output.

### Сравнение 4 конфигураций через RAGAS и DeepEval

В проект включены фиктивный корпус `evaluation/synthetic_corpus` и 12 эталонных
вопросов `evaluation/synthetic_testset.jsonl`. Эксперимент сравнивает:

1. `vector_only` — только dense vector search;
2. `bm25_only` — только лексический BM25;
3. `hybrid` — vector + BM25 через weighted RRF;
4. `hybrid_reranker` — hybrid retrieval и CrossEncoder reranker.

Сначала проиндексируйте синтетические документы:

```bash
docker compose build app
docker compose up -d qdrant vllm
docker compose run --rm app index /evaluation/synthetic_corpus --recreate
```

Затем выполните общий benchmark:

```bash
docker compose run --rm app benchmark \
  /evaluation/synthetic_testset.jsonl \
  --threshold 0.7 \
  --output-dir /evaluation/benchmark-results
```

Benchmark также принимает `--skip-answer-relevancy` и `--skip-context-precision`;
соответствующая метрика будет пропущена и в RAGAS, и в DeepEval, чтобы сравнение
оставалось симметричным.
Флаги `--artifact-cache-dir`, `--no-artifact-cache` и `--refresh-artifact-cache`
работают аналогично. В Docker Compose кэш по умолчанию находится в
`/evaluation/artifact-cache` и поэтому сохраняется в локальной папке
`./evaluation/artifact-cache`.

Для быстрой проверки соединения сначала добавьте `--limit 1`. Полный запуск всё равно
делает много judge-вызовов и на локальной модели может занять заметное время; только
независимые проверки чанков RAGAS Context Precision выполняются параллельно.

Ответ RAG для каждой пары «конфигурация × вопрос» генерируется один раз. RAGAS и
DeepEval получают один и тот же `response`, `reference` и список `retrieved_contexts`.
Это исключает случайное различие ответов между двумя evaluator-ами.

Результаты:

- `benchmark-details.json` — ответы, контексты, оценки, причины и ошибки;
- `benchmark-comparison.csv` — строки-конфигурации и столбцы-метрики обеих библиотек;
- `benchmark-report.md` — описание метрик, сводная таблица и автоматический вывод о
  лучшей конфигурации.

Сравниваются одинаковые понятия: Faithfulness, Context Recall, Answer Accuracy,
Context Precision и Answer Relevancy. Первые три обязательны; последние две
диагностические. Для DeepEval Answer Accuracy реализована как reference-based GEval.
Формулы и judge-промпты библиотек различаются, поэтому абсолютные баллы не обязаны
совпадать. Для выбора конфигурации основной средний балл считается только по трём
обязательным метрикам.

Исходный ноутбук не изменяется, проект не зависит от состояния его ячеек.
