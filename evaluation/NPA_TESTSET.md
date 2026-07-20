# Набор тестов по нормативным правовым актам

Набор построен по документам из архива `НПА для чат-бота.zip` и предназначен
для проверки retrieval, генерации ответа и judge-метрик RAGAS/DeepEval.

## Что подготовлено

- `data/npa/НПА для чат-бота` — распакованные исходники: RTF, ODT и реестр XLSX;
- `data/npa_text` — полный текст 102 нормативных документов в UTF-8 TXT;
- `data/npa_text/manifest.json` — соответствие исходного файла и TXT-копии;
- `evaluation/npa_testset.jsonl` — 60 эталонных вопросов по 30 документам.

TXT-файлы содержат полный извлеченный текст, а не короткие выдержки. Проект теперь
также умеет индексировать исходные RTF напрямую; нормализованные TXT оставлены как
воспроизводимый корпус, на имена которого ссылается `npa_testset.jsonl`.

## Состав тестов

Вопросы проверяют определения, перечни, сроки, числовые ограничения, условия,
исключения, требования и объединение нескольких фактов. Каждая строка JSONL
содержит:

- `question` — вопрос пользователю;
- `reference` — эталонный ответ для judge-модели;
- `answer_terms` — точные контрольные фрагменты ожидаемого ответа;
- `context_terms` — фрагменты, которые должны быть найдены retriever-ом;
- `expected_sources` — ожидаемый TXT-файл;
- `legal_basis` и `category` — поясняющая разметка; загрузчик проекта их
  безопасно игнорирует.

Контрольные термины и ожидаемые источники автоматически сверены с полными
извлеченными текстами. Все 60 вопросов уникальны и имеют эталонный ответ.

## Индексация

Команда `--recreate` удаляет текущую коллекцию Qdrant и создает ее заново.

Локальный запуск:

```powershell
rag index data\npa_text --recreate
```

Запуск через Docker Compose:

```bash
docker compose up -d qdrant vllm
docker compose run --rm app index /data/npa_text --recreate
```

## Быстрая проверка

Сначала удобно выполнить только пять вопросов:

```bash
docker compose run --rm app evaluate \
  /evaluation/npa_testset.jsonl \
  --limit 5 \
  --output /evaluation/npa-results-smoke.json
```

Семантическая оценка RAGAS той же моделью vLLM:

```bash
docker compose run --rm app evaluate-ragas \
  /evaluation/npa_testset.jsonl \
  --limit 5 \
  --skip-answer-relevancy \
  --threshold 0.7 \
  --output /evaluation/npa-ragas-smoke.json
```

После smoke-теста уберите `--limit 5` для полного прогона. Чтобы измерять
диагностическую Answer Relevancy, уберите `--skip-answer-relevancy`.

## Сравнение четырех конфигураций

```bash
docker compose run --rm app benchmark \
  /evaluation/npa_testset.jsonl \
  --threshold 0.7 \
  --skip-answer-relevancy \
  --output-dir /evaluation/npa-benchmark
```

Оценка отражает содержание переданных версий документов. Для юридически
значимых выводов нужно отдельно проверять актуальную редакцию НПА.
