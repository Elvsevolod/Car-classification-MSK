# Краткая архитектура Vehicle ReID

## Назначение

Сервис по изображению автомобиля и BBox строит 512-D embedding, ищет похожие объекты в статичной gallery и возвращает Top-N либо отказ. Конкурсный batch-пайплайн создаёт `submission.csv`, `embeddings.npy` и `candidates.csv`.

## Границы модулей

| Слой | Файлы | Ответственность |
|---|---|---|
| HTTP/API | `backend/app.py` | Валидация upload и BBox, JSON-контракт, изображения, OpenAPI и offline Swagger. |
| ML и поиск | `backend/core.py`, `backend/rerank.py` | EXIF/RGB/crop/preprocess, ONNX OSNet, L2-normalization, confidence и k-reciprocal reranking. |
| Сборка зависимостей | `backend/bootstrap.py` | Создаёт PostgreSQL-репозиторий из окружения; доменный поиск не зависит от переменных окружения. |
| Данные | `backend/postgres_gallery_repository.py`, `alembic/` | PostgreSQL + pgvector, fingerprint gallery, exact cosine search и сохранение embedding. |
| Batch и оценка | `backend/infer.py`, `backend/evaluate.py`, `evaluate.py` | Автономный inference без БД/train, общий экспорт/валидатор и отдельно локальная размеченная оценка. |
| Калибровка | `backend/calibration.py`, `models/calibration.json` | Bundled порог и локальный отчёт с проверкой совместимости модели, preprocessing и reranking. |
| Интерфейс | `web-ui/src/App.tsx` | Белое минималистичное рабочее место на React/shadcn/ui, BBox на canvas и вызовы только к `/api/*`. |
| Поставка | `Dockerfile`, `docker-compose.yml`, `docker/entrypoint.sh` | Сборка frontend, Python runtime, PostgreSQL, миграции и запуск одной командой. |

## Поток одного запроса

```text
Изображение + BBox
  → FastAPI проверяет формат, размер и границы
  → Encoder: crop → preprocessing → OSNet → L2 embedding
  → PostgreSQL/pgvector: полный exact cosine-рейтинг статичной gallery
  → reranking определяет порядок; max raw cosine определяет confidence/refusal
  → JSON Top-N или отказ
```

## Важные инварианты

- Каждый query обрабатывается независимо; query expansion не используется.
- Gallery статична для одного запуска; query не добавляются в неё.
- Отказ применим только в `candidates`-режиме и основан на пороге raw cosine.
- PostgreSQL — постоянное runtime-хранилище gallery в web-демо; автономный batch использует память процесса, SQLite сохранён только для parity-тестов.
- API и batch читают один bundled manifest калибровки. Runtime-volume не является источником порога; несовместимая калибровка вызывает явную ошибку.
- В runtime нет скачивания моделей или пакетов. Офлайн запуск использует заранее загруженные образы и `--pull never`.

## Docker-режимы

- Demo: `docker compose up` — `dataset-init` сначала переносит внешний dataset во внутренний volume, затем запускаются FastAPI + React + PostgreSQL + локальный Swagger. Это снимает зависимость runtime от прав исходной папки dataset.
- Mandatory batch-export: `docker compose --profile inference run --rm --no-deps --pull never inference` — `backend.infer` читает `./dataset:/data:ro`, пишет в `./artifacts:/out`, не запускает PostgreSQL, миграции, `dataset-init` или калибровку.
- Полный offline demo после `docker load`: `docker compose up -d --no-build --pull never`.

## Frontend и тесты

Production Docker всегда собирает `web-ui/` и отдаёт `frontend/dist`. `frontend/index.html` и `frontend/app.js` остаются только документированным fallback для source-level Python-тестов. Docker test target также собирает React bundle, а `web-ui/e2e/search.smoke.spec.ts` проверяет реальный путь браузера: успешный Top-N и отказ на мобильной ширине 390 px.

Интерфейс различает ranking без порога, принятых кандидатов и отказ; список похожих объектов сам по себе не означает подтверждённую идентичность. Качество модели и особенности evaluator описаны отдельно в [MODEL_REPORT.md](MODEL_REPORT.md).
