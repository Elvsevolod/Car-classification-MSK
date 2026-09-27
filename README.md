# Кейс от ASU_TEAM — Vehicle ReID

Сервис ищет похожие автомобили: получает изображение и ограничивающую рамку автомобиля (BBox), строит embedding, ищет кандидатов в статичной gallery и возвращает Top-N либо отказ.
В поставку входят FastAPI API, React-интерфейс, PostgreSQL 16 + pgvector, офлайн Swagger UI и конкурсный batch-экспорт.

Два независимых пути запуска: **автономный batch-export без БД и `train.csv`** и демонстрационное web-приложение с PostgreSQL. Активная модель и порог фиксированы; протокол, локальные метрики и ограничения — в [отчёте о модели](docs/MODEL_REPORT.md).

## Что нужно заранее

- Docker Desktop (macOS/Windows) или Docker Engine + Docker Compose plugin (Linux);
- датасет организаторов в папке `dataset/` рядом с `docker-compose.yml`;
- для web-демо — свободное место Docker под внутреннюю копию датасета (около 7 ГБ для текущего набора); автономный batch читает исходный каталог напрямую.

Локальные Python, `venv`, Node.js и npm для обычного запуска **не нужны**.

## Структура датасета

Папка `dataset/` не хранится в Git. Перед запуском она должна содержать:

```text
dataset/
├── images/
├── test_gallery.csv
└── test_query.csv
```

`train.csv` нужен только для отдельной локальной оценки/калибровки разработчиком. Для web-демо и batch-export он не требуется.

Автономный конкурсный inference требует минимум 10 объектов gallery, чтобы каждый query получил полный Top-10.

Проверка структуры:

```bash
ls dataset/images dataset/test_gallery.csv dataset/test_query.csv
```

## Запуск демо одной командой

Из корня репозитория:

```bash
docker compose up --build
```

После строки `Application startup complete` откройте:

- приложение: http://127.0.0.1:8000;
- Swagger API: http://127.0.0.1:8000/docs;
- healthcheck: http://127.0.0.1:8000/api/health.

При первом запуске сервис `dataset-init` автоматически копирует dataset во внутренний Docker volume. Поэтому права доступа исходной папки, включая `700` на Linux, не мешают основному контейнеру. Затем запускаются PostgreSQL, миграции и заполнение gallery (750 объектов в текущем наборе). Кэш датасета проверяется по содержимому CSV и изображений, а не только размерам файлов; галерея дополнительно проверяет модель и preprocessing. Замена изображения тем же числом байт также обновляет кэш.

Порог и локальный отчёт входят в образ как `models/calibration.json`: предварительный export и заполненный artifacts-volume не нужны. При несовместимом manifest запуск завершается явной ошибкой; порог не выдумывается и не перекалибровывается на тестовых данных.

Запуск в фоне:

```bash
docker compose up -d --build
docker compose ps
```

Остановка контейнеров без удаления данных:

```bash
docker compose stop
```

Полный сброс локальных данных (удаляет PostgreSQL, внутреннюю копию dataset и runtime-artifacts):

```bash
docker compose down -v
```

## Как пользоваться интерфейсом

Интерфейс — белое минималистичное рабочее место на shadcn/ui: изображение и BBox слева, результаты справа; на узком экране блоки идут последовательно. Числовые координаты, Top-N и ручной порог находятся в дополнительных настройках.

1. Выберите официальный query или загрузите JPEG/PNG.
2. Для официального query BBox подставляется из CSV. Для своего изображения нарисуйте рамку на canvas или заполните `x`, `y`, `w`, `h` вручную.
3. Нажмите «Найти автомобиль».
4. В режиме «Ранжирование» отображается Top-N похожих объектов gallery.
5. В режиме «С порогом» по умолчанию используется сохранённый порог `0.5948754549026489`. При необходимости его можно переопределить вручную: если максимум cosine ниже порога, API вернёт отказ и пустой список кандидатов.
6. При необходимости скачайте JSON ответа.

`confidence` — максимальный raw cosine similarity, а не вероятность. Reranking влияет на порядок кандидатов, но не на решение об отказе. Режим «Ранжирование» не подтверждает совпадение: он возвращает Top-N без проверки порога. Даже принятые кандидаты требуют визуальной проверки.

## API

Swagger находится по адресу `/docs`, OpenAPI JSON — `/openapi.json`.

| Метод | Адрес | Назначение |
|---|---|---|
| `GET` | `/api/health` | Готовность сервиса, модель, порог, размер gallery |
| `GET` | `/api/queries` | Официальные query и их BBox |
| `POST` | `/api/search` | Загруженное изображение + BBox → поиск |
| `POST` | `/api/search/query` | Поиск по `query_id` из `test_query.csv` |
| `POST` | `/api/embedding` | Получить 512-D L2-нормированный embedding |
| `GET` | `/api/images/...` | Исходный кадр или crop query/gallery |
| `GET` | `/api/metrics` | Зафиксированные локальные метрики модели |

Пример проверки API:

```bash
curl http://127.0.0.1:8000/api/health
curl 'http://127.0.0.1:8000/api/queries?limit=1'
```

`POST /api/search` принимает `multipart/form-data` с полями `image`, `x`, `y`, `w`, `h`, а также необязательными `top_k`, `mode` (`ranking`/`candidates`) и `threshold`. Изображения ограничены JPEG/PNG, 15 МиБ и 25 мегапикселями. Некорректный BBox не исправляется молча — API возвращает ошибку 4xx.

## Экспорт файлов сдачи

Сначала соберите образ (нужна сеть для отсутствующих образов/зависимостей):

```bash
docker compose build inference
```

Затем создайте обязательные файлы одной командой. Web, PostgreSQL, миграции, `dataset-init` и `train.csv` не нужны:

```bash
docker compose --profile inference run --rm --no-deps --pull never inference
```

Сервис запускает `python -m backend.infer --dataset /data --output /out`, читает `./dataset` через read-only mount и пишет в `./artifacts`. Gallery хранится в памяти процесса; encoder, ranking, отказ и экспортный формат общие с основным приложением. Порог берётся из bundled manifest, калибровка не запускается.

В папке `artifacts/` появятся:

```text
artifacts/
├── submission.csv
├── embeddings.npy
├── candidates.csv
└── export_manifest.json
```

Три конкурсных файла автоматически проверяются после экспорта. `export_manifest.json` — дополнительный служебный отчёт с порядком ID, хэшами и настройками. Повторный запуск заменяет файлы в выходной папке; важные исторические результаты следует хранить отдельно.

Проверка их структуры без повторного вычисления embeddings:

```bash
docker compose --profile inference run --rm --no-deps --pull never \
  -e DATASET_DIR=/data --entrypoint python inference \
  -m backend.evaluate --validate-only --output /out
```

Ожидаемые размеры для текущего датасета: 1110 query, 750 gallery и `embeddings.npy` формы `(1860, 512)` типа `float32`: сначала query, затем gallery в порядке CSV. `submission.csv` не имеет заголовка и содержит query ID плюс 10 gallery ID. Отказ отражается отсутствием строк соответствующего query в `candidates.csv`; принятый query содержит одного верхнего reranked-кандидата с `confidence=(max_raw_cosine+1)/2`, не вероятностью. Файл Top-10 при отказе не сокращается.

При установленных Python-зависимостях тот же путь доступен без Docker:

```bash
python -m backend.infer --dataset ./dataset --output ./artifacts
```

`python -m backend.evaluate` остаётся инструментом разработки для локального размеченного `train.csv`, а не обязательным шагом перед экспортом. Подробности воспроизведения — в [MODEL_REPORT.md](docs/MODEL_REPORT.md).

## Офлайн запуск на стенде

Во время работы сервис не скачивает модели или Python-пакеты. Однако `docker compose up --build` может скачивать базовые образы и зависимости при **сборке**. Для стенда без сети нужно заранее собрать Linux `amd64` образы, сохранить их через `docker save`, доставить вместе с исходниками и выполнить `docker load`.

После загрузки образов запуск выглядит так:

```bash
docker compose --profile inference run --rm --no-deps --pull never inference
# Необязательное web-демо, дополнительно требует образ PostgreSQL:
docker compose up -d --no-build --pull never
```

Полная пошаговая процедура, контрольные SHA-256 и перечень образов — в [docs/CONTEST_IMAGE_DELIVERY.md](docs/CONTEST_IMAGE_DELIVERY.md). Перед передачей жюри этот путь нужно обязательно прогнать на чистом Linux `amd64` стенде.

## Тесты: оставлять ли их в проекте?

**Да, тесты нужно оставить в репозитории.** Они не являются частью долгоживущего production-контейнера и не запускаются при `docker compose up`. Отдельный target Dockerfile создаёт временную БД и проверяет BBox, API, PostgreSQL + pgvector, экспортные форматы и отказ. Это доказательство воспроизводимости для команды и жюри.

Полный прогон:

```bash
docker compose -p vehicle-reid-tests -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
docker compose -p vehicle-reid-tests -f docker-compose.test.yml down -v
```

Вторая команда удаляет только временные ресурсы с префиксом `vehicle-reid-tests`.

Browser smoke-тесты React-интерфейса запускаются только для разработки и требуют Node.js:

```bash
cd web-ui
npm ci
npm run test:e2e
```

Перед ними запустите приложение; адрес по умолчанию — `http://127.0.0.1:8000`, другой задаётся через `E2E_BASE_URL`. Для первого запуска тестов также нужен установленный Chromium Playwright (`npx playwright install chromium`).

Browser-тесты проверяют интерфейс и его контракт с API. Это инженерные проверки; результаты качества модели приведены отдельно в [MODEL_REPORT.md](docs/MODEL_REPORT.md).

## Архитектура

```text
Изображение + BBox
  → FastAPI: проверка формата и границ
  → OSNet: crop, preprocessing, 512-D L2 embedding
  → статичная gallery: PostgreSQL + pgvector в web / память процесса в batch
  → k-reciprocal reranking: порядок Top-N
  → max raw cosine: confidence и решение об отказе
  → React UI / JSON API / конкурсные CSV и NPY
```

- `backend/` — FastAPI, обработка изображений, поиск, экспорт и PostgreSQL-репозиторий;
- `web-ui/` — белый минималистичный React/TypeScript/Tailwind интерфейс с shadcn/ui;
- `frontend/vendor/swagger-ui/` — локальные Swagger assets без CDN;
- `alembic/` — миграции PostgreSQL + pgvector;
- `tests/` — unit, API и integration-тесты;
- `docs/` — архитектура, runbook, тестовый отчёт и аудит требований.

Подробнее: [архитектура](docs/ARCHITECTURE.md), [runbook](docs/RUNBOOK.md), [итоги тестирования](docs/TEST_SUMMARY.md), [аудит конкурса](docs/CONTEST_COMPLIANCE_AUDIT.md).

## Статус и ограничения

- Текущая поставка — **CPU-MVP**: FastAPI, React, PostgreSQL + pgvector для web, Docker Compose, офлайн Swagger и автономный batch-export. Наличие этих компонентов не означает завершённую приёмку на конкурсном стенде.
- PostgreSQL — постоянное runtime-хранилище gallery для web; batch хранит её только в памяти процесса. `embeddings.npy` — сдаваемый артефакт.
- Каждый query обрабатывается независимо; query expansion, OCR, номерные знаки и детектор не используются.
- Модель, калибровка, метрики и ограничения описаны в [docs/MODEL_REPORT.md](docs/MODEL_REPORT.md). Validation mAP@10 — 81,47%, candidate F1 — 72,86%, TNR — 79,03%; checkpoint выбирался по этой validation, поэтому это не независимый финальный тест и не результат организаторов.
- CUDA/GPU profile и benchmark на RTX A5000, а также презентация, ещё не подготовлены.
# Изолированная интеграция frozen R1

После подтверждения v25 активен **MVP_fusion_v25**: ranking по смеси MVP/R1 50/50,
кандидат/отказ full-train R1 неизменны. Откат — MVP_dual_role_v24; MVP_legacy также сохранён.
Доказательство переноса и команды запуска: [V25_PROMOTION.md](docs/V25_PROMOTION.md).
Переключённое демо на localhost:8000 и точные команды отката описаны в
[docs/V24_PROMOTION.md](docs/V24_PROMOTION.md). Руководство по профилям, notebook Run All,
offline-поставке и незакрытым проверкам: [docs/RELEASE_INTEGRATION.md](docs/RELEASE_INTEGRATION.md).
Сервис этой копии по умолчанию использует порт 8017 и собственный Docker image.
Ниже сохранена документация исходного приложения; при различиях runtime
актуален контракт из RELEASE_INTEGRATION.md.
