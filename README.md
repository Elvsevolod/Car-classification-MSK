# Кейс от ASU_TEAM — Vehicle ReID

Сервис ищет похожие автомобили: получает изображение и ограничивающую рамку автомобиля (BBox), строит embedding, ищет кандидатов в статичной gallery и возвращает Top-N либо отказ.
В поставку входят FastAPI API, React-интерфейс, PostgreSQL 16 + pgvector, офлайн Swagger UI и конкурсный batch-экспорт.

## Что нужно заранее

- Docker Desktop (macOS/Windows) или Docker Engine + Docker Compose plugin (Linux);
- датасет организаторов в папке `dataset/` рядом с `docker-compose.yml`;
- свободное место Docker: на первом запуске требуется дополнительно около 7 ГБ для внутренней копии датасета.

Локальные Python, `venv`, Node.js и npm для обычного запуска **не нужны**.

## Структура датасета

Папка `dataset/` не хранится в Git. Перед запуском она должна содержать:

```text
dataset/
├── images/
├── train.csv
├── test_gallery.csv
└── test_query.csv
```

Проверка структуры:

```bash
ls dataset/images dataset/train.csv dataset/test_gallery.csv dataset/test_query.csv
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

При первом запуске сервис `dataset-init` автоматически копирует dataset во внутренний Docker volume. Поэтому права доступа исходной папки, включая `700` на Linux, не мешают основному контейнеру. Затем запускаются PostgreSQL, миграции и заполнение gallery из 750 объектов. На повторных запусках copy и вычисление gallery переиспользуются, если исходные данные и модель не изменились.

Запуск в фоне:

```bash
docker compose up -d --build
docker compose ps
```

Остановка без удаления данных:

```bash
docker compose down
```

Полный сброс локальных данных (удаляет PostgreSQL, внутреннюю копию dataset и runtime-artifacts):

```bash
docker compose down -v
```

## Как пользоваться интерфейсом

1. Выберите официальный query или загрузите JPEG/PNG.
2. Для официального query BBox подставляется из CSV. Для своего изображения нарисуйте рамку на canvas или заполните `x`, `y`, `w`, `h` вручную.
3. Нажмите «Запустить поиск».
4. В режиме «Ранжирование» отображается Top-N похожих объектов gallery.
5. В режиме «Кандидаты с отказом» задайте порог cosine: если максимум ниже порога, API вернёт отказ и пустой список кандидатов.
6. При необходимости скачайте JSON ответа.

`confidence` — максимальный raw cosine similarity, а не вероятность. Reranking влияет на порядок кандидатов, но не на решение об отказе.

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

После первого запуска Compose создайте обязательные файлы одной командой:

```bash
docker compose --profile inference run --rm --pull never inference
```

В папке `artifacts/` появятся:

```text
artifacts/
├── submission.csv
├── embeddings.npy
└── candidates.csv
```

Проверка их структуры без повторного вычисления embeddings:

```bash
docker compose run --rm --no-deps --pull never --entrypoint python inference -m backend.evaluate --validate-only
```

Ожидаемые размеры для текущего датасета: 1110 query, 750 gallery и `embeddings.npy` формы `(1860, 512)` типа `float32`. Отказ отражается отсутствием строк соответствующего query в `candidates.csv`.

## Офлайн запуск на стенде

Во время работы сервис не скачивает модели или Python-пакеты. Однако `docker compose up --build` может скачивать базовые образы и зависимости при **сборке**. Для стенда без сети нужно заранее собрать Linux `amd64` образы, сохранить их через `docker save`, доставить вместе с исходниками и выполнить `docker load`.

После загрузки образов запуск выглядит так:

```bash
docker compose up -d --no-build --pull never
docker compose --profile inference run --rm --pull never inference
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

Они проверяют два сценария через реальный API: Top-N по официальному query и отказ в режиме candidates на ширине 390 px.

## Архитектура

```text
Изображение + BBox
  → FastAPI: проверка формата и границ
  → OSNet: crop, preprocessing, 512-D L2 embedding
  → PostgreSQL + pgvector: exact cosine search по статичной gallery
  → k-reciprocal reranking: порядок Top-N
  → max raw cosine: confidence и решение об отказе
  → React UI / JSON API / конкурсные CSV и NPY
```

- `backend/` — FastAPI, обработка изображений, поиск, экспорт и PostgreSQL-репозиторий;
- `web-ui/` — React/TypeScript/Tailwind интерфейс;
- `frontend/vendor/swagger-ui/` — локальные Swagger assets без CDN;
- `alembic/` — миграции PostgreSQL + pgvector;
- `tests/` — unit, API и integration-тесты;
- `docs/` — архитектура, runbook, тестовый отчёт и аудит требований.

Подробнее: [архитектура](docs/ARCHITECTURE.md), [runbook](docs/RUNBOOK.md), [итоги тестирования](docs/TEST_SUMMARY.md), [аудит конкурса](docs/CONTEST_COMPLIANCE_AUDIT.md).

## Статус и ограничения

- Текущая поставка — проверенный **CPU-MVP**: FastAPI, React, PostgreSQL + pgvector, Docker Compose, офлайн Swagger, batch-export и тесты.
- PostgreSQL — единственное runtime-хранилище gallery; `embeddings.npy` — сдаваемый артефакт, не замена базы.
- Каждый query обрабатывается независимо; query expansion, OCR, номерные знаки и детектор не используются.
- Модель и локальные метрики описаны в [docs/TEST_SUMMARY.md](docs/TEST_SUMMARY.md); это не результат скрытого теста организаторов.
- CUDA/GPU profile и benchmark на RTX A5000, а также презентация, ещё не подготовлены.
