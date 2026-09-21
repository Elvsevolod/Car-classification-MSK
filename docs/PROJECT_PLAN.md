# План развития Vehicle ReID MVP

## Контекст

Проект решает задачу Vehicle Re-Identification: по изображению автомобиля и BBox получить embedding, найти похожие объекты в gallery, вернуть Top-10 и корректно отказаться при отсутствии уверенного совпадения.

Рабочая ветка: `devops`.

## Текущее состояние

- FastAPI backend, ONNX OSNet, crop по BBox, L2-normalization, cosine search и k-reciprocal reranking реализованы.
- Есть веб-интерфейс: загрузка JPEG/PNG, выбор BBox, Top-N, confidence/refusal и экспорт JSON.
- Реализованы экспорт и проверка `submission.csv`, `embeddings.npy`, `candidates.csv`.
- Docker CPU-MVP подготовлен: `Dockerfile`, `docker-compose.yml`, `.dockerignore`.
- Dataset находится в `dataset/` и исключён из Git.
- Единственное runtime-хранилище gallery: PostgreSQL + pgvector exact cosine search.

## Цель ближайшего этапа

Сделать воспроизводимый Docker-запуск сервиса и экспорта на реальном датасете через PostgreSQL + pgvector без изменения публичного API.

## Этап 1. Проверка Docker MVP

1. Запустить `docker compose up --build` с локальным датасетом.
2. Проверить `GET /api/health`, загрузку gallery и открытие интерфейса на `http://127.0.0.1:8000`.
3. Выполнить тестовый поиск через `POST /api/search` и сценарий отказа.
4. Выполнить `docker compose exec vehicle-reid python -m backend.evaluate --export`.
5. Проверить три артефакта и скопировать их из контейнера.

Критерий готовности: сервис запускается одной командой, не требует локального Python/venv, а экспорт создаёт валидные файлы сдачи.

## Этап 2. PostgreSQL + pgvector

### Инфраструктура

1. Добавить в `docker-compose.yml` сервис PostgreSQL на образе с pgvector.
2. Добавить named volume для данных PostgreSQL.
3. Передать backend настройки через `DATABASE_URL` и отдельные переменные пользователя, пароля и имени БД.
4. Добавить healthcheck PostgreSQL и ожидание его готовности перед инициализацией backend.

### Схема данных

Создать расширение `vector` и таблицу `gallery_items`:

```text
image_id            text primary key
x, y, w, h          integer not null
embedding           vector(512) not null
encoder_fingerprint text not null
image_sha256        text not null
created_at          timestamptz
```

### Backend

1. Добавить PostgreSQL driver и слой доступа к данным.
2. Заменить SQLite-класс `Gallery` на репозиторий, совместимый с текущим HTTP-контрактом.
3. При первом запуске вычислять embeddings gallery и сохранять их в PostgreSQL.
4. При повторном запуске сверять fingerprint модели, CSV и hash изображений; при изменении пересобирать gallery.
5. Выполнять cosine search в pgvector: получить Top-50, затем применять текущий k-reciprocal reranking и возвращать Top-10.
6. Сохранить raw cosine для confidence и существующую логику refusal.

### Индексы

- Для текущей gallery из 750 объектов использовать exact search как эталон воспроизводимости.
- Добавить HNSW-индекс `vector_cosine_ops` как опциональный режим для демонстрации масштабирования.
- Не использовать приближённый индекс как единственный источник результатов export, пока не подтверждена идентичность Top-10.

Критерий готовности: API, ranking, отказ и экспорт дают те же или эквивалентные результаты, а gallery и metadata сохраняются в PostgreSQL.

## Этап 2.5. Multi-stage Docker deployment и one-command inference (TASK 8, 11 — выполнено)

Источник задачи: CODEX_DOCKER_MULTISTAGE_DEPLOYMENT_PLAN.md. Выполняется после TASK 4–6: PostgreSQL gallery lifecycle и exact pgvector search.

1. Перевести Dockerfile на multi-stage build: builder собирает зафиксированные Python-зависимости, runtime содержит только необходимые runtime-библиотеки, backend, frontend, ONNX-веса и миграции.
2. Проверить dockerignore, non-root runtime, read-only dataset mount, отдельные volumes postgres_data и artifacts; секреты и env-файл не включать в image.
3. Добавить явный startup/entrypoint: проверка конфигурации → alembic upgrade head → exec FastAPI. PostgreSQL ожидается через Compose healthcheck, без sleep.
4. Добавить backend Docker healthcheck через API health, startup/shutdown-логи и проверить graceful shutdown.
5. Провести cold start с пустым postgres_data, warm restart с cache hit без повторного inference и offline runtime-проверку уже собранных образов.
6. Добавить profile `inference`: одна команда с PostgreSQL + pgvector, но без UI, генерирует submission.csv, embeddings.npy и candidates.csv из mounted dataset.
7. Зафиксировать размер образа до/после; CPU-сценарий оставить основным, GPU вынести в отдельный Compose profile.

Критерий готовности: docker compose up --build поднимает PostgreSQL и API одной командой; после сборки runtime не обращается к PyPI или источникам весов; cold/warm start и API health воспроизводимы.

## Этап 3. Тестирование (TASK 9 — выполнено)

1. Unit-тесты BBox, preprocessing, L2-normalization, стабильной сортировки и threshold/refusal.
2. Интеграционные тесты PostgreSQL: миграция, загрузка gallery, повторный старт без перерасчёта, invalidation кэша.
3. API-тесты: невалидные файлы, BBox вне границ, Top-K, ranking и candidates mode.
4. Тесты форматов `submission.csv`, `embeddings.npy`, `candidates.csv`.
5. Smoke-test Docker Compose на реальном датасете.
6. Замеры latency batch=1 и throughput; отдельно зафиксировать CPU/GPU окружение.

## Этап 4. Frontend и offline-документация

1. Доработать UX отображения результатов, ошибок и статуса загрузки gallery.
2. Показать явно: rank, raw cosine, rerank score, confidence и отказ.
3. Сделать Swagger UI offline либо оставить доступным только `/openapi.json` в offline-режиме.
4. Дополнить README: архитектура, запуск, конфигурация БД, схема данных, API, метрики, порог отказа и внешние лицензии.

## Передача образа жюри

Пошаговый план сборки и передачи prebuilt Linux x86_64 Docker-образа находится в [CONTEST_IMAGE_DELIVERY.md](CONTEST_IMAGE_DELIVERY.md). Он является дополнением к исходному коду, а не заменой репозитория.

## Этап 5. Подготовка сдачи и защиты

1. Проверить offline-запуск уже собранных образов.
2. Сформировать валидные артефакты сдачи.
3. Подготовить презентацию: проблема, архитектура, pipeline, метрики, ошибки, отказ, Docker и масштабирование pgvector/HNSW.
4. Подготовить демонстрацию: запуск → загрузка изображения/BBox → Top-10 → отказ → экспорт.

## Порядок работы

1. Проверить Docker с добавленным датасетом.
2. Реализовать PostgreSQL + pgvector и миграцию gallery.
3. Запустить unit- и PostgreSQL integration-тесты, включая экспорт и API.
4. Доработать frontend и offline Swagger.
5. Подготовить артефакты, README и презентацию.

## Ограничения

- Не коммитить `dataset/`, `artifacts/`, `.venv/` и локальные результаты.
- Не использовать номерные знаки, OCR, детектор или информацию других test-query.
- Query обрабатывать независимо; re-ranking разрешён только относительно статичной gallery.
- Вес всех моделей инференса должен оставаться менее 2 ГБ.
