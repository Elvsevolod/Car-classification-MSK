# Инструкция по запуску

## Что нужно подготовить

- Docker Desktop или Docker Engine с Docker Compose v2;
- папка `dataset/` в корне репозитория;
- внутри `dataset/`: `images/`, `train.csv`, `test_query.csv`, `test_gallery.csv`.

Датасет и результаты `artifacts/` игнорируются Git. При первой сборке Docker может скачать базовые образы и зависимости. Для запуска на закрытом стенде заранее загрузите образы по [инструкции передачи](CONTEST_IMAGE_DELIVERY.md).

## Запуск веб-приложения

Из корня репозитория:

```bash
docker compose up --build
```

Compose сначала запускает `dataset-init`: он читает `./dataset` и один раз копирует его в именованный Docker volume. Поэтому сервис работает и при правах `700` на исходной папке в Linux; первый запуск требует ещё около 7 ГБ Docker-диска и занимает больше времени. Затем Compose запускает PostgreSQL 16 с pgvector, ждёт healthcheck БД, применяет Alembic-миграции и запускает FastAPI. Откройте http://127.0.0.1:8000.

### Закрытый стенд без интернета

После `docker load` заранее подготовленных образов используйте:

```bash
docker compose up -d --no-build --pull never
```

Команда запрещает сборку и скачивание образов.

Проверка готовности:

```bash
curl http://127.0.0.1:8000/api/health
```

В ответе должны быть `"status":"ready"`, `"gallery_storage":"PostgresGalleryRepository"` и `"gallery_size":750` для выданного датасета.

Остановка приложения без удаления данных:

```bash
docker compose down
```

## Создание файлов для сдачи

Команда использует тот же PostgreSQL + pgvector runtime, но не запускает веб-интерфейс:

```bash
docker compose --profile inference run --rm --pull never inference
```

Файлы появятся в локальной папке `artifacts/`:

- `submission.csv`;
- `embeddings.npy`;
- `candidates.csv`.

Проверка уже созданных файлов:

```bash
docker compose run --rm --no-deps --pull never --entrypoint python inference -m backend.evaluate --validate-only
```

## Тестирование

```bash
docker compose -p vehicle-reid-tests -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
docker compose -p vehicle-reid-tests -f docker-compose.test.yml down -v
```

Вторая команда удаляет только временные ресурсы тестового проекта `vehicle-reid-tests`.
