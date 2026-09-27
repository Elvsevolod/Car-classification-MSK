# Инструкция по запуску

## Что нужно подготовить

- Docker Desktop или Docker Engine с Docker Compose v2;
- папка `dataset/` в корне репозитория;
- внутри `dataset/`: `images/`, `test_query.csv`, `test_gallery.csv`; `train.csv` нужен только для локальной разработки/оценки модели.

Датасет и результаты `artifacts/` игнорируются Git. При первой сборке Docker может скачать базовые образы и зависимости. Для запуска на закрытом стенде заранее загрузите образы по [инструкции передачи](CONTEST_IMAGE_DELIVERY.md).

## Запуск веб-приложения

Из корня репозитория:

```bash
docker compose up --build
```

Compose сначала запускает `dataset-init`: он читает `./dataset` и один раз копирует его в именованный Docker volume. Поэтому сервис работает и при правах `700` на исходной папке в Linux; первый запуск требует ещё около 7 ГБ Docker-диска и занимает больше времени. Затем Compose запускает PostgreSQL 16 с pgvector, ждёт healthcheck БД, применяет Alembic-миграции и запускает FastAPI. Откройте http://127.0.0.1:8000.

При следующих запусках `dataset-init` сравнивает хэши содержимого CSV и изображений, включая изменения без изменения размера файла. Порог и отчёт загружаются из bundled `models/calibration.json`, а не из runtime-volume: отдельная калибровка перед поиском не нужна. Отсутствующий, повреждённый или несовместимый manifest останавливает запуск с ошибкой; локальный `artifacts/baseline_metrics.json` его не переопределяет.

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

В ответе должны быть `"status":"ready"`, `"gallery_storage":"PostgresGalleryRepository"`, `"gallery_size":750` для выданного датасета и `"default_threshold":0.5948754549026489` для текущей модели. `/api/metrics` возвращает сохранённый локальный отчёт, не измерения скрытого теста.

Остановка приложения без удаления данных:

```bash
docker compose stop
```

## Создание файлов для сдачи

Сначала соберите образ `docker compose build inference` либо загрузите готовый. Автономный batch использует общую модель/поиск, но gallery хранит в памяти процесса: PostgreSQL, web, миграции, `dataset-init` и `train.csv` не нужны.

```bash
docker compose --profile inference run --rm --no-deps --pull never inference
```

Файлы появятся в локальной папке `artifacts/`:

- `submission.csv`;
- `embeddings.npy`;
- `candidates.csv`.

Входной `./dataset` монтируется read-only в `/data`, выходной `./artifacts` — в `/out`. Порог берётся из `models/calibration.json`, не подбирается на входном наборе. Экспорт также сохраняет `export_manifest.json` и автоматически проверяет структуру трёх конкурсных файлов.

Проверка уже созданных файлов:

```bash
docker compose --profile inference run --rm --no-deps --pull never \
  -e DATASET_DIR=/data --entrypoint python inference \
  -m backend.evaluate --validate-only --output /out
```

Локальный эквивалент без Docker при установленных Python-зависимостях:

```bash
python -m backend.infer --dataset ./dataset --output ./artifacts
```

Повторный запуск заменяет экспортные файлы в выбранной папке. Исследовательская оценка `python -m backend.evaluate --output artifacts/local-evaluation` отделена от batch и требует размеченный `train.csv`; она не заменяет bundled калибровку автоматически. Протокол и ограничения — в [отчёте о модели](MODEL_REPORT.md).

## Тестирование

```bash
docker compose -p vehicle-reid-tests -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
docker compose -p vehicle-reid-tests -f docker-compose.test.yml down -v
```

Вторая команда удаляет только временные ресурсы тестового проекта `vehicle-reid-tests`.
