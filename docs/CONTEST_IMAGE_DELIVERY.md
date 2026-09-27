# Передача Docker-образа для конкурсной проверки

## Цель

Передать жюри воспроизводимый Linux-образ приложения. Обязательный автономный запуск формирует `submission.csv`, `embeddings.npy` и `candidates.csv` из смонтированных `images/`, `test_query.csv` и `test_gallery.csv` без доступа в интернет, PostgreSQL или `train.csv`. Образ PostgreSQL + pgvector дополнительно нужен для необязательного web-демо.

Исходный код, Dockerfile, docker-compose.yml, README и файл зависимостей остаются обязательной частью репозитория. Docker-архив — дополнительный способ исключить скачивание образов и Python-пакетов на закрытом стенде.

## Важное ограничение платформы

Локальная разработка ведётся на macOS ARM64. Стенд организаторов — Linux x86_64. Поэтому передавать локальный тег `vehicle-reid:local` нельзя: нужен отдельный образ `linux/amd64`.

Текущая поставка использует CPU ONNX Runtime. Не заявлять GPU-ускорение, пока не будет создан и проверен отдельный GPU-образ на Linux с NVIDIA runtime.

## Подготовка образа

Выполняется на машине с доступом к Docker registry. Для итоговой передачи предпочтительна нативная Linux x86_64 машина; buildx с эмуляцией допустим только после полного теста.

```bash
docker buildx build --platform linux/amd64 \
  --tag vehicle-reid:contest-amd64 \
  --load .

docker image inspect vehicle-reid:contest-amd64 \
  --format '{{.Os}}/{{.Architecture}}'
docker pull --platform linux/amd64 pgvector/pgvector:0.8.6-pg16-bookworm
docker image inspect pgvector/pgvector:0.8.6-pg16-bookworm \
  --format '{{.Os}}/{{.Architecture}}'
```

Ожидаемый результат обеих проверок архитектуры: `linux/amd64`.

## Проверка inference до передачи

Подготовить рядом с репозиторием `dataset/`, затем запустить конкурсный pipeline:

```bash
docker tag vehicle-reid:contest-amd64 vehicle-reid:local
docker compose --profile inference run --rm --no-deps --pull never inference
```

Batch запускает `python -m backend.infer --dataset /data --output /out`, читает исходный датасет read-only и пишет напрямую в `./artifacts`. Он не запускает `dataset-init`, миграции или БД и не требует дополнительной копии датасета. Порог и отчёт калибровки входят в образ (`models/calibration.json`); пересчёта порога на тестовых данных нет.

Убедиться, что в `./artifacts/` появились:

- `submission.csv`;
- `embeddings.npy`;
- `candidates.csv`.

Проверить форматы после экспорта:

```bash
docker compose --profile inference run --rm --no-deps --pull never \
  -e DATASET_DIR=/data --entrypoint python inference \
  -m backend.evaluate --validate-only --output /out
```

Внешние загрузки при runtime не требуются. Для batch достаточно заранее загруженного образа приложения; для web-демо дополнительно нужен образ PostgreSQL. Проверка `--pull never` исключает pull, но полную автономность нужно отдельно подтвердить запуском без доступа в сеть на целевом стенде.

## Упаковка и передача

```bash
docker save --output vehicle-reid-contest-amd64.tar \
  vehicle-reid:contest-amd64 \
  pgvector/pgvector:0.8.6-pg16-bookworm
shasum -a 256 vehicle-reid-contest-amd64.tar > vehicle-reid-contest-amd64.tar.sha256
```

Передать архив и checksum через GitHub Release, облачное хранилище или носитель. Не коммитить `.tar` в Git-репозиторий.

## Действия жюри

```bash
docker load --input vehicle-reid-contest-amd64.tar
docker tag vehicle-reid:contest-amd64 vehicle-reid:local
docker compose --profile inference run --rm --no-deps --pull never inference
```

Для полного demo-сервиса дополнительно нужен локально доступный образ `pgvector/pgvector:0.8.6-pg16-bookworm`; обязательный inference от него не зависит. Архив выше включает оба образа для удобства запуска обоих сценариев.

## Финальный чек-лист

- [ ] Git-репозиторий содержит исходный код, Dockerfile, docker-compose.yml, README и документацию.
- [ ] `vehicle-reid:contest-amd64` имеет архитектуру `linux/amd64`.
- [ ] Проверен one-command inference в `./artifacts/`.
- [ ] Валидатор проходит через Compose после автономного inference без `train.csv`, БД и внешних загрузок.
- [ ] Архив образа и SHA-256 переданы отдельно от Git.
- [ ] В README указаны внешние модели, библиотеки и версии.
