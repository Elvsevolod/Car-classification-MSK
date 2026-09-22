# Передача Docker-образа для конкурсной проверки

## Цель

Передать жюри воспроизводимые Linux-образы приложения и PostgreSQL + pgvector. Обязательный запуск формирует `submission.csv`, `embeddings.npy` и `candidates.csv` из смонтированных `images/` и CSV без доступа в интернет.

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
docker compose --profile inference run --rm --pull never inference
```

Перед export Compose автоматически выполнит `dataset-init`: сервис копирует внешний датасет во внутренний Docker volume, поэтому права исходной папки на Linux не влияют на non-root runtime. На первом запуске требуется дополнительное место, примерно равное размеру датасета.

Убедиться, что в `./artifacts/` появились:

- `submission.csv`;
- `embeddings.npy`;
- `candidates.csv`.

Проверить форматы после экспорта:

```bash
docker compose --profile inference run --rm --pull never \
  --entrypoint python inference -m backend.evaluate --validate-only
```

Контейнеры общаются только по внутренней Docker-сети; внешние загрузки при runtime не требуются. Поэтому перед передачей оба образа должны быть уже загружены локально.

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
docker compose --profile inference run --rm --pull never inference
```

Для inference и полного demo-сервиса требуется также локально доступный образ `pgvector/pgvector:0.8.6-pg16-bookworm`; он является частью поставки PostgreSQL-only.

## Финальный чек-лист

- [ ] Git-репозиторий содержит исходный код, Dockerfile, docker-compose.yml, README и документацию.
- [ ] `vehicle-reid:contest-amd64` имеет архитектуру `linux/amd64`.
- [ ] Проверен one-command inference в `./artifacts/`.
- [ ] Валидатор проходит через Compose после PostgreSQL-backed inference без внешних загрузок.
- [ ] Архив образа и SHA-256 переданы отдельно от Git.
- [ ] В README указаны внешние модели, библиотеки и версии.
