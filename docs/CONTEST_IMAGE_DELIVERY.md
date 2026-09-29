# Offline-поставка: приложение и PostgreSQL

Актуальная модель — MVP_fusion_v25. Конкурсный образ — `vehicle-reid:cuda12.2`.
Сборка с сетью разрешена, inference выполняется без сети. Предварительно проверьте
GPU по [README](../README.md). Проверка на RTX 4060 не является замером на A5000.

## 1. Подготовка на машине с сетью

PostgreSQL работает отдельным контейнером. Его образ собирается целью `postgres`
в Dockerfile из pgvector/PostgreSQL, закреплённого SHA256, и получает локальный тег
`vehicle-reid-postgres:pg16-pgvector0.8.6`. **Полная `compose build` готовит оба образа**;
`docker build .` и `compose build inference` готовят только приложение.

Для NVIDIA/Linux amd64, из корня main (Bash/WSL2):

```bash
dcgpu() { docker compose -f docker-compose.yml -f docker-compose.gpu.yml "$@"; }
dcgpu build
docker image inspect vehicle-reid:cuda12.2 vehicle-reid-postgres:pg16-pgvector0.8.6 \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'
docker save -o vehicle-reid-gpu-stack.tar \
  vehicle-reid:cuda12.2 vehicle-reid-postgres:pg16-pgvector0.8.6
shasum -a 256 vehicle-reid-gpu-stack.tar > vehicle-reid-gpu-stack.tar.sha256
```

Оба образа должны быть `linux/amd64`. Перенесите TAR, SHA256, исходники main
(включая оба Compose-файла), README и фиксированный обучающий код из
[исследовательской ветки](BRANCH_LAYOUT.md). Dataset передаётся отдельно.
Архив образа не коммитится в Git. Запишите `git rev-parse HEAD` и ID образа.
Используйте заново собранный образ с NVRTC 12.2.140 из актуального GPU lock,
а не старый TAR: обновление исходников само по себе содержимое образа не меняет.

Для CPU (например, Mac → Mac той же архитектуры):

```bash
docker compose build
docker save -o vehicle-reid-cpu-stack.tar \
  vehicle-reid:release-integration vehicle-reid-postgres:pg16-pgvector0.8.6
shasum -a 256 vehicle-reid-cpu-stack.tar > vehicle-reid-cpu-stack.tar.sha256
```

Архитектура образов должна совпадать с целевой машиной. Если сборка и запуск
проходят на одном компьютере, `save/load` не нужны — после успешной сборки можно
отключить интернет и перейти к запуску. TAR создаётся вне Git и не содержит БД
с данными: она инициализируется при первом запуске, затем хранится в volume.

## 2. На offline-машине

Должны быть заранее установлены Docker, NVIDIA driver и работающий доступ
контейнеров к GPU (WSL2/Docker Desktop либо Linux/NVIDIA Container Toolkit).

```bash
shasum -a 256 -c vehicle-reid-gpu-stack.tar.sha256
docker load -i vehicle-reid-gpu-stack.tar
docker image inspect vehicle-reid:cuda12.2 vehicle-reid-postgres:pg16-pgvector0.8.6 \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

Для CPU загрузите аналогично `vehicle-reid-cpu-stack.tar` и используйте основной
Compose без `-f docker-compose.gpu.yml`. Docker load восстанавливает локальные
теги обоих образов; разрешать registry-digest через интернет при запуске не нужно.

## 3. Обязательный batch-инференс

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile inference \
  run --rm --no-deps --pull never inference --dataset /data --output /out/contest-run1 \
  --profile MVP_fusion_v25 --provider CUDAExecutionProvider
```

Это единственная команда inference после загрузки образа. Она не собирает образы,
не запускает PostgreSQL или web, читает `./dataset` read-only и пишет
`./artifacts/contest-run1`. Сеть процесса отключена (`network_mode: none`).
Каталог результата должен быть новым/пустым. Программа сама проверяет три файла
и сохраняет export manifest с хэшами. Порог берётся из frozen R1 bundle;
никакой калибровки или train.csv не требуется.

## 4. Полное offline web-демо

После полной сборки или загрузки **обоих** образов, одной командой:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --no-build --pull never
```

CPU:

```bash
docker compose up -d --no-build --pull never
```

UI находится на порту 8017 по умолчанию. `pull_policy: never` запрещает скачивание
сервисных образов; `--no-build` исключает случайную сборку. Если образ отсутствует,
запуск завершается ошибкой — вернитесь к подготовке или загрузите полный TAR.
Dataset-init имеет `network_mode: none`; web и БД используют локальную bridge-сеть
Compose. Ей не нужен интернет для связи приложения с PostgreSQL по имени `postgres`
и доступа пользователя к опубликованному HTTP-порту. Сама bridge-сеть не запрещает
внешний трафик: для web отключение интернета обеспечивает окружение стенда.
Не задавайте `network_mode: none` веб-сервису: он потеряет связь с БД.
Зависимости, веса, UI,
шрифты и Swagger уже находятся в образе. Не удаляйте volumes при обновлении.

Для основного трёхфайлового inference
образ PostgreSQL и его volume не нужны. Исторический инструмент
`tools.build_offline_release` собирает **только CPU batch-поставку**, не полный
web-стек. Для полного CPU/GPU-демо используйте команды этого документа.

Основание: [ответы организаторов, вопросы 11 и 44](../ORGANIZER_QA.md),
[build/image в Compose](https://docs.docker.com/reference/compose-file/build/).
