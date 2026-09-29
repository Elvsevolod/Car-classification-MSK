# Offline-поставка для NVIDIA / Linux amd64

Актуальная модель — MVP_fusion_v25. Конкурсный образ — `vehicle-reid:cuda12.2`.
Сборка с сетью разрешена, inference выполняется без сети. Предварительно проверьте
GPU по [README](../README.md). Проверка на RTX 4060 не является замером на A5000.

## Подготовка на машине с сетью

Из корня main (Bash/WSL2):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build inference
docker image inspect vehicle-reid:cuda12.2 --format '{{.Os}}/{{.Architecture}}'
docker save -o vehicle-reid-cuda12.2.tar vehicle-reid:cuda12.2
sha256sum vehicle-reid-cuda12.2.tar > vehicle-reid-cuda12.2.tar.sha256
```

Архитектура должна быть `linux/amd64`. Перенесите TAR, SHA256, исходники main
(включая оба Compose-файла), README и фиксированный обучающий код из
[исследовательской ветки](BRANCH_LAYOUT.md). Dataset передаётся отдельно.
Архив образа не коммитится в Git. Запишите `git rev-parse HEAD` и ID образа.
Используйте заново собранный образ с NVRTC 12.2.140 из актуального GPU lock,
а не старый TAR: обновление исходников само по себе содержимое образа не меняет.

## На offline-машине

Должны быть заранее установлены Docker, NVIDIA driver и работающий доступ
контейнеров к GPU (WSL2/Docker Desktop либо Linux/NVIDIA Container Toolkit).

```bash
sha256sum -c vehicle-reid-cuda12.2.tar.sha256
docker load -i vehicle-reid-cuda12.2.tar
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

Для необязательного offline web-демо дополнительно сохраните/загрузите точный
образ PostgreSQL из `docker-compose.yml` и убедитесь, что его digest разрешается
локально. После загрузки всех образов:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --no-build --pull never
```

UI находится на порту 8017 по умолчанию. Для основного трёхфайлового inference
образ PostgreSQL и его volume не нужны. Исторический инструмент
`tools.build_offline_release` собирает CPU-поставку; для NVIDIA используйте
команды этого документа.
