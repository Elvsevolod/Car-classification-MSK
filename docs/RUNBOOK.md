# Запуск Vehicle ReID

Команды выполняются из корня репозитория в Bash, Zsh или терминале WSL. Профиль по умолчанию: `MVP_fusion_v25`.

## 1. Подготовить данные

Нужны Docker Engine или Docker Desktop с Compose v2. Сборка скачивает базовые образы и зависимости; модели уже находятся в репозитории.

```text
dataset/
  images/             # JPG/JPEG/PNG, имя файла соответствует image_id
  test_query.csv      # image_id,x,y,w,h
  test_gallery.csv    # image_id,x,y,w,h
```

Изображения предоставляет организатор. Для поиска `train.csv` не нужен. Датасет не публикуется в Git.

По умолчанию используются `./dataset`, `./artifacts` и порт `8017`. Другие пути и порт можно задать в `.env`:

```dotenv
HOST_DATASET_DIR=/absolute/path/to/dataset
OUTPUT_DIR=/absolute/path/to/results
PORT=8017
```

Для каждого полного экспорта выбирайте новую или пустую выходную папку. Существующие результаты не перезаписываются.

## 2. Получить конкурсные файлы на CPU

```bash
docker compose build inference
docker compose --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/run1
```

В `artifacts/run1/` появятся `submission.csv`, `candidates.csv`, `embeddings.npy`, манифест и замер полного экспорта. Batch работает без сети, PostgreSQL и веб-приложения. Входной каталог доступен только для чтения.

Проверка форматов готового экспорта:

```bash
docker compose --profile inference run --rm --no-deps --pull never \
  -e DATASET_DIR=/data --entrypoint python inference \
  -m backend.evaluate --validate-only --output /out/run1
```

Здесь `backend.evaluate` используется только как валидатор файлов. Его отдельный legacy-режим оценки не измеряет качество v25.

## 3. Запустить на NVIDIA

Нужен Linux x86_64 с доступной для Docker NVIDIA GPU либо Docker Desktop с WSL2 и GPU. Сначала проверьте доступность устройства в своей среде. GPU-конфигурация собирает образ `linux/amd64` с зависимостями из `requirements-gpu.txt`.

После изменения зависимостей 29 сентября образ нужно пересобрать: явно добавлена NVRTC. Это исправление окружения ещё не проверено повторным запуском на NVIDIA; локальные CPU-проверки не заменяют GPU-preflight.

```bash
dcgpu() { docker compose -f docker-compose.yml -f docker-compose.gpu.yml "$@"; }
dcgpu build inference
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.gpu --profile MVP_fusion_v25 --output /out/gpu/preflight.json
dcgpu --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/gpu/run1 \
  --profile MVP_fusion_v25 --provider CUDAExecutionProvider
```

Preflight должен подтвердить `preflight_passed: true`, профиль `MVP_fusion_v25`, размерность 2048 и работу вычислительных операторов всех четырёх моделей на CUDA. При отсутствии нужного provider программа сообщает ошибку. Наличие GPU-конфигурации само по себе не подтверждает скорость на конкурсном стенде.

Для повторного экспорта используйте `/out/gpu/run2`. Сравнение сохранённых результатов:

```bash
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.compare_exports --dataset /data \
  --first /out/gpu/run1 --second /out/gpu/run2 \
  --output /out/gpu/repeatability.json
```

Отдельный замер извлечения признаков, когда второй экспорт уже выполнен:

```bash
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.benchmark --dataset /data --profile MVP_fusion_v25 \
  --provider CUDAExecutionProvider --output /out/gpu/benchmark.json
```

Методика включает 50 прогревов, 300 измерений batch=1, median/p95 и throughput для batch 1/8/16/32 не менее десяти секунд на batch. В измерение входят чтение, декодирование, кроп, preprocessing, все четыре модели и нормализация; поиск по gallery не входит. Эти команды описывают процедуру, а не уже полученный результат. [Локальное сравнение до/после оптимизации](CPU_OPTIMIZATION_2026-09-29.md) не является официальным GPU-бенчмарком.

## 4. Открыть веб-приложение

CPU:

```bash
# Подготовка с интернетом: приложение и PostgreSQL.
docker compose build
# Выполнение без интернета:
docker compose up -d --no-build --pull never
```

GPU, после определения `dcgpu` из предыдущего раздела:

```bash
dcgpu build
dcgpu up -d --no-build --pull never
```

Откройте <http://127.0.0.1:8017>. Swagger доступен по <http://127.0.0.1:8017/docs>.

Полная сборка обязательна: `build inference` из разделов 2–3 не готовит PostgreSQL. Цель `postgres` в Dockerfile фиксирует образ pgvector/PostgreSQL по SHA256; Compose собирает его под локальным тегом `vehicle-reid-postgres:pg16-pgvector0.8.6`. В runtime скачивание образов запрещено. Web/БД используют локальную сеть Compose и могут работать на отключённом от интернета стенде; `dataset-init` и batch не имеют сети. Для web интернет отключается на уровне стенда, не правилом Compose `internal`.

Первый запуск копирует датасет во внутренний volume, запускает PostgreSQL, применяет миграции и строит gallery. Понадобится дополнительное место примерно размером датасета. До завершения индексации сервис может быть не готов.

```bash
curl http://127.0.0.1:8017/api/health
```

Ожидаемые поля: `status=ready`, `profile=MVP_fusion_v25`, `embedding_dim=2048`, `default_threshold=0.534365177154541`. Для выданной gallery ожидается `gallery_size=750`. Provider должен соответствовать выбранному CPU/GPU-режиму.

Выберите готовый query или загрузите изображение, задайте BBox и выполните поиск. В режиме кандидатов отказ показывается отдельно от списка похожих машин. Ручная смена порога в демо не меняет экспорт с зафиксированным профилем.

Остановка без удаления данных: `docker compose stop` или `dcgpu stop`.

## 5. Запуск без интернета

На той же машине достаточно заранее выполнить полную сборку из раздела 4. Для другого компьютера перенесите **приложение и PostgreSQL**, а не только образ приложения, по [инструкции передачи Docker](CONTEST_IMAGE_DELIVERY.md). После `docker load`:

```bash
docker compose up -d --no-build --pull never
```

Для GPU используйте ту же команду через `dcgpu`. Веса и Swagger включены в образ. Batch-команды выше уже содержат `--pull never` и запускаются с `network_mode: none`.

## 6. Проверки разработчика

Изолированные тесты с временной PostgreSQL:

```bash
docker compose -p vehicle-reid-tests -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
docker compose -p vehicle-reid-tests -f docker-compose.test.yml down -v
```

Последняя команда удаляет ресурсы тестового проекта. Состав результатов и границы измерений описаны в [документации](DOCUMENTATION.md).
