# ASU_TEAM — Vehicle ReID

Поиск одного автомобиля на снимках с разных камер по изображению и предоставленному BBox. В `main` находятся актуальный React-интерфейс и последняя утверждённая модель **MVP_fusion_v25**: четыре ONNX-энкодера, зафиксированный preprocessing, ranking и порог отказа. Все веса входят в Git обычными файлами; Git LFS и скачивание моделей при запуске не нужны.

**Основной конкурсный путь:** каталог изображений + два CSV → одна команда offline inference → `submission.csv`, `embeddings.npy`, `candidates.csv`. Web/API с PostgreSQL — отдельное демонстрационное приложение. Для inference не нужны `train.csv`, исследовательская ветка или обучение.

Текущий статус проверки и незакрытые условия: [GPU_READINESS.md](docs/GPU_READINESS.md). Изменения без замены модели и локальное сравнение скорости: [CPU_OPTIMIZATION_2026-09-29.md](docs/CPU_OPTIMIZATION_2026-09-29.md). Замеры Mac или RTX 4060 нельзя выдавать за результат на конкурсной RTX A5000 24 ГБ.

Для передачи проверки другой нейросети: [готовый промпт для RTX 4060](docs/RTX_4060_TEST_PROMPT.md) с командами запуска, критериями проверки и форматом отчёта в Git.

## Материалы для сдачи

- [Документация решения](docs/DOCUMENTATION.md), [пошаговый запуск](docs/RUNBOOK.md) и [PDF-снимок документации от 28 сентября](docs/ASU_Team_Vehicle_ReID.pdf). Актуальные команды после оптимизации 29 сентября — в README и Runbook.
- Презентация ASU Team: [PDF](presentation/ASU_Team.pdf) и [редактируемый PPTX](presentation/ASU_Team.pptx), 15 слайдов, версия от 29 сентября.
- [Готовые конкурсные результаты v25](submission/MVP_fusion_v25/README.md): три файла, паспорт, проверка и контрольные суммы. Это результаты на выданных изображениях, не оценка скрытого теста.
- [Исходники воспроизведения четырёх моделей, коммит 35e6e1f ветки fine-tuning](https://github.com/Elvsevolod/Car-classification-MSK/tree/35e6e1f0ffc6f378e924da930fdeccc5841273d1/reproduction-kit), [история экспериментов](docs/EXPERIMENT_HISTORY.md).
- [Памятка по ссылкам для формы сдачи](docs/SUBMISSION_PACKAGE_GUIDE.md). Адрес размещённого прототипа команда указывает отдельно; localhost не является публичной ссылкой.
- [Повторная проверка комплекта перед публикацией](docs/PUBLICATION_CHECK_2026-09-29.md): тесты main/research, воспроизведение из Git и согласованность конкурсных файлов.

## Быстрый запуск для экспертов: Linux amd64 + NVIDIA

Нужны Docker с Compose, NVIDIA driver и настроенный [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Python, CUDA Toolkit и Node.js отдельно на хосте не нужны. Команды выполняются из корня поставленной версии `main` в Bash.

1. Поместите выданные тестовые данные в `dataset/`: `images/`, `test_query.csv`, `test_gallery.csv` (подробный формат ниже).
2. Один раз соберите образ с доступом к сети:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build inference
```

3. Выполните конкурсный запуск одной командой:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile inference \
  run --rm --no-deps --pull never inference \
  --dataset /data --output /out/contest-run1 \
  --profile MVP_fusion_v25 --provider CUDAExecutionProvider
```

Результаты: `artifacts/contest-run1/{submission.csv,candidates.csv,embeddings.npy}`. Программа обрабатывает весь набор, сама проверяет форматы и сохраняет дополнительные manifest/timing-файлы. Повторный запуск требует нового выходного каталога. Inference работает **без сети, обучения, калибровки, PostgreSQL и UI**; отсутствие CUDA — ошибка, не скрытый переход на CPU. Для полностью offline-стенда вместо сборки загрузите заранее переданный образ: [инструкция поставки](docs/CONTEST_IMAGE_DELIVERY.md).

Для измерения скорости extractor по методике организаторов есть отдельная команда; время полного экспорта её не заменяет:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile inference \
  run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.benchmark --dataset /data --profile MVP_fusion_v25 \
  --provider CUDAExecutionProvider --output /out/contest-benchmark.json
```

UI предназначен для демонстрации поиска оператором: загрузка отдельного фото/BBox и просмотр результатов. Эксперт передаёт новую gallery через входной каталог и CSV; загружать библиотеку через браузер для конкурсного тестирования не требуется.

## 1. Входные данные и альтернативный запуск Windows + RTX 4060

1. Установите актуальный Windows-драйвер NVIDIA с поддержкой WSL2 и Docker Desktop. В PowerShell выполните `wsl --update`; если WSL ещё не установлен — сначала `wsl --install`, затем завершите настройку Ubuntu и перезагрузку, если она запрошена.
2. В Docker Desktop включите **Use the WSL 2 based engine**, Linux containers и **Resources → WSL Integration → Ubuntu**. GPU поддерживается именно через WSL2: [инструкция Docker](https://docs.docker.com/desktop/features/gpu/).
3. Все последующие команды выполняйте в **терминале Ubuntu/WSL (Bash)**. Не устанавливайте Linux-драйвер NVIDIA внутрь WSL. Отдельные Python, CUDA Toolkit и Node.js на хосте для запуска проекта не нужны.
4. Держите проект и dataset в файловой системе WSL (`~/...`), а не в `/mnt/c/...`: чтение и декодирование изображений входят в замер скорости.

```bash
nvidia-smi
docker version
docker compose version
git clone --branch main https://github.com/Elvsevolod/Car-classification-MSK.git
cd Car-classification-MSK
git rev-parse HEAD
```

Если репозиторий уже клонирован, используйте чистую копию `main` и `git pull --ff-only`. Незакоммиченные исследования сохраняйте отдельно. На Linux x86_64 требуется NVIDIA driver и настроенный [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Папка `dataset/` рядом с Compose-файлами должна содержать:

```text
dataset/
├── images/             # JPG/JPEG/PNG, имя файла = image_id
├── test_query.csv      # image_id,x,y,w,h
└── test_gallery.csv    # image_id,x,y,w,h
```

Дополнительные колонки метаданных не подаются модели. ID уникальны, query/gallery не пересекаются. Для конкурсного Top-10 нужны минимум 10 объектов gallery. Пропавшее/повреждённое изображение или неверный BBox завершают запуск ошибкой, а не пропускаются.

```bash
ls dataset/images dataset/test_query.csv dataset/test_gallery.csv
```

Датасет не публикуется в Git. Другой каталог можно задать через `HOST_DATASET_DIR` в `.env`; каталог результатов — через `OUTPUT_DIR`. Без этих настроек используются `./dataset` и `./artifacts`.

## 2. Сборка и проверка CUDA

Определите сокращение в текущем терминале Bash; в новом терминале выполните его снова:

```bash
dcgpu() { docker compose -f docker-compose.yml -f docker-compose.gpu.yml "$@"; }
dcgpu build inference
```

Сборка требует сети. Она создаёт `vehicle-reid:cuda12.2` для **linux/amd64**, включает UI, все веса и точные версии зависимостей из [requirements-gpu.txt](requirements-gpu.txt). ONNX Runtime GPU 1.20.2 использует CUDA 12.2/cuDNN 9.1, включая NVRTC 12.2.140 и путь к `libnvrtc.so.12` внутри образа. Он выбран с учётом заявленного организаторами драйвера CUDA 12.2; нельзя без повторной проверки заменять его последним GPU-пакетом. После обновления зависимостей пересоберите образ: старый образ не получает исправления из нового README. В одном окружении не устанавливаются одновременно CPU и GPU пакеты ONNX Runtime. Матрица совместимости: [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

Проверьте доступ к GPU и запуск всех четырёх моделей без сети:

```bash
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint nvidia-smi inference
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.gpu --output /out/rtx4060/preflight.json
```

Ожидается `preflight_passed: true`, профиль `MVP_fusion_v25`, размерность `2048`, `CUDAExecutionProvider` первым в каждой из четырёх сессий и `cpu_compute_fallback: false`. Проверяется фактическое распределение операторов, а не только наличие CUDA в списке провайдеров. ONNX Runtime может выполнять служебный `Shape` на CPU; свёртки и остальные вычисления сети на CPU запрещены. Отсутствие драйвера/библиотеки или GPU приводит к ошибке. Используется float32 без TF32; веса и порог не меняются.

## 3. Конкурсный offline inference одной командой

После сборки и подготовки dataset:

```bash
dcgpu --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/rtx4060/run1
```

Это самостоятельный batch-процесс с `network_mode: none`, read-only dataset, без web, PostgreSQL, миграций и калибровки. Все query обрабатываются независимо относительно статичной gallery. По умолчанию batch size — 16; уменьшить его при нехватке памяти можно через `--batch-size 8` или `1`, записав выбранное значение в отчёт.

В `artifacts/rtx4060/run1/` появятся:

| Файл | Контракт |
|---|---|
| `submission.csv` | Без заголовка; query ID и ровно 10 разных gallery ID. Все query, включая отказы, в исходном порядке CSV. |
| `embeddings.npy` | `float32`, `(N_query + N_gallery, 2048)`: сначала query, затем gallery, строго в порядке CSV. Для выданного набора — `(1860, 2048)`. |
| `candidates.csv` | Заголовок `query_id,gallery_id,confidence`. Один принятый R1 raw-top1 кандидат с raw cosine; при отказе строка отсутствует. |
| `export_manifest.json` | Профиль, провайдер, хэши входов/кода, порядок ID и результаты автоматической проверки форматов. |
| `runtime_timing.json` | Время загрузки и полного экспорта; это не latency extractor. |

Сдаются первые три файла. Остальные сохраняются для проверки воспроизводимости. Выходной каталог должен быть новым или пустым: программа **не перезаписывает** прежние результаты. Для повторного теста выбирайте новое имя, например `rtx4060-test2/run1`.

Вектор содержит два нормированных блока MVP512 + R1_1536, общая норма `sqrt(2)`. Общая L2-нормализация необязательна по Q&A №8; размерность фиксирована для всех объектов. Raw cosine в `candidates.csv` не является вероятностью и не преобразуется в `(cosine+1)/2` для v25.

## 4. Повторяемость и benchmark на RTX 4060

Выполните второй независимый offline запуск и сравнение:

```bash
dcgpu --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/rtx4060/run2
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.compare_exports --dataset /data \
  --first /out/rtx4060/run1 --second /out/rtx4060/run2 \
  --output /out/rtx4060/repeatability.json
```

По умолчанию требуется точное совпадение embeddings, confidence, Top-10 и кандидата/отказа. Несовпадение — ошибка проверки, не успешный результат. Между разными GPU/CPU побитовая идентичность заранее не обещается.

Перед benchmark закройте игры и другие GPU-задачи; web-приложение запускайте после замеров. Если оно уже запущено через `dcgpu`, остановите его: `dcgpu stop vehicle-reid`.

```bash
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.benchmark --dataset /data --profile MVP_fusion_v25 \
  --provider CUDAExecutionProvider --output /out/rtx4060/benchmark.json
```

Методика Q&A №38: 50 прогревов, 300 измерений batch=1 с `cudaDeviceSynchronize` до/после; median и p95; batches 1/8/16/32 не менее 10 секунд каждый. Таймер включает чтение, декодирование, EXIF, BBox, preprocessing, передачу данных, все четыре forward и нормализацию. Поиск/gallery reranking в extractor timing не входят. Отчёт содержит GPU/драйвер, версии, размер весов, память и отдельное время полного извлечения признаков.

RAM/VRAM снимаются периодически: это наблюдавшийся максимум, не гарантированный точный пик. В WSL2 NVML может не показывать память отдельного процесса; тогда VRAM остаётся `null` с предупреждением и требуется дополнительный замер на Linux/NVIDIA. Ошибка/OOM на batch=32 не должна скрываться уменьшением списка батчей или выдачей частичного замера за полный.

Для сравнения с CPU-сборкой создайте отдельный эталон; это может занять заметно больше времени:

```bash
docker compose build inference
docker compose --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/rtx4060/cpu-reference --provider CPUExecutionProvider
dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.compare_exports --dataset /data \
  --first /out/rtx4060/cpu-reference --second /out/rtx4060/run1 \
  --atol 0.0002 --output /out/rtx4060/cpu-cuda-parity.json
```

Численный допуск `2e-4` применяется к vectors/confidence; **Top-10, ID кандидата и отказ должны совпасть точно**. Даже небольшое численное отличие, изменившее решение, требует разбора. Эта проверка одновременно сравнивает CPU ORT 1.30.0 с GPU ORT 1.20.2. Сохраните весь каталог `artifacts/rtx4060/` и SHA коммита из шага 1.

## 5. Новый web-интерфейс на GPU

Подготовка с интернетом (собирает **и приложение, и PostgreSQL**):

```bash
dcgpu build
```

После сборки интернет больше не нужен. Запуск всех сервисов одной командой:

```bash
dcgpu up -d --no-build --pull never
dcgpu ps
curl http://127.0.0.1:8017/api/health
```

- Интерфейс: <http://127.0.0.1:8017>.
- Swagger: <http://127.0.0.1:8017/docs> — assets включены в образ, CDN не нужен.
- Health: <http://127.0.0.1:8017/api/health> — `status=ready`, `profile=MVP_fusion_v25`, `provider=CUDAExecutionProvider`, `embedding_dim=2048`, gallery соответствует CSV.

Порт по умолчанию **8017**, другой задаётся `PORT` в `.env`. При первом запуске `dataset-init` копирует dataset во внутренний volume (понадобится дополнительное место примерно размером dataset), затем выполняются миграции и индексирование gallery. Дождитесь healthy; первое индексирование может занять несколько минут. Кэш учитывает байты изображений, BBox, модель, provider и реализацию preprocessing/runtime. Обновление кода оптимизации или смена CPU/GPU создаёт новое пространство кэша; прежние таблицы и результаты сохраняются. Первое построение после обновления не является замером поиска с готовым кэшем.

В UI выберите query либо загрузите JPEG/PNG, задайте BBox и выполните поиск. Ranking — смесь MVP/R1 50/50 и streaming k-reciprocal (20/3/0.50). Решение «совпадение/отказ» принимает отдельная ветвь R1 с порогом **0.534365177154541**. Принятый кандидат может отличаться от первого в ranking и показывается отдельно. При отказе `accepted_candidate=null`; ranking остаётся доступным для просмотра. Ручной threshold в UI — только демо и не меняет конкурсный профиль.

Основные API: `GET /api/health`, `/api/queries`, `/api/metrics`; `POST /api/search`, `/api/search/query`, `/api/embedding`; `GET /api/images/...`. Embedding API возвращает 2048D для v25. `POST /api/search` принимает image + x/y/w/h, опционально top_k/mode/threshold. JPEG/PNG ограничены 15 МиБ и 25 мегапикселями. Можно сравнивать кандидатов и скачать JSON ответа.

Остановка без удаления данных: `dcgpu stop`. Команда `dcgpu down -v` удаляет БД, кэш датасета и внутренние runtime-artifacts; она не нужна для обычного перезапуска или отката модели. Сохранённые bind-mount результаты в `./artifacts` остаются на хосте.

## 6. CPU-запуск и автономная поставка

На macOS или компьютере без NVIDIA используйте основной Compose без GPU overlay:

```bash
# Один раз, пока есть интернет: все образы, включая PostgreSQL.
docker compose build
# Запуск без сборки и без скачивания образов:
docker compose up -d --no-build --pull never
# Или только конкурсный экспорт, без web/БД:
docker compose build inference
docker compose --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/cpu-run1
```

PostgreSQL — отдельная цель `postgres` в Dockerfile и отдельный контейнер. `docker compose build` готовит его локальный образ `vehicle-reid-postgres:pg16-pgvector0.8.6`; версия исходного образа закреплена SHA256. Только `docker build .` или `docker compose build inference` готовят приложение, **но не БД**: этого достаточно для batch, недостаточно для web.

Для переноса полного демо на другой offline-стенд сохраните **оба образа**: приложение и PostgreSQL. При сборке на той же машине перенос не нужен. Точные команды CPU/GPU и контрольные суммы — [CONTEST_IMAGE_DELIVERY.md](docs/CONTEST_IMAGE_DELIVERY.md). При запуске pull запрещён; отсутствующий образ — ошибка, а не попытка скачивания. Batch и подготовка данных работают с `network_mode: none`. Web и PostgreSQL используют локальную сеть Compose; UI доступен через опубликованный порт хоста. Интернет для работы демо не требуется. Обычная bridge-сеть не является запретом внешнего трафика: отключение интернета для web-стенда обеспечивает окружение организаторов.

## 7. Модель, качество и источники

Архитектура: вход/BBox → проверка и crop → OSNet-AIN x1.0 (одна MVP и три R1) → фиксированные признаки → статичная gallery → независимые ranking и отказ → UI/API/конкурсные файлы. Web хранит gallery в PostgreSQL 16 + pgvector; batch — в памяти процесса. Нет OCR, детектора, трекинга, query expansion или использования других query. Камера, время и география не поступают модели.

Наблюдавшаяся development-validation на 309 query / 896 gallery: **mAP@10 82.90%, Rank-1 83.00%, Rank-5 87.85%, F1 79.55%, TNR 70.97%**. Это сохранённые результаты выбора модели, не новый независимый тест и не оценка организаторов. Порог выбран на calibration по максимуму `0.7×F1 + 0.3×TNR`, при равенстве — F1 и затем больший порог. Закрытый тест не используется для настройки. Методика, ограничения и исходные отчёты: [MODEL_REPORT.md](docs/MODEL_REPORT.md), [V25_PROMOTION.md](docs/V25_PROMOTION.md), [profiles.json](models/profiles.json).

Источники и фиксированные версии:

- Backbone OSNet-AIN x1.0, публичная инициализация OpenVINO Open Model Zoo **vehicle-reid-0001, 2022.1**, авторы исходного vehicle-ReID порта: [sovrasov/deep-person-reid](https://github.com/sovrasov/deep-person-reid/tree/vehicle_reid). MIT, локально [LICENSE.osnet](models/LICENSE.osnet). URL весов, SHA-256/SHA-384 и preprocessing: [models/README.md](models/README.md). Четыре дообученных ONNX имеют проверяемые checksum в frozen bundles; [ASSET_PROVENANCE.json](docs/ASSET_PROVENANCE.json) хранит их происхождение.
- Дообучение активных весов: выданный организаторами dataset. Внешние экспериментальные NiVe/TransReID не входят в активный v25. Данные организаторов получают отдельно; пути прошлой разработки в provenance не нужны для запуска.
- Обучающий код и команды воспроизведения: [зафиксированный комплект в fine-tuning, 35e6e1f](https://github.com/Elvsevolod/Car-classification-MSK/tree/35e6e1f0ffc6f378e924da930fdeccc5841273d1/reproduction-kit). В нём сохранены исторические исходники [8fc310c](https://github.com/Elvsevolod/Car-classification-MSK/tree/8fc310ca6ac1d8467ef8bfdc5ce4c1d1afa4498b), точные рецепты и хеши. При сдаче передайте ссылку на этот код/архив вместе с main, а не только контейнер inference. [Разделение веток](docs/BRANCH_LAYOUT.md).
- Полный список Python-библиотек с точными версиями: [CPU requirements.txt](requirements.txt), [GPU requirements-gpu.txt](requirements-gpu.txt). Ключевые: Python 3.11, FastAPI 0.141.1, NumPy 2.4.6, Pillow 12.3.0, ONNX 1.22.0; версии ORT и CUDA описаны выше. Node 24.15.0 нужен только при сборке UI. Полный список frontend-зависимостей и версий: [package-lock.json](web-ui/package-lock.json); `npm ci` использует lock. Версии и digest контейнеров зафиксированы в Dockerfile/Compose.

## 8. Проверки и границы готовности

Изолированные тесты с временной PostgreSQL (не затрагивают БД демо):

```bash
docker compose -p vehicle-reid-tests -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
docker compose -p vehicle-reid-tests -f docker-compose.test.yml down -v
```

Browser-тесты требуют Node.js на машине разработчика и работающий UI:

```bash
cd web-ui
npm ci
npx playwright install chromium
E2E_BASE_URL=http://127.0.0.1:8017 npm run test:e2e
```

Требования проверяются по [ORGANIZER_QA.md](ORGANIZER_QA.md), исходному ТЗ и [позднейшим уточнениям чата](docs/ORGANIZER_CLARIFICATIONS_2026-09-28.md). **Обязательное требование `extractor.py` отменено организаторами 22 сентября**; используйте описанный выше batch CLI. Обучение на выданных тестовых изображениях запрещено даже без разметки. Полная готовность к сдаче требует фактического прогона на NVIDIA, повторяемости, замеров всех батчей, проверки на контрольном маскировании номеров, презентации и ссылок на поставку. Неоднозначность junk/Top-10 сохраняется в [release_decision.json](release_decision.json); официальный `evaluate.py` не изменён.

Передайте доступные экспертам ссылки на репозиторий, документацию, презентацию PDF/PPTX и работающий прототип. Для кейса нужна полная презентация с обязательным блоком 7–11 официального шаблона. Достаточность одного Docker/скринкаста и требуемый срок работы онлайн-прототипа отдельно не уточнены. **Стоп-код: 29 сентября 2026, 23:59 МСК**; после него сданные ветки, документацию, презентацию и стенд менять нельзя. Основания и граница актуальности — в [уточнениях](docs/ORGANIZER_CLARIFICATIONS_2026-09-28.md).
