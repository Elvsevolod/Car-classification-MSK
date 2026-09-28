# ASU_TEAM — Vehicle ReID

Поиск одного автомобиля на снимках с разных камер по изображению и предоставленному BBox. В `main` находятся актуальный React-интерфейс и последняя утверждённая модель **MVP_fusion_v25**: четыре ONNX-энкодера, зафиксированный preprocessing, ranking и порог отказа. Все веса входят в Git обычными файлами; Git LFS и скачивание моделей при запуске не нужны.

**Основной конкурсный путь:** каталог изображений + два CSV → одна команда offline inference → `submission.csv`, `embeddings.npy`, `candidates.csv`. Web/API с PostgreSQL — отдельное демонстрационное приложение. Для inference не нужны `train.csv`, исследовательская ветка или обучение.

Текущий статус проверки и незакрытые условия: [GPU_READINESS.md](docs/GPU_READINESS.md). RTX 4060 позволяет проверить CUDA и получить замеры на своём компьютере. Эти цифры нельзя выдавать за результат на конкурсной RTX A5000 24 ГБ.

Для передачи проверки другой нейросети: [готовый промпт для RTX 4060](docs/RTX_4060_TEST_PROMPT.md) с командами запуска, критериями проверки и форматом отчёта в Git.

## 1. Подготовка Windows + RTX 4060

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

Сборка требует сети. Она создаёт `vehicle-reid:cuda12.2` для **linux/amd64**, включает UI, все веса и точные версии зависимостей из [requirements-gpu.txt](requirements-gpu.txt). ONNX Runtime GPU 1.20.2 использует CUDA 12.2/cuDNN 9.1. Он выбран с учётом заявленного организаторами драйвера CUDA 12.2; нельзя без повторной проверки заменять его последним GPU-пакетом. В одном окружении не устанавливаются одновременно CPU и GPU пакеты ONNX Runtime. Матрица совместимости: [ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).

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

```bash
dcgpu up -d --build
dcgpu ps
curl http://127.0.0.1:8017/api/health
```

- Интерфейс: <http://127.0.0.1:8017>.
- Swagger: <http://127.0.0.1:8017/docs> — assets включены в образ, CDN не нужен.
- Health: <http://127.0.0.1:8017/api/health> — `status=ready`, `profile=MVP_fusion_v25`, `provider=CUDAExecutionProvider`, `embedding_dim=2048`, gallery соответствует CSV.

Порт по умолчанию **8017**, другой задаётся `PORT` в `.env`. При первом запуске `dataset-init` копирует dataset во внутренний volume (понадобится дополнительное место примерно размером dataset), затем выполняются миграции и индексирование gallery. Дождитесь healthy; первое индексирование может занять несколько минут. Кэш учитывает байты изображений, BBox, модель и provider; смена CPU/GPU создаёт совместимое пространство заново.

В UI выберите query либо загрузите JPEG/PNG, задайте BBox и выполните поиск. Ranking — смесь MVP/R1 50/50 и streaming k-reciprocal (20/3/0.50). Решение «совпадение/отказ» принимает отдельная ветвь R1 с порогом **0.534365177154541**. Принятый кандидат может отличаться от первого в ranking и показывается отдельно. При отказе `accepted_candidate=null`; ranking остаётся доступным для просмотра. Ручной threshold в UI — только демо и не меняет конкурсный профиль.

Основные API: `GET /api/health`, `/api/queries`, `/api/metrics`; `POST /api/search`, `/api/search/query`, `/api/embedding`; `GET /api/images/...`. Embedding API возвращает 2048D для v25. `POST /api/search` принимает image + x/y/w/h, опционально top_k/mode/threshold. JPEG/PNG ограничены 15 МиБ и 25 мегапикселями. Можно сравнивать кандидатов и скачать JSON ответа.

Остановка без удаления данных: `dcgpu stop`. Команда `dcgpu down -v` удаляет БД, кэш датасета и внутренние runtime-artifacts; она не нужна для обычного перезапуска или отката модели. Сохранённые bind-mount результаты в `./artifacts` остаются на хосте.

## 6. CPU-запуск и автономная поставка

На macOS или компьютере без NVIDIA используйте основной Compose без GPU overlay:

```bash
docker compose up -d --build
# Или только конкурсный экспорт, без web/БД:
docker compose build inference
docker compose --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/cpu-run1
```

Для стенда без сети заранее соберите и перенесите Linux amd64 GPU-образ через `docker save`/`docker load`. На стенде используйте `--pull never` и `--no-build` для web. Точная процедура и контрольная сумма архива — [CONTEST_IMAGE_DELIVERY.md](docs/CONTEST_IMAGE_DELIVERY.md). Сеть разрешена при сборке; конкурсный inference выполняется с полностью отключённой сетью. Web использует локальную сеть Compose для PostgreSQL.

## 7. Модель, качество и источники

Архитектура: вход/BBox → проверка и crop → OSNet-AIN x1.0 (одна MVP и три R1) → фиксированные признаки → статичная gallery → независимые ranking и отказ → UI/API/конкурсные файлы. Web хранит gallery в PostgreSQL 16 + pgvector; batch — в памяти процесса. Нет OCR, детектора, трекинга, query expansion или использования других query. Камера, время и география не поступают модели.

Наблюдавшаяся development-validation на 309 query / 896 gallery: **mAP@10 82.90%, Rank-1 83.00%, Rank-5 87.85%, F1 79.55%, TNR 70.97%**. Это сохранённые результаты выбора модели, не новый независимый тест и не оценка организаторов. Порог выбран на calibration по максимуму `0.7×F1 + 0.3×TNR`, при равенстве — F1 и затем больший порог. Закрытый тест не используется для настройки. Методика, ограничения и исходные отчёты: [MODEL_REPORT.md](docs/MODEL_REPORT.md), [V25_PROMOTION.md](docs/V25_PROMOTION.md), [profiles.json](models/profiles.json).

Источники и фиксированные версии:

- Backbone OSNet-AIN x1.0, публичная инициализация OpenVINO Open Model Zoo **vehicle-reid-0001, 2022.1**, авторы исходного vehicle-ReID порта: [sovrasov/deep-person-reid](https://github.com/sovrasov/deep-person-reid/tree/vehicle_reid). MIT, локально [LICENSE.osnet](models/LICENSE.osnet). URL весов, SHA-256/SHA-384 и preprocessing: [models/README.md](models/README.md). Четыре дообученных ONNX имеют проверяемые checksum в frozen bundles; [ASSET_PROVENANCE.json](docs/ASSET_PROVENANCE.json) хранит их происхождение.
- Дообучение активных весов: выданный организаторами dataset. Внешние экспериментальные NiVe/TransReID не входят в активный v25. Данные организаторов получают отдельно; пути прошлой разработки в provenance не нужны для запуска.
- Полный обучающий код и воспроизведение модели: [зафиксированный исследовательский коммит 8fc310c](https://github.com/Elvsevolod/Car-classification-MSK/tree/8fc310ca6ac1d8467ef8bfdc5ce4c1d1afa4498b), папка `reproduction/source/`. При сдаче приложите этот код/архив вместе с main, а не только контейнер inference. [Разделение веток](docs/BRANCH_LAYOUT.md).
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
