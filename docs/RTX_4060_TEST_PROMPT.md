# Промпт для проверки проекта на RTX 4060

Передай другой нейросети весь текст ниже. Она должна работать на компьютере с
RTX 4060, Windows, Docker Desktop и Ubuntu/WSL2, иметь доступ к репозиторию и
предоставленному датасету. Результат работы — фактический прогон и отчёт в Git.

---

Ты проверяешь проект **ASU_TEAM / Vehicle ReID**:
[Elvsevolod/Car-classification-MSK](https://github.com/Elvsevolod/Car-classification-MSK).
Выполни проверку актуального `main` на **RTX 4060 через Docker Desktop / WSL2**.
Не ограничивайся чтением кода или составлением плана: собери образ, выполни CUDA
inference, измерь производительность, проверь UI и сохрани доказательства.

## 1. Цель и границы проверки

Нужно установить:

1. Собирается ли текущий проект из Git и запускаются ли все четыре сети на CUDA.
2. Получаются ли три конкурсных файла на полном предоставленном наборе.
3. Совпадают ли два независимых GPU-запуска и результаты CPU/CUDA.
4. Каковы latency, throughput и потребление памяти именно на этой RTX 4060.
5. Работает ли актуальный UI с CUDA-профилем и совпадает ли инструкция с реальностью.

RTX 4060 проверяет работоспособность CUDA и скорость на данном компьютере.
Она **не подтверждает скорость, потребление памяти или официальный балл на
RTX A5000 24 ГБ**. Не пересчитывай FPS через число CUDA-ядер или отношение
характеристик видеокарт.

Сначала прочитай `AGENTS.md`, если он есть, и следующие файлы репозитория:

- `README.md`;
- `ORGANIZER_QA.md` и `docs/ORGANIZER_CLARIFICATIONS_2026-09-28.md`;
- `docs/GPU_READINESS.md` и `docs/CONTEST_IMAGE_DELIVERY.md`;
- `models/profiles.json`, `docs/V25_PROMOTION.md`, `release_decision.json`;
- `Dockerfile`, оба Compose-файла, оба файла `requirements*.txt`.

Последний утверждённый продуктовый профиль — **MVP_fusion_v25**:
четыре активных ONNX, выход 2048D, порог отказа **0.534365177154541**.
Более высокий номер исследовательской модели не означает разрешение заменить v25.
Если актуальный main явно изменил утверждённый профиль, опиши расхождение до запуска;
не выбирай модель самостоятельно.

Предыдущая проверка проводилась на Mac/CPU. Она не подтвердила успешную чистую
GPU-сборку или реальное выполнение CUDA. Старые результаты не засчитывай за новые.

Не меняй веса, preprocessing, ranking, threshold, официальный evaluator и тестовые
данные. Не обучай и не калибруй модель на тесте. Не ослабляй проверки, чтобы получить
PASS. Цель этой задачи — проверка и отчёт; исправление кода при обнаружении ошибки
выноси в отдельную задачу, сохранив воспроизведение исходной проблемы.

## 2. Подготовка и фиксация версии

Все shell-команды ниже выполняются в **Bash внутри Ubuntu/WSL2**, из корня проекта.
Репозиторий и dataset желательно держать в файловой системе WSL, например `~/work`,
поскольку чтение изображений входит в замеры.

Если копии проекта нет:

```bash
git clone --branch main https://github.com/Elvsevolod/Car-classification-MSK.git
cd Car-classification-MSK
```

Если копия уже есть, сначала проверь её ветку и незакоммиченные изменения.
Используй чистый main с `git pull --ff-only` либо отдельную чистую копию.
Не делай `reset --hard`, `clean`, принудительный checkout или stash чужой работы.
После начала измерений не обновляй проверяемый коммит.

Убедись, что Windows-драйвер NVIDIA установлен, WSL2 обновлён, Docker Desktop
использует WSL2 engine, Linux containers и интеграцию с Ubuntu.
Linux-драйвер NVIDIA внутрь WSL не устанавливай.

```bash
git status --short --branch
git rev-parse HEAD
nvidia-smi
docker version
docker compose version
```

Зафиксируй в отчёте:

- дату/время UTC, полный SHA проверяемого коммита и чистоту дерева;
- Windows, WSL, Ubuntu, kernel, Docker Desktop/Engine и Compose;
- точное имя GPU, Desktop/Laptop, объём VRAM, драйвер, CPU, RAM хоста и лимиты WSL;
- питание от сети/режим питания, фоновые GPU-задачи, расположение dataset;
- версии Python, ONNX Runtime, CUDA runtime и cuDNN **в собранном контейнере**.

Строка CUDA Version в `nvidia-smi` характеризует возможности драйвера;
она не заменяет проверку установленных библиотек контейнера.
Если доступа к RTX 4060 или датасету нет, запиши BLOCKED и причину.
Запуск на CPU не закрывает этот блокер.

Dataset должен содержать `images/`, `test_query.csv`, `test_gallery.csv`.
Проверь наличие изображений, колонки `image_id,x,y,w,h`, валидность BBox,
уникальность ID, отсутствие пересечения query/gallery и минимум 10 gallery.
Запиши фактические количества строк и SHA-256 обоих CSV.
В исходном наборе 1110 query и 750 gallery; для другого набора используй реальные
количества, не подгоняй файлы под 1860 объектов.

Создай уникальный каталог результатов и отдельный Compose-проект:

```bash
RUN_ID="rtx4060-$(date -u +%Y%m%d-%H%M%S)"
TESTED_SHA="$(git rev-parse HEAD)"
export COMPOSE_PROJECT_NAME="reid-$RUN_ID"
export REID_PROFILE=MVP_fusion_v25
export HOST_DATASET_DIR="$PWD/dataset"
export OUTPUT_DIR="$PWD/artifacts/$RUN_ID"
export PORT=8017
mkdir -p "$OUTPUT_DIR/logs"
dcgpu() { docker compose -f docker-compose.yml -f docker-compose.gpu.yml "$@"; }

run_logged() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "$OUTPUT_DIR/logs/$name.log"
  local rc="${PIPESTATUS[0]}"
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$name" "$rc" >> "$OUTPUT_DIR/exit-codes.tsv"
  return "$rc"
}
```

Если dataset расположен иначе, до запуска замени `HOST_DATASET_DIR` на его
абсолютный путь. Если 8017 занят, выбери свободный порт, обнови `PORT` и запиши его.
Переменные и функции нужны в каждом новом терминале. Логи могут содержать
конфиденциальные сведения: оригиналы храни локально, перед публикацией проверь.

Выполняй этапы последовательно и проверяй exit code после каждого.
При ошибке не запускай зависимые этапы как будто всё прошло; независимые проверки
продолжи. Для повторной попытки выбирай новые пути и имена логов.

## 3. Чистая сборка и доказательство CUDA

```bash
run_logged gpu-build dcgpu build --no-cache inference
run_logged image docker image inspect vehicle-reid:cuda12.2 --format '{{.Id}} {{.Os}}/{{.Architecture}}'
run_logged container-gpu dcgpu --profile inference run --rm --no-deps --pull never --entrypoint nvidia-smi inference
run_logged gpu-packages dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference -m pip freeze
run_logged gpu-pip-check dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference -m pip check
run_logged preflight dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.gpu --profile MVP_fusion_v25 --output /out/preflight.json
```

Ожидается linux/amd64 и зависимости из `requirements-gpu.txt`.
На момент подготовки промпта это ORT GPU 1.20.2, CUDA runtime 12.2, cuDNN 9.1.
CPU-пакет `onnxruntime` не должен быть установлен рядом с `onnxruntime-gpu`.

Проверь в `preflight.json`:

- `preflight_passed=true`, нужный профиль, размерность 2048;
- четыре сессии, в каждой CUDA — первый фактический provider;
- реальные CUDA kernel events, `cpu_compute_fallback=false`;
- на CPU разрешён только служебный `Shape`, вычисления сетей идут через CUDA;
- все веса присутствуют, их checksum совпадают с frozen bundles;
- суммарные поставляемые веса не превышают 2 000 000 000 байт.

Наличие слова CUDA в настройках, списка доступных providers или окна UI само по
себе не доказывает выполнение сетей на GPU. Сохрани ID образа и не подменяй его
старым образом с похожим тегом. Если сборка прервалась из-за сети, это BLOCKED,
а не успешная сборка.

## 4. Два полных offline-запуска и форматы сдачи

```bash
run_logged gpu-run1 dcgpu --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/run1 --profile MVP_fusion_v25 --provider CUDAExecutionProvider --batch-size 16
run_logged gpu-run2 dcgpu --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/run2 --profile MVP_fusion_v25 --provider CUDAExecutionProvider --batch-size 16
run_logged repeatability dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.compare_exports --dataset /data --first /out/run1 --second /out/run2 --output /out/repeatability.json
```

Подтверди `network_mode: none`, read-only dataset и отсутствие зависимости inference
от PostgreSQL, web, train.csv, скачивания весов или обучения. Проверь эффективную
конфигурацию Compose с профилем inference; в отчёт вынеси только относящиеся к этим
пунктам поля, не публикуй конфигурацию с паролями.

У каждого запуска проверь пять файлов:

| Файл | Что должно быть |
|---|---|
| `submission.csv` | Без заголовка; каждая query в порядке CSV; ровно 10 различных существующих gallery ID; строки есть и для отказов |
| `embeddings.npy` | float32, конечные значения, форма `(N_query + N_gallery, 2048)`; сначала все query, потом gallery в порядке CSV |
| `candidates.csv` | Заголовок `query_id,gallery_id,confidence`; максимум одна строка на query; при отказе строки нет; confidence — raw cosine |
| `export_manifest.json` | Правильные профиль/provider, fingerprints, хэши входов, порядок объектов, успешная проверка форматов |
| `runtime_timing.json` | Реальное время данного запуска, включая полный pipeline; не выдавать его за latency extractor |

Для v25 нормированы два блока 512D и 1536D, общая норма около sqrt(2);
не исправляй её до 1. Кандидат/отказ определяются отдельной ветвью R1 и могут
отличаться от первого элемента общего ranking. Отказ не удаляет Top-10.

Ожидается `repeatability.json: passed=true`, нулевой допуск, точные embeddings,
confidence, Top-10, принятые кандидаты и отказы. Если сравнение падает, сохрани
исходную ошибку и диагностику различий; не расширяй допуск.

При OOM сначала сохрани ошибку и исходный batch. Можно проверить запуск с batch=8
или 1 в новых каталогах, но отдельно укажи, что batch=16 не прошёл. Повторяемость
сравнивай для двух запусков с одинаковыми настройками.

## 5. Производительность и память

Останови web этого тестового Compose-проекта, если он уже работает. Убедись, что
на GPU нет конкурирующей нагрузки; чужие процессы самостоятельно не завершай.

```bash
run_logged benchmark dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.benchmark --dataset /data --profile MVP_fusion_v25 \
  --provider CUDAExecutionProvider --output /out/benchmark.json
```

Проверь методику и фактические поля JSON: **50 прогревов, 300 измерений batch=1,
cudaDeviceSynchronize до/после замера, median и p95; batches 1/8/16/32,
каждый не менее 10 секунд**. Таймер включает чтение, decode/EXIF, BBox,
preprocessing, передачу данных, все четыре forward и нормализацию.
Поиск и reranking в latency extractor не входят.

В отчёте приведи FPS и длительность для каждого batch, время загрузки модели,
полного извлечения признаков и полного конкурсного экспорта.
Не меняй FP32 на FP16/TF32 и не измеряй одну сеть вместо всего профиля.
OOM на batch=32 — не пройденный полный benchmark; не скрывай его исключением batch.
Если итоговый JSON не появился из-за ошибки, приложи лог и отметь отсутствие JSON.

Память из benchmark — наблюдавшийся максимум периодических samples, а не точный
пик аллокатора. Разделяй RAM процесса, доступную WSL RAM и VRAM.
Если в WSL память GPU-процесса недоступна, сохрани `null` и warning, напиши
«не измерено». Общая занятость GPU из nvidia-smi не равна памяти одного процесса.
Не ставь 0 вместо отсутствующего измерения.

## 6. Сравнение CPU и CUDA

Создай CPU-эталон на тех же входах и том же коммите, отдельно от GPU-образа:

```bash
run_logged cpu-build docker compose build inference
run_logged cpu-reference docker compose --profile inference run --rm --no-deps --pull never inference \
  --dataset /data --output /out/cpu-reference --profile MVP_fusion_v25 --provider CPUExecutionProvider --batch-size 16
run_logged cpu-cuda-parity dcgpu --profile inference run --rm --no-deps --pull never --entrypoint python inference \
  -m backend.compare_exports --dataset /data --first /out/cpu-reference --second /out/run1 \
  --atol 0.0002 --output /out/cpu-cuda-parity.json
```

Ожидается `passed=true`. Абсолютный допуск 2e-4, rtol=0 — только для embeddings
и confidence. Top-10, ID кандидатов и отказы должны совпасть точно.
На момент подготовки это также сравнение CPU ORT 1.30.0 и GPU ORT 1.20.2:
зафиксируй обе реальные версии. Если CPU-эталон не выполнен, поставь NOT RUN;
успешный preflight не заменяет parity.

## 7. UI, API и регрессии

После измерений запусти уже проверенный GPU-образ:

```bash
run_logged web-start dcgpu up -d --no-build
run_logged web-status dcgpu ps -a
run_logged health curl --fail --show-error "http://127.0.0.1:$PORT/api/health"
```

Дождись завершения dataset-init, миграций и индексирования, затем healthy.
Первый запрос до готовности не засчитывай как окончательный результат.
Health должен показывать ready, v25, CUDAExecutionProvider, 2048D и правильное
число gallery. Сохрани тело успешного ответа отдельно как `health.json`.

Через браузер проверь:

- загрузку актуального UI, отсутствие ошибок JavaScript и запросов;
- выбор query, загрузку JPEG/PNG, установку/изменение BBox и поиск;
- Top-10, отдельного принятого кандидата, confidence и состояние отказа;
- сохранение ranking при отказе, `accepted_candidate=null` в API;
- сравнение/увеличение изображений и скачивание JSON ответа;
- понятные ошибки для повреждённого файла и неправильного BBox;
- доступность Swagger `/docs` без CDN;
- обычный stop/start с сохранением БД и кэша.

Для демонстрации отказа/принятия можно менять threshold только в отдельном
UI-запросе; после верни значение профиля. Не сохраняй это в модели и экспортах.
API/скриншот одной страницы не заменяет фактическую проверку сценариев браузером.

Запусти Python/integration-тесты по разделу 8 README на отдельном временном
Compose-проекте и Playwright-тесты с `E2E_BASE_URL=http://127.0.0.1:$PORT`.
Укажи passed/failed/skipped, версии и логи. Эта Python-suite использует CPU:
не выдавай её за доказательство CUDA. Если инструмента браузера/Node нет,
зафиксируй ограничение и фактический объём ручной проверки.

После проверки останови только созданный тобой тестовый проект. Не удаляй
чужие контейнеры, images, volumes или исходные данные. `down -v` допустим только
для заведомо одноразового проекта тестов, не для БД демонстрации.

## 8. Требования организаторов и итоговый вердикт

Сверь фактическое поведение с прочитанными требованиями. В частности:

- три формата сдачи, полностью offline inference, фиксированные веса/порог;
- независимые query относительно статичной gallery; без cross-query обработки,
  обучения на тесте, использования камеры/времени/географии и OCR;
- лимит весов, методика timing, честная фиксация оборудования;
- доступность воспроизводимого запуска по README.

Отсутствие `extractor.py` **не является блокером**: требование отменено
организаторами 22.09.2026. Не создавай адаптер из-за старого ответа в Q&A.
Не объявляй закрытыми конфликт junk/Top-10, проверку остаточного номерного сигнала,
презентацию, доступность прототипа или другие пункты `release_decision.json`,
если эта работа их фактически не проверяла.

Статусы каждого пункта: **PASS / FAIL / BLOCKED / NOT RUN** с доказательством.
Отдельно сформулируй готовность CUDA-запуска, полноту замеров и состояние UI.
«Полная готовность к конкурсу» не следует автоматически из успешного теста 4060.
Не меняй `official_gpu_verified` на true и не придумывай официальный score.

## 9. Что сохранить локально и отправить в Git

Все исходные результаты, три файла каждого экспорта, полные manifests и логи
сохрани локально в `artifacts/<RUN_ID>/`. В Git отправь компактный отчёт:

```text
docs/gpu-tests/<RUN_ID>/
├── REPORT.md
├── environment.md
├── commands.md
├── exit-codes.tsv
├── checksums.sha256
├── evidence/
│   ├── preflight.json
│   ├── benchmark.json
│   ├── repeatability.json
│   ├── cpu-cuda-parity.json
│   ├── health.json
│   ├── run1-runtime_timing.json
│   └── run2-runtime_timing.json
└── logs/
    └── <логи сборки, запусков, тестов и ошибок>
```

Не создавай фиктивные JSON для упавших этапов. В REPORT.md перечисли отсутствующие
артефакты с причинами. `commands.md` содержит фактически выполненные команды,
параметры, время и повторные попытки. `checksums.sha256` — хэши локальных трёх
файлов сдачи каждого запуска, входных CSV и проверенных весов; подпиши пути так,
чтобы было ясно, какие файлы остаются локально. Сохрани fingerprints из manifests
в отчёте, чтобы связать входы, модель и результаты.

Перед публикацией проверь каждый файл. Не коммить dataset, изображения теста,
raw embeddings, копии весов, Docker TAR, базы данных, .env, токены, пароли,
персональные пути или полный экспорт чата организаторов.
Для больших логов публикуй относящиеся к проверке выдержки с пометкой об усечении,
хэшем оригинала и его локальным расположением без персональных данных.
Скриншоты UI прикладывай только при допустимости публикации показанных изображений;
иначе опиши сценарии и сохрани скриншоты локально. Не ретушируй ошибки из доказательств.

Структура REPORT.md:

```markdown
# Проверка RTX 4060 — <UTC-дата>

## Вердикт
CUDA: <статус>. Экспорт: <статус>. Повторяемость: <статус>.
CPU/CUDA parity: <статус>. Benchmark: <статус>. UI: <статус>.
Главные блокеры и что остаётся непроверенным.

## Версия и стенд
Проверенный commit SHA; image ID; профиль и fingerprints; окружение;
N_query/N_gallery; batch; ссылки на environment.md, команды и checksum.

## Результаты
Таблица: проверка | статус | наблюдение | ссылка на JSON/лог.
Отдельно: сколько тестов passed/failed/skipped и какие сценарии не выполнены.

## Производительность RTX 4060
Latency median/p95; warmup/samples; синхронизация и границы таймера.
Таблица: batch 1/8/16/32 | images/s | секунды | статус/OOM.
Load/full extract/full export time. RAM/VRAM и метод измерения/ограничения.
Это собственные замеры RTX 4060; официальная A5000 не проверена.

## Форматы и воспроизводимость
Количество строк, форма/dtype/finite embeddings, принятые query/отказы.
Результат точного GPU/GPU сравнения.
Результат CPU/CUDA, допуск и максимальное отличие; совпадение решений.

## UI и инструкция
Проверенные сценарии; ссылки на доказательства; расхождения с README.

## Ошибки и оставшаяся работа
Для каждой: команда, exit code, существенный текст ошибки, воспроизведение,
влияние, диагностические попытки и следующий шаг.
Открытые требования конкурса перечислены отдельно от GPU-ошибок.

## Артефакты
Что приложено в Git, что осталось локально, что не создано и почему.
```

## 10. Публикация и ответ пользователю

Публикация отчёта в Git входит в эту задачу. Используй новую ветку
`reports/<RUN_ID>` от проверенного коммита. Не коммить в main, не делай
force-push и не объединяй отчёт с изменениями модели или чужими файлами.

**Перед публикацией проверь дату и актуальные правила стоп-кода.**
В имеющихся уточнениях дедлайн — **29.09.2026 23:59 МСК**
(30.09.2026 03:59 UTC+07). После него запрещено менять сданные ветки,
документацию, презентацию и стенд. Этот промпт не отменяет запрет.
Если дедлайн прошёл, сохраняй отчёт локально; публикацию делай только в явно
разрешённом, не входящем в сдачу месте. Не запускай обновление сданного стенда.

До стоп-кода, при отсутствии другого ограничения, опубликуй подготовленный отчёт:

```bash
REPORT_DIR="docs/gpu-tests/$RUN_ID"
REPORT_BRANCH="reports/$RUN_ID"
# Сначала убедись, что отчёт уже создан, HEAD = TESTED_SHA и нет чужих изменений.
git switch -c "$REPORT_BRANCH" "$TESTED_SHA"
git add -- "$REPORT_DIR"
git diff --cached --stat
git diff --cached --check
# Просмотри staged diff: только отчёт, без секретов и запрещённых payload.
git commit -m "Report RTX 4060 CUDA validation $RUN_ID"
git push -u origin "$REPORT_BRANCH"
```

Проверь, что remote-ветка указывает на опубликованный commit. Если push заблокирован
авторизацией/сетью, сохрани локальный commit и прямо сообщи, что публикации не было.
Не заявляй, что отчёт на GitHub, только потому что существует локальный файл.

В финальном ответе пользователю напиши по-русски:

1. Запустился ли проект на CUDA RTX 4060 и что осталось неготовым.
2. Полный SHA проверенного кода, ветку/commit отчёта и **прямую ссылку на REPORT.md**.
3. Median/p95 latency, FPS для 1/8/16/32 и доступные измерения RAM/VRAM.
4. Результаты GPU/GPU, CPU/CUDA и UI; критичные ошибки и непроверенные пункты.
5. Где локально лежат три файла сдачи и полные логи.

Если этап не выполнен, сообщи это вместо предполагаемого результата.
