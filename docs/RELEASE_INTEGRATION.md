# Интеграция frozen R1: запуск и границы приёмки

Работа находится в отдельной копии Car-classification-MSK-release-integration,
ветка codex/r1-release-integration. Ничего не слито и не отправлено в Git.
Исходная review-fixes копия, её незакоммиченные изменения, данные, веса и
история исследований сохранены. SOURCE_APPLICATION_SNAPSHOT.json фиксирует
исходное состояние, ASSET_PROVENANCE.json — источник и SHA256 каждого актива.

## Профили

| Профиль | Вход / D | Ranking | Кандидат | Frozen threshold |
|---|---|---|---|---:|
| MVP_fusion_v25 (по умолчанию) | 208+256 / 2048 | смесь MVP/R1 50/50, 20/3/0.50 | R1 raw_top1 | 0.534365177154541 |
| MVP_dual_role_v24 (откат) | 208+256 / 2048 | MVP legacy 20/3/0.50 | R1 raw_top1 | 0.534365177154541 |
| MVP_legacy (исторический) | 208 / 512 | legacy 20/3/0.50 | ranking_top1 | 0.5948754549026489 |
| RC_R1_equal3_v18 | 256 / 1536 | less_graph 20/3/0.75 | raw_top1 | 0.534365177154541 |
| RC_R1_single_v18 | 256 / 512 | less_graph 20/3/0.75 | raw_top1 | 0.5522230267524719 |

Single — seed 20260915, threshold именно schema-2 v18, не schema-1 v16.
Ансамбль — равная конкатенация трёх нормированных ветвей, не среднее координат.
Все ONNX и исходные JSON bundles скопированы без изменения байтов.
При загрузке проверяются hashes, preprocessing, размерности, политика и метрики.

backend/runtime.py — общий вычислительный путь API, экспорта и benchmark.
Runtime не импортирует training, torch, backend.evaluate или PostgreSQL.
Загрузка frozen calibration.json — проверка описания, не калибровка.
ORT telemetry отключена. CUDA запрашивается явно; fallback запрещён.
Все входы — действительные изображения с исходным bbox. Нет OCR/детектора/
plate masking; identity, camera, имя файла и другие метаданные не подаются модели.

## Notebook Run All

Откройте notebooks/verify_release_run_all.ipynb в kernel новой копии.
В этой рабочей копии отдельная .venv; прежняя среда обучения не изменяется.
Подготовка на другой машине:

    python3.11 -m venv .venv
    .venv/bin/python -m pip install -r requirements-verification.txt
    npm ci --prefix web-ui
    cd web-ui && npx playwright install chromium

Потом выберите этот Python в Jupyter/IDE. В первой ячейке пути DATASET,
RESEARCH, LEGACY_APP и REFERENCE_PYTHON; по умолчанию они указывают на соседние
локальные копии проекта. Для эталонного v18 используется прежняя research
среда, поскольку исторический модуль импортирует training-зависимости.
Новый runtime отдельно тестируется с запретом этих импортов и сетевых вызовов.
Эталон ничего не обучает и пишет только в новый каталог проверки.

SMOKE=False — полная выборка, без общего лимита времени. Каждый Run All
создаёт новый artifacts/release_verification/<дата>_<id>.
Ошибочный этап записывается в отчёт; следующие этапы продолжаются.
В самом конце notebook сообщает о непрошедшей приёмке, сохранив все отчёты.

Проверки: компоненты/API, новые экспорты и независимый эталон, все query с
batch 1/8/16/32, перестановка/удаление соседей, кэш/откат, отдельная временная
PostgreSQL, браузер, offline Docker, новые замеры CPU.
Промежуточные логи показывают этап, профиль, прогресс, время.
2e-4 — прежний фиксированный допуск эмбеддингов, не разрешение менять решения.
Top-10 и ID принятого кандидата/отказ должны совпасть точно.
Для короткого инженерного прогона есть SMOKE=True (32 query / 64 gallery);
это явно помечается как ограниченное покрытие, не приёмка полной выборки.

## Автономный конкурсный запуск

Из новой копии, новый каталог результата:

    .venv/bin/python -m backend.infer --dataset /absolute/dataset --output /absolute/new_output
    .venv/bin/python -m backend.infer --dataset /absolute/dataset --output /absolute/new_r1_output --profile RC_R1_equal3_v18

Вход: images/ и два CSV test_query.csv, test_gallery.csv. Train, labels,
PostgreSQL и UI не нужны. Общий resolver принимает JPG/JPEG/PNG без
чувствительности к регистру расширения. Два файла с одним ID — ошибка.
Повреждённое/отсутствующее изображение, недопустимый bbox, веса с неверной
контрольной суммой — ошибка. Ничего не пропускается и не заменяется фиктивным
вектором. Для конкурсного запуска gallery должна содержать хотя бы 10 ID.

- submission.csv: без заголовка, ровно 10 уникальных gallery-ID на каждый
  query, включая отказы; порядок query совпадает с CSV.
- candidates.csv: заголовок query_id,gallery_id,confidence; одна строка
  только для принятого кандидата. Отказ — отсутствие строки.
- embeddings.npy: float32, конечные ненулевые векторы, query затем gallery.
  D=512/1536 для старых профилей; v24/v25 — блоки MVP512 + R1_1536 (D=2048),
  единичные по отдельности, общая норма sqrt(2). Общая L2 не обязательна (Q&A №8).
  Число строк берётся из CSV.

Для MVP сохранено историческое преобразование confidence=(raw+1)/2 в CSV.
R1 экспортирует raw cosine. Оба не являются вероятностями, порог применяется
до преобразования. Конкурсный CLI не имеет аргумента ручного threshold.
export_manifest.json и runtime_timing.json — дополнительные локальные сведения;
при сдаче обязательны именно три названных файла.

## PostgreSQL, сервис и откат

Миграция 0002 создаёт reid_gallery_spaces/reid_gallery_vectors; старые
gallery_items/gallery_state не изменяет. Векторные размерности не смешиваются:
составной foreign key связывает namespace и D, CHECK проверяет vector_dims.
Fingerprint включает модель, preprocessing, D, provider/ORT, порядок gallery,
bbox и SHA256 байтов изображений. Успешная сборка публикуется одной транзакцией.
Кэш immutable: конфликтные данные не перетираются.
PostgreSQL хранит; общий NumPy scorer принимает решения, без ANN/SQL-ranking.

Профиль и provider выбираются при запуске процесса:

    REID_PROFILE=RC_R1_equal3_v18 REID_PROVIDER=CPUExecutionProvider docker compose up --build

У запуска по умолчанию MVP_fusion_v25/CPU. Горячего переключения нет.
Возврат REID_PROFILE=MVP_dual_role_v24 и перезапуск восстанавливают решения v24.
REID_PROFILE=MVP_legacy возвращает исторический MVP. Вектора v24/v25 идентичны и могут
использовать общее пространство хранения, но profile fingerprint и ranking различаются.
Старый образ также может читать нетронутые старые таблицы.
Но его Alembic ещё не знает revision 0002: для запуска старого образа
на уже обновлённой БД нужно обойти его migration-entrypoint
(--entrypoint python, затем -m backend), не выполнять старый alembic upgrade.
Обычный проверяемый откат MVP → R1 → MVP выполняется в НОВОМ образе через
REID_PROFILE, без downgrade и без удаления новых пространств.
Не используйте docker compose down -v для отката: это удаляет volumes.

API сохраняет results как ranking даже при отказе. accepted_candidate=null
при отказе или в режиме ranking; в candidates он может отличаться от
results[0] и даже отсутствовать в показанном top-K. Интерфейс показывает его
отдельно, с его собственной фотографией и cosine.
Ручной threshold помечается «только демо», не сохраняется, не меняет bundle.
Health и ответы показывают профиль/fingerprint и политики. Метрики берутся
только из проверенного отчёта выбранного профиля, не из метрик MVP для всех.

## Офлайн-поставка одной командой

В процессе сборки сеть разрешена, при инференсе нет.

    .venv/bin/python -m tools.build_offline_release --output /absolute/new_delivery --platform linux/amd64

Получатся сжатый Docker image, source ZIP (включая незакоммиченный код и
reproduction), MIT license, manifest с checksums, inventory всего models/,
решение о релизе и run-offline.sh. Dataset/история/среды/секреты не включаются.
Веса включены в Docker image; source ZIP содержит их описания и checksums,
но не вторую копию ONNX. Извлечение весов описано в reproduction/README.md.
Архитектура образа записывается; сборка amd64 на Mac не означает проверку
на native Linux amd64 или измерение производительности официального железа.
Используйте каталог вне дерева репозитория для поставки.

После переноса на офлайн-машину:

    sh /absolute/delivery/run-offline.sh /absolute/dataset /absolute/new_output
    sh /absolute/delivery/run-offline.sh /absolute/dataset /absolute/new_output_r1 RC_R1_equal3_v18

Скрипт загружает локальный образ и запускает --network none, без БД.
Лимит весов — 2 000 000 000 bytes; inventory консервативно считает ВСЕ
файлы models/, включая неактивные веса, JSON и лицензию.
Происхождение и код обучения: reproduction/README.md.
Python лицензии сохраняются в dist-info внутри образа; OSNet MIT — в models/.
Лицензии точных frontend-зависимостей собираются в frontend/dist/THIRD_PARTY_NOTICES.txt.

## Benchmark и GPU

    .venv/bin/python -m backend.benchmark --profile RC_R1_equal3_v18 --dataset /absolute/dataset --output /absolute/new_benchmark.json

50 прогревов; 300 новых latency; throughput batch 1/8/16/32 минимум 10 секунд;
полный новый extract. Включены чтение/decode, crop, resize, нормализация,
transfers, все member forwards, L2. Загрузка модели указана отдельно.
Runner измеряет полный экспорт как время нового процесса и сопоставляет с
ориентиром median(batch1) × (n_query+n_gallery) × 3, без изменения скорости
по предыдущим runs. Notebook сам не ограничен этим временем.

Тот же runner поддерживает --provider CUDAExecutionProvider. На GPU требуется
совместимый onnxruntime-gpu вместо CPU wheel, CUDA/cuDNN и nvidia-smi.
Не устанавливать оба ORT wheel одновременно. Точный GPU environment должен
быть зафиксирован и проверен на целевой машине; здесь CUDA не подтверждена.
RAM — process RSS; VRAM — NVML process allocation через nvidia-smi каждые
0.25 секунды. Это sampled peak, не гарантированный пик и не torch allocator.
Отсутствующий PID/драйвер даёт unknown, не нулевое потребление.

## Незакрытые требования

- GPU и native Linux amd64. Наши скорости — оценочные; баллы назначают организаторы.
- Отсутствие OCR не доказывает отсутствие остаточного сигнала номерной зоны.
- Точная сигнатура extractor.py не предоставлена. Файл с выдуманным контрактом
  не добавлен; подтверждён только известный однокомандный экспорт.
- Junk/top-10: evaluator удаляет junk перед своим усечением. Из десяти
  экспортированных ID нельзя восстановить 11-й. Evaluator неизменён,
  расширенный ranking не сдаётся, неизвестные test-метки не используются.
- Презентация и ссылки на материалы — отдельная финальная проверка.

release_decision.json фиксирует одобренное пользователем продвижение v25;
v24 и MVP_legacy сохранены для отката. Детали: V25_PROMOTION.md.
Успех smoke-тестов не объявляется полной приёмкой соревнования.
