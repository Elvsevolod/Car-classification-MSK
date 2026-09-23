# Проверка реализации NiVe transfer — 23 сентября 2026

Полное обучение на NiVe/данных организаторов **не запускалось**. Подготовлен первый
изолированный эксперимент: B0, двухэтапный matched-budget C1 и NiVe-transfer E1.
Смешанное обучение и маски частей остаются отдельными следующими экспериментами.

## Автоматические проверки

Из корня репозитория:

```sh
.venv/bin/python -m pytest tests/test_nive_transfer.py tests/test_osnet_ablation_suite.py tests/test_training.py tests/test_stage6.py tests/test_masked_hpo.py tests/test_official_evaluation.py tests/test_baseline.py -q -k 'not notebook or test_nive_notebook_is_valid_and_compiles_with_source_guard or test_historical_notebook_is_valid_and_compiles_without_clearing_results'
```

Результат: **168 passed, 2 deselected**, 56.60 s. Предупреждения относятся к существующим
FastAPI/Starlette API и legacy ONNX exporter/InstanceNorm; тесты численного соответствия проходят.

Исключены две исторические проверки notebook. В предварительном прогоне старый тест
`test_repair_notebook_is_unexecuted_valid_and_compiles` упал: он требует пустых outputs,
но пользователь уже выполнил этот notebook. Результаты пользователя не очищались,
старый тест не изменялся. Проверки компиляции исторического основного notebook и нового
NiVe notebook включены в итоговый прогон.

Проверено на синтетических CPU-примерах:

- отсутствие NiVe test и organizer holdout в обучающих стадиях;
- namespace identity, корректный loader, обнаружение подмены файлов и исходного checkpoint;
- фиксированная source-стадия без выбора по validation;
- перенос backbone/BNNeck, новый classifier и пустое состояние target optimizer;
- раздельный C1 для каждого fold и безопасное переиспользование NiVe source;
- точное совпадение B0 с прежним trainer на одинаковом синтетическом запуске;
- совпадение model/optimizer после прерывания и возобновления source/target стадий;
- отсутствие повторных updates для завершённой стадии;
- pilot без outer-оценки/экспорта; confirm с freeze выбора и всех final checkpoint до outer;
- блокировка неподтверждённого источника и активного variant 14 без записи в его файлы;
- реальный OSNet: перенос между classifier разного размера, загрузка checkpoint,
  CPU ONNX parity для batch 1/3/8, максимальная абсолютная погрешность ниже 2e-4.

Новый notebook прошёл `nbformat.validate` и компиляцию всех пяти code cells. На момент
передачи execution_count пусты, outputs отсутствуют. MPS/CUDA training в этой реализации
не проверялся: пользовательская серия variant 14 продолжала занимать устройство.

## Реальные данные: только аудит и загрузка

Артефакты: `runs/preflight_20260923/manifest.json` и `verification.json`.
Этот каталог не является training run. Каталог `runs/nive_pilot_v1` не создан.

| Разбиение NiVe | Изображения | Identity | Используется для обучения |
|---|---:|---:|---|
| train | 17 070 | 703 | да |
| test/query | 2 887 | 600 | нет |
| test/gallery | 11 778 | 600 | нет |

Проверены SHA256 и заголовки всех 31 735 JPEG. Точных дубликатов и пересечений по SHA256
с 9556 изображениями организаторов не обнаружено; near-duplicates этим не исключаются.
Три реальных train-фото прошли обе трансформации loader: конечные тензоры 3×208×208.
Маски NiVe и upload-sidecar файлы не используются.

Fingerprint списка NiVe-файлов:
`345a5d0e940e7c14065f2cec96ea787a84400c92bf553e49907a0fee7b136137`.

Исходное organizer-разбиение сохранено: train 925, calibration 307, validation 309 identity.
Снимки SHA256 до/после совпали для всех 14 защищаемых файлов: семь общих training-модулей,
organizer CSV, splits, baseline metrics, две существующие ONNX-модели и два notebook variant 14.
Активная серия может продолжать записывать собственные outputs; её работу не останавливали.

Официальная карточка источника указана, но происхождение локальной копии пользователь ещё
не подтвердил. Поэтому audit manifest честно содержит `source_confirmed: false`
(в полном manifest — `local_copy_source_confirmed_by_user: false`). Перед первым настоящим
запуском нужно подтвердить источник в notebook, а не переиспользовать каталог preflight.

## Что запускать

После завершения variant 14 открыть `train_nive_transfer.ipynb`, проверить источник,
оставить `PHASE = 'pilot'` и выполнить **Restart Kernel → Run All**.
Точные настройки и ограничения описаны в `README.md`.
