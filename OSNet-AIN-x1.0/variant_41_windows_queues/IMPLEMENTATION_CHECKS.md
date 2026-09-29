# Проверки реализации — 28.09.2026

Это проверка кода до запуска экспериментов, не новый результат качества.
Рабочий MVP_fusion_v25, ветка main, разметка и исторические веса не изменялись.

## Пройдено на исходном Mac

Python 3.11, torch 2.14.0 / torchvision 0.29.0, CPU для unit/smoke.
Целевая Windows-сборка имеет другие закреплённые версии; её проверяет собственный preflight.

```text
ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python -m pytest -q tests/test_parallel_research.py tests/test_transreid_night.py
  tests/test_transreid_system.py tests/test_transreid_full_train.py
  tests/test_transreid_graph.py tests/test_nive_mixed.py
160 passed
```

Проверены G1/G2 weight0 против v25, query permutation/subset, целостность результатов,
запрет CPU fallback, переносимые пути, независимость BN-доменов, target-extra sampling,
точное продолжение синтетического обучения с optimizer, activation checkpoint gradients,
архив с реальными features/checkpoints, импорты без fcntl и синтаксис/формат notebooks.

Дополнительные smoke:

- Все пять исторических OSNet checkpoints строго загрузились; на четырёх настоящих
  изображениях каждый вернул конечные нормированные 512-векторы.
- Исходные N1/T12 caches calibration/validation проверены по SHA; их первые 2048
  координат контроля совпадают побитово. Нулевой вес эксперта воспроизвёл полный
  порядок v25 на обоих splits. Метрики новых fusion/BN-опытов ещё не вычислялись.
- Все три BN-режима OSNet прошли synthetic aux+main forward/backward без NaN/Inf.
- Настоящий DINOv2-S14 загрузился из официального источника и дал `(2,384)` при 280×280.
  SHA256 скачанных весов: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- Настоящий ImageNet IBN-Net загрузился строго с прежним закреплённым SHA256 и дал
  `(2,2048)` при 256×256.
- Служебные заголовки всех трёх официальных автомобильных checkpoints прочитаны
  ограниченным metadata-reader без исполнения сериализованного кода: подтверждены
  one-channel non-local VeRi, старые имена classifier/BN counter и наличие learned
  GeM у VeRi/VehicleID (у VERI-Wild — AvgPool). Добавлена явная миграция;
  остальные поля/формы по-прежнему проверяются строго. Это не полный forward-parity.
- Входной ZIP полностью проверен по CRC и SHA каждого файла; его pin в INPUT_PACKAGE.json.
  Датасеты, ZIP, новые веса и environments исключены из Git.

## Найденное ограничение старых тестов

В расширенном запуске с `tests/test_nive_low_aux.py` осталась одна прежняя ошибка:
`test_notebook_has_clean_run_all_and_new_entrypoint` требует пустой execution_count/outputs
у уже выполненного v33 notebook. Это не ошибка новых очередей. Исторический notebook
и его результаты не очищались; этот старый тест не входит в preflight v40/v41.
Аналогичная проверка в ещё не опубликованном v39 была исправлена, сохранив его результаты.

## Не проверено на Mac

Native Windows PowerShell, CUDA 12.4/PyTorch 2.6, реальные 8 GB, длительные optimizer-runs,
полное совпадение автомобильных FastReID моделей с upstream на реальных весах,
DINOv2-B full/partial run и окончательное качество всех 55 условий.
OOM/ошибки источников должны оставаться явными failed trials, не скрытыми заменами.
Полный Run All и финальный перенос архивов выполняет пользователь на целевых машинах.

## Дополнение: перенос на Mac mini M4 / 16 ГБ

Целевая машина изменена пользователем на Mac mini M4 с 16 ГБ unified memory.
Добавлены `train_mac_m4_queues.ipynb`, `Start_v41_Mac.command`, `setup_mac.sh`,
`requirements-mac.txt` и отдельный run `mac_m4_v1`. Историческое имя каталога
`variant_41_windows_queues` сохранено; Windows-сценарий остаётся резервным.
Рабочий MVP и выполненный пользователем v40 notebook не редактировались.

Проверено после изменений:

- Расширенная регрессия: **165 passed, 7 skipped** (25.46 s). Команда выше плюс
  `tests/test_research_mac.py`. Семь пропусков — намеренно opt-in MPS smokes,
  а не скрытые ошибки; обычный Run All не запускает дополнительное тестовое обучение.
- `bash -n` для обоих Mac-скриптов, формат/синтаксис нового notebook и паритет
  научной конфигурации с Windows. Новый notebook не содержит выполненных ячеек.
- Явные отказы при Rosetta, недоступном MPS, CPU fallback, неограниченном allocator;
  защита от использования перенесённого окружения и старых абсолютных путей.
- Реальный MPS preflight: autograd + AdamW без CPU fallback.
- **6 passed** в реальных MPS forward/backward: TransReID SupCon/soft-triplet,
  NiVe shared/target-updates-only/domain-specific BN, DINOv2-S last4. Батч 4,
  размеры 256 (DINO 280), конечные loss/градиенты/эмбеддинги после optimizer step.
- **1 passed** в MPS resume smoke: четыре шага подряд против двух шагов,
  сохранения, загрузки model/optimizer через CPU и ещё двух шагов. Параметры,
  loss/LR и остальные научные значения history совпали **точно**, без допусков.
  Только allocator samples исключены из сравнения history: число одновременно
  живых тестовых моделей различается. CPU-тест по-прежнему проходит точно.
- Разрешение закреплённых зависимостей через `pip --isolated install --dry-run
  --only-binary=:all: -r .../requirements-mac.txt` завершилось успешно. Это не
  установка с нуля: часть версий уже была в базовом Python. Новая `.venv-v41-m4`
  здесь не создавалась; установка и запуск Jupyter проверяются на целевой машине.
- `git diff --check` без ошибок. Исходные данные, веса и исторические runs не менялись.

Для повторения семи коротких MPS проверок в окружении с входными данными и
перенесённым кешем официального DINOv2-S (из корня проекта):

```bash
ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 RUN_REID_MPS_SMOKE=1 \
  .venv-v41-m4/bin/python -m pytest -q -s \
  tests/test_parallel_research.py tests/test_research_mac.py \
  -k 'real_mps or (resume and mps)'
```

**Граница подтверждения:** GPU проверки выполнены на доступном **Apple M4 Pro,
24 ГБ, macOS 15.7.3**, Python 3.11 / torch 2.14.0 / torchvision 0.29.0, а не на
целевом M4 с 16 ГБ. Это подтверждает работу MPS-пути, но не вместимость всех
P16K4, DINOv2-B, полных NiVe-runs или новых автомобильных весов на 16 ГБ.
Длительное обучение, полный AirDrop, запуск из Finder на другом Mac, установка
чистого окружения и окончательное качество пока не проверены.
MPS memory samples на концах шагов не объявляются точным пиком или полной RAM.
При OOM параметры опыта не меняются автоматически. Ограничения времени нет,
но завершение всех очередей за одну ночь не обещается.
