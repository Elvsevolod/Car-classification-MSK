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
