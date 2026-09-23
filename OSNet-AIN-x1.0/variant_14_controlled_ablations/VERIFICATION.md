# Проверка реализации — 22.09.2026

Статус: notebook подготовлен; **полные обучения на данных организаторов не запускались**.
Никаких выводов о приросте качества variant 14 пока нет.

## Выполнено

- 42 новых теста variant 14 + 86 существующих regression tests: **128 passed, 1 deselected**.
  Исключён только старый тест пустых outputs исторического notebook. Его результаты не очищались.
- Все 22 конфигурации: stock initialization, конечный loss, backward, полный охват параметров optimizer.
- B0: точное совпадение state, loss и classifier gradients с существующей реализацией при одном seed.
- BN: проверка running statistics после последовательности eval → train, которую использует consistency.
- Прерывание после atomic checkpoint: повторный запуск восстанавливает model, optimizer, EMA, XBM и best state;
  совпадает с непрерывным синтетическим обучением. Готовые обучения повторно не выполняются.
- Проверки неизменности исходного изображения/строки при masking, cross-camera sampler, detached memory,
  блокировки параллельного запуска, отклонения изменённого config, freeze выбора до outer и неизменных thresholds.
- ONNX для B0, 256px, local128, color32 и GeM: batch 1/3/8, ошибка <2e-4, конечные L2 embeddings.
- Новый notebook валиден, code cells компилируются; setup/config/preflight реально выполнены без training cell.
- `git diff --check` прошёл. Ранее существовавшие изменения training/hpo.py, pipeline.py и preprocessing.py не правились.

Команда regression:

```sh
.venv/bin/python -m pytest tests/test_osnet_ablation_suite.py tests/test_training.py tests/test_stage6.py tests/test_masked_hpo.py tests/test_official_evaluation.py tests/test_baseline.py -q -k 'not notebook or test_notebook_is_unexecuted_valid_and_compiles'
```

## Проверка реальных входов

Все **9556** кадров сверены по SHA256 с сохранёнными splits, затем проверены повторно после preflight.
`train.csv`, stock/MVP weights, evaluator, baseline reference и исходные query/gallery прошли проверку.
Ничего не исключено и не исправлено.

| Partition | Identity | Query | Gallery |
|---|---:|---:|---:|
| Outer train | 925 | — | — |
| Primary inner holdout | 185 (inner train 740) | 185 | 496 |
| Alternate inner holdout | 185 (inner train 740) | 185 | 520 |
| Original calibration | 307 | 307 | 895 |
| Original validation | 309 | 309 | 896 |

Manifest диагностического preflight находится в `runs/verification_only_20260922/manifest.json`.
Он не резервирует пользовательский `suite_v1` и не содержит результатов обучения.
Fingerprint manifest: `cb2c67073984734f490458d002e88ab5da1ac65f091429c9a28490b9dfc77d08`.
Aggregate frame digest: `dc4cabbc0d4f0dead1f859236070c2bc46a67801914d3f1674eca417710dce45`.

## Ограничения проверки

- Локально проверен CPU; полный CUDA/MPS run не выполнен. Время обучения и VRAM на целевой машине не измерялись.
- Есть предупреждения PyTorch о legacy ONNX export/InstanceNorm tracing. Экспорт использует существующий
  в проекте путь `dynamo=False`; численная parity разных batch проверена, предупреждения не скрываются.
- Context7 и сетевой доступ к документации были недоступны; использованы существующие вызовы проекта,
  локальные signatures/docstrings установленного PyTorch и исполняемые проверки.
- Это не полный запуск всех тестов репозитория и не измерение конкурсного performance score.
- Original validation не стала новым независимым тестом: она уже использовалась в исследованиях.
