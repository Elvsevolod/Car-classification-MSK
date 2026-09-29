# Проверки реализации v43 — 2026-09-29

Полное сравнение R03 на original validation **не запускалось**: notebook оставлен
для пользовательского Run All. Новых научных результатов v43 пока нет.
Каталог `runs/` не создан. Notebook проходит nbformat validation, все code cells
компилируются, execution_count/outputs пустые.

## Компоненты и регрессия

**86 passed, 7 skipped**, 7.53 s:

```bash
ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q \
  tests/test_osnet_r1_validation.py tests/test_osnet_r1_search.py \
  tests/test_dual_role_inference.py tests/test_map_search.py tests/test_parallel_research.py
```

В новом файле v43 — 18 тестовых случаев. Проверены роли моделей, буквальное
совпадение контроля с v25 scorer, замена только первого ranking-R1, реальные
float32-блоки, сохранение raw-кандидата и его confidence, отказ с полным top-10,
NPY replay, отсутствие обучения/калибровки/сетевого запроса в synthetic Run All,
возобновление без повторного извлечения, повреждённые веса/кэш, atomically committed
stage, query permutation/removal, batch 1/8/16/32, явное устройство без fallback,
JPEG/PNG и неоднозначное/отсутствующее изображение, identity leakage.

Пропуски принадлежат opt-in hardware/integration тестам существующих модулей;
это не семь проваленных конфигураций. Реальная MPS-проверка v43 выполнена отдельно.

## Реальные локальные файлы и веса

- Read-only preflight: 309 query / 896 gallery, защищены 1284 файла.
- Полная строгая загрузка конкретного checkpoint R03: 740-классовая исходная
  модель, step400 / 12800 предъявлений; eval-mode, параметры requires_grad=False.
- CPU: 32 реальные primary-изображения; max error к v42 **9.648501873e-7**.
- CPU: fresh v25 на 8 изображениях, max error к прежнему кэшу **0.0**.
- Из прежних validation-векторов повторён **только контроль v25**:
  submission.csv и candidates.csv совпали побайтно, mAP@10 **0.8289573904235558**,
  F1 **0.7954545454545456**. Это replay старого контроля, не новый результат R03
  и не измерение скорости продукта.
- Источники и защищённые файлы после проверки не изменились.

## Реальный MPS — Apple Silicon / исходный Mac

Без нового обучения и без оценки R03 на original validation. На 32 primary
изображениях из уже завершённого v42 при FP32 и отключённом CPU fallback:

| Batch | Максимальная ошибка к сохранённым v42-признакам |
|---:|---:|
| 1 | 1.899898052e-7 |
| 8 | 1.713633537e-7 |
| 16 | 0.0 |
| 32 | 0.0 |

Обратный порядок изображений также прошёл прежний допуск **2e-5**.
Буферы BN побитно неизменны. Допуски, порог и данные не подгонялись.
Это проверка численной совместимости, не бенчмарк RTX/GPU организаторов.

## Изоляция

Добавлены только новые v43-файлы: runner, тесты, notebook, config, README,
этот протокол и локальный `.gitignore`. Ранее существующие изменения сохранены.
Не выполнялись commit/push, переключение MVP, новый refit или изменение v41.

Сохранённые hashes прежних runner:

- v42 `osnet_r1_search.py`: `74c16ffc432c9a33fc0251db6d12c04b15ea2a47ca933dedf7c2ee8a3830d66e`;
- v41 `research_gpu.py`: `2d8297a6a65bafb72e1ef308f185f913cb88c3dacb979601e1d40531204c8028`;
- `research_training.py`: `5ee205a6c8389957356e539100f6668263305b96c4e350b9393efa66d3ebda1d`.
