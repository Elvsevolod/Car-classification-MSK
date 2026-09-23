# GeM / MixStyle repair — проверка 2026-09-23

Полное обучение на данных организаторов **не запускалось**. Точка входа пользователя:
`train_osnet_gem_mixstyle_repair.ipynb`, Restart Kernel → Run All в прежнем `.venv`.
Каталог `runs/suite_v2_gem_mixstyle_fix` до передачи пользователю не создавался.

## Исправление и регрессии

`AblationModel` теперь явно передаёт `use_bnneck`, `pooling`, `resize_mode`,
`use_mixstyle`, `mixstyle_probability`, `mixstyle_alpha` родительскому конструктору.
Инициализация проверяет фактическое наличие GeM/MixStyle до обучения.

До исправления пять добавленных регрессионных проверок завершались ошибкой;
после исправления проходят. Проверяются реальный тип модулей, обновление параметра
GeM `p`, включение MixStyle в обоих местах backbone при train и отключение при eval.
B0 сопоставлен с существующей реализацией: точное равенство начальных весов,
значений loss и градиента classifier при одинаковом seed.

Команды из корня проекта:

```bash
.venv/bin/python -m pytest tests/test_osnet_ablation_suite.py -q
.venv/bin/python -m pytest tests/test_training.py tests/test_stage6.py tests/test_masked_hpo.py tests/test_official_evaluation.py tests/test_baseline.py -q -k 'not notebook'
```

Результаты: **61 passed** и **86 passed, 1 deselected** соответственно.
Один исторический notebook-тест второй команды исключён; новый notebook валидируется
и компилируется в первой команде. Выводы исторического variant 14 notebook не очищены.
Предупреждения legacy ONNX exporter / InstanceNorm / FastAPI не являются падениями тестов.

ONNX parity проверена для B0, R1, R2, K1, **фактических GeM и MixStyle**:
batch 1/3/8, L2-норма, конечность значений, max absolute difference < 2e-4.
Экспорт выполняется только в тестовые каталоги; MVP не заменяется.

На MPS вне песочницы выполнен один optimizer step для каждого исправленного варианта
на синтетических тензорах batch 4 × 3 × 208 × 208. Loss и нормы градиентов конечны:

| Вариант | Loss | Gradient norm | Диагностика |
|---|---:|---:|---|
| G1_gem | 1.299381 | 243.356491 | GeM p: 3.0 → 2.999899864 |
| S2_mixstyle | 2.090865 | 392.901672 | MixStyle включён, probability=1 для smoke test |

Это проверка исполнения, не измерение качества и не полное обучение. CUDA не проверялась.

## Реальные артефакты и неизменность данных

Выполнены первые три кодовые ячейки нового notebook с отдельным именем
`repair_preflight_20260923`, без вызова `run_suite`. Проверены SHA256 всех 9556 изображений,
CSV, splits, stock/MVP, recipe, evaluator, mask provenance и runtime.
Реальные модули: B0 = AdaptiveAvgPool2d/Identity; G1 = GeM/Identity;
S2 = AdaptiveAvgPool2d/MixStyle.

Семь старых B0 summary/checkpoint прошли проверку совместимости и исходных training signatures:
три primary, три alternate и final refit (1600 steps). Final B0 успешно загружен.
Primary B0: mean 0.820925479, sample std 0.004201895 — без нового обучения.
Протокол повторного использования разрешает только B0; старые G1/S2 не импортируются.
Регрессии проверяют отказ при изменении runtime, бюджета, seed, split, recipe signature,
абляции, посторонних исходников и reference manifest/summary/history/checkpoint.

Preflight-отчёт: `runs/repair_preflight_20260923/verification.json`.
Хеши исходного notebook, train.csv, splits, baseline_metrics и stock/MVP совпали
со снимком до правок. Размеры и mtime всех **166 файлов** `runs/suite_v1` не изменились.
Все исходные изображения повторно проверены по SHA256 после preflight.

Гарантии ограничены этими проверками: ещё неизвестно, улучшат ли G1/S2 метрики поиска.
Исходная validation остаётся development-набором, а не независимым финальным тестом.
