# Вариант 2: HPO + P=16/K=2 + BNNeck/SupCon

## Цель

Проверить, можно ли улучшить mAP и качество отказа за счёт:

- большего числа negative identity в batch: `P=16`, `K=2`;
- positive-пар с разных камер, когда это возможно;
- раздельных learning rate для encoder и новой головы;
- warmup + cosine decay;
- BNNeck;
- Supervised Contrastive Loss как альтернативы Batch Hard Triplet;
- Optuna TPE + median pruning после четырёх полных эпох;
- последовательного отбора: 16 trials → top-4 → top-2 с несколькими seed;
- раздельного выбора checkpoint по mAP, Candidate F1 и TNR.

## Протокол

Optuna не видит внешние 307 calibration и 309 validation identity. 925 train identity внутренне
делятся 80/20. Сначала запускаются 16 trials максимум по 8 эпох; слабый trial можно остановить
не раньше четвёртой эпохи. Четыре лучших завершённых trial продолжаются с их checkpoint до 20 эпох.
Две лучшие конфигурации затем независимо обучаются с нуля до 30 эпох на трёх seed. Победитель
выбирается по среднему best-mAP, а не по одному удачному запуску. Только после этого он обучается
на всех 925 train identity и оценивается на прежней calibration/validation.

Финальные seed-запуски намеренно начинаются со стокового ONNX, а не с top-4 checkpoint. Иначе смена
seed только на последних эпохах не была бы независимой проверкой устойчивости конфигурации.

`camera_id` используется только для выбора cross-camera positive в train-batch. Он не подаётся модели и не
нужен при inference.

## Запуск

```bash
uv pip install --python .venv/bin/python -r requirements-train.txt
uv pip check --python .venv/bin/python
.venv/bin/python -m jupyter lab
```

Открыть `train_osnet_hpo.ipynb` и выполнить `Run All`. По умолчанию:

- `TARGET_TRIALS = 16`;
- `TRIAL_EPOCHS = 8`;
- `TOP_CANDIDATES = 4`, `PROMOTION_EPOCHS = 20`;
- `FINALISTS = 2`, `FINALIST_EPOCHS = 30`;
- три seed: `20260915`, `20260916`, `20260917`;
- `OUTER_EPOCHS = 30`;
- HPO, полное обучение и ONNX-экспорт включены.

Study возобновляется из `results/optuna.sqlite3`. Перед новым полным запуском нужно изменить
`RUN_NAME`, чтобы notebook не перезаписал прошлую историю.

## Checkpoint policy

- `weights/<RUN_NAME>/last.pt` — последняя завершённая эпоха;
- `best_map.pt` — лучший retrieval;
- `best_f1.pt` — лучший баланс кандидатов;
- `best_tnr.pt` — лучший отказ;
- `epoch_XX.pt` — каждая эпоха, которая улучшила хотя бы одну из трёх метрик;
- `osnet_best_map.onnx` — экспорт best-mAP encoder с BNNeck, если он выбран.

Промежуточные веса лежат в `weights/stage1`, `weights/stage2_top4` и
`weights/stage3_top2_seeds`. Бинарные веса варианта 2 и Optuna SQLite исключены из Git, чтобы
поиск не раздувал репозиторий.

## Результат

Пока не получен: notebook подготовлен, но полное HPO-обучение ещё не запускалось. После запуска итоги
будут в `results/study_summary.json`, `results/stage2_top4_summary.json`,
`results/stage3_top2_seeds_summary.json` и `results/<RUN_NAME>/summary.json`; по ним этот раздел
нужно обновить.
