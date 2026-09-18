# Вариант 4: обучающая стратегия

## Статус

Оговорка аудита 18 сентября: encoder исходного checkpoint уже обучался на всех
925 outer-train identity, включая будущие inner-validation identity. Сброс головы
не устраняет эту утечку во внутреннем отборе. Внешнее разбиение остаётся чистым,
но отрицательные результаты screening не доказывают бесполезность стратегий.
Архивные код и веса не перезаписываются; новые абляции должны начинаться с
инициализатора, не видевшего внутреннюю validation.

Полные эксперименты завершены. Ни одна новая стратегия не превзошла baseline,
поэтому активные веса MVP не изменены.

Inner screening выбрал сам baseline на первой эпохе. В matched-сравнении по трём
seed контроль получил mean mAP@10 `0.796863 ± 0.007137`, mean Candidate score
`0.752816` и mean quality score `0.433870`. Это ниже активной модели по mAP@10
и quality score, поэтому `beats_active_reference=false`.

Этап проверяет четыре гипотезы поверх победившей конфигурации варианта 2:

1. MixStyle в двух ранних/средних уровнях OSNet;
2. формирование P×K batch из визуально похожих identity;
3. постепенное увеличение веса metric loss;
4. `CE + Circle Loss` вместо `CE + SupCon`.

Все изменения проверяются последовательно. Circle и SupCon не складываются в
одном запуске, а MixStyle и hard negatives объединяются только если каждый из
них отдельно превзошёл контроль.

Каждый запуск начинается из одного и того же
`variant_02_hpo_bnneck_supcon/weights/selected_run_02/best_map.pt`. Для
внутреннего screening переносятся encoder и BNNeck, а голова создаётся
заново под 740 inner-train identity. В финальном matched-сравнении
переносится весь checkpoint, включая голову на 925 identity.

## Исторический протокол (ограничение inner отбора описано выше)

Screening использует только внутреннее identity-disjoint разбиение 80/20 внутри
925 train identity. Внешние calibration и validation на этом шаге не кодируются
и не участвуют в выборе стратегии.

После screening его best epoch замораживается. Контроль и победившая
стратегия обучаются ровно столько эпох по трём seed: `20260915`, `20260916`,
`20260917`, не заглядывая в outer validation между эпохами. Затем параметры
реранкинга и порог отказа выбираются только на calibration, а validation
каждого seed оценивается один раз. Test query и test gallery нигде в обучении
и выборе параметров не участвуют.

Победителем стратегия считается только если относительно matched-контроля она:

- улучшила средние mAP@10 и quality score;
- не потеряла больше `0.005` среднего candidate score.

Отдельно отчёт сообщает, превзошёл ли победитель текущий активный MVP:
`mAP@10=0.814689`, `candidate_score=0.747121`, `quality_score=0.441322`.

## Что именно реализовано

- MixStyle работает только в режиме `train`, смешивает отсоединённые статистики
  признаков после `pool2` и `pool3`; на inference слой тождественный.
- Hard-negative sampler строит прототип каждой identity активным checkpoint варианта 2 и
  группирует ближайшие прототипы, сохраняя `P=16`, `K=2` и приоритет
  cross-camera positive.
- Dynamic loss начинает с 25% настроенного веса metric loss и линейно доводит
  его до 100% за четыре эпохи.
- Circle Loss использует margin `0.25`, gamma `64` и все positive/negative пары
  текущего P×K batch.
- Все screening- и seed-запуски сохраняют `last.pt` и историю и умеют
  продолжаться после прерывания; screening также хранит `best_map.pt`,
  а финальные seed — замороженный `final.pt`.
- Для итогового кандидата сохраняются настроенный реранкинг, ONNX и
  PyTorch/ONNX parity-check.

## Запуск

Из корня репозитория:

```bash
.venv/bin/python -m jupyter lab
```

Открыть `train_osnet_strategy.ipynb` и выполнить `Run All`. По умолчанию
выбирается MPS, если он доступен. Контролировать выполнение не требуется:
завершённые эксперименты пропускаются, незавершённые продолжаются.

Порядок ноутбука:

1. smoke test одного шага MixStyle + Circle;
2. screening по внутреннему split, максимум 8 эпох на конфигурацию;
3. matched-сравнение контроля и стратегии по трём seed на
   замороженном inner-selected числе эпох;
4. экспорт репрезентативного seed в ONNX.

Этап длиннее предыдущего: screening содержит 5–6 запусков, финальное сравнение —
ещё 6. Не удаляйте `results/` и `weights/`, если хотите продолжить прерванный
расчёт.

## Артефакты

- `results/screening_summary.json` — все screening-кандидаты и победитель;
- `results/final_comparison.json` — среднее и разброс по трём seed;
- `results/export_summary.json` — checkpoint, параметры retrieval и ONNX parity;
- `weights/osnet_stage4_selected.onnx` — экспериментальный экспорт;
- `weights/screening/**/best_map.pt`, `weights/final_seeds/**/final.pt` и `last.pt`
  — PyTorch checkpoint, исключённые из Git.

Экспорт этого каталога не подключается к backend сам. Сначала нужно
проанализировать `final_comparison.json`; если `beats_active_reference=false`,
активную модель следует оставить без изменений.
