# Этап 2: Flip TTA и потоковый k-reciprocal reranking

Этап не меняет веса OSNet. Проверены четыре режима на тех же calibration и
validation identity, что использовались в этапе 1:

1. baseline;
2. Flip TTA;
3. baseline + reranking;
4. Flip TTA + reranking.

Полный результат 108 calibration-конфигураций и четырёх финальных режимов сохранён
в [`results.json`](results.json). Эксперимент воспроизводится командой:

```bash
.venv/bin/python -m backend.compare_inference
```

## Потоковое ограничение

За основу взят метод [Re-ranking Person Re-identification with k-Reciprocal
Encoding](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html)
и [официальная реализация авторов](https://github.com/zhunzhong07/person-re-ranking).

Оригинальный алгоритм адаптирован под требования организаторов:

- заранее строится только граф `gallery → gallery`;
- при поиске используется один текущий query;
- другие test query не загружаются в реранкер и не влияют на результат;
- Jaccard distance смешивается с исходной cosine distance;
- параметры подбираются только на calibration.

## Сетка параметров

- `k1`: 10, 20, 30;
- `k2`: 1, 3, 6;
- `lambda`: 0.2, 0.3, 0.5;
- отказ: raw cosine или reranking score.

Критерий выбора на calibration:

```text
0.45 * mAP@10 + 0.10 * (0.7 * F1 + 0.3 * TNR)
```

Validation не использовалась для подбора параметров.

## Результаты validation

| Режим | mAP@10 | Candidate F1 | TNR | Балл кандидатов |
|---|---:|---:|---:|---:|
| Baseline | 79,02% | 72,55% | 79,03% | 74,49% |
| Flip TTA | 79,18% | 74,82% | 77,42% | 75,60% |
| Baseline + reranking | **81,47%** | 72,86% | **79,03%** | 74,71% |
| Flip TTA + reranking | 80,75% | **75,12%** | 77,42% | **75,81%** |

Победитель по зафиксированному quality-критерию: baseline + reranking.

Выбранные параметры:

```text
k1 = 20
k2 = 3
lambda = 0.5
refusal confidence = maximum raw cosine
cosine threshold = 0.5954670310020447
```

Реранкинг поднял validation mAP@10 на 2,45 процентного пункта и full mAP на
2,46 пункта. Отказ оставлен на raw cosine: reranking score хуже разделял запросы
с существующей парой и open-set query.

## Скорость

Локальный CPU, batch=1, включая JPEG decode, crop, preprocessing, ONNX и L2:

| Режим encoder | Median | p95 |
|---|---:|---:|
| Baseline | 16,06 мс | 16,67 мс |
| Flip TTA | 39,79 мс | 40,80 мс |

Для выбранного реранкинга:

- построение validation gallery-графа: около 0,105 с;
- обработка одного query: около 0,315 мс.

Flip TTA не активирован: выигрыш без реранкинга составил только 0,17 п.п.
mAP@10, комбинация с реранкингом уступила победителю, а p95 локально превысил
целевые 40 мс.

## Активный пайплайн

MVP и экспорт используют k-reciprocal порядок. `submission.csv` содержит результат
после реранкинга. `embeddings.npy` содержит исходные OSNet-эмбеддинги без TTA.
Решение об отказе и `confidence` в `candidates.csv` основаны на максимальном raw
cosine, а не на reranking score.
