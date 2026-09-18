# Вариант 1: первое development-обучение

## Что было сделано

- инициализация из стокового `vehicle-reid-0001`;
- 925 train identity, 307 calibration identity, 309 validation identity;
- PK-batch `P=8`, `K=4`, batch size 32;
- AdamW, LR `3e-4`, weight decay `5e-4`, cosine scheduler, 20 эпох;
- `CrossEntropy + 1.0 × BatchHardTriplet + 0.2 × consistency`;
- blur, JPEG, downscale, Random Erasing и lower-center occlusion;
- checkpoint выбирался только по validation mAP.

## Результат

Лучший mAP-checkpoint: эпоха 18.

| Метрика | Calibration | Validation |
|---|---:|---:|
| mAP | 74,36% | **77,40%** |
| Rank-1 | 74,80% | **77,33%** |
| Rank-5 | 87,80% | **88,66%** |
| mINP | 67,04% | 70,52% |
| Candidate F1 | 57,78% | 59,77% |
| TNR | 47,54% | 46,77% |

Calibration cosine threshold: `0.6030338406562805`.

Эпоха 20 имела чуть меньший mAP (`76,94%`), но лучшие Candidate F1 (`62,57%`) и TNR
(`59,68%`). Её веса не сохранились, потому что первая версия сохраняла только лучший mAP.
Это исправлено в варианте 2.

## Файлы

- `train_osnet.ipynb` — выполненный notebook с выводами;
- `weights/best_epoch_18.pt` — PyTorch checkpoint;
- `weights/osnet_epoch_18.onnx` — encoder, который сейчас подключён к MVP;
- `results/history.json` — история 20 эпох;
- `results/metrics.json` — повторная ONNX-оценка и latency.

Notebook в этой папке использует `STOCK_MODEL` для повторной инициализации, а не текущий
дообученный ONNX.
