# OSNet-AIN x1.0

Каталог хранит независимые варианты дообучения `vehicle-reid-0001 / OSNet-AIN x1.0`.
Каждый вариант имеет свои notebook, описание, веса и результаты.

| Вариант | Статус | Отличия |
|---|---|---|
| [`variant_01_initial`](variant_01_initial/) | Сохранён как baseline | P=8/K=4, один LR, CE + Triplet + consistency |
| [`variant_02_hpo_bnneck_supcon`](variant_02_hpo_bnneck_supcon/) | Обучен, checkpoint эпохи 5 активен в MVP | P=16/K=2, разные LR, 16→4→2 staged HPO, BNNeck, SupCon |
| [`stage_02_inference`](stage_02_inference/) | Выполнен, reranking активен в MVP | Flip TTA и потоковый k-reciprocal reranking |
| [`variant_03_gem`](variant_03_gem/) | Обучен; GeM отклонён | Square победил letterbox; AvgPool `0.800250` против GeM `0.791110` mean mAP@10 |
| [`variant_04_training_strategy`](variant_04_training_strategy/) | Реализован, ожидает обучения | MixStyle, hard negatives, dynamic metric weight, Circle Loss, три seed |

Новый вариант не заменяет активную модель автоматически. Сначала сравниваются mAP,
Rank-1/5, Candidate F1, TNR, устойчивость и ONNX parity.
