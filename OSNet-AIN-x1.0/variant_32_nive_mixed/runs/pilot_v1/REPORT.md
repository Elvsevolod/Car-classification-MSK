# v32 — N0/N1: mixed NiVe pilot

Статус: complete. Исходная validation не оценивалась; v25 не изменён.

| Модель | Выбранный update | Mean raw mAP | Mean fixed-graph mAP |
|---|---:|---:|---:|
| N_ref | 0 | 0.832842 | 0.844099 |
| N0_target_extra | 800 | 0.833170 | 0.844361 |
| 50% N_ref + 50% N0_target_extra | 800 | 0.833888 | 0.845909 |
| N1_nive_mixed | 0 | 0.832842 | 0.844099 |
| 50% N_ref + 50% N1_nive_mixed | 0 | 0.832842 | 0.844099 |

Рекомендация screening: **stop**, не решение о релизе.

Все назначенные checkpoint:

| Arm | Update | Raw | Graph |
|---|---:|---:|---:|
| N0_target_extra | 0 | 0.832842 | 0.844099 |
| N0_target_extra | 400 | 0.833500 | 0.841344 |
| N0_target_extra | 800 | 0.833170 | 0.844361 |
| N0_target_extra | 1200 | 0.831079 | 0.842637 |
| N0_target_extra | 1600 | 0.832107 | 0.840464 |
| N0_target_extra | 1800 | 0.831284 | 0.840066 |
| N1_nive_mixed | 0 | 0.832842 | 0.844099 |
| N1_nive_mixed | 400 | 0.812937 | 0.826124 |
| N1_nive_mixed | 800 | 0.811222 | 0.822410 |
| N1_nive_mixed | 1200 | 0.807804 | 0.823739 |
| N1_nive_mixed | 1600 | 0.806673 | 0.821080 |
| N1_nive_mixed | 1800 | 0.821663 | 0.831807 |

## Бюджет и ограничения

1800 updates на ветвь: 1600 joint + 200 target-only.
108800 логических предъявлений на ветвь; N1: 57600 organizer + 51200 NiVe; N0: 108800 organizer.
Clean/robust дают 217600 image-forwards, а не удвоенное число уникальных фото.
Aux-loss .25 не означает 25% градиента; реальные нормы backbone-градиентов записаны каждые 25 updates.
Основные organizer-пачки/аугментации одинаковы; общее target-exposure различается намеренно.
BN обновляется одинаково: aux robust, затем main robust; clean eval; tail только target в обеих ветвях.
Primary draws используют одни identity и не являются независимыми датасетами. Это адаптивный development-screening.
Frozen step/50:50 mixture не подбирают граф/порог. Active v25 не является inner-контролем (его R1 видел holdout).
Negative pilot не доказывает бесполезность всех внешних данных; positive pilot не означает прирост над v25.
NiVe test/маски не обучались; bbox/evaluator не менялись. Номерная устойчивость и GPU не подтверждены.

## Покрытие NiVe

17070/17070 уникальных train-фото использовано.
Подробности: manifest.json, training/*/history.json, evaluation/*/metrics.json, frozen_selection.json, results.json.
