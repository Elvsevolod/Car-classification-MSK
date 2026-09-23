# Как улучшить OSNet и качество поиска: исследование и программа экспериментов

Дата проверки: 22 сентября 2026 года. Статус: исследование, не внедрение.

> Дополнение после обсуждения 22.09.2026: пользователь решил сохранить bbox, метки и все train-примеры
> организаторов без исправлений/исключений. Предложение D1 ниже остаётся частью исторического исследования,
> но **не включено в согласованный запуск**. Подготовлен отдельный
> [variant 14 notebook](OSNet-AIN-x1.0/variant_14_controlled_ablations/train_osnet_ablation_suite.ipynb):
> 22 контролируемые конфигурации на неизменённых данных, обучение запускает пользователь.
> Внешний teacher и faithful RPTM пока не реализованы; T1 — самостоятельный similarity sampler.

## Короткий вывод

**Продолжать развитие OSNet разумно. Но первым экспериментом должна быть проверка качества обучающих bbox/identity, а не ещё один loss или ансамбль.**

В ходе исследования обнаружены конкретные несоответствия между вырезанным автомобилем и другими изображениями той же identity. Они есть не только в validation, но и в development train. При таких примерах cross-camera SupCon и hard mining могут усиливать ошибочный обучающий сигнал.

После аудита данных наиболее обоснованная последовательность: согласованный контроль обучения → BatchNorm/consistency → разнообразие позитивов и негативов → angular-margin classification → разрешение и локальные признаки → при необходимости дистилляция в OSNet. Цветовая ветка — отдельная перспективная гипотеза: действующий encoder почти полностью инвариантен к поканальному аффинному преобразованию RGB, что подтверждено проверкой ONNX.

Ниже отделены факты из файлов, новые диагностические наблюдения и предлагаемые эксперименты. Ни один предложенный прирост не считается доказанным.

## 1. Объём и источники проверки

Изучены ТЗ, все 54 ответа организаторов, официальный evaluator, текущий inference/training-код, сводки и результаты всех найденных семейств экспериментов: OSNet variants 01–04, 08, 12, 13; inference stage 02; audits 07, 09–11; обе версии ResNet; обе версии CLIP; два benchmark готовых масочных детекторов и обе версии YOLO11.

Дополнительно программно проверены **70 файлов training history, суммарно 1544 записи**, а при финальной сверке — ещё три fold-history OOF по пять эпох: всего **73 файла / 1559 записей**. Это не 73 независимых эксперимента: среди историй есть этапы HPO, продолжения, несколько seed и разные стадии обучения. Evidence JSON хранит исходный индекс 70 историй; поздняя сверка OOF опирается на его отдельный report.json и fold histories.

Приоритет доказательств: фактический JSON/история/экспорт → актуальный код → README → старые планы и память. Например, прежние записи «stage 4 не обучен», «YOLO holdout не запускался», «masked HPO подготовлен» уже не отражают текущие результаты. У variant 13 README тоже остался на стадии подготовки, но финальный RESULTS.md записан 22 сентября в 15:42; его готовые результаты включены в этот отчёт.

Основные локальные источники:

- [ТЗ](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/7. Фалькон Тех.pdf>), [ответы организаторов](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/ORGANIZER_QA.md>), [официальный evaluator](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/evaluate.py>).
- [Активные метрики](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/artifacts/baseline_metrics.json>), [разбиения](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/artifacts/splits.json>).
- [Архитектура OSNet](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/training/osnet.py>), [HPO и loss](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/training/hpo.py>), [аугментации](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/training/pipeline.py>), [локальный протокол](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/backend/evaluate.py>).
- [Новые численные диагностики и hashes](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNET_SEARCH_QUALITY_EVIDENCE_2026-09-22.json>).
- [Кандидаты аудита train](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNET_SEARCH_QUALITY_TRAIN_AUDIT_2026-09-22.json>).

В этой работе не запускалось обучение, не менялись исходные изображения, CSV, маски, веса, MVP или параметры поиска. Новые файлы — только исследовательские материалы. Закрытый test не использовался для подбора.

## 2. Ограничения ТЗ, которые определяют полезность улучшений

| Требование | Следствие для исследования |
|---|---|
| Поиск конкретного ТС, включая ранее не встречавшиеся identity | Точность классификации train не заменяет проверку retrieval на новых identity |
| BBox предоставлен, отдельная детекция ТС не требуется | Сначала проверять соответствие bbox целевому ТС; нельзя незаметно заменить постановку поиском любой машины на полном кадре |
| Каждый query обрабатывается независимо | Запрещены cross-query clustering, query expansion через другие запросы, их накопление и совместное test-time обучение |
| Вся gallery статична и доступна | Gallery-only граф и per-query reranking допустимы; при 750 объектах ANN не является обязательной оптимизацией |
| camera_id — обучающая/оценочная метаинформация | Использовать для sampler и cross-camera оценки, но не передавать модели на inference |
| Основная метрика — официальный mAP@10 | Не смешивать её с историческим mAP, full mAP и mINP; проверять реальные файлы submission |
| submission всегда содержит 10 gallery ID на query | Отказ не сокращает submission; он означает отсутствие строк query в candidates |
| Отказ оценивается на уровне query по лучшему кандидату | Отдельно проверять порядок и confidence; рост F1 при падении TNR может быть невыгоден |
| Offline inference, суммарно до 2 GB участвующих весов | Training-only teacher можно не включать в deployment; все реально поставляемые участвующие веса учитывать вместе |
| RTX A5000, 24 GB; полный extract входит в замер | Учитывать чтение, декодирование, bbox, preprocessing, все encoder и L2; CPU-цифры не доказывают прохождение GPU-нормативов |
| Полные баллы: latency ≤40 ms, throughput ≥100 FPS | Усложнение модели оценивать по итоговому score, а не только mAP |
| Внешние воспроизводимые данные/веса разрешены | Доступность, происхождение, лицензия, версия и checksum — обязательный предварительный фильтр |

Качество даёт `45*mAP@10 + 10*(0.7*F1+0.3*TNR)` баллов. Поэтому +1 процентный пункт mAP — это +0.45 балла; +1 п.п. F1 — +0.07; +1 п.п. TNR — +0.03. Потеря баллов за скорость может перекрыть небольшой retrieval gain.

Важная тонкость evaluator: junk — **та же identity и та же камера**, а не все машины с камеры query. Запрос без валидного позитива исключается из mAP/Rank. Код очищает переданный список из десяти ID; недостающий хвост из позиций 11+ ему неизвестен. Локальные проверки должны повторять именно фактический evaluator. GT-фильтрация кандидатов на inference недопустима.

Wrong-match принятие известного query в официальной query-level схеме считается FP, но не дополнительным FN. Поэтому привычные формулы классификации, подставленные вместо организаторских, могут давать другие F1.

## 3. Что является текущей точкой отсчёта

Действующий файл: `models/osnet_ain_x1_0_vehicle_reid_hpo_best_map.onnx`.

SHA-256: `01466f503232467224774b6e3bafe6c4393b1de30b47b1bd714908a62f2006a2`.

Размер 8 754 346 bytes, около 8.35 MiB. Вход RGB 208×208, square resize, выход 512D. Reranking: k1=20, k2=3, lambda=0.5, без flip TTA. Текущий порог `0.5948754549026489`; в старых описаниях встречается другое значение.

| Метрика | Calibration | Validation |
|---|---:|---:|
| Query / gallery | 307 / 895 | 309 / 896 |
| Известные / неизвестные query | 246 / 61 | 247 / 62 |
| Raw cosine mAP@10 | 75.6050% | 79.0162% |
| С действующим reranking | 78.7185% | **81.4689%** |
| Raw Rank-1 | — | 78.9474% |
| Reranked Rank-1 | — | 80.1619% |
| Reranked Rank-5 | — | 88.2591% |
| Candidate F1 | — | 72.8606% |
| TNR | — | 79.0323% |
| 0.7F1+0.3TNR | — | 74.7121% |

На validation: TP=149, FP=38, FN=73, TN=49; 13 FP относятся к неизвестным query. Raw full mAP=80.3597%, mINP=74.0733% — отдельные диагностические метрики, не основной конкурсный результат.

Это хороший рабочий baseline, но **не независимая оценка будущего качества**: outer validation уже участвовала в истории выбора checkpoint и неоднократного сравнения методов.

## 4. Разбор проведённых экспериментов

### 4.1. OSNet: обучение и preprocessing

| Эксперимент | Что фактически получилось | Корректный вывод |
|---|---|---|
| Variant 01, исходный fine-tune | 925 ID / 5717 изображений; CE+triplet+consistency, P8K4, 20 эпох; best epoch 18; исторический mAP≈77.40% | Первая рабочая версия. Старая семантика метрик не позволяет напрямую вычитать результат из нынешних 81.47% |
| Variant 02, HPO BNNeck/SupCon | 16 trials, top-4 до 20 эпох, top-2 на 3 seed до 30. Trial 5: историческое inner mean 84.6167%, std 1.4759 п.п.; trial 12: 84.5428%, std 1.9038 п.п. | Разница лидеров всего 0.074 п.п. Отбор одного trial не доказывает уникальность его коэффициентов |
| Variant 02, selected run 02 | Обучение на всех 925 ID; текущий checkpoint epoch 5 из 30. Старый отчёт mAP≈81.06%, официальный raw пересчёт 79.0162% | Разница чисел связана в том числе со сменой метрик, а не с деградацией ONNX |
| Inference stage 02 | Raw 79.0162%; flip≈79.18%; rerank 81.4689%; flip+rerank≈80.75%. 108 calibration конфигураций | Reranking полезен; flip не дал устойчивого выигрыша и увеличил CPU-время |
| Variant 03, letterbox только на inference | Square+r=81.47%, letterbox+r≈66.76% на том же checkpoint | Это сильный train/inference mismatch. Не доказано, что согласованно обученный letterbox плох |
| Variant 03, matched avg/GeM fine-tune | Avg 80.0250±0.5829%; GeM 79.1110±0.7347%, по 3 seed | GeM в этом режиме хуже; обе группы ниже MVP. Не менять pooling «на лету» |
| Variant 04, MixStyle/hard negatives/dynamic weights/Circle | Inner baseline 94.8353%; MixStyle 94.1944%; hard negatives 94.7580%; dynamic 94.4568%; Circle 94.2552%; best epoch 1 у всех | Screening использовал encoder, уже видевший inner holdout. Эти 94–95% не независимая оценка |
| Variant 04, final | Выбран baseline; control и strategy фактически один рецепт. Среднее 79.6863±0.7137%, по 3 seed, checkpoint epoch 1 | Это не шесть независимых подтверждений новой стратегии. Успешная инфраструктура, но улучшение обучения не установлено |
| Variant 08, masked HPO | Полностью выполнен: 16 trials, 6 complete/10 pruned; trial 5 inner mean 81.6313±0.6992%. Selected epoch 16 выбран по calibration | README «не запускалось» устарел. Inner split иной; напрямую сравнивать с прежними 84.62% нельзя |
| Variant 08, итог на полном validation | Raw 75.6399%; rerank **77.6518%**; F1 73.5577%, TNR 74.1935% | Эта masked-система не лучше MVP. Почти совпадает с inference-only YOLO masking, но сравнение не причинно чистое |

Подробные первоисточники: [variant 02](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/variant_02_hpo_bnneck_supcon/results>), [variant 03](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/variant_03_gem/results>), [variant 04](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/variant_04_training_strategy/results>), [masked HPO](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/variant_08_masked_hpo/results/masked_run_01>).

Выбранный рецепт: encoder LR≈1.194e-4, head LR×10, weight decay≈6.07e-5, P16K2, label smoothing 0.1, SupCon temperature 0.1, metric weight≈1.4949, consistency≈0.2257, warmup 2 эпохи. Это отправная точка, не доказанный глобальный оптимум.

Дополнительные ограничения старого HPO:

- Seed обучения одновременно влияет на построение query/gallery и open-set части. Его std смешивает стохастику обучения и сложность протокола.
- При продвижении 8→20→30 эпох меняется горизонт LR schedule. «Продолжение» не равно первым N шагам одного заранее фиксированного расписания.
- В `fit_selected` validation проверяется каждую эпоху и используется для best checkpoint. Отдельного нетронутого outer теста после этого нет.
- GeM и stage 4 стартовали из уже локально обученного encoder; чистый stock-start эксперимент может вести себя иначе.
- Лучшая эпоха 1 или 5 сама по себе не доказывает ни недообучение, ни переобучение. Важны число обновлений, data exposure, train loss и независимая retrieval-кривая.

### 4.2. Другие encoder

| Эксперимент | Результат | Что он действительно проверяет |
|---|---|---|
| ResNet50-IBN v05 | Outer mean 27.2971±1.7324%; около 285 обновлений по перенесённому OSNet-рецепту | ImageNet-инициализация, короткий неподходящий рецепт; не доказательство общего превосходства OSNet-архитектуры |
| ResNet50-IBN v06 | Чистый frame-grouped inner; бюджет 4000 steps. SupCon варианты 41.52–42.80%; triplet **48.1342%** на step 3200 | Более корректный контроль, но gate 50% не пройден; final seeds/outer/export не проводились |
| CLIP-ReID v01, vehicle transfer | Stock CLIP 37.5252%, stock OSNet 65.2115%; обученный CLIP 71.8209%, чистый OSNet control **77.8715%** на одном inner | Реальный vehicle checkpoint CLIP дообучился, но OSNet сильнее в этом сравнении |
| CLIP-ReID v02, HPO | Best trial 11, epoch 33, **75.7994%**; 3 seed mean **75.5566±0.2380%** | HPO улучшил CLIP относительно v01, но не обошёл чистый OSNet control |
| CLIP в общем outer audit 11 | Raw 74.0080%, reranked **74.9476%** | Это уже сравнение готовых систем на общей validation; CLIP обучен на 741 ID, OSNet на 925 |

Источники: [ResNet v06](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/ResNet50-IBN/variant_06_controlled_training/results>), [CLIP v01](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/CLIP-ReID-ViT-B-16/variant_01_vehicle_transfer/results>), [CLIP HPO](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/CLIP-ReID-ViT-B-16/variant_02_hpo/results>).

Вывод: менять OSNet на ещё один ImageNet-backbone с тем же бюджетом сейчас малообоснованно. Источник предобучения и процедура адаптации могут быть важнее размера архитектуры.

### 4.3. Маски и детекторы

| Проверка | Результат | Ограничение |
|---|---|---|
| Audit 07, ручные маски | 126 crops: 32 query, 94 gallery, 26 known, 6 unknown. Raw 97.2283→96.6728%; rerank 99.0476→99.4505% | Маленький лёгкий набор с насыщенным Rank-1; не доказательство отсутствия зависимости от номера |
| Benchmark 01: YOLOv9-T / YOLO11n | Покрытие ручных пикселей 12.95% / 35.37%; ни одной из 140 областей не покрыто на ≥90% | Детектирование пластины не равно покрытию всей пикселизации; ручные области включают не только номера |
| Benchmark 02: DeepMosaics / EgoBlur | DeepMosaics даёт больше покрытия, но много лишнего; EgoBlur LP≈22.79%, combined≈24.69% покрытия | Готовые specialist-модели не решают надёжно локальную задачу анонимизации |
| YOLO11 pilot 01 | Train 160 crops/191 regions; validation 40/49; holdout 40/45. Best detector epoch 16, mAP50≈87.41%, mAP50:95≈52.34% | Detector mAP — не ReID mAP |
| YOLO calibration | conf=0.2, margin=0.1; validation coverage 84.11→91.51%; regions≥90% 30/49→43/49 | Порог выбран на validation детектора, не на retrieval validation |
| YOLO holdout — уже выполнен | Coverage **92.30→97.60%**; regions≥90% **37/45→42/45**; все области закрыты≥90% в **32/40→37/40** crops | Вне эталона 0.84→2.50% площади crops. Нет negative images: false-mask rate на чистых изображениях неизвестен |
| YOLO variant 02, полный ReID validation | MVP+r 81.4689→≈77.66%; Δ≈−3.81 п.п., CI [−6.07;−1.60] | Это одновременно закрытие сигнала, лишней площади и domain shift |
| Audit 09, исправленный ручной эталон | 157 rectangles; masked-trained raw: original 98.6395%, manual 98.5806%, YOLO 98.3189%; reranked почти насыщен | Нужна большая новая масочная оценка; нельзя выбирать модель по этому маленькому набору |

Источники: [benchmark 01](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/Mask-Detectors/benchmark_01_pretrained/RESULTS.md>), [benchmark 02](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/Mask-Detectors/benchmark_02_mosaic_egoblur/RESULTS.md>), [YOLO holdout](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/YOLO11/variant_01_anonymized_regions/runs/pilot_01/calibration_v1/holdout_final.json>), [full masking ablation](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/YOLO11/variant_02_mvp_reid_ablation>), [audit 09](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/audit_09_model_mask_comparison/results/comparison_v1/comparison.json>).

В masked HPO обе ветки получают маскированные изображения. Он **не проверял полноценное обучение на смеси original+masked**. Успешная детекция анонимизированных областей не означает, что их тотальное закрытие оптимально для retrieval.

Нужна матрица 2×2 train original/mixed × inference original/masked, с одинаковыми split, бюджетом и checkpoint selection. Масочный стресс-тест обязан закрывать полную область номера, но следует отдельно измерять вред от чрезмерного расширения и от закрытия лиц/посторонних машин.

### 4.4. Reranking, ансамбль, обучаемая голова

| Система, общий validation | mAP@10 | F1 | TNR | Вывод |
|---|---:|---:|---:|---|
| MVP | 81.4689% | 72.8606% | 79.0323% | Контроль |
| Audit 10, neighbor support | 81.7191% | 72.8606% | 79.0323% | +0.2502 п.п.; одна дополнительная правильная top-1 |
| Audit 11, OSNet+CLIP | 81.3549% | 80.1782% | 64.5161% | Прирост calibration не перенёсся; validation Δ−0.1140 п.п. |
| Variant 12, ranking-only | 81.7764% | 72.8606% | 79.0323% | +0.3076 к MVP, но лишь **+0.0574 к neighbors** |
| Variant 12, ranking+новый отказ | 81.7764% | 73.5577% | 74.1935% | Итоговая метрика отказа хуже: 73.7484 против 74.7121% |
| Variant 13, OOF | 81.8113% | 72.8606% | 79.0323% | +0.0922 п.п. к neighbors; преимущество OOF не подтверждено |
| Variant 13, matched in-sample | 81.9056% | 72.8606% | 79.0323% | Численно лучший из этих скалярных reranker, но маленький локальный gain |

Для ансамбля 95% paired CI прироста mAP: [−2.3160; +2.3816] п.п.; для pair-head относительно neighbors: [−0.1026; +0.2358] п.п. Эти интервалы условны на фиксированной gallery и не учитывают всю историю выбора моделей.

Ансамбль использовал 25% OSNet / 75% CLIP в weighted concatenation и union top-50. На calibration top-1 исправлялся только CLIP в 14 случаях, только OSNet — в 26. Union увеличивал покрытие, но этого не хватило для устойчивого общего улучшения. CPU extract вырос примерно с 17.23 до 59.72 ms; веса 337.66 MiB.

Pair-head — линейный 8→1 классификатор скалярных признаков, а не визуальный эксперт, заново рассматривающий детали автомобилей. MLP с координатами embeddings не выиграл. Маленькая голова почти не меняет top-1 относительно neighbor-контроля.

OOF variant 13 уже выполнен: три независимых stock-start encoder, frame/identity grouping, фиксированные 5 эпох с LR-горизонтом 30, только скалярные межмодельно сопоставимые признаки, matched in-sample контроль, старый отказ. В каждом fold было 190 updates против 285 у исторического MVP epoch 5. Raw OOF mAP по трём разным held-out частям: 76.5542%, 73.8828%, 70.1955%; это не seed-variance на одном наборе.

OOF-голова обучалась 7 фиксированных эпох; beta=0.1 выбрана на calibration. Gain к neighbors +0.0922 п.п. с 95% CI [−0.0174; +0.2305]; к previous_head +0.0349 [−0.0596; +0.1350]; к matched in-sample −0.0943 [−0.2231; +0.0227]. Преимущество OOF не установлено. Более сложные обучающие пары действительно появились: OOF hit@50 99.19/97.15/96.34% против 100/99.59/100% у MVP на тех же q/g, однако в конечный gain это почти не превратилось.

Полный запуск занял около 680 s, обучение encoder — 470 s; inference-веса по-прежнему 8.35 MiB, CPU reranking median/p95 0.317/0.633 ms. **Не нужно повторно реализовывать OOF или считать его ещё не проверенной основной гипотезой.** Обнаруженные ошибки train могут ухудшать и его признаки, но это объяснение пока не проверено отдельным экспериментом.

Источники: [audit 10](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/audit_10_rerank_score_control/results/run_01/RESULTS.md>), [audit 11](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/audit_11_osnet_clip_ensemble/results/run_01/RESULTS.md>), [variant 12](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/variant_12_pair_reranker/results/run_01/RESULTS.md>), [результаты variant 13](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNet-AIN-x1.0/variant_13_oof_pair_reranker/results/run_01/RESULTS.md>).

## 5. Новые находки: данные важнее ещё одного коэффициента

### 5.1. Четыре сильных подозрения на неверный target crop в validation

Я воспроизвёл mAP@10=81.4689% из сохранённых embeddings и официального evaluator. У пяти из 247 известных query нет ни одного правильного автомобиля в raw top-50. Затем просмотрел crops и полные исходные кадры.

| Query, начало ID | Vehicle ID | Первая GT-positive позиция | Что видно |
|---|---:|---:|---|
| cd6712e74b… | 1463 | 739 | BBox вырезает тёмный седан; другие снимки identity — синий SUV. Синий SUV виден на полном query вне основной вырезанной области |
| 5a9d755ffd… | 180 | 580 | BBox [653,825,1017,254] вырезает нижнюю чёрную машину; GT-позитивы — белый фургон, который виден выше на том же полном кадре |
| 6a35d140d5… | 714 | 769 | Crop содержит переднюю часть чёрного автомобиля; GT-позитив — серебристый SUV с боковой рекламой, видимый в центре полного кадра |
| 290fcee71d… | 1184 | 94 | Crop показывает крупный SUV, GT-позитив — другой автомобиль; на полном кадре есть отдельная машина, визуально согласующаяся с остальными снимками identity |
| a76c648b5c… | 405 | 179 | Несколько перекрывающихся машин, человек, яркая рекламная машина. Это скорее загрязнение crop/окклюзия; автоматически объявлять ошибкой identity нельзя |

[Визуальное сопоставление query / неверного top-1 / GT-positive](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNET_SEARCH_QUALITY_ERRORS_2026-09-22.png>).

Это проверка соответствия визуальных объектов, а не чтение номерных знаков. Перед исправлением нужна ручная проверка всей identity, исходного кадра и эталонного CSV. Пример с ID 180 особенно показателен: проблема не в неспособности OSNet сопоставить два ракурса одного объекта — модель получает другой объект.

**Не следует удалять эти query из validation, менять GT и публиковать «улучшенный score».** Основная оценка остаётся на исходных данных. Отдельная маркированная диагностическая подвыборка допустима, но не заменяет benchmark. Четыре query — максимум 4/247≈1.62 п.п. их индивидуального вклада в среднее; это не установленный потолок всего датасета и не измеренный выигрыш исправления.

### 5.2. Такие же случаи обнаружены в train

Использован существующий cache из variant 12: 3600 изображений, 925 development identity. Для каждого изображения вычислено среднее cosine с изображениями той же identity с других камер. Выбраны шесть разных identity с самым низким значением; затем визуально просмотрены все доступные снимки этих identity из train.csv.

| Vehicle ID | Отмеченный image_id | Наблюдение |
|---|---|---|
| 936 | 64017d118a8a40d4818bdfc29999fe8e | Яркий рекламный автомобиль среди изображений тёмного SUV |
| 1317 | af9c329528224fcba61d9856282a1b88 | Грузовой автомобиль занимает crop среди изображений тёмного седана |
| 538 | 8d90b954abde478bb14ca0404f8aa1e2 | Белый SUV среди изображений чёрного седана |
| 1476 | 77bd3557bf6a42f9ac8f375020f5abb4 | Красный автомобиль среди изображений рекламного кроссовера |
| 1513 | 7a450c66d40c425aac9c3c46ba01ffbb | Жёлтое такси среди изображений тёмного универсала/минивэна |
| 1118 | 05cd40d3848e46d4b5f520fc743be8dd | Бежевый седан доминирует в crop среди изображений тёмного SUV |

[Контактный лист всех шести identity](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNET_SEARCH_QUALITY_TRAIN_AUDIT_2026-09-22.png>), [100 кандидатов для последующего просмотра](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNET_SEARCH_QUALITY_TRAIN_AUDIT_2026-09-22.json>).

Это **целенаправленно отобранные аномалии**, не случайная выборка. Нельзя утверждать, что все или определённая доля 5717 обучающих изображений ошибочны. Низкий cosine также может означать честную сложную пару, ночь, окклюзию, редкий ракурс. Cache получен encoder, уже видевшим train; такой скоринг пригоден для triage, но не для автоматической очистки и не для несмещённой оценки label noise.

Почему это особенно важно для OSNet:

- P16K2 даёт каждому anchor ровно один позитив. Если это другой автомобиль из ошибочного bbox, весь положительный сигнал anchor противоречив.
- Batch-hard усиливает наиболее далёкие positive-пары, среди которых могут быть именно ошибки.
- Большой SupCon coefficient сильнее стягивает несовместимые объекты.
- Большее разрешение не исправляет факт выбора другой машины.
- XBM без защиты от шума будет многократно возвращать ошибочный образец в последующие batch.

**Первый приоритет:** train-only ручной аудит подозрительных crop с manifest решений. Сначала сравнить исходный train и train без подтверждённых неверных примеров, не меняя validation. Коррекцию bbox хранить отдельным overlay; не переписывать оригиналы. При неоднозначности исключать из экспериментального train, а не придумывать новый ID. Проверять, что после фильтрации sampler всё ещё имеет достаточно корректных изображений и камер на identity.

Пакет примеров для вопроса организаторам можно подготовить отдельно; в этой работе сообщения им не отправлялись.

## 6. Что ещё ограничивает действующий OSNet

### 6.1. Это не обычный person-OSNet из Model Zoo

Наш backbone соответствует специализированному public vehicle encoder: ранний InstanceNorm, omni-scale блоки и два 256D FC+BN выхода, объединённые в 512D. Затем локальный fine-tune добавляет BNNeck и classifier. Подмена на стандартный `osnet_ain_x1_0` из другой библиотеки не гарантирует совпадения графа, pretraining или embeddings.

[OpenVINO vehicle-reid-0001](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/README.md) документирует 208×208, 512D и примерно 2.18 млн параметров. Его VeRi776 результаты относятся к другому датасету и протоколу. Название «OSNet» не делает ImageNet/person/vehicle checkpoint взаимозаменяемыми.

### 6.2. Проверенная потеря поканального аффинного сигнала

В `VehicleOSNet` перед первой свёрткой стоит `InstanceNorm2d(3, affine=True)`. В упрощении для каждого канала:

`IN(x_c) = gamma_c * (x_c - mean(x_c)) / sqrt(var(x_c) + eps) + beta_c`.

Поэтому при положительном масштабе `x'_c=a_c*x_c+b_c` результат почти не меняется. Обучаемые gamma/beta постоянны между изображениями и не восстанавливают удалённые индивидуальные mean/std.

Проверка действующего ONNX на 12 детерминированно выбранных development crops:

| Преобразование нормализованного RGB | Максимальное отличие координаты embedding | Минимальный cosine с исходным |
|---|---:|---:|
| Поканальный сдвиг [+0.30, −0.25, +0.20] | 1.84e−6 | 0.99999994 |
| Масштаб [0.90,1.10,1.05] плюс тот же сдвиг | 1.60e−6 | ≈1.0 |

Источник — [численный probe](</Users/elvsevolod/Desktop/учеба/Учёба 4 курс/Хакатон мск сентябрь/Car-classification-MSK/OSNET_SEARCH_QUALITY_EVIDENCE_2026-09-22.json>). Это синтетическое преобразование тензоров без clipping, **не реалистичная перекраска и не измерение retrieval gain**. Пространственные цветовые паттерны не исчезают полностью; говорить «OSNet не видит цвет» неверно.

Мой вывод: есть основание проверить контролируемый цветовой bypass. Но устойчивость к камерам и освещению — ценное свойство, которое легко потерять. Простой inference switch `input_IN=Identity` нарушит распределение всех последующих слоёв.

Предлагаемый безопасный эксперимент: оставить основной 512D encoder, добавить малую ветку цветовых статистик/низкоуровневых признаков **до input_IN**, обучить её только на train и сравнить с нулевым весом. Например, 32D branch и нормализованная конкатенация; два заранее выбранных малых веса вместо большой сетки. Для background-heavy crop этот путь опасен: сначала аудит bbox, затем устойчивость к свету и маскам. Если branch действительно полезен, проверить сжатие обратно в 512D или distillation; не добавлять его навсегда только ради красивой гипотезы.

### 6.3. «Эпоха» существенно меньше прохода по всем изображениям

Development train содержит 925 identity / 5717 изображений. На identity 4–8 снимков, медиана 6; камер 2–8, медиана 2.

Текущий sampler P16K2 формирует **57 batch по 32**, то есть 1824 предъявления за эпоху, а не 5717. На один anchor в SupCon приходится **1 позитив и 30 негативов**.

Replay текущего sampler с seed 20260915:

| Эпох | Обновлений | Предъявлений | Уникальных изображений |
|---|---:|---:|---:|
| 1 | 57 | 1824 | 1824 / 5717 = 31.90% |
| 5 | 285 | 9120 | 4747 / 5717 = 83.03% |
| 10 | 570 | 18240 | 5517 / 5717 = 96.50% |
| 30 | 1710 | 54720 | 5715 / 5717 ≈99.97% |

Это воспроизведение нынешнего sampler, не доказательство точного исторического RNG state. Однако оценка масштаба обучения точная: «30 эпох» здесь порядка 1710 updates.

Из этого **не следует**, что нужно просто увеличить число эпох. Чистый OSNet control уже обучался до 3200 steps, достиг train accuracy≈1 и не показал монотонного роста validation. Проверять надо качество сигнала и траекторию, не только длительность.

Осмысленные ablation после проверки noisy labels:

- P8K4, batch32: 3 позитива и 28 негативов; больше позитивов, меньше identity.
- P16K4, batch64: 3 позитива и 60 негативов.
- P32K2, batch64: 1 позитив и 62 негатива — отдельный контроль количества identity.
- XBM с ограниченной очередью только train embeddings — если недостаточно in-batch negatives.

При изменении batch нельзя одновременно сохранить и число updates, и число предъявлений. Основное сравнение вести по фиксированным updates, дополнительно показывать samples-seen, время и кривые при matched data exposure. Gradient accumulation сам по себе **не** увеличивает множество сравнений внутри contrastive loss.

### 6.4. Consistency сейчас не является дистилляцией от стабильного teacher

`experiment_losses` делает:

1. `model.eval()`, no-grad embedding «clean» ветки текущей модели.
2. `model.train()`, forward robust ветки.
3. CE и metric loss на robust; consistency между post-BN robust и detached clean.

Teacher здесь — тот же изменяющийся encoder, не зафиксированная более сильная модель. Кроме того, «clean» preprocessing включает случайные преобразования; это не всегда исходный неизменённый crop. BN running statistics в eval и batch statistics в train различаются.

Это не доказанный баг: такой consistency может работать. Но есть конкретные проверяемые альтернативы:

- consistency=0 как однофакторный контроль;
- замороженные running mean/var **backbone BN**, оставляя отдельно управляемый новый BNNeck;
- небольшой head-only warmup перед изменением pretrained encoder;
- после положительного сигнала — EMA teacher или train-only пересчёт BN.

Важно для будущей реализации: простой внешний вызов `bn.eval()` недостаточен, потому что `model.train()` вызывается внутри loss. Freeze-политику нужно применять после переключения режима. Не замораживать случайно весь training graph.

Стандартные средства EMA/SWA и обновления BN есть в [PyTorch AveragedModel](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html). BN recalibration — только на разрешённом train; не на нескольких test query и не через адаптацию к их общей статистике. Новая статистика требует нового train-only эксперимента, экспорта и parity.

### 6.5. Сильные аугментации не всегда сохраняют identity-сигнал

В robust ветке присутствуют blur, downscale, JPEG, random erasing, lower-center occlusion, flip и color jitter. Вероятности random erase 0.30 и lower-center 0.35 дают вероятность хотя бы одного такого закрытия 54.5%, если соответствующие случайные события независимы.

Для машины похожей модели отличия могут находиться в дисках, рейлингах, решётке, наклейке или малом элементе кузова. Уничтожить их — не всегда полезная регуляризация. Особенно если crop уже содержит другую машину.

Предложение: отдельно проверить отключение lower-center occlusion и отдельно смесь слабой/сильной supervised-ветки. Не менять одновременно все blur/JPEG/resize/mask коэффициенты. Внешний полезный сигнал: автор Torchreid обсуждает разные эффекты random erasing в разных режимах, то есть универсального «чем сильнее, тем лучше» нет. [Комментарий автора](https://github.com/KaiyangZhou/deep-person-reid/issues/221).

### 6.6. Разрешение и локальность ещё не проверены полноценно

По development bbox медиана короткой стороны **485 px**, минимум 147 px; медиана sqrt площади≈574.8 px. В исходных данных достаточно пикселей для проверки 256/320. Но большой crop может оставаться размытым или закрытым; пиксельный размер не гарантирует полезные детали.

Вход 208 даёт последнюю пространственную карту примерно 13×13, 256 — 16×16, 320 — 20×20. Потенциальный путь — сохранить детали до pooling и дать модели локальную supervision.

Предлагаемые проверки:

- Согласованное обучение и inference 256×256 против 208×208.
- 320×320 только после сигнала на 256.
- Одно дополнительное локальное attention-pooling ответвление из последней feature map; global branch сохранить.
- Training-only локальные ID-heads с проверкой, остаётся ли выигрыш после удаления вспомогательных голов.

Грубая оценка свёрточных затрат по площади: 256/208 даёт ×1.51, 320/208 — ×2.37. Это не прогноз полного extract: decode, память, kernel efficiency и batch меняют время. Историческая неудача GeM не опровергает локальную supervision: GeM лишь меняет агрегацию, не учит отдельное соответствие деталей.

Жёсткие горизонтальные полосы из person ReID не равны «перед/зад/бок» машины при произвольном ракурсе. Attention тоже может выбрать номер или соседний объект; нужны масочные ablation и просмотр ошибок, а не только красивые heatmap.

## 7. Проекты других авторов: что переносить, а что не переносить

Проверялись первичные публикации и авторские репозитории. Ни один внешний benchmark не переводится напрямую в ожидаемые проценты на Falcon Tech. Наличие открытого кода не гарантирует доступности всех весов, данных или современного окружения.

| Источник | Полезная идея для нас | Готовность и граница переноса |
|---|---|---|
| [OSNet / Torchreid](https://github.com/KaiyangZhou/deep-person-reid), [OSNet-AIN paper](https://arxiv.org/abs/1910.06827) | Omni-scale backbone, нормализация, recipes обучения, mutual learning | Главный источник архитектуры; большинство готовых checkpoint относятся к людям |
| [OSNet-IAP + AM-Softmax](https://arxiv.org/html/2003.07618v2) | Angular-margin classification, identity sampling, head warmup; баланс дискриминативности и generalization | Исследование person cross-domain, не vehicle-результат. Даёт обоснование ablation, не обещание прироста |
| [OpenVINO deep-object-reid](https://github.com/openvinotoolkit/deep-object-reid/blob/ote/README.md) | Реализация AM-Softmax engine и связанных training tricks | Проверять конкретный commit: README/ветки исторические, это не гарантированный drop-in в наш граф |
| [FastReID](https://github.com/JDAI-CV/fast-reid), [VeRi config](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml) | Vehicle-oriented recipes, freeze/warmup, BN, tooling | Apache-2.0. VeRi config использует square256, batch64 и длительный warmup; числа нельзя механически переносить на 925 ID |
| [FastDistill](https://github.com/JDAI-CV/fast-reid/tree/master/projects/FastDistill) | Обучение компактного student от teacher | Подходящий инженерный донор для сохранения OSNet на inference |
| [regob/vehicle_reid](https://github.com/regob/vehicle_reid) | Прозрачные vehicle ablation: IBN, metric losses, разрешение, batch | Полезен как пример сравнения. В его таблицах больший batch не всегда лучше; нет универсального winning recipe |
| [RPTM, WACV 2023](https://github.com/adhirajghosh/RPTM_reid) | Выбор positive-пар с учётом геометрических отношений, а не слепой hardest-positive | MIT; есть GMS preprocessing script и vehicle результаты. Полезнее после очистки неверных пар; GMS чувствителен к blur/окклюзии |
| [Self-supervised Geometric Features, ICCV 2021](https://github.com/ming1993li/Self-supervised-Geometric) | Локальные/геометрические attention features для vehicle ReID | Направление для облегчённой OSNet local-head; старый небольшой repo, лицензию и воспроизводимость надо проверить перед заимствованием |
| [PLR-OSNet, PRCV 2020](https://github.com/AI-NERC-NUPT/PLR-OSNet) | Непосредственное расширение OSNet part-level признаками | Person ReID; использовать архитектурную идею, не переносить разбиение тела человека на автомобиль буквально |
| [UFDN, ECCV 2022](https://github.com/damo-cv/UFDN-Reid) | Разделение vehicle features, работа с неоднородностью визуальных признаков | Код и vehicle checkpoints заявлены; PyTorch1.7.1-era. Это отдельный более сложный метод, не первый пилот |
| [AICity 2021 DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT) | Crop quality, многообразие train data, сильная адаптация и complementary representations | Победитель track2, MIT. Восемь backbone, camera/view/track компоненты нельзя переносить целиком в наш streaming-контракт |
| [XBM, CVPR 2020](https://github.com/msight-tech/research-xbm) | Очередь прошлых train embeddings увеличивает число сравнений без второго encoder на inference | Реальный авторский код; stale features и шумные пары требуют контроля |
| [PRISM, CVPR 2021](https://arxiv.org/abs/2103.16047) | Noise-resistant metric learning через оценку надёжности примеров | Особенно релевантно найденным несовпадениям. Не заменяет ручную проверку: редкие ракурсы могут ошибочно выглядеть шумом |
| [LCNL, IJCV 2024](https://github.com/XLearning-SCU/2024-IJCV-LCNL) | Надёжность меток и соответствий в ReID, адаптивная работа с noisy pairs | README всё ещё относит выпуск vehicle-кода к TODO. Это научный ориентир, не готовый vehicle training script |
| [Proxy Anchor, CVPR 2020](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020) | Class proxies как альтернатива зависимости от малого batch | Авторский MIT repo; Cars196 — категории моделей машин, не тот же instance-ReID. После более прямого AM-Softmax пилота |
| [Smooth-AP, ECCV 2020](https://github.com/Andrew-Brown1/Smooth_AP) | Differentiable ranking loss ближе к поисковой цели | Оптимизирует surrogate AP, не точный официальный top10+junk протокол; малый batch и label noise остаются |
| [Light-ReID](https://github.com/wangguanan/light-reid) | Компактные representations и distillation | Для нас интереснее distillation, чем binary hashing: gallery из 750 не требует экономить каждый cosine |
| [VehicleNet](https://arxiv.org/abs/2004.06305) | Разнообразное vehicle pretraining с последующей target-адаптацией | Сначала выбрать конкретные доступные внешние источники; не начинать массовую загрузку без решения о provenance |
| [VehicleMAE, ICCV 2025](https://iccv.thecvf.com/virtual/2025/poster/1394) | View-asymmetric masked pretraining и mutual distillation; авторы описывают DiffVERI более 1.7 млн изображений | Современное направление для teacher/pretraining. В проверенных материалах не подтверждён готовый доступный нам комплект code+weights+data |
| [VehicleMAE, AAAI 2024](https://github.com/Event-AHU/VehicleMAE) | Structural/multimodal vehicle-centric pretraining | **Другая работа**, несмотря на одинаковое имя. Нельзя приписывать этому repo результаты ICCV2025 |
| [DN-ReID, CVPR 2024](https://github.com/chenjingong/DN-ReID) | Day/night-specific vehicle representation | Рассматривать только при подтверждённом ночном срезе ошибок; не универсальный следующий backbone |

### 7.1. Что брать в первую очередь

**AM-Softmax на нашем OSNet.** Нормализованные веса classifier и embeddings, `logit_y=s*(cos(theta_y)-m)`; штраф заставляет формировать угловое разделение identity. Мой стартовый пилот: scale 30, margin 0.2 с коротким ramp; остальные настройки контролируемые. Это предлагаемые параметры, не воспроизведение чисел статьи. Сначала заменить только CE-head, сохранив остальную supervision; если возникает конфликт с SupCon — отдельный контроль веса, не одновременно новая сетка из десятков комбинаций.

**RPTM-подобная работа с позитивами.** Наш camera-aware sampler уже полезен, но другой camera_id не гарантирует ни иной ракурс, ни корректный объект. Сначала верифицировать labels, затем балансировать простые и сложные cross-camera positives. Реализация собственного pose-aware sampler не должна называться воспроизведением RPTM без его GMS/threshold алгоритма.

**Небольшая локальная ветка.** Моя предлагаемая адаптация: один shared OSNet trunk, global512 и один attention/local128 head. Учить оба через identity supervision; отдельно проверить ортогональность/разнообразие только если головы действительно дублируют друг друга. Не начинать с нескольких segmentation/keypoint teacher и сложной смеси loss.

**Training-only distillation.** Более сильный teacher может давать pairwise отношения `T_ij=cos(t_i,t_j)`; OSNet учится воспроизводить отношения между изображениями, не складывать несовместимые координаты. Проекция или relational loss снимают требование одинаковой размерности. Учитель должен быть сильнее или полезнее на честном inner benchmark, устойчив к маскам и не видеть holdout identity через локальное обучение. Нынешний CLIP слабее; его использование как teacher по умолчанию не обосновано.

### 7.2. Внешние данные — отдельный этап, не бесплатный прирост

Рассматривать VeRi-776, VehicleID, VERI-Wild или VRIC как разные источники, а не одну большую безусловно полезную смесь. Официальный [VERI-Wild](https://github.com/PKU-IMRE/VERI-Wild) описывает разнообразные условия и значительный объём; это не означает автоматического соответствия нашему домену.

Перед тренировкой на любом источнике:

1. Проверить официальный origin, доступ и разрешённый способ получения, условия использования весов и данных отдельно.
2. Сохранить URL/revision/checksum, splits и лицензию; repo license не заменяет dataset license.
3. Проверить exact/near duplicates с локальным development; не трогать закрытые test labels.
4. Не смешивать одинаковые числовые ID разных датасетов; namespace identity по источнику.
5. Сначала короткий vehicle-transfer контроль, затем target adaptation с малым LR; сравнить с текущим stock vehicle initialization при равном target-бюджете.
6. Проверять исходные и непрозрачно замаскированные номера, не реконструировать их.
7. В deployment оставить только нужный encoder; вспомогательные pretraining checkpoints не включать в участвующие inference weights.

Самостоятельное masked-pretraining гигантского teacher на миллионах изображений — не ближайшая итерация. Для нашего масштаба данных более реалистичны подтверждённый готовый vehicle checkpoint и последующая дистилляция.

## 8. Чего не стоит делать сейчас

- Ещё один большой Optuna по тем же нескольким scalar weights на многократно использованной validation.
- Автоматическое удаление всех hard positives: среди них есть честные редкие ракурсы.
- Усиление batch-hard или XBM до проверки ошибочных bbox.
- Перенос letterbox/GeM/отключения IN только на inference с прежними весами.
- Объявление MixStyle/Circle бесполезными вообще на основании leaked stage 4.
- Использование исходного CLIP в ансамбле только потому, что модель крупнее.
- Полное копирование AICity camera/track-aware postprocessing.
- Увеличение top-K как основная стратегия: большинство positives уже внутри top50; 4 из 5 экстремальных промахов подозрительны по bbox.
- Генеративное super-resolution для восстановления индивидуальных деталей: оно может придумать нужные для ReID признаки.
- OCR или извлечение следов номера, «размытых символов», fingerprint пикселизации.
- Отбор confidence по F1 без TNR и конкурсной формулы.
- Объявление масочного набора с 26 known query независимым доказательством устойчивости.

## 9. Конкретная программа следующих экспериментов

Это предложение порядка работы. В этой исследовательской сессии перечисленные обучения **не выполнялись**.

### 9.1. Нулевой этап: сделать результат интерпретируемым

1. Зафиксировать текущий MVP, stock initializer, evaluator, dataset hashes, split и конфигурацию. Не менять рабочий API.
2. Сверить минимум 100 найденных train-кандидатов, дополнительно случайную контрольную выборку. Просматривать полный кадр, crop и остальные изображения identity. Отдельно отмечать: чужой bbox, смешанная identity, окклюзия, честный сложный ракурс, сомнение.
3. Не устанавливать автоматический порог «удалить cosine<X». Ошибочный экземпляр подтверждает человек; для оценки распространённости шума нужна случайная выборка, а не top-outliers.
4. Сохранить `original_annotation`, `decision`, `reason`, `reviewer`, `image_sha256`, экспериментальный overlay/ignore list. Все оригиналы неизменны.
5. Проверить identity- и full-frame-disjoint folds, включая транзитивные связи нескольких ID на одном исходном кадре. Exact SHA не ловит почти одинаковые соседние кадры: нужен дополнительный near-duplicate аудит, без автоматического объединения всех похожих автомобилей.
6. Отдельно сохранить список подозрительных validation-примеров только для анализа. Основные метрики по ним не переписывать.
7. Воспроизвести raw 79.0162% и reranked 81.4689% как baseline integrity test.

### 9.2. Первая очередь — максимум шесть конфигураций

Сначала control и проверка данных, затем четыре небольших однофакторных сравнения. Это более информативно, чем одновременно добавлять новую архитектуру, loss и masks.

| ID | Изменение | Что проверяем | Условия |
|---|---|---|---|
| B0 | Чистый stock-start OSNet с текущим рецептом на фиксированном inner split | Контроль в новом step-based протоколе | Без использования локального MVP как initializer |
| D1 | B0, но исключены только подтверждённые неверные примеры fold-train | Вредит ли реально label/bbox noise | Inner/outer evaluation остаётся исходной; бюджет matched |
| N1 | Выбранный data-control + frozen backbone BN statistics | Вредит ли drift статистик при маленьком PK batch | BNNeck регулируется отдельно; всё остальное неизменно |
| C1 | Тот же data-control + consistency=0 | Нужен ли текущий self-consistency | Не менять одновременно teacher/augmentation |
| P1 | Тот же data-control + P16K4 вместо P16K2 | Помогают ли дополнительные positives/negatives | Batch64, сопоставление по updates и отдельно data exposure |
| A1 | Тот же data-control, но lower-center occlusion отключён | Не уничтожаем ли identity-сигнал | Blur/JPEG/другие erase не менять |

D1 нельзя объявлять победителем потому, что стали проще тренировочные пары: выигрывать он должен на **неизменённом held-out retrieval**. Если удалено мало изображений, показать matched exposure и sensitivity, чтобы не спутать эффект очистки с эффективным oversampling остальных.

Для B0/D1 разумен общий заранее зафиксированный горизонт, например 1700 optimizer steps с оценкой каждые 100–200 шагов, а не «5/30 эпох» разных размеров. Это предложенный пилотный бюджет, не найденный optimum. Внутренний selection выбирает checkpoint; предел и LR horizon не изменяются после просмотра кривой. Дополнительно воспроизвести прежние 285 updates как точку сравнения, сохранив исходный LR-горизонт: cosine до нуля за 285 шагов — другой рецепт.

Один seed и один внутренний split допустимы только для screening. Для двух лучших настроек — минимум три training seed при **одинаковых query/gallery**, затем проверка на другом frame-grouped development split. В каждом fold encoder и любой teacher стартуют без локального знания его holdout ID.

Не объявлять повторную CV «новым независимым тестом»: гипотезы уже сформированы на данных проекта. Это способ проверить устойчивость, а окончательная независимая оценка остаётся у организаторов.

### 9.3. Вторая очередь — representation

Запускать после первой, не весь список сразу.

| ID | Гипотеза | Начальная проверка | Причина остановиться |
|---|---|---|---|
| M1 | Angular margins лучше plain CE | AM-Softmax вместо CE classifier, controlled SupCon | Выигрыша нет на нескольких seed или сильнее проседает mask-slice |
| M2 | Batch всё ещё мал | XBM queue 1024; 4096 только при сигнале, train-only warmup | Шумные labels усиливаются, drift очереди ухудшает retrieval |
| R1 | 208 теряет мелкие детали | 256 train/inference; тот же split, comparable exposure | Нет gain или extract съедает performance score |
| R2 | Global pooling смешивает объект и фон | Одна local/attention head на shared OSNet | Голова выбирает plate/background, gain исчезает при masking |
| K1 | Ранний IN теряет полезный цвет | Небольшой bypass до IN, основной путь сохранён | Ложные совпадения из-за фона/света; нет gain на новых камерах |
| T1 | Различия ракурсов требуют лучшей positive selection | Verified cross-camera positives + balanced easy/hard; затем RPTM | Метод фактически отбрасывает все сложные реальные ракурсы |
| S1 | Нужна устойчивость к маскам без постоянного удаления деталей | Смешанное original/masked обучение и полная 2×2 оценка | Обычный или масочный retrieval деградирует |
| KD1 | Сильный teacher можно сжать в OSNet | Сначала доказать teacher advantage, затем relational distillation | Teacher слабее, licence/access не подтверждены или gain не переносится |

Сочетать N1/C1/P1/A1/M1 только после их отдельных ablation. Сумма выигрышей отдельных методов не равна выигрышу комбинации.

### 9.4. Минимальная карточка каждого запуска

- Название гипотезы и ровно одно основное отличие от control.
- Commit/code hash, seed модели, seed sampler, отдельный seed протокола.
- Initializer URL/hash и перечень identity, использованных для локального обучения.
- Train/calibration/validation identity и frame group hashes.
- Data overlay hash, число исключённых/исправленных изображений, камер и ID.
- Steps, samples-seen, batch P/K, LR trajectory; не только epochs.
- Train CE/metric/consistency отдельно, gradient norms, norm/variance embeddings, доля активных triplets.
- Доля корректных cross-camera positives, повторов одного кадра и hard negatives.
- Raw mAP@10, Rank-1/5, full mAP, mINP, any-positive hit@10/50 и доля всех найденных positives.
- Reranked mAP@10 при одинаковом frozen control; затем отдельная calibration нового reranker.
- F1, TNR, итоговый candidate score, TP/FP/FN/TN.
- Per-query AP delta и тип ошибки, а не только среднее.
- Масочный/ночной/окклюзионный/ракурсный slices; пустые и малые slices явно помечены.
- CPU smoke и официальный по составу extract benchmark на A5000: batch1 median/p95, пакетный throughput, VRAM, суммарные weights.
- Выбор checkpoint/threshold зафиксирован **до** финального evaluation этого этапа.

### 9.5. Критерии принятия

Рабочий критерий для дорогого representation change: воспроизводимый прирост порядка **+1 п.п. mAP@10** на сопоставимых development проверках без существенного провала mask robustness и общего candidate score. Это критерий инвестиций, не обещание точности.

Бесплатное по inference изменение можно рассматривать при меньшем gain, если он повторяется и не объясняется несколькими подозрительными query. В случае дорогого encoder/ensemble сначала пересчитать полный конкурсный score.

Проверки:

1. Парный bootstrap по query/identity при фиксированной gallery; для нескольких query одного ID — группировать по identity.
2. Показывать диапазон по seeds и альтернативным gallery, а не только условный bootstrap CI.
3. Сравнивать с сильнейшим простым контролем: для pair-head это neighbors, не только MVP.
4. Проверять train-only загрязнение и эффект confirmed-noise samples.
5. Не продвигать модель автоматически; новый ONNX/weights хранить изолированно.
6. ONNX parity на нескольких batch, row order, конечность/L2 embeddings, offline smoke, обязательные форматы.
7. Финальный untouched test не имитировать повторным использованием outer validation под новым именем.

## 10. Протоколу оценки тоже нужна проверка реалистичности

Сейчас `make_protocol` выбирает один query на identity и убирает из gallery **все изображения этой identity с выбранной камеры**. В результате local gallery вообще не содержит same-ID/same-camera junk для этого query. Это корректный cross-camera diagnostic, но он не покрывает все случаи официального evaluator.

Новые диагностические числа на текущей validation:

- Any positive raw top-50: **242/247 = 97.98%**.
- Средняя доля всех валидных positives в raw top-50: **96.3563%**.
- Oracle top-50 mAP@10: **96.3563%** при идеальном GT-порядке; недостижимая на практике диагностическая граница, не прогноз.
- Только четыре known query имеют один позитив; остальные 243 — несколько.
- Bbox aspect ratio>1.5: 92 query, mAP≈79.15%; ≤1.5: 155 query, ≈82.85%.

Разница aspect-ratio **не доказывает вред square resize**: широкие bbox здесь могут содержать неправильный объект или сильную обрезку. Найденные extreme failures делают этот confound особенно важным.

Что добавить на development, без использования closed test:

- Gallery около 750, несколько фиксированных пересэмплирований.
- Режим с одним позитивом на query и с несколькими, отчёт отдельно.
- Добавление реального same-ID/same-camera junk с расчётом официальным скриптом.
- Изменение доли неизвестных, например 10/20/30%, как sensitivity, а не три новые возможности подобрать лучший threshold.
- Камерные срезы и, при достаточном числе ID, более трудный held-out-camera diagnostic; не выдавать его за идентичную конкурсную постановку.
- Набор verified-correct bbox как **дополнительный** срез, неизменённый полный набор как основной.
- Несколько query одной identity допустимы в evaluation, но inference каждого строго независим.

Исторический reranking поднял mAP, однако raw Rank-5≈90.28% выше reranked≈88.26%. Значит, «rerank помогает» не означает улучшение любого ранга и каждого запроса. Следует смотреть, не поднимает ли gallery graph плотные группы похожих машин над единичным честным позитивом.

## 11. Отказ: полезный, но отдельный резерв

В прежнем полном аудите было 73 ошибочных отказа, и в 49 случаях правильная машина уже стояла первой. Значит, исправление embedding/ranking и исправление отказа — не одна задача.

Но последние проверки показывают цену снижения порога: у ансамбля вырос F1 и упал TNR; у pair-head новый отказ ухудшил `0.7F1+0.3TNR`. Поэтому не переносить threshold на новые embeddings и не обучать «вероятность совпадения» по pairwise balanced batches без query-level calibration.

Ближайший разумный путь:

1. Пока исследуем OSNet, сравнивать порядок со старым контролируемым механизмом отказа, отдельно показывая новый calibration-optimal threshold.
2. Для новой модели выбирать порог на calibration по официальной целевой комбинации.
3. Если простого cosine недостаточно, использовать очень компактный query-level calibrator: top cosine, gap, локальная поддержка gallery. OOF-сигналы предпочтительнее train-seen, но сперва нужны валидные labels.
4. Не выдавать sigmoid за калиброванную вероятность.
5. Проверять unknown с похожей моделью/цветом автомобиля, а не только случайные легко различимые negatives.
6. Публиковать кривую trade-off F1/TNR и чувствительность к gallery size, а не один лучший порог.

У этого направления малый inference overhead, но вклад в общий конкурсный score меньше, чем у сопоставимого прироста mAP.

## 12. Итоговая рекомендация

Если выбрать одну следующую содержательную итерацию, я рекомендую **OSNet data-quality + controlled training**, а не новую архитектуру и не очередной reranker.

Последовательность решений:

1. Подтвердить и документировать неверные train crops; сделать отдельный очищенный training overlay.
2. Сравнить его с исходным train при одинаковом stock-start OSNet и фиксированном step schedule.
3. Раздельно проверить BN, consistency, PK-состав и lower-center augmentation.
4. На устойчивой основе — AM-Softmax и 256×256.
5. Если остаются ошибки различения похожих машин — маленькая local-head и контролируемый color bypass.
6. Более крупный teacher/внешние данные рассматривать как источник знаний для OSNet, а не как обязательный второй encoder на inference.
7. Выполненный OOF-reranker оставить отдельной веткой: заметного преимущества OOF он не показал, неверные target crops не исправляет и недостающий визуальный сигнал не создаёт.

**Самая сильная новая улика исследования — конкретные несовместимые training positives, а не название очередной статьи.** Самая перспективная архитектурная улика — проверенная инвариантность раннего IN и отсутствие полноценного эксперимента с локальными признаками/разрешением. Величина возможного улучшения пока неизвестна; обещать 85/90/95% по этим данным нельзя.

## Приложение. Что проверено и что осталось неизвестным

### Новые проверки этой работы

- Проверены hashes MVP, evaluator, splits и ключевого training-кода; они сохранены в evidence JSON.
- Официальный mAP@10 текущей validation воспроизведён из cache.
- Просмотрены 70 training histories и дополнительно три fold-history OOF; компактный индекс основного архива сохранён в evidence JSON.
- Воспроизведён P16K2 sampler и посчитан exposure.
- Выполнен affine-input ONNX probe на 12 development crops, без градиентов.
- Вычислены top50 coverage, oracle и простые validation slices.
- Визуально проверены пять extreme validation ошибок; для четырёх дополнительно просмотрены полные кадры.
- На существующих train embeddings найден shortlist, визуально проверены шесть training identity.
- Проверено фактическое наличие YOLO holdout и завершённого variant 13 вопреки устаревшим README.

### Ограничения

- Ни одна новая training-гипотеза здесь не проверена обучением.
- Процент ошибочной разметки всего train неизвестен.
- Не проверены все 9556 изображений вручную, не оценён независимый test.
- Внешние проекты не запускались в нашем окружении и не дали новых локальных метрик.
- Не все внешние checkpoints/data доступны без регистрации; лицензии потребуется проверять для конкретных выбранных артефактов.
- Новые статьи рассмотрены по доступным первичным материалам; для ICCV2025 VehicleMAE подтверждён метод по странице конференции, но не готовность полного release.
- A5000 benchmark не запускался, а CPU measurements прошлых экспериментов остаются CPU measurements.
- Изменения датасета, inference и гиперпараметров требуют отдельной реализации; исследовательские файлы ничего не внедряют.

### Как воспроизвести диагностики без обучения

Базовая логика зафиксирована в evidence и приведена здесь, чтобы не зависеть от временного рабочего скрипта:

1. Прочитать train.csv через `backend.core.read_rows`, train identity из artifacts/splits.json.
2. Для sampler: `CameraAwarePKBatchSampler(rows,16,2,seed=20260915)`; set_epoch=0…29, накапливать индексы.
3. Для affine probe: отсортировать development по image_id, взять 12 равномерно расположенных записей; использовать штатные bbox/preprocess и Encoder. Применить указанные scale/shift к нормализованным float32 CHW-тензорам, без clipping; L2-нормализовать штатным encoder.
4. Для validation: взять q/g CSV и embeddings.npy из audit_10/results/run_01/validation; строки связать с train.csv по image_id. Первые len(q) embeddings — query, остальные — gallery.
5. Посчитать cosine, stable descending sort, cross-camera positives по GT, mAP переданного baseline/submission.csv через официальный evaluate.py.
6. Для train shortlist: взять variant_12/results/run_01/cache/development.npz, сопоставить ids и vectors. Для каждого crop усреднить cosine только по same-ID/different-camera peers. Отсортировать, выбрать первые шесть разных identity; это не правило очистки.
7. Для визуальной проверки использовать исходные full frames и bbox в формате x,y,w,h, то есть crop(x,y,x+w,y+h), с тем же EXIF handling, что в backend. Не интерпретировать x,y,w,h как x1,y1,x2,y2.

Полные абсолютные пути к исходникам и новым диагностическим файлам приведены в начале отчёта.
