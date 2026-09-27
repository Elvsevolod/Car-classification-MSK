# Variant 16: изменения по аудиту 24 сентября 2026

Точка входа: **`train_osnet_review_protocol.ipynb` → Restart Kernel → Run All** в прежнем `.venv`.
По умолчанию запускается только pilot. Обучение запускает пользователь.
NiVe не используется и остаётся локально, текущий MVP и backend не меняются.

## Что исправлено

| Замечание аудита | Решение |
|---|---|
| MVP выбирался по outer validation | 81,47% — исторический development reference, не независимый тест. Новые рецепты сравниваются с заново обученным B0 по одинаковому протоколу. |
| Средние индивидуальных максимумов не соответствуют одному final-рецепту | Один общий шаг на рецепт: среднее по трём фиксированным draws внутри seed, затем по трём seed. Все selectable checkpoints сохраняются. |
| Перенос длительности между splits | Alternate и final обучаются с нуля ровно до выбранного primary шага, с полным исходным LR horizon 1700. Alternate ничего не выбирает. |
| Общий BN меняет соотношение 80/20 | Отдельные BN → L2 ветвей → sqrt(0.8)/sqrt(0.2) → concat, без BN после. Геометрия одинакова для metric loss и inference. |
| Нужны целевые проверки вместо широкого HPO | Только исправленная color-ветвь, resolution256, triplet и два необходимых контроля. |
| Один query/gallery draw недостаточен | Три фиксированных regular draws; на alternate также same-camera junk и confidence по known/unknown, без настройки порога. |
| Автоматические маски не доказывают plate-only эффект | На alternate/final есть masked query/gallery/both, явно помеченные как automatic-region diagnostics. Исходные bbox/кадры не меняются. |
| CPU-only inference и размерность 512 | Отдельный opt-in runtime с явным CPU/CUDA provider и размерностью из ONNX. MVP Encoder остаётся прежним. |
| Экспорт повторно требует train/calibration | Frozen bundle хранит модель, preprocessing, порог, reranking и provenance. Submission-export читает только test query/gallery и изображения. |

Новые модули: `training/osnet_review_protocol.py`, `review_selection.py`,
`osnet_fixed_fusion.py`, `frozen_inference.py`.
Исторические training/backend-модули не переписываются: они входят в fingerprints прежних запусков.

## Условия и этапы

| Условие | Единственная целевая разница относительно B0 |
|---|---|
| B0_control | Общий step-based контроль, stock → organizer train |
| K1_color32_legacy | Прежняя color32-ветвь с общим BN — архитектурный контроль |
| K2_color32_fixed | Color32 с независимыми BN/L2 и фиксированным cosine 80/20 |
| R1_resolution256 | Размер 256 вместо 208 |
| M3_triplet | Batch-hard triplet вместо исходного metric loss |

Геометрия K2 меняется намеренно и для metric loss: это проверка целого согласованного
рецепта fusion, не чисто inference-time перевзвешивание уже обученного K1.
При нулевой/численно вырожденной ветви используется детерминированный единичный вектор;
это защита от NaN, а не восстановление отсутствующего cosine или свидетельство качества.
Исправленный модуль также поддерживает local128, но она не входит в этот пилот.

1. **`PHASE='pilot'`**: все пять условий на первом seed, 5 × 1700 = **8500 updates**.
   Три regular draws считаются на каждой границе; отчёт показывает общий последний шаг,
   а не выбирает победителя по одному seed. Outer/calibration и export не запускаются.
2. **`PHASE='confirm_inner'`**, только после разбора пилота: primary на трёх seed,
   один общий step на рецепт и фиксация выбора в `selection.json` **до** alternate.
   Затем все пять условий × три alternate seed до своих замороженных шагов.
   Всего с пилотом до **30 стадий / 51 000 updates**; после пилота до **42 500 новых updates**.
   Завершённые стадии переиспользуются после проверки хешей. Это отдельный долгий запуск,
   не автоматическое продолжение. Resolution256 и несколько draws увеличивают время.
3. **`PHASE='final'`, `ALLOW_OUTER_EVALUATION=True`**, только после отдельного решения:
   B0 + заранее выбранный primary победитель; если победил K2, также K1 для прямого сравнения.
   Один заранее фиксированный seed, refit на outer train. При победе B0 — только B0.
   Все веса фиксируются до первой outer-оценки. Пороги выбираются только на calibration,
   затем без изменений используются на original/masked outer. ONNX сохраняется только в run.

Ни одна фаза автоматически не запускает следующую и не продвигает модель в MVP.
При равенстве primary score выбирается более ранний шаг; при равенстве рецептов — B0.
Alternate проверяет переносимость выбранного правила; выбирать другого победителя по его максимуму нельзя.
Primary/alternate находятся внутри outer train и частично пересекаются: это не независимый тест.
Исторические результаты уже влияли на выбор гипотез, поэтому и этот прогон не отменяет прошлую адаптацию к данным.

## Сохранность и воспроизводимость

- Не редактируются organizer CSV, bbox, identity, camera, исходные изображения и splits.
- Manifest фиксирует recipes, budgets, три training seed, три draw seed, данные/код/runtime.
- Seed: 20260915, 20260916, 20260917. Draw seed: 20260915, 20261016, 20261117.
- Стандартная процедура query/gallery и официальный evaluator остаются прежними.
  Same-camera junk добавляется только в diagnostic draws, metadata не подаётся в inference.
- Resume идёт с authoritative `last.pt`; прерванный блок до 200 updates повторяется.
  Блоки используют воспроизводимый seed. CPU-синтетические проверки сравнивают также веса/optimizer.
  Побитовая детерминированность GPU/MPS не обещается: notebook разрешает предупреждения unsupported ops.
- Проверяются checkpoint, summary и результаты завершённой фазы, а не только наличие JSON.
- Изменение кода/данных/runtime требует нового RUN_NAME. Старые manifests не удалять.
- Перед обучением проверяются активные lock-файлы вариантов 14–16 без изменения чужих файлов.
  Это best-effort проверка, не общий межпроцессный lock для старых trainers; не запускать два notebook одновременно.
- На каждый primary selectable шаг сохраняется модель. Предусмотреть несколько ГБ диска.

## Как читать результаты

`PILOT_RESULTS.md` / `pilot.json`, затем `CONFIRM_INNER_RESULTS.md` / `confirm_inner.json`.
В confirmation есть paired-seed разницы относительно B0 и K2−K1.
Три draws не превращают три seed в девять независимых повторений.
В alternate `summary.json` каждого условия: regular/junk mAP и распределения raw maximum cosine
для known/unknown; для regular draws дополнительно три automatic-mask diagnostics.
Confidence не вероятность. Порог на этих диагностиках не подбирается.

`FINAL_RESULTS.md` / `final.json`: outer ranking и refusal отдельно, per-query AP и paired
identity bootstrap относительно B0. Интервал условен на фиксированной gallery и не исправляет
многократный подбор по ранее просмотренной validation. Автоматического критерия продвижения нет.

## Frozen inference после final

Bundle: `runs/<RUN_NAME>/final/<VARIANT>/seed_20260915/bundle.json`.
Он фиксирует SHA256 ONNX, square/letterbox, размер изображения, нормализацию, размерность,
calibration-only порог по maximum raw cosine, reranking k1=20/k2=3/lambda=0.5.
Submission всегда содержит Top-10; отказ — отсутствие строки query в `candidates.csv`.

Из корня репозитория, подставив существующий bundle и dataset:

```bash
.venv/bin/python -m training.frozen_inference export --bundle PATH_TO_BUNDLE --dataset PATH_TO_DATASET --output NEW_OUTPUT_DIR --provider CPUExecutionProvider
.venv/bin/python -m training.frozen_inference parity --bundle PATH_TO_BUNDLE --dataset PATH_TO_DATASET
.venv/bin/python -m training.frozen_inference benchmark --bundle PATH_TO_BUNDLE --dataset PATH_TO_DATASET --provider CUDAExecutionProvider
```

Export не читает train и не пересчитывает calibration. Размерность 544/640 поддерживается;
проверка PyTorch/ONNX выполняется на batch 1/3/8 до фиксации bundle.
CUDA требует окружения с CUDAExecutionProvider. Скрытый CPU fallback запрещён, в том числе
для CPU-only подграфов: такая модель остановится при создании сессии и потребует отдельного разбора.
Проверка CPU/GPU и официальный GPU benchmark здесь **не выполнены**.
Benchmark включает decode/EXIF/bbox/resize/normalize/transfers/forward/L2, но не загрузку модели,
поиск по gallery и reranking. Он не заменяет официальный speed evaluator.
Это отдельный experimental runtime, не переключатель существующего web/backend на GPU.

## Что сознательно не включено

- Автоматическое продолжение NiVe: пилот отрицательный; старый confirm notebook требует явного opt-in.
- Повтор GeM/MixStyle: исправленная реализация уже проверялась; новых доказательств пользы нет.
- Буквальное воспроизведение 5 эпох исторического MVP: B0/step285 им не объявляются.
  Для такого диагностического контроля нужен отдельный epoch-based рецепт с прежним LR horizon 30.
- EMA encoder для inference: отдельная следующая гипотеза, не смешанная с уже проверенной EMA-consistency.
- Подтверждённые plate-only маски: нужны проверенные масочные аннотации; имеющиеся автоматические
  маски не переименовываются в plate-only и не используются для исправления исходной validation.
- Новые rerank-параметры, online/cross-query методы, deployment и отправка датасета в Git.

## Проверено локально 24 сентября 2026

- **224 tests passed**: новые protocol/fusion/runtime/export проверки и связанные regression
  tests NiVe, обучения, variant14 и официального evaluator. Это не полный тест всего приложения.
- Проверены обычное и прерванное обучение на синтетических данных, сохранность completed phases,
  контроль доступа к outer, ветви 544/640D, ONNX CPU parity batch 1/3/8 и export без train.csv.
- Preflight проверил все **9556 исходных изображений**. Primary и alternate:
  740 train identity / 185 validation identity; на каждом три regular и три junk draws.
- 76 защищённых файлов MVP, исходников и исторических артефактов сохранили исходные SHA256.
  Пользовательские outputs не очищались. Проверка старого repair notebook больше не требует
  удалять outputs уже завершённого обучения; проверка структуры и компиляции сохраняется.
- На реальных данных **0 optimizer updates**, outer не оценивалась. Проверочный каталог
  `runs/verification_only_20260924` содержит только manifest/lock; пользовательский `review_v1`
  не запускался. Прирост качества пока неизвестен. GPU/MPS training и GPU parity этим не подтверждены.
