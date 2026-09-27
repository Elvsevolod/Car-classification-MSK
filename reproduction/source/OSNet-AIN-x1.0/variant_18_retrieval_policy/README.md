# Variant 18: разделение encoder / ranking / candidate policy

Точка входа: **`evaluate_retrieval_policy.ipynb` → Restart Kernel → Run All** в прежнем `.venv`.
По умолчанию `PHASE='inner'`, `ALLOW_OUTER_EVALUATION=False`. Запускает пользователь.
Нет optimizer updates, новых весов, изменений MVP/backend, bbox, CSV, изображений или split.

## Почему этот этап, а не EMA

Учтены `Car_ReID_research_2026-09-25.md` и `notebook_final_seed_results.md`.
В v17 R1 лучше B0 по raw, но эффект старого reranking меньше, а TNR ниже.
Средний кандидатский C при этом не хуже: R1 reranked 0.75160 против MVP 0.74712;
R1 raw — 0.75546. C = 0.7 F1 + 0.3 TNR, вес кандидатского блока — **10%**.
Это не полный балл соревнования и не обещание выигрыша на закрытом тесте.

Прочитаны и доступные локально результаты, которых не было в архиве аудитора:
v16 primary K2 = 0.72847 против K1 = 0.81246, R1 = 0.83228, M3 = 0.82557.
K2 слабее в этом рецепте; синтетическая гипотеза масштаба CE остаётся отдельным
экспериментом, не доказанной причиной и не объяснением поведения R1.

## Реализовано

- Пять заранее фиксированных ranking-политик: legacy 20/3/0.5, 20/1/0.5,
  20/3/0.75, 20/1/0.75 и прямой raw cosine (без graph при lambda=1).
- Только один query и статическая gallery. Identity/camera используются лишь
  официальным evaluator и диагностикой, не формированием графа или выдачи.
- Независимые top-10 и candidate: `raw_top1` либо прежний `ranking_top1`.
  Confidence в обоих случаях — maximum raw cosine; каждому правилу свой calibration.
- Равновесный ансамбль **всех трёх** R1: concat нормированных признаков / sqrt(3),
  размерность 1536. Не среднее 512 координат и не среднее отдельных метрик.
  Граф строится в новом пространстве; старые пороги не усредняются.
- AP каждого query, raw/reranked top-1 и ошибки перехода, confidence всех query
  до порога (включая unknown), PR-AUC до порога, пересечение ошибок seed.
- Частота появления gallery-кадров в соседях и состав reciprocal neighborhood,
  с отдельным подсчётом same-identity/same-camera junk. Это диагностика, не удаление кадров.
- Threshold curve на calibration, диапазон C в пределах 0.5 п.п. от максимума,
  контрольные точки TNR 0.7/0.8/0.9. По outer порог не выбирается.
- Schema-2 bundle фиксирует список/hash исходных ONNX bundles, ranking, candidate policy,
  новый порог и calibration provenance. Schema-1 и старый runtime не переписаны.
- Проверка hybrid CSV официальными `load_submission/load_candidates`, ranking/candidate metrics.
  Отказ — отсутствие строки candidates; submission всегда содержит 10 ID.

## Три раздельные фазы

### inner — следующий запуск

1. Проверить хеши всех завершённых фаз v16/v17 и всех 9556 organizer frames.
2. На **шаге 800** извлечь признаки B0/R1 для трёх seed primary, без переобучения.
3. Сравнить пять политик на трёх regular draws. Сначала среднее draws внутри seed,
   затем трёх seed. Для equal3 — один ансамбль на трёх draws, без фиктивного n=3.
4. Выбрать политику отдельно для каждого encoder/system по конечному mAP@10.
   При точном равенстве оставить legacy (фиксированный порядок POLICIES).
5. Сохранить `selection.json` **до чтения alternate-эмбеддингов и оценок**.
6. На alternate проверить выбранное правило плюс raw/legacy controls, включая junk draws.
   Не менять выбор по alternate. Не использовать calibration/outer.

Основной результат: `runs/policy_v1/INNER_RESULTS.md` и `inner.json`.
Полные диагностические записи — `reports/`; кеш признаков — `cache/`.

### checkpoints — опциональная отдельная диагностика

Сохранены 11 primary steps × 2 рецепта × 3 seed = 66 состояний encoder.
Эта фаза может быть заметно дольше inner, хотя обучения нет. Кеш шага800 переиспользуется.
Выбирается только **primary-предложение** общего step/ranking на рецепт, не лучший seed.

У alternate/final сохранился **только шаг800**. Поэтому предложение другого шага
не подменяет `selection.json`, не выбирает старый final и не запускает обучение:
для него нужен отдельный план refit/confirmation. Это ограничение существующих весов,
а не разрешение достроить поздний checkpoint по outer.

### final — только после разбора inner и отдельного разрешения

Нужно вручную `PHASE='final'`, `ALLOW_OUTER_EVALUATION=True`.
Проверяется завершение inner и его frozen selection.
Шаг800 остаётся прежним. Все три seed сохраняются, «лучший seed» не выбирается.

Final использует сохранённые ONNX на **CPU batch16**; это явно отличается от MPS
primary-evaluation и позволяет проверять именно экспортируемую модель.
Сначала calibration для всех систем/политик, затем замораживаются все threshold/bundles
в `final_selection.json`, и лишь после этого код извлекает outer embeddings.

Сравниваются legacy и primary-selected ranking, каждый с raw/ranking candidate.
Ни candidate policy, ни ensemble не выбираются по outer: показаны все заранее заданные
контроли. Одиночные модели усредняются по трём seed; equal3 показан отдельно.
Полный балл и победитель соревнования из этих метрик не выводятся.

`FINAL_RESULTS.md`, `final.json`, `final/<system>/<ranking>/<candidate>/bundle.json`
и `csv/` сохраняются только в новом run. Пороги и веса старых runs не меняются.

## Frozen export, отдельно от приложения

Новый export использует только frozen bundle + test_query/test_gallery + изображения.
Не читает train.csv и не запускает calibration. Пример из корня репозитория:

```bash
.venv/bin/python -m training.policy_inference \
  --bundle PATH_TO_NEW_SCHEMA2_BUNDLE \
  --dataset PATH_TO_DATASET --output NEW_EMPTY_OUTPUT \
  --provider CPUExecutionProvider
```

Создаёт submission/candidates, embeddings.npy с реальной размерностью и manifest
с fingerprint всех encoder, preprocessing, policies, CSV и кадров.
Провайдер явный; скрытый CPU fallback при CUDA запрещён прежним FrozenEncoder.
Gallery embeddings строятся заново для выбранного fingerprint; старый backend-cache
в этот путь не подключается. Это **не переключает backend/web** на новую модель.

`policy_inference.compare_decisions()` проверяет не только близость tensors, но и
top-10/accepted set на одном фиксированном протоколе. На реальных CPU/GPU решениях эта
проверка пока не выполнена. Официальный extract benchmark также не выполнен;
equal3 требует трёх forward на каждый кадр, поэтому нельзя продвигать его только по mAP.

## Восстановление и ограничения

Restart Kernel → Run All с тем же RUN_NAME/кодом. Кеш embeddings и отчётов проверяется
по хешам. Прерванная незакоммиченная cache-запись пересчитывается; готовые не перезаписываются.
CSV публикуются после полной записи; прерывание до отчёта не заставляет менять готовые CSV.
Завершённая фаза имеет `<phase>_complete.json` с хешами всех её файлов.
При несовпадении остановиться и разобрать причину, не удалять manifest/receipt.
Не запускать старые и новые notebooks одновременно; межверсионная lock-проверка best-effort.

Original validation уже просмотрена и остаётся development, не независимым тестом.
Новые выборы не отменяют историю адаптации экспериментов к данным. Три draws/seed
не создают новых независимых test datasets. Данные и старые результаты не исправляются.

K2 CE-scale с logging CE/logits/градиентов, EMA/soup, low-LR tail и перенос training clock
**отложены** согласно порядку нового исследования. Их нельзя смешивать с этим этапом:
сначала выясняем, что дают границы encoder/reranker/candidate без нового обучения.
NiVe остаётся локально и не включается. Commit/push/автоматическое продвижение не выполняются.

## Проверено перед передачей, 25.09.2026

- **340 tests passed**, в том числе 36 новых: hybrid CSV через официальный evaluator,
  streaming-независимость query, равновесное пространство ансамбля, schema-2 export
  без train/labels/calibration, cache/receipt/resume и порядок freeze перед alternate/outer.
- Исправлен один устаревший тест variant17: выполненный пользовательский notebook
  теперь проверяется без требования стереть его outputs. Сам notebook не изменён.
- Реальный preflight `verification_only_20260925` успешно выполнен дважды:
  9556 кадров, 352 protected inputs, неизменный manifest при повторении.
- Отдельный снимок **374** защищённых файлов до/после совпал:
  `0062293dc998896b718f5254e92ac5816fb7992748068e392566a6d49779a39f`.
- На реальных данных не выполнялись ни обучение, ни encoder inference, ни новые оценки.
  Пользовательский `policy_v1` не запускался; новый notebook валиден и не выполнен.
  Улучшение качества variant18 пока неизвестно.
