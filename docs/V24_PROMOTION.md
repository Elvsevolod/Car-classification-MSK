# Продвижение v24 — 26 сентября 2026

По явному запросу пользователя активным стал `MVP_dual_role_v24`.
Это улучшение двух взвешенных компонентов качества, не новый рекорд mAP и не полный конкурсный балл.

- Top-10: прежний MVP, input208, legacy k1=20/k2=3/λ0.50.
- Кандидат, confidence и отказ: full-train R1 equal3 v18, input256, raw_top1.
- Порог: исходный 0.534365177154541, не подбирался повторно.
- Веса: прежние четыре ONNX без изменения байтов; нового обучения нет.
- Эмбеддинг: настоящий float32 MVP512 + R1_1536, без глобальной L2 (норма sqrt(2)).
  Отдельные блоки нормированы, API `/api/embedding` честно возвращает `l2_normalized=false` и layout.
- Confidence принятого кандидата — R1 cosine; similarity в ranked results — MVP cosine.

Validation: mAP 0.8146886982413298, F1 0.7954545454545456, TNR 0.7096774193548387,
C 0.7697214076246335. Против старого MVP: ΔmAP=0, ΔC=+0.022600183553334485.
Правильных совпадений 149→175, ложных срабатываний неизвестных 13→18.
Данные уже наблюдались; превосходство на скрытом тесте не доказано.

## Проверка переноса

`tools/promote_dual_role.py` проверяет исходный v24 и фиксирует profile/metrics/evidence.
`docs/V24_QUALITY_EVIDENCE.json` хранит исходное доказательство качества; прежнее решение
сохранено в `docs/RELEASE_DECISION_BEFORE_V24.json`. Исторические исследовательские runs не изменены.

Новый проход через приложение на полном исходном validation:
`artifacts/v24_promotion/validation_v1/verification.json`.
309 query / 896 gallery, embeddings (1205,2048): ошибка признаков 0,
изменённых top-10 — 0, кандидатов/отказов — 0. Максимальное расхождение confidence
1.2517e-6 из-за batched/streaming cosine, в заранее фиксированном допуске 2e-5;
порог не менялся. Это не заявление о побайтной идентичности confidence CSV.

API, экспорт, кэш и benchmark используют общий runtime; runtime не импортирует training/torch.
Кэш v24 имеет отдельный fingerprint и dimension2048, старые таблицы gallery_items/gallery_state
сохранены. Возврат legacy проверяется отдельно; галерея не смешивает размерности.

Проверки после переноса: 89 тестов приложения прошли с отдельной временной PostgreSQL,
включая кэш2048, сохранность legacy-таблиц и возврат к прежним решениям. Тестовый контейнер
удалён, рабочая БД не использовалась для тестовых записей. Ещё 125 целевых research-тестов прошли.
В настоящем браузере проверены выбор примера, поиск с порогом, отдельный принятый кандидат
и метрики нового профиля. Снимок: `output/playwright/v24-live.png`.

## Запущенное локальное демо

Адрес: http://127.0.0.1:8000. Контейнер `vehicle-reid-fixes-vehicle-reid-1`,
образ `vehicle-reid:dual-role-v24`. PostgreSQL и volumes проекта `vehicle-reid-fixes` не заменялись.
Добавочная миграция создаёт новые пространства; старые 750 gallery_items и 1 gallery_state сохранены.
Код новой сборки находится в `Car-classification-MSK-release-integration`.
Файлы `Car-classification-MSK-review-fixes` и research backend не перезаписывались.

Повторное включение из корня integration-копии (образ сначала собрать):

```bash
docker build -t vehicle-reid:dual-role-v24 .
PORT=127.0.0.1:8000 REID_PROFILE=MVP_dual_role_v24 docker compose -p vehicle-reid-fixes \
  -f ../Car-classification-MSK-review-fixes/docker-compose.yml \
  -f ../Car-classification-MSK-review-fixes/artifacts/verification/compose.yml \
  -f docker/v24-demo-override.yml up -d --no-deps --no-build --pull never vehicle-reid
```

Откат той же командой с `REID_PROFILE=MVP_legacy`: новый образ содержит старые веса и профиль.
Это перезапуск одного приложения, без удаления БД/томов и без миграции downgrade.
Не использовать `docker compose down -v`. Старый образ `vehicle-reid:review-fixes` тоже сохранён,
но для него действуют ограничения старого Alembic после новой миграции (см. RELEASE_INTEGRATION.md).

При обычном запуске integration `docker compose up --build` профиль v24 используется по умолчанию,
но отдельный проект слушает порт8017. Исторический offline-архив сам собой не обновился:
для поставки нужно собрать новый архив из актуальной integration-копии.

## Дальше

Исследования mAP продолжаются в research `variant_25_map_search/search_map.ipynb`:
84 calibration-сравнения, validation только baseline+победитель. Кандидаты R1 фиксированы.
Результаты будущего поиска не меняют MVP автоматически. Push/merge не выполнялись.
GPU, native Linux amd64, остаточный номерной сигнал и остальные открытые пункты сдачи
остаются незакрытыми; локальная promotion не объявляет полную конкурсную готовность.
