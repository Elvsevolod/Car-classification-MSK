# Продвижение v25

По явному запросу пользователя активным профилем выбран `MVP_fusion_v25`.
Ranking: равная смесь нормированных признаков MVP и full-train R1, затем
streaming k-reciprocal k1=20/k2=3/λ0.50. Кандидат/confidence/отказ: прежний raw R1.
Порог 0.534365177154541 не меняется. Четыре ONNX-веса и preprocessing не меняются.
Эмбеддинги сохраняют прежние действительные блоки MVP512+R1_1536 без общей L2.
Веса смешивания применяются только при ranking, не к кандидатской ветке.

## Качество и доказательство переноса

На исходном validation mAP@10 0.8146886982 → 0.8289573904 (+1.427 п.п.),
Rank-1 0.8016194332 → 0.8299595142. Rank-5 немного снизился: 0.8825910931 → 0.8785425101.
F1=0.7954545455 и TNR=0.7096774194 неизменны, принято 218 / отказ 91.
На calibration вариант выбран среди заранее зафиксированных 84 сравнений.
Это уже наблюдавшиеся development-данные, не подтверждение hidden-test результата.

`tools/promote_v25.py` проверяет 86 task receipts, исходники, 2226 защищённых файлов,
выбор на calibration и неизменность кандидатского режима. Доказательство качества:
`V25_QUALITY_EVIDENCE.json`. Решение до переключения: `RELEASE_DECISION_BEFORE_V25.json`.

Свежий проход приложения на всех 309 query / 896 gallery воспроизвёл исследовательский v25:
ошибка эмбеддингов 0, изменённых top-10 0, кандидатов/отказов 0.
Максимальное различие confidence 1.2517e-6 в прежнем допуске 2e-5
(batched/streaming cosine); не заявляется побайтное совпадение confidence CSV.
Свидетельство: `V25_APPLICATION_EVIDENCE.json`.

Прошли 91 тест приложения с отдельной временной PostgreSQL (включая v24 → v25 → v24),
134 исследовательских теста. Отдельный свежий CLI reference на 1205 изображениях также
воспроизвёл эмбеддинги, top-10 и отказы в том же фиксированном допуске.
В браузере проверены выбор примера, поиск, принятый кандидат и mAP=82.90% в метриках.
Снимок: `output/playwright/v25-live.png`. Контейнер healthy; старые таблицы сохраняют
750 gallery_items и 1 gallery_state. Временная тестовая БД удалена, рабочая БД сохранена.

## Запуск и откат

Код сборки находится в `Car-classification-MSK-release-integration`.
Локальное демо сохраняет адрес http://127.0.0.1:8000 и существующие PostgreSQL/volumes.

```bash
docker build -t vehicle-reid:fusion-v25 .
PORT=127.0.0.1:8000 REID_PROFILE=MVP_fusion_v25 docker compose -p vehicle-reid-fixes \
  -f ../Car-classification-MSK-review-fixes/docker-compose.yml \
  -f ../Car-classification-MSK-review-fixes/artifacts/verification/compose.yml \
  -f docker/v25-demo-override.yml up -d --no-deps --no-build --pull never vehicle-reid
```

Откат: та же команда с `REID_PROFILE=MVP_dual_role_v24` (или `MVP_legacy`).
Не удалять volumes, не выполнять downgrade. Новый образ содержит все прежние профили;
старый образ `vehicle-reid:dual-role-v24` также сохранён.
Одинаковые вектора v24/v25 безопасно используют общий кэш; scorer и fingerprint профиля различны.
Старые таблицы и веса не удаляются.

## Следующий эксперимент

В research подготовлен `variant_26_fusion_extension/search_fusion.ipynb`:
84 смеси с долей R1 от 40 до 100%, без обучения и без изменения кандидатов/порога.
Контроль — текущий конкретный v25. На validation — контроль и один calibration-победитель.
Полный поиск пользователь запускает самостоятельно; автоматической смены MVP нет.

Исторические runs и v25 notebook не редактируются. Push/merge не выполняются.
GPU/native Linux amd64 и остальные открытые вопросы сдачи остаются незакрытыми;
исторический offline-архив для поставки нужно пересобрать отдельно.
