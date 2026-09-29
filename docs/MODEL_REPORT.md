# Активная модель MVP_fusion_v25

Профиль по умолчанию в runtime, Docker Compose и `release_decision.json` —
**MVP_fusion_v25**. Метод и инструкция запуска описаны в [документации](DOCUMENTATION.md).

## Два назначения признаков

- **Ranking:** равная смесь нормированных признаков MVP_legacy (512D, 208×208)
  и ансамбля трёх R1 (1536D, 256×256), затем независимый от соседних query
  k-reciprocal: k1=20, k2=3, lambda=.50.
- **Кандидат и отказ:** отдельный raw top-1 ансамбля R1; его cosine и фиксированный
  порог **0.534365177154541**. Кандидат может отличаться от первого результата ranking.
- **Embeddings:** действительные float32-блоки MVP512 + R1_1536 = 2048D,
  без общей L2-нормализации; смешивание применяется внутри scorer к ranking.
- `submission.csv` всегда содержит десять разных gallery-ID на query,
  включая отказы. При отказе строка в `candidates.csv` отсутствует.
- API, экспорт и benchmark используют общий runtime. Данные камер/identity не
  подаются модели; нет OCR, детектора или обработки номерной зоны.

Все четыре используемых ONNX лежат в Git: legacy HPO-вес в `models/` и три R1
в `models/frozen/`. Stock-инициализация и прежние профили также сохранены для
происхождения/отката. Точные SHA256 и относительные связи — в
[profiles.json](../models/profiles.json), frozen bundles и
[ASSET_PROVENANCE.json](ASSET_PROVENANCE.json).
Последний документ содержит исторические локальные пути происхождения, не
обязательные runtime-пути.

## Сохранённое качество

Это **наблюдавшаяся development-validation**, не новый замер и не скрытый тест.
На прежних 309 query / 896 gallery:

| Метрика | Значение |
|---|---:|
| mAP@10 | 0.8289573904 |
| Rank-1 | 0.8299595142 |
| Rank-5 | 0.8785425101 |
| Candidate F1 | 0.7954545455 |
| TNR | 0.7096774194 |
| Принято / отказ | 218 / 91 |

Ranking v25 выбран на calibration из 84 зафиксированных сравнений.
Предыдущий legacy mAP@10 был 0.8146886982; прирост — около 1.427 п.п.
Кандидатская ветвь относительно v24 не менялась. Не переносить этот порог
на новую модель, другие веса смешивания или другую нормировку.

Источники: [метрики профиля](../models/MVP_fusion_v25.metrics.json),
[V25_QUALITY_EVIDENCE.json](V25_QUALITY_EVIDENCE.json),
[V25_APPLICATION_EVIDENCE.json](V25_APPLICATION_EVIDENCE.json),
[описание продвижения](V25_PROMOTION.md).
Legacy-отчёт с другим порогом и цифрами сохранён отдельно:
[LEGACY_MODEL_REPORT.md](LEGACY_MODEL_REPORT.md).

## Запуск и ограничения

Основной путь без обучения, train.csv, PostgreSQL и сети:

```bash
python -m backend.infer --dataset /absolute/dataset --output /absolute/new_output --profile MVP_fusion_v25
```

Docker-эквиваленты и web-приложение описаны в [README](../README.md).
Исходники выбранных рецептов находятся в [reproduction-kit ветки fine-tuning, коммит 35e6e1f](https://github.com/Elvsevolod/Car-classification-MSK/tree/35e6e1f0ffc6f378e924da930fdeccc5841273d1/reproduction-kit).
История поиска сохранена отдельно: [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md).
`backend.evaluate` сохранён как отдельный legacy-инструмент разработки;
его результат не заменяет оценку ансамбля v25.

Готовность всех условий сдачи не заявляется: остаются Linux amd64/NVIDIA GPU,
остаточный номерной сигнал, неоднозначность junk/top-10,
презентация и ссылки. Полный список — [release_decision.json](../release_decision.json).
Исторические CPU-замеры одной OSNet не являются скоростью четырёхмодельного v25.
Ручной threshold разрешён только в демо и не изменяет конкурсный профиль.

Требование отдельного `extractor.py` отменено позднейшим ответом организаторов:
[уточнение от 22.09.2026](ORGANIZER_CLARIFICATIONS_2026-09-28.md).
