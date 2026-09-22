# Vehicle ReID: MVP с дообученной OSNet

Минимальный локальный сервис: Python/FastAPI + обычные HTML/JavaScript.
Модель — **автомобильная OSNet-AIN x1.0 (`vehicle-reid-0001`),
дообученная на 925 train identity**. MVP использует HPO best-mAP checkpoint
эпохи 5 с BNNeck, Supervised Contrastive Loss и потоковым
k-reciprocal reranking. Стоковый ONNX сохранён только как начальные веса для новых
экспериментов. Случайные веса, детектор и OCR не используются.

## Быстрый запуск в Docker

Проект запускается через Docker Compose: FastAPI, PostgreSQL 16 + pgvector, локальный веб-интерфейс и offline Swagger UI находятся в контейнерах. Локальный Python/venv для обычного запуска не нужен.

1. Поместите датасет в `./dataset`. Внутри должны быть `images/`, `train.csv`, `test_query.csv` и `test_gallery.csv`.
2. Из корня репозитория выполните одну команду:

```bash
docker compose up --build
```

Интерфейс и API: http://127.0.0.1:8000. Swagger: http://127.0.0.1:8000/docs. Перед стартом приложения `dataset-init` автоматически копирует внешний датасет в именованный Docker volume: это устраняет зависимость от прав исходной папки (включая Linux `700`), но на первом запуске требует ещё около 7 ГБ Docker-диска. Затем Compose ждёт готовности PostgreSQL, применяет миграции и создаёт/переиспользует gallery из 750 объектов.

Для закрытого стенда используйте заранее загруженные локальные образы и запускайте `docker compose up -d --no-build --pull never`. Эта команда была проверена: она запрещает и сборку, и pull. Runtime не загружает веса или Python-пакеты. Порядок подготовки Linux x86_64 образов для жюри описан в [docs/CONTEST_IMAGE_DELIVERY.md](docs/CONTEST_IMAGE_DELIVERY.md).

### Экспорт файлов сдачи

```bash
docker compose --profile inference run --rm --pull never inference
```

Команда создаёт в `./artifacts/` три обязательных файла: `submission.csv`, `embeddings.npy` и `candidates.csv`. Экспорт использует PostgreSQL + pgvector, но не поднимает веб-интерфейс; подготовленный dataset volume переиспользуется без повторного копирования.

Проверка созданных файлов:

```bash
docker compose run --rm --no-deps --pull never --entrypoint python inference -m backend.evaluate --validate-only
```

Полная инструкция, включая остановку и тесты, находится в [docs/RUNBOOK.md](docs/RUNBOOK.md). Последний проверенный прогон — в [docs/TEST_SUMMARY.md](docs/TEST_SUMMARY.md).

## Тестирование

Полный набор unit-, API-, экспортных и PostgreSQL + pgvector integration-тестов запускается в отдельной временной БД; production-образ и рабочие Docker volumes не затрагиваются:

```bash
docker compose -p vehicle-reid-tests -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
docker compose -p vehicle-reid-tests -f docker-compose.test.yml down -v
```

Вторая команда удаляет только временные ресурсы проекта `vehicle-reid-tests`.

## Что умеет страница

1. Загрузить JPEG/PNG или выбрать любой из 1110 официальных query-примеров.
2. Выделить автомобиль мышью или ввести `x, y, w, h`. Для query BBox подставляется из CSV.
3. Получить Top-N кропов галереи, rerank score, raw cosine и ссылку на полный кадр.
4. Включить режим кандидатов: вернуть только результаты выше порога или отказ.
5. Скачать JSON результата и посмотреть измеренные метрики активной модели.

Один объект галереи — **конкретный BBox в конкретном кадре**, не установленная
личность автомобиля. API не возвращает выдуманные номера или `vehicle_id` теста.
Query не добавляются в галерею; поиск всегда проводится только по `test_gallery.csv`.

## Пайплайн и хранение

EXIF-ориентация → RGB → строгий кроп BBox без отступов → bilinear 208×208 →
float32 / 255 → ImageNet normalization → ONNX OSNet → L2-нормализация 512-D вектора.
Одинаковая функция используется для галереи, запросов API и оценки модели.
Конкретный ONNX ожидает RGB; не путать с BGR у конвертированного OpenVINO IR.

В PostgreSQL + pgvector хранятся метаданные и float32-векторы объектов gallery. При старте из них строится статический k-reciprocal граф; raw cosine source — exact pgvector search. Каждый query
обрабатывается независимо: порядок задаёт смесь Jaccard и cosine distance,
а отказ — максимальный raw cosine. Другие query не используются.
Отдельная векторная СУБД или приближённый индекс этой версии не нужны.

`artifacts/embeddings.npy` — отдельный артефакт для организаторов, а не рабочая
база: сначала query, затем gallery, строго в порядке соответствующих CSV.

## Контракт API

Спецификация: `/openapi.json`; интерактивная документация: `/docs`.

| Метод | Адрес | Назначение |
|---|---|---|
| GET | `/api/health` | Готовность, модель, размер галереи, порог |
| GET | `/api/queries?offset=0&limit=50` | Query ID и BBox для интерфейса |
| POST | `/api/search` | Изображение + BBox → результаты поиска |
| POST | `/api/search/query` | Query ID из CSV → результаты поиска |
| POST | `/api/embedding` | Изображение + BBox → L2-вектор, 512 float32 |
| GET | `/api/images/{query или gallery}/{image_id}?crop=true` | Кроп или исходный кадр |
| GET | `/api/metrics` | Сохранённый отчёт активной модели |

`POST /api/search` принимает `multipart/form-data`:

- `image`: JPEG/PNG, максимум 15 МиБ и 25 мегапикселей;
- `x, y, w, h`: целые пиксели исходного изображения после EXIF-ориентации;
- `top_k`: 1–100, по умолчанию 10;
- `mode`: `ranking` (Top-N) или `candidates` (Top-N с порогом);
- `threshold`: необязательный cosine-порог от −1 до 1, только для `candidates`.

Нулевые/отрицательные размеры, выход BBox за границы, повреждённые изображения
отклоняются. Вход не обрезается молча до границ. Рамку детектор не угадывает.

Пример запроса по существующему query:

```bash
curl 'http://127.0.0.1:8000/api/queries?limit=1'
```

Скопировать `image_id` из ответа и отправить:

```json
{
  "query_id": "<image_id из test_query.csv>",
  "top_k": 10,
  "mode": "ranking"
}
```

на `POST /api/search/query` с `Content-Type: application/json`.
Ответ содержит `results`: `rank, image_id, x, y, w, h, similarity, rerank_score, crop_url`,
а также `query_id`, `mode`, `confidence`, `refused`, `threshold`, `threshold_source`,
`gallery_size`, `elapsed_ms`, `encoder_fingerprint`.
Для загруженного файла `query_id=null`.
`similarity` и `rerank_score` — **не вероятности совпадения**.

Режим ранжирования не применяет порог. В режиме кандидатов пустой `results`
сопровождается `refused=true`. Без отчёта калибровки и без ручного порога API
вернёт 409: числовой порог не выдумывается. Ручной порог отмечается `manual`.

## Оценка активной модели

```bash
.venv/bin/python -m backend.evaluate --export
```

Команда фиксирует разбиение, прогоняет активную модель, выбирает порог
на calibration, оценивает отдельную validation и создаёт тестовые артефакты.
Без `--export` выполняется только локальная оценка. Повторный запуск заменяет
сгенерированные файлы в `artifacts/`; для сравнения экспериментов сохраняйте копию отчёта.
Экспорт завершается строгой проверкой трёх файлов. Уже существующие артефакты
можно проверить отдельно командой `.venv/bin/python -m backend.evaluate --validate-only`.

Файлы:

- `artifacts/splits.json` — списки identity и query/gallery ID, seed, хэши train-кадров;
- `artifacts/baseline_metrics.json` — метрики активной модели, порог, веса/preprocessing и локальное время;
- PostgreSQL volume `postgres_data` — единственная рабочая gallery и метаданные;
- `artifacts/submission.csv` — без заголовка: 1110 query, по 10 gallery ID;
- `artifacts/embeddings.npy` — `(1860, 512)`, L2-нормированный `float32`;
- `artifacts/candidates.csv` — принятые кандидаты; отсутствие строк query означает отказ;
- `artifacts/export_manifest.json` — порядок ID, хэши, параметры экспорта.

### Протокол оценки

Источник формул — опубликованный организаторами [`evaluate.py`](evaluate.py).
Он и [`example_submission/`](example_submission/) сохранены без изменений.
`backend/scoring.py` только адаптирует предсказания к его функциям; своей копии
формул метрик больше нет. Это относится и к MVP, и к обучению/HPO.

Локальная оценка передаёт ровно первые `min(10, размер gallery)` кандидатов
без предварительного удаления junk: фильтрацию выполняет официальный скрипт.
F1/TNR/PR-AUC также считаются его функцией, включая обработку same-camera
кандидатов и отсутствующих confidence при отказе. Справочные full mAP и mINP
считаются по исходным эмбеддингам, не по реранкингу. Неопределённые `NaN`
в наших JSON/API записываются как `null`, без изменения численных результатов.

Пример содержит 5 query, 8 gallery и векторы `(13, 16)` — это только образец
формата. Проверка: `.venv/bin/python -c "from pathlib import Path; from backend.evaluate import validate_artifacts; p = Path('example_submission'); print(validate_artifacts(p, p))"`.
Для реального датасета с 750 gallery экспорт по-прежнему содержит Top-10.

При наличии размеченного ground truth готовые файлы можно проверить напрямую:

```bash
.venv/bin/python evaluate.py --gt /path/to/ground_truth.csv \
  --submission artifacts/submission.csv --candidates artifacts/candidates.csv \
  --embeddings artifacts/embeddings.npy \
  --query dataset/test_query.csv --gallery dataset/test_gallery.csv
```

Ground truth должен относиться именно к переданным query/gallery. Тестовой
разметки организаторов в репозитории нет; пример не содержит эталонных метрик.

У официальных test query/gallery нет `vehicle_id`, поэтому их mAP локально
вычислить нельзя. Метрики ниже получены из размеченного `train.csv`:

- seed `20260915`, примерно 60/20/20 по группам identity;
- 925 identity использованы для development-обучения, 307 — для calibration,
  309 — для validation; calibration/validation identity не попадали в обучение;
- все identity, связанные одинаковыми SHA-256 кадров, остаются в одном разделе;
- по одному query на identity, галерея содержит остальные камеры этой identity;
- у примерно 20% identity все gallery-снимки убраны: проверяем отказ при отсутствии совпадения;
- как junk исключаются только объекты с одновременным совпадением `vehicle_id` и
  `camera_id`; негативы той же камеры остаются в ранжировании;
- `camera_id` используется только для этого протокола, не передаётся в OSNet и не нужен API;
- mAP@10, Rank-1/5 и mINP считаются по query с хотя бы одним допустимым совпадением;
  query без совпадений участвуют в F1/TNR;
- AP@10 нормируется на `min(n_pos, 10)`; справочный full mAP и mINP считаются
  по полному cosine-ранжированию исходных эмбеддингов;
- candidate F1 — micro F1 на уровне query, в расчёт входит только кандидат с
  максимальным confidence;
- TNR — доля отказов среди query без совпадений;
- cosine-порог максимизирует `0.7 * F1 + 0.3 * TNR` **только на calibration**.
  При равенстве выбирается больший порог. Он затем фиксируется для validation/test.

Текущий HPO best-mAP checkpoint обучен только на `identities.train` из сохранённого
разбиения. Финальную модель на train+validation нельзя после этого оценивать на той же
validation как на невиденных автомобилях. Хэши находят точные копии кадров,
но не гарантируют отсутствия похожих соседних кадров.

### HPO best-mAP checkpoint + streaming reranking, официальный evaluator

| Метрика | Calibration | Validation |
|---|---:|---:|
| mAP@10 | 78,72% | **81,47%** |
| Full mAP исходных эмбеддингов, справочно | 77,29% | 80,36% |
| Rank-1 | 79,27% | **80,16%** |
| Rank-5 | 87,80% | **88,26%** |
| mINP исходных эмбеддингов | 69,44% | 74,07% |
| Candidate F1 | 72,64% | 72,86% |
| TNR | 72,13% | 79,03% |
| `0.7 * F1 + 0.3 * TNR` | 72,49% | **74,71%** |
| Query с совпадениями / без | 246 / 61 | 247 / 62 |

Cosine-порог после повторной калибровки официальным кодом: **0.5948754549026489**.
Эти значения — локальная оценка
HPO best-mAP checkpoint с `k1=20`, `k2=3`, `lambda=0.5`, не оценка
закрытого теста и не обещание качества
на произвольных фотографиях. Порог применён к Top-1; изменение состава галереи
может изменить распределение score и качество отказа.

CPU, macOS ARM64, 2 потока, batch=1, 30 повторов после 3 прогревов:
медиана **15,35 мс**, p95 **16,44 мс**, включая JPEG decode, crop/resize, OSNet
и L2-нормализацию. Это локальный справочный замер одного изображения,
не замер GPU, серверной пропускной способности или результат жюри. Реранкинг
измеряется отдельно (в предыдущем замере — **0,315 мс/query**).

### Соглашения для организаторов

`query_id`/`gallery_id` трактуются как `image_id`. В `candidates.csv` поле
`confidence=(maximum_raw_cosine+1)/2` — монотонный score в [0,1], **не калиброванная
вероятность**. Соответствующий порог score — примерно 0.79743773.
Отказ кодируется полным отсутствием строк этого query в `candidates.csv`.
Ни test-разметка, ни номерные знаки для подбора порога не используются.

## Проверки и структура

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Тесты проверяют BBox/RGB/normalization, L2 и стабильную сортировку, mAP@10 и
query-level F1/TNR на примерах с известным ответом, junk-фильтр, форматы трёх
файлов сдачи, разделение identity/кадров, реальную OSNet, согласованность
batch/single inference, кэш, API и отказ.

- `backend/core.py` — обработка изображений, OSNet, repository selection и поиск;
- `backend/app.py` — HTTP-контракт и выдача HTML;
- `evaluate.py`, `example_submission/` — неизменённые файлы организаторов;
- `backend/evaluate.py` — разбиение, запуск инференса, экспорт и проверка файлов;
- `backend/scoring.py` — вызовы официальных метрик и подбор порога на calibration;
- `frontend/index.html`, `frontend/app.js` — простой интерфейс;
- `models/` — исходный ONNX, лицензия, контрольные суммы;
- `tests/` — автоматические проверки.

Версии всех зависимостей зафиксированы в `requirements.txt` и
`requirements-dev.txt`; `.in` содержат исходные ограничения для обновления lock-файлов.
Новый ML-код можно подключать за интерфейсом Encoder, сохранив HTTP-контракт.
При замене весов обязательно пересчитать эмбеддинги и порог.

## Источники

- [OSNet vehicle-reid-0001 / Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/README.md).
- [Официальный ONNX и checksum](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/vehicle-reid-0001/model.yml).
- [Исходная vehicle-ReID ветка](https://github.com/sovrasov/deep-person-reid/tree/vehicle_reid), MIT.
- [K-reciprocal reranking, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.html),
  [код авторов](https://github.com/zhunzhong07/person-re-ranking).
- [Официальный preprocessing](https://github.com/sovrasov/deep-person-reid/blob/vehicle_reid/torchreid/data/transforms.py).
- [ONNX Runtime](https://onnxruntime.ai/docs/api/python/api_summary.html), [FastAPI](https://fastapi.tiangolo.com/).
- Данные: официальный дополненный датасет организаторов; дополнительные датасеты не загружались.

Весь `dataset/` (изображения, CSV и README датасета), временные результаты
и окружение исключены из Git. После клонирования датасет нужно разместить локально.
Стоковый и активный HPO checkpoint входят в репозиторий.
Автоматических commit/push нет.
