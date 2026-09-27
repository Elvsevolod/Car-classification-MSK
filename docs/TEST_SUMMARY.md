# Итоги тестирования

## Улучшение UI: 25 сентября 2026 года

- Обновлённый production UI: build прошёл, lint без ошибок (три прежних Fast Refresh warnings shadcn).
- На Docker-сервисе с полной gallery (750) и query (1110): **18 браузерных тестов passed, 25,5 с**, Chromium; 10 обновлённых сценариев и 8 новых.
- Новые проверки: сравнение/crop/EXIF, независимое увеличение, закрытие и фокус, смена исходника, ошибки изображений, 8/+8 миниатюр и поиск ID, значения validation-метрик и неблокирующее отсутствие отчёта.
- Белая тема проверена на 390/768/1440 CSS px и при системной dark theme; отдельно проверен reflow-эквивалент 200% (720 CSS px, device scale factor 2), мобильный переход к результатам и возврат к исходнику.
- Визуальная проверка подтвердила работу desktop sticky-crop и читаемость окон сравнения/примеров/метрик. Backend и модель не менялись; полный backend regression и GPU benchmark в этот этап не входили.

Подробности реализации, образ и снимки — в [UI_DOCKER_REVIEW.md](UI_DOCKER_REVIEW.md#выполнение-согласованного-ui-плана--25-сентября-2026-года). Отложенные замечания не закрывались этим UI-этапом.

## Исправления ревью: 24 сентября 2026 года

Проверки выполняются в изолированном worktree от `devops`; активные веса и исходные сохранённые артефакты MVP не заменяются. Это инженерные проверки, не новая ML-оценка. Протокол и качество модели — в [MODEL_REPORT.md](MODEL_REPORT.md).

| Проверка | Подтверждённый результат |
|---|---|
| Локальные calibration / baseline / official-evaluator / inference / dataset-cache тесты | 72 passed, 2 dependency deprecation warnings |
| Полный Docker regression с PostgreSQL 16 + pgvector | 74 passed, 2 upstream deprecation warnings; включая настоящий ONNX и InMemory/PostgreSQL parity |
| Свежие web-volumes, test-only fixture (3 query / 127 gallery) | API ready; default threshold `0.5948754549026489`, без предварительной калибровки или export |
| PostgreSQL против frozen SQLite fixture на полной gallery (750 объектов) | Все 3 query: raw Top-50, reranked Top-10 и refusal идентичны; max raw cosine difference `1.4901161193847656e-7` |
| Полный batch в Docker с `--network none` | Успешно; смонтированы только test CSV и `images/`, без `train.csv` и БД |
| Выходные данные | 1110 query, 750 gallery; `embeddings.npy` — `(1860, 512)`, `float32` |
| Refusal | 1051 принятый query, 59 отказов |
| Top-10 против ранее сохранённого MVP | `submission.csv` побайтно совпадает |
| Candidate пары | Все 1051 пары query/gallery совпадают с сохранённым MVP |
| Численная погрешность confidence | Максимальная абсолютная разница `1.7881393432617188e-7`, ниже допуска `2e-6` |
| Численная погрешность embeddings | Максимальная абсолютная разница `3.2782554626464844e-7`, ниже допуска `2e-6` |
| Финальный runtime image, автономный smoke-export | `--network none`, без bind-подмены кода/модели и без БД/train: 3 query / 127 gallery, три файла валидны |
| React production build / lint | Build проходит; lint без ошибок, 3 Fast Refresh предупреждения в компонентах shadcn (`button`, `badge`, `toggle`) |
| Playwright на полной gallery (750) и query (1110) | 10 passed: 6 real-API сценариев и 4 контролируемых проверки ошибок/гонок |
| Белая тема и адаптивность | 390 / 768 / 1440 px, без горизонтального overflow, остаётся светлой при системной dark theme |
| UI-контракт | Upload/query, BBox мышью и с клавиатуры, default/manual threshold, ranking/accept/refusal, JSON download, изображения и метрики |
| Устаревшие результаты | Изменение параметров очищает выдачу; поздний ответ поиска, fetch изображения и decode не заменяют актуальное состояние |

SHA-256 совпавшего `submission.csv`: `b1dff828048a5247ee97846e453c2f42a4262339af7c0806cbf50b1d0cb447a8`.

Локальный набор проверок после установки Python 3.11 и зависимостей:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_calibration.py tests/test_baseline.py \
  tests/test_official_evaluation.py tests/test_infer.py tests/test_dataset_init.py
```

Browser-проверки воспроизводятся командой `cd web-ui && npm ci && npx playwright install chromium --only-shell && npm run test:e2e` при запущенном приложении на `http://127.0.0.1:8000` (другой адрес задаётся через `E2E_BASE_URL`). Скриншоты сохраняются в игнорируемой папке `web-ui/test-results/`; desktop и mobile просмотрены визуально. Дополнительная ручная проверка выполнена через Playwright CLI.

Docker-проверки выше выполнены в Linux ARM64 (OrbStack на macOS), не на целевом Linux `amd64`; GPU benchmark в этот этап не входит. Исторические результаты ниже не заменяют новые проверки.

## Предыдущая версия devops: исторический отчёт

Приведённые ниже результаты относятся к версии до исправлений batch/калибровки/кэша и белого UI и не подтверждают регрессию текущих изменений.

Дата последнего полного прогона: 23 сентября 2026 года. Тесты запускались в изолированном Docker Compose-проекте с временной PostgreSQL 16 + pgvector БД; рабочие контейнеры и volumes не использовались.

| Проверка | Результат |
|---|---|
| Unit и API-тесты | 41 passed |
| Предупреждения | 2 upstream deprecation warnings (`Starlette TestClient` / `anyio`) |
| PostgreSQL + pgvector | миграция, запись gallery, exact cosine search и invalidation cache проверены |
| Docker Compose | `docker compose --profile inference config --quiet` прошёл |
| Offline Swagger UI | `/docs` ссылается только на `/static/vendor/swagger-ui/`, без CDN |
| API после перезапуска | `/api/health`: `PostgresGalleryRepository`, 750 объектов gallery |
| React production build | TypeScript и Vite build прошли; lint без ошибок (одно известное предупреждение Fast Refresh) |
| Browser smoke-test | 2/2: официальный query → Top-N и `candidates` → отказ на ширине 390 px без горизонтального overflow |
| Экспорт артефактов | успешно через PostgreSQL + pgvector |
| `submission.csv` | 1110 строк, у каждого query Top-10 без заголовка |
| `embeddings.npy` | shape `(1860, 512)`, `float32`, L2-нормированные векторы |
| `candidates.csv` | 1051 принятый query, 59 отказов (отсутствующие строки) |

Команда полного регрессионного прогона:

```bash
docker compose -p vehicle-reid-tests -f docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
docker compose -p vehicle-reid-tests -f docker-compose.test.yml down -v
```

Это не GPU-бенчмарк. Текущая модель запускается через CPU ONNX Runtime; измерение latency/FPS на NVIDIA RTX A5000 остаётся отдельной незавершённой задачей.
