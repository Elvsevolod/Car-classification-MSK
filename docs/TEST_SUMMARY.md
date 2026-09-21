# Итоги тестирования

Дата последнего полного прогона: 21 сентября 2026 года. Тесты запускались в изолированном Docker Compose-проекте с временной PostgreSQL 16 + pgvector БД; рабочие контейнеры и volumes не использовались.

| Проверка | Результат |
|---|---|
| Unit и API-тесты | 41 passed |
| Предупреждения | 2 upstream deprecation warnings (`Starlette TestClient` / `anyio`) |
| PostgreSQL + pgvector | миграция, запись gallery, exact cosine search и invalidation cache проверены |
| Docker Compose | `docker compose --profile inference config --quiet` прошёл |
| Offline Swagger UI | `/docs` ссылается только на `/static/vendor/swagger-ui/`, без CDN |
| API после перезапуска | `/api/health`: `PostgresGalleryRepository`, 750 объектов gallery |
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
