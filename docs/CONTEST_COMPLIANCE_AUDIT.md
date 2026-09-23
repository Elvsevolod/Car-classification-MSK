# Аудит соответствия конкурсным условиям

Проверено 21 сентября 2026 года по конкурсному PDF и `ORGANIZER_QA.md` (вопросы 7-11, 36-41, 44 и 47).

| Условие | Статус | Доказательство |
|---|---|---|
| Pipeline формирует `submission.csv`, `embeddings.npy`, `candidates.csv` | Выполнено | `backend.evaluate`, строгая `validate_artifacts`, успешный экспорт 1110/1860/1051/59 |
| Формат: Top-10, float32, порядок query затем gallery, отказ без строк | Выполнено | `backend.evaluate.export` и `validate_artifacts`; есть тесты форматов |
| Независимая обработка query и статичная gallery | Выполнено | `Gallery.search_with_confidence` использует только текущий embedding и gallery; k-reciprocal работает внутри gallery |
| BBox -> embedding -> поиск -> confidence/refusal | Выполнено | `POST /api/search`, `POST /api/search/query`, `Gallery` |
| Dockerfile и `docker compose up` | Выполнено для CPU-MVP | multi-stage `Dockerfile`, `dataset-init` для прав dataset, PostgreSQL healthcheck и entrypoint с миграциями |
| PostgreSQL + pgvector | Выполнено | миграция Alembic, `PostgresGalleryRepository`, health API и проверка 750 строк |
| Одна команда для экспортного pipeline | Выполнено | `docker compose --profile inference run --rm inference` |
| Отсутствие скачивания весов/пакетов во время runtime | Выполнено для подготовленных образов | ONNX-веса и venv копируются в runtime image; Docker build может использовать сеть, что разрешено Q&A |
| Документация и запуск | Выполнено | `README.md`, `docs/RUNBOOK.md`, `docs/TEST_SUMMARY.md` |
| Интерактивный OpenAPI/Swagger без сети | Выполнено | `/docs` использует локальные `frontend/vendor/swagger-ui/` assets, включённые в Docker-образ |
| React-интерфейс и браузерные smoke-тесты | Выполнено для MVP | production bundle из `web-ui/`; тесты успешного поиска и отказа на ширине 390 px |
| Чистая офлайн-проверка на Linux `amd64` с заранее загруженными образами | Не подтверждено | инструкция и `--pull never` есть, но прогон на целевом стенде ещё нужно зафиксировать |
| GPU-инференс и benchmark на RTX A5000/CUDA 12.2 | Не выполнено | текущий `Encoder` использует `CPUExecutionProvider`; отсутствуют CUDA-образ и GPU benchmark |
| Презентация и сценарий защиты | Не выполнено | презентационный файл в репозитории отсутствует |

## Итог

Обязательный экспортный pipeline, форматы файлов, контейнеризация, API, PostgreSQL + pgvector, React-интерфейс и offline Swagger UI реализованы и протестированы для CPU-MVP. Проект **не следует заявлять как полностью готовый по всем пунктам инженерной части**, пока не будут закрыты прогон на целевом Linux `amd64`, GPU-профиль с замерами и презентация.
