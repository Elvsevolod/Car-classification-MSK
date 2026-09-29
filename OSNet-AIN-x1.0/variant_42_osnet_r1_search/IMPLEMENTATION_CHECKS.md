# Проверки v42 — 28 сентября 2026

Это приёмка реализации, **не результат 18 полноценных обучений**.

## Компонентные проверки

В существующем Python 3.11 / torch 2.14.0 окружении проекта:

```bash
ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  .venv/bin/python -m pytest -q \
  tests/test_osnet_r1_search.py tests/test_parallel_research.py tests/test_research_mac.py \
  tests/test_transreid_night.py tests/test_transreid_system.py \
  tests/test_transreid_full_train.py tests/test_transreid_graph.py tests/test_nive_mixed.py
```

**182 passed, 13 skipped**. Пропуски — opt-in GPU/MPS smoke, не скрытые падения.
Новые v42 unit-тесты: 17 passed; ещё шесть настоящих MPS проверок выполнены отдельно.

Проверено: 18 различных условий; точное совпадение числа предъявлений; неизменный
LR horizon между оценками; CE/metric/consistency; обе loss-функции; CPU resume с
точным совпадением параметров, BN buffers и научных полей history; обнаружение
повреждённого checkpoint; сохранение лучшего раннего checkpoint вместо последнего;
предварительный статус при незавершённой серии; продолжение очереди после OOM,
остановка при нарушении целостности; отказ от CPU fallback/Rosetta; контроль места;
корректность notebook Run All. Допуски для восстановления не увеличивались.

## Реальные одношаговые MPS проверки

**Apple M4 Pro / 24 GiB**, не Mac mini M4/16. Из исходного fold-matched R1 загружены
настоящие веса; использованы только настоящие организаторские primary-train crop.
Для P16K2, P16K4, P32K2 × SupCon/soft-triplet выполнены clean/robust forward,
backward, AdamW step и повторное извлечение embedding. Во всех случаях конечные
loss/градиенты/512D-векторы, фактические батчи 32/64, FP32 без CPU fallback.

```bash
ORT_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 RUN_R1_MPS_SMOKE=1 \
  .venv/bin/python -m pytest -q tests/test_osnet_r1_search.py -k real_r1_mps
```

**6 passed** (19.74 s для этой проверки, не прогноз длительности полной серии).
Веса изменялись только в оперативной памяти тестового процесса, исходные checkpoints
не перезаписывались. Полные 18 тренировок не запускались.

## Проверка scorer на настоящем сохранённом feature bank v40

Проверены hashes `quick_v1/bn/baseline`, использованы 1036 реальных feature rows,
три draw по 185 query. Новый scorer воспроизвёл **2775 top-10 точно**: raw, graph,
control, G2, no-op replacement. Перестановка/удаление query отдельно покрыты unit-тестом.

| Величина | Воспроизведённый mAP@10 |
|---|---:|
| R1 raw | 0.8328422682886969 |
| R1 less_graph | 0.8440989714203999 |
| C_primary | 0.8494544927580643 |
| C_primary + исходный R1, G2 .10 | 0.8495929645632027 |
| Замена R1 исходным R1 | 0.8494544927580643 |

Это **повторная проверка сохранённых признаков**, не новое обучение, новый extract,
замер производительности или метрика релиза v25. В пользовательском v42 контрольные
эмбеддинги извлекаются заново на выбранном runtime.

## Сохранность и ограничения

- До/после разработки совпал SHA256 всего ранее существовавшего `git diff --binary`:
  `2c7b129789751c819635d391a9ba9e1c4172a3b6b4b1fc0822d4ed7c5e19f459`.
- Отдельно совпали hashes двух v41 runner-файлов, Mac notebook и корневого manifest
  переносимой папки V41_Mac_M4. Изменения этого этапа — только новые файлы v42 и тест.
- Main, рабочий MVP, bbox/validation, датасеты, исторические runs и исходные веса
  не редактировались. Commit/push не выполнялись.
- Полный пользовательский Run All, окончательный mAP, длительность и итоговый
  объём диска ещё не проверены. Запуск на другом аппаратном классе не подтверждён.
- На исходной машине после MPS-проверок оставалось примерно **2.7 GiB** свободно:
  перед полной серией нужно освободить место самостоятельно, предпочтительно до
  5–6 GiB или больше. Preflight остановится при значении меньше 3 GiB.
