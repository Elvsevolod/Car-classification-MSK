# v32: готовность N0/N1 notebook — 2026-09-27

Реализован изолированный первый quality-пилот. **Полное обучение N0/N1 не запускалось.** Для него открыть `train_nive_mixed.ipynb`, выбрать проектный `.venv`, выполнить Restart Kernel → Run All. Оставить `RUN_NAME='pilot_v1'`, `DEVICE='mps'`. При завершении появится `runs/pilot_v1/REPORT.md`.

## Проверено

- 111 тестов прошли: `test_nive_mixed.py`, `test_nive_transfer.py`, `test_overnight_training.py`, `test_retrieval_policy.py` (14,09 с). Пять предупреждений относятся к старому ONNX exporter/InstanceNorm; ошибок нет. `ORT_DISABLE_TELEMETRY=1` задан до импортов.
- Тесты проверяют namespaces, отсутствие holdout в обучении, сопоставимость основных пачек/аугментаций, раздельные головы и loss, один optimizer step, BN-порядок, точное CPU-resume с optimizer/RNG, прерывание между записью resume-слота и переключением указателя, повреждённые файлы и полный синтетический цикл.
- На настоящих изображениях MPS выполнил **по три одноразовых update** для N0 и N1: два mixed и один target-only. Loss и градиенты конечны. Smoke-веса не являются родителями полного опыта.
- Для обеих smoke-моделей проверены head-free PyTorch/ONNX экспорт, 512D, единственный вход — изображения, CPU ONNX parity и batch=1 против batch=8. Допуск фиксирован: абсолютная ошибка не более `2e-5`, без относительного допуска.

| Smoke-ветвь | Максимальная ошибка ONNX против PyTorch | Batch 1 против 8 |
|---|---:|---:|
| N0_target_extra | 5.77419996e-7 | 1.49011612e-7 |
| N1_nive_mixed | 6.33299351e-7 | 1.56462193e-7 |

Среда: Python 3.11.9, PyTorch 2.14.0, NumPy 2.4.6, macOS 15.7.3 arm64, MPS; ONNX проверялся на CPU. Эти проверки не подтверждают CUDA/Linux и не являются замером конкурсной скорости.

## Реальные данные и reference

Подтверждён parent R1 primary, seed 20260915, step800, обученный на 740 identity; 185 inner-holdout identity в его обучение не входят. Внешний источник: только NiVe train, 17070 фото / 703 identity. Test и маски не участвуют в loss.

Byte-аудит повторён на окончательной версии. Сохранённый dHash-аудит переиспользован после проверки fingerprint изображений, split и кода аудитора; это не выдано за новый визуальный осмотр. По критерию dHash64/Hamming<=3: **0 cross-domain подозрений**, 567 внутренних похожих пар NiVe. Метод не исключает все возможные near-duplicates. Planned auxiliary coverage: N0 — 4612/4612 organizer train-фото; N1 — 17070/17070 NiVe train-фото. Фактическое покрытие полного обучения будет в его отчёте.

На окончательной версии выполнена оценка **неизменённого N_ref**, без нового обучения:

| Primary draw | Raw mAP@10 | Fixed graph mAP@10 |
|---|---:|---:|
| regular_20260915 | 0.826654 | 0.836130 |
| regular_20261016 | 0.841372 | 0.853779 |
| regular_20261117 | 0.830501 | 0.842389 |
| Среднее | 0.832842 | 0.844099 |

В каждом draw 185 query, 148 оцениваемых known-query; gallery содержит 496/501/505 фото. Граф: 20/3/0.75. Это inner reference, **не новая лучшая метрика и не сравнение с v25 на исходной validation**. Разбиения делят identity и не являются независимыми наборами.

## Артефакты и неизменность

Окончательная проверка: `runs/implementation_verified_v1/`, manifest signature:
`18cf2c779202cbcf09e42541a4f789716d6a31ceacdfcb232243c465c23c7ce1`.

- `technical_smoke/report.json`: реальные update/loss/градиенты, export receipts.
- `evaluation/N_ref/`: признаки, порядок изображений, raw/ranking метрики и per-query AP/top-10.
- `manifest.json`, `domain_audit.json`: происхождение, фиксированный план, fingerprints и диагностика данных.
- `implementation_preflight_v1/` — сохранённая предыдущая техническая проверка до изменения atomic-resume; не использовать её для продолжения окончательным кодом.

В конце повторно проверены 44227 защищённых файлов и хеши исходников. Active profile остаётся `MVP_fusion_v25`. `release_decision.json` SHA256: `c95695b3eb3613c1d436e4622647e32f925c7cad814b723105f2a13a632767be`; `models/profiles.json`: `b91bd027779247fe6726b4b84229bb5495e5b59daa4ea60788392b073f7207b4`.

Рабочее приложение, его веса/порог, bbox/CSV, evaluator и прежние runs не изменялись. Исходная validation не оценивалась, внешние данные не публиковались, push/merge не выполнялись. Все новые результаты находятся в v32. Следующий шаг — пользовательский Run All и сравнение N_ref/N0/N1; обещания прироста пока нет.
