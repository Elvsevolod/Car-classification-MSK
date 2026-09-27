# v33 — проверка готовности, 2026-09-27

**Готов для пользовательского Run All. Полный N0/N1 с auxiliary 0.025 не запускался.** Рабочий v25, код/результаты v32, bbox и исходная validation не изменялись.

## Проверки

- 130 тестов прошли за 17,30 с: `test_nive_low_aux.py`, `test_nive_mixed.py`, `test_nive_transfer.py`, `test_overnight_training.py`, `test_retrieval_policy.py`. Пять прежних предупреждений ONNX/InstanceNorm, ошибок нет.
- Проверены единственное изменение коэффициента, запрет других изменений условий, десятикратное уменьшение auxiliary-градиента на одинаковых входах в обеих ветвях, точное CPU-resume, отдельный smoke, полный синтетический цикл с историческим сравнением и защита при изменении N_ref.
- Настоящий preflight на MPS подтвердил одинаковые parent, данные, индексы main/auxiliary, LR/optimizer/scheduler, BN, draw-протоколы и runtime. В плане различаются только `experiment` и `aux_weight` (0.25 → 0.025).
- Выполнено по три одноразовых update на ветвь: два joint, один target-only. Это проверка trainer/resume/export на коротком техническом scheduler, не полный quality-run и не его начальные веса.
- Обе smoke-модели экспортированы без классификационных голов, 512D, единственный вход — изображения. CPU ONNX/PyTorch и batch1/batch8 parity прошли с прежним atol 2e-5, rtol 0.

| Smoke | Максимальная ONNX ошибка | Batch1/8 ошибка |
|---|---:|---:|
| N0_target_extra | 9.23871994e-7 | 1.56462193e-7 |
| N1_nive_mixed | 6.40749931e-7 | 1.56462193e-7 |

N_ref извлечён заново: **точное совпадение признаков, порядка и метрик с v32**. Mean fixed-graph mAP 0.8440989714203999, raw 0.8328422682886969. Это повтор исходной контрольной точки, не прирост качества.

В конце проверены 44321 защищённый файл и хеши исходников. Предыдущий `training/nive_mixed.py` остался побайтно прежним: SHA256 `6a31b74b9f03294393508ede5216e5174a4f825f304de77d77f840621157cbb8`.

## Где доказательства

Технический run: `runs/implementation_verified_v1/`. Manifest signature:
`77582e1b6f82b0e316cb41a36b60fb4e682fa78fbdd466e6f09be0c292883f40`.

- `manifest.json`: сопоставимость и hashes, источник v32, повторно проверенный сохранённый dHash-аудит.
- `technical_smoke/report.json`, `technical_smoke/training/*/history.json`: обновления и экспорт обеих ветвей.
- `evaluation/N_ref/`: свежий неизменённый reference.

Среда: Python 3.11.9, PyTorch 2.14.0, NumPy 2.4.6, macOS 15.7.3 arm64, MPS. Это не CUDA/Linux или конкурсный benchmark. Полный `runs/low_aux_v1/` появится после запуска пользователем `train_nive_low_aux.ipynb`.
