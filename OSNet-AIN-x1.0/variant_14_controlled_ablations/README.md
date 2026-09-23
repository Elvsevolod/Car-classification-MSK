# Variant 14: контролируемый перебор улучшений OSNet

Текущая точка входа: **`train_osnet_gem_mixstyle_repair.ipynb`**. Полные обучения запускает пользователь.
Исследовательская основа: `../../OSNET_SEARCH_QUALITY_RESEARCH_2026-09-22.md`, раздел 9.

## Повторная проверка GeM / MixStyle — 2026-09-23

В завершённом `suite_v1` настройки передавались как `super().__init__(num_classes, config)`,
хотя второй аргумент родительского конструктора — `use_bnneck`, а не объект конфигурации.
Поэтому G1 оставался с average pooling, S2 — без MixStyle. Их checkpoint совпадали с B0;
эти результаты **не проверяют гипотезы GeM/MixStyle**. Исправлена явная передача настроек,
добавлены проверки фактической архитектуры, градиента GeM и действия MixStyle только в train mode.
B0 с его исходными настройками не изменился; forward/loss/градиенты проверяются на точное совпадение.

Что запускать сейчас:

1. Открыть `train_osnet_gem_mixstyle_repair.ipynb` в прежнем окружении `.venv`.
2. **Restart Kernel → Run All**. По умолчанию: MPS, `RUN_NAME = 'suite_v2_gem_mixstyle_fix'`.
3. Дождаться `runs/suite_v2_gem_mixstyle_fix/RESULTS.md` и `summary.json`.

Это отдельный эксперимент **B0 + исправленные G1/S2**, а не повторный перебор всех 22 вариантов.
Оба исправленных варианта обучаются на трёх primary seed; выбранный по primary победитель
проходит три alternate seed и refit с заранее выбранным числом шагов. До **10 новых обучений**.
Если B0 выигрывает на primary, дополнительных обучений на alternate/refit не требуется.
Прогресс остаётся прежним: variant/seed, steps, inner mAP, best, elapsed и ETA.

`REUSE_CONTROL_FROM = 'suite_v1'` повторно использует только семь завершённых B0-обучений:
три primary, три alternate и final refit. Проверяется конкретный исторический manifest по SHA256,
равенство данных, splits, recipe, seed, бюджета, runtime и исходников вне двух исправленных файлов.
Хеши manifest, summary/history/best.pt сохраняются в новом manifest и проверяются при продолжении.
Новый `summary.json` ссылается на исходный B0 checkpoint и сохраняет его подпись; `last.pt`
не копируется и не используется для продолжения старого обучения. G1/S2 из старой серии не импортируются.
Итоговые evaluation/export B0 пересчитываются в **новый** каталог, старые файлы не переписываются.

Для иного устройства/окружения/бюджета нужен новый `RUN_NAME` и `REUSE_CONTROL_FROM = None`;
тогда B0 тоже обучается заново (до 17 обучений). Не обходить проверку несовместимости и не
редактировать старый manifest. Исторический `train_osnet_ablation_suite.ipynb` с выводами и
`runs/suite_v1` сохранены: **не запускать их заново под прежним RUN_NAME после исправления**.
Прежний `VERIFICATION.md` — исторический отчёт: его shape/parity-тесты не обнаруживали выключенные модули.

## Решение пользователя о данных

**Не исправлять bbox, vehicle_id, camera_id, не удалять примеры и не менять состав исходных splits.**
Предложенный ранее D1 с исключением подтверждённых ошибок отменён и в код не включён.
`train.csv`, оригинальные изображения, `artifacts/splits.json`, исходная validation, MVP и backend не переписываются.
Обычные тренировочные аугментации выполняются только в памяти. Маски S1 непрозрачные, crop-local;
они не меняют bbox и не пытаются восстанавливать номера. Clean view остаётся исходным.

## Полный перебор (исходный протокол)

`train_osnet_ablation_suite.ipynb` описывает первоначальный полный перебор.
Для нового полного перебора после исправления обязателен новый `RUN_NAME`; текущий repair-запуск описан выше.

1. Открыть notebook в окружении проекта с PyTorch, torchvision, ONNX, ONNX Runtime, pandas, tqdm, optuna и Jupyter.
   Никаких новых весов/пакетов notebook автоматически не скачивает.
2. В первой конфигурационной ячейке проверить `DEVICE`, `RUN_NAME`, список вариантов и бюджет.
3. `Run All`. На CPU полный перебор по умолчанию заблокирован: для осознанного CPU-запуска есть `ALLOW_CPU_TRAINING`.
4. Результат: `runs/<RUN_NAME>/RESULTS.md`, подробные JSON, checkpoint и экспериментальные ONNX.
5. После прерывания снова `Run All` с теми же настройками. Максимум один незавершённый блок шагов повторится.
   Завершённые обучения не запускаются заново; оценки/export могут быть пересчитаны.
   Не менять код/данные/версии библиотек в процессе. Для другого эксперимента использовать новый `RUN_NAME`.

Можно перенести весь проект на CUDA-машину, включая dataset, stock/MVP ONNX, splits, исторический recipe,
готовый cache масок variant 08 и YOLO11 pilot_01 с provenance. Интернет не требуется.
Уже начатый run нельзя молча продолжать на другом backend/версии библиотек: нужен новый RUN_NAME.

## Что перебирается

Каждый вариант сравнивается с B0; это **не** полный декартов набор комбинаций.

| ID | Единственное основное изменение |
|---|---|
| B0 | Stock-start, BNNeck + CE/SupCon/consistency, 208 square |
| N1 | Frozen running statistics backbone BN; affine обучается, BNNeck обычный |
| N2 | Frozen running statistics всех BN, включая BNNeck; InstanceNorm не меняется |
| C1 | Consistency weight = 0 |
| C2 | EMA target для consistency, decay 0.99; buffers копируются от student |
| P1 | P16K4 (batch 64) |
| P2 | P32K2 (batch 64) |
| A1 | Lower-center occlusion p=0 |
| A2 | Lower-center occlusion p=0.1 вместо 0.35 |
| A3 | RandomErasing p=0.1 вместо 0.3 |
| M1 | AM-Softmax: scale 30, margin 0→0.2 за 100 updates; SupCon сохранён |
| M2 | SupCon memory 1024, detached train features, включение с шага 100 |
| M3 | Batch-hard triplet вместо SupCon |
| M4 | Circle вместо SupCon |
| R1 | 256 вместо 208 на train **и** inference |
| R2 | Shared OSNet trunk + spatial softmax attention/local 128D, итог 640D |
| K1 | Mean/std RGB до input IN → color 32D, итог 544D |
| T1 | Cross-camera positive из easy/hard половины stock cosine similarity |
| S1 | 50% robust views с готовой непрозрачной маской; clean view исходный |
| G1 | GeM, обучается вместе с моделью |
| L1 | Letterbox на train **и** inference |
| S2 | MixStyle; остальные настройки B0 |

R2/K1 объединяют нормализованные main/aux признаки с начальными множителями sqrt(0.8)/sqrt(0.2),
затем обучают общий BNNeck. Это **не** обещание неизменных 80/20 cosine-весов после обучения.
Все параметры новых голов входят в optimizer и checkpoint. Они не требуют второго encoder в inference.

T1 — собственный similarity-proxy sampler, **не** pose estimator, не verified-positive mining и не воспроизведение RPTM.
Он не исключает ни одну identity/строку. Подозрительные примеры остаются; hard mining и XBM могут усилить шум.
M2 исключает из пар одинаковые image_id, не хранит validation и не распространяет градиент через очередь.
M3/M4 используют прежний metric_weight: отрицательный результат относится к этому пилоту, не ко всему семейству loss.

## Протокол

1. Полная проверка SHA256 train.csv и всех 9556 исходных изображений против сохранённого split.
2. Проверка исходных outer partitions и exact-frame-disjoint групп.
   Для всех вариантов общий inner split отделяет также все identity/кадры, использованные локальным YOLO,
   от inner holdout. Это необходимо для честной S1: обычный случайный inner split мог бы протечь через detector.
   Ничего не исключается: inner train ∪ inner validation = исходный outer train.
3. Один фиксированный evaluation seed и одинаковые query/gallery для всех training seed.
   Near-duplicate кадры не проверяются этим exact-SHA аудитом; полной гарантии отсутствия near duplicates нет.
4. Инициализация только из stock vehicle OSNet, никогда из локально обученного MVP.
5. B0 и 21 однофакторный вариант: 1700 updates, warmup 100, cosine LR с неизменным горизонтом 1700.
   Оценка каждые 200 updates, плюс точки 285, 850 и конец. Early stopping отсутствует.
   Best checkpoint выбирается по **raw inner mAP@10**; step 0 тоже допустим.
   Точка 285 не воспроизводит старый epoch-based LR: это явно другой протокол.
6. Два лучших НЕ-control варианта + B0: три одинаковых training seed. Выбор по среднему inner mAP@10;
   при точном равенстве предпочтение B0. Сохраняются sample std и best steps.
7. B0 и выбранный вариант: второй frame-grouped inner split, три seed. Это проверка устойчивости,
   а не новая возможность выбрать победителя. Отрицательный результат не скрывается.
8. До outer оценки фиксируются название варианта, первый training seed, median best step по трём primary seed,
   неизменный LR horizon и final checkpoint. Refit на **всём исходном outer train**.
9. Порог отказа выбирается только на original calibration по 0.7F1+0.3TNR, отдельно для raw/reranked.
   Reranking заморожен: k1=20, k2=3, lambda=0.5, gallery-only, каждый query независим.
10. Исходная outer validation — основная; masked-query, masked-gallery, masked-both — дополнительные 2×2 diagnostics.
    Для них сохраняются original-calibration thresholds, без подбора по масочным validation-результатам.
    Метрики использует неизменённый organizer evaluator с junk filtering.
11. Парный bootstrap по identity, условный на фиксированную gallery; исходные per-query AP сохраняются.
12. ONNX parity batch 1/3/8 на реальных crops, размер весов и CPU forward median/p95.
    Это **не** официальный A5000 полный extract benchmark: его нужно выполнить до решения о внедрении.

Максимум 36 уникальных обучений: 22 screening + 6 дополнительных primary seed + 6 alternate + 2 refit.
Бюджет до 61 200 updates; фактическое время выводится по блокам. Нужны несколько GB свободного места.
P1/P2 видят вдвое больше изображений при одинаковых updates. `samples_seen` фиксируется;
для exposure-matched диагностики сравнивать P1/P2 step 850 с B0 step 1700 (LR horizon остаётся 1700).
Это другая перспектива сравнения, не идентичная траектория оптимизации.

## Где смотреть

- `manifest.json`: данные, код, source hashes, runtime, splits, фиксированные query/gallery и весь план.
- `<primary|alternate|final>/<variant>/seed_<seed>/history.json`: loss, norms, exposure, metric curve, время.
- `last.pt`: единственный источник восстановления (model, optimizer, memory, EMA, best state, история).
- `best.pt`, `summary.json`: производные артефакты, восстанавливаются из last.pt.
- `selection.json`: замороженный выбор до alternate/outer; `final_selection.json`: конкретные final checkpoints.
- `final/.../thresholds.json`, `evaluation.json`, `per_query_*.json`, `encoder.onnx`, `export.json`.
- `paired_bootstrap.json`, `RESULTS.md`, итоговый `summary.json`.

Checkpoint фиксируется атомарно перед производными файлами. Блоки переигрываются с тем же RNG seed;
битовая идентичность между разными GPU/backend не обещается. В notebook отключён cuDNN benchmark,
включены deterministic algorithms (warn-only: неподдержанные операции сообщают предупреждение).
Одновременный запуск одного RUN_NAME в двух kernel запрещён lock-файлом.

## Что сознательно НЕ запускается автоматически

- D1/очистка/исправление разметки — отменены пользователем.
- Внешние датасеты и внешний сильный teacher: сначала выбрать точный source/checkpoint, проверить лицензию,
  provenance и преимущество на inner. EMA C2 не заменяет relational distillation сильного teacher.
- Точное RPTM/GMS: нужен отдельный faithful port и аудит зависимостей; T1 — самостоятельная гипотеза.
- XBM4096 и комбинации победивших изменений: после результатов screening/confirmation, отдельные новые runs.
  Сумма отдельных улучшений не гарантирует улучшение комбинации.
- Автоматическая замена MVP/порогов/API, подбор по closed test, изменение validation, cross-query inference.

Эта validation уже участвовала в исследованиях; результаты остаются development evidence.
Перед продвижением нужны устойчивость по seeds/splits, отсутствие сильного mask-regression, полный performance
benchmark и отдельное решение пользователя. Статус до пользовательского запуска: **подготовлено, не обучено**.
