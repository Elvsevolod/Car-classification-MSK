# Вариант 1 — CLIP-ReID, двухэтапный vehicle transfer

Дата реализации: 19 сентября 2026 года. Запуск:
`train_clip_reid.ipynb`, kernel репозиторного `.venv`, затем **Run All**.
Установлены существующие `requirements-train.txt`; новых пакетов не требуется.

По умолчанию Run All выполняет полное обучение: **60 prompt-эпох + 60 image-эпох**,
после него — отдельный чистый OSNet-контроль. Для проверки без обучения установи
`RUN_TRAINING=False`. `RUN_OSNET_CONTROL=False` пропустит контроль, но сравнение
дообученных моделей тогда останется неполным. При недоступном GPU CPU-training
не включается молча. MPS smoke с batch 64/32 уже прошёл на локальном Mac 24 ГБ.

## Что проверяем

Способен ли автомобильный CLIP-ReID после переноса на наши данные превзойти
свою исходную версию и честный OSNet-контроль. Исходные веса и SHA-256 описаны
в [SOURCES.md](../SOURCES.md). Не используем raw внешние датасеты или точные
маски в первом обучении: это разные эксперименты.

### Данные и защита от утечки

- Читаем только train.csv. Старый outer train содержит 925 identity.
- Внутри него 741 train-ID / 184 validation-ID; группы общих точных кадров
  остаются в одной части. Проверяются identity и SHA-256 полных кадров.
- Обучение: 4605 кропов. Inner evaluation: 184 query, 512 gallery,
  147 известных query / 37 no-match; фиксированный SEED=20260915.
- Ни calibration, ни outer validation, ни test не участвуют в выборе эпохи/LR.
- Camera ID нужен только sampler и evaluator, не модели.
- Активная дообученная OSNet видела все 925 ID: её нельзя использовать
  как честный inner-контроль. Контроль здесь стартует заново от stock ONNX.

### Предобработка и обучение

На входе — точный crop по BBox, не весь кадр. Evaluation: bilinear 256×256,
mean/std 0.5. Train: bicubic resize → horizontal flip 0.5 → pad10/random crop →
нормализация → RandomErasing 0.5 с pixel noise. Это типовые аугментации, не
точное маскирование номера и не доказательство независимости от анонимизации.

Stage 1 замораживает оба encoder. Проекционные train-image features кешируются
один раз. Учим четыре контекстных вектора для каждого нашего train-ID посредством
двунаправленного multi-positive image/text loss; temperature=1, raw dot product,
как в авторском коде. Старые 576 VeRi ID-контекстов и обе classifier-head сброшены.

Stage 2 фиксирует текстовые признаки классов и учит visual encoder/BN/classifiers:

`loss = 0.25 * (ID_CE_768 + ID_CE_512) + sum(Triplet_3_features) + ImageToText_CE`

CE label smoothing=0.1; triplet — batch-hard, margin0.3, ненормализованные
Euclidean признаки; признаки после блока 11, после блока 12/LN и проекция.
P=8,K=4, batch32; ~144 шага/эпоху. Adam LR5e-6, bias×2, weight decay1e-4,
warmup10, LR drops30/50. Все 60 эпох проходят по фиксированному рецепту;
на этом этапе нет Optuna или автоматического early stopping CLIP.

На инференсе остаются только визуальные признаки: concat(768,512) → L2.
Текст, ID-классификаторы, camera/view SIE, OCR и другие query не используются.

## Отбор и сохранение

- Отбор — максимальный **официальный raw inner mAP@10**. Включена эпоха 0:
  если перенос не помог, сохраняется исходный visual encoder.
- `prompt_last.pt`: prompts, Adam state, история и подпись запуска.
- `image_last.pt`: последний encoder, Adam state, история и лучший encoder
  в одном атомарном checkpoint. Он может занимать около 1.4 ГБ; это training-артефакт,
  не размер сдаваемого inference решения.
- `image_best.pt`: выбранные визуальные веса; контрольные snapshots10/30/50/60.
- `results/prompt_history.json`, `image_history.json`, `training_summary.json`,
  `baselines.json`, `protocol.json`, `RESULTS.md`.
- Консоль: эпоха/всего/осталось, loss, шаги внутри tqdm, время эпохи, прошедшее
  время и ETA текущего этапа. В epoch time входят train+evaluation, но не запись
  больших файлов; ETA ориентировочная и не включает ещё не начатые этапы.
- Необходимы несколько ГБ свободного диска; сохраняется один best, а не каждый
  улучшившийся checkpoint. Все метрики эпох сохраняются.

### Если ноутбук прервался

Restart Kernel → Run All без изменения папки/config. Незавершённая эпоха повторяется
с последней сохранённой границы. Seeds задаются по эпохе; на CPU это проверено
сравнением весов непрерывного и возобновлённого запуска. Побитовая идентичность
разных устройств/версий PyTorch не обещается.

Checkpoint является источником истины, JSON восстанавливаются из него.
Смена данных/config/кода/torch приводит к остановке resume, а не смешению запусков.
Для нового эксперимента укажи новую папку VARIANT. Не удаляй прошлые JSON/веса.
Не запускай два kernel одновременно в одну папку.

## Чистый OSNet-контроль

От stock OSNet, inner-only, Avg/BNNeck/SupCon, P16×K2, LR1e-4/head×10,
до4000 шагов; используются уже проверенные train/augmentation/step-resume helpers.
Это отдельный baseline-рецепт, а не равный compute-бюджет CLIP.
Результаты и веса — `results/osnet_control/`, `weights/osnet_control/`.

## Что уже проверено и что пока не сделано

- Реальный source checkpoint: strict load + зафиксированный checksum.
- Реальные MPS backward и Adam steps, prompt batch64/image batch32.
- ONNX: только image encoder, 1280D, batch1/2, PyTorch parity; около345МБ.
  Это технический экспорт source-весов в `results/cache/`, **не веса для MVP**.
- Baseline на inner: CLIP-ReID37.53%, stock OSNet65.21% mAP@10.
- Полное локальное обучение и чистый обученный OSNet-контроль ещё не запускались.
- Три seed, calibration, outer evaluation, отказ, reranking и production-экспорт
  дообученной модели — следующий этап только при перспективном результате.
- Рабочий MVP, его веса/галерея, файлы организаторов и прошлые опыты не изменены.

## Интерпретация

37.53% — слабее stock OSNet65.21%, поэтому улучшение нельзя обещать заранее.
Если после переноса CLIP остаётся хуже честного OSNet-контроля, не запускаем
автоматически большой поиск и не переносим его в продукт. Если становится лучше,
проверяем несколько seed, устойчивость к маскам, отказ и скорость на стенде.
Сравнивать этот inner result с outer81.47% активного OSNet напрямую нельзя.
