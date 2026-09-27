# v32 — сопоставимое совместное дообучение R1 с NiVe

Первый quality-этап плана `Car_ReID_DEEP_RESEARCH_AND_NEXT_PLAN_2026-09-27.md`, разделы 10–12 и задачи 0, 2–5. GPU/Linux приёмка, DINO, alternate fold, full refit, смена candidate/threshold и promotion в этот запуск не входят.

Открыть `train_nive_mixed.ipynb` с проектным `.venv` и нажать **Run All**. На этом Mac `DEVICE='mps'`. При недоступном выбранном устройстве запуск завершится с ошибкой, без тихой замены на CPU. Лимита времени нет; показываются этап, ветвь, update, loss и затраченное время. Результаты — `runs/pilot_v1/`. Полное обучение запускает пользователь.

Результаты проверки реализации: [IMPLEMENTATION_CHECKS.md](IMPLEMENTATION_CHECKS.md) — 111 тестов, реальный MPS smoke обеих ветвей и оценка неизменённого inner N_ref. Это ещё не результаты полного N0/N1.

## Зафиксированный опыт

| Условие | Parent | Main | Auxiliary |
|---|---|---|---|
| N_ref | R1 primary, seed 20260915, step800 | Без обучения | — |
| N0_target_extra | Тот же parent | Organizer train | Другие organizer train фото |
| N1_nive_mixed | Тот же parent | Те же organizer train пачки | NiVe train |

Parent из v16 обучен на 740 identity, inner-holdout — 185 других ID. Проверяются hash manifest/summary/checkpoint, подпись обучения, train-строки, labels, fold и отсутствие пересечения. Full-train encoder из v25 не используется для inner-контроля.

Вход 256, encoder 512D, прежние R1 аугментации и CE/SupCon/consistency. Общие backbone и BNNeck; main-head сохраняется из parent, независимая auxiliary-head создаётся заново: N0 — 740 классов, N1 — 703 до исключения подтверждённых дубликатов. Namespace `organizer:<id>` и `nive:<id>` не смешиваются. Metric-loss только внутри домена, без cross-domain negatives. Heads не входят в экспорт.

По 1600 joint + 200 target-only updates на условие. Обе пачки P16K2=32. Auxiliary forward/backward выполняется первой, main — второй; один optimizer.step на update. Loss `L_main + 0.25*L_aux`; это не обещание 25% градиента. BN running stats обновляются на robust views одинаково в N0/N1, clean-consistency views используют eval/no_grad. На target-only конце auxiliary отсутствует в обеих ветвях.

Fresh AdamW, LR backbone **1.1940564013110388e-5**, головы **1.1940564013110388e-4**, weight decay **6.069870050850335e-5**, warmup 100 updates, cosine до 1800, min LR ratio 0.02. Остальные численные loss/augmentation параметры сохраняются в manifest из конкретного parent-рецепта, не из дефолтов другой серии.

Всего **108800 логических предъявлений на ветвь**. N1: 57600 organizer + 51200 NiVe; N0: все 108800 organizer. Main-exposure совпадает, полное target-exposure — намеренно нет. С clean/robust — 217600 image-forwards, но не столько уникальных фото. Smoke и evaluation считаются отдельно от этого тренировочного бюджета.

Main PK-сэмплер прежний, одинаковые индексы и per-update augmentation seeds для N0/N1. Auxiliary в обеих ветвях циклически обходит фотографии identity, предпочитает другую camera/view-группу, N0 не повторяет main-фото того же update. NiVe prefix — proxy ракурса, не доказанный camera-ID. Planned и actual coverage сохраняются отдельно. Градиенты общего backbone от weighted auxiliary и main логируются каждые 25 updates, BN drift — каждые 100.

## Отбор без новой сетки

Сохраняются точки **0/400/800/1200/1600/1800**. На каждой: raw и фиксированный single-R1 graph `20/3/lambda=.75`, три ранее определённых primary query/gallery draws, Rank-1/5, Hit@10/50 и per-query AP/top-10. Нет threshold fitting, обхода исходной calibration/validation, подбора графа или выбора по train-loss.

Каждая ветвь выбирает свой конкретный checkpoint по среднему graph mAP этих draws; при равенстве — меньший update. Затем ровно две фиксированные 50/50 cosine-смеси: N_ref+N0 и N_ref+N1. Векторы конкатенируются после поблочной нормировки, координаты независимых encoder не усредняются.

`continue` означает положительный screening single или смеси относительно matched controls, не разрешение релиза и не обещание hidden-test прироста. При отсутствии выигрыша и coverage ниже 90% — `diagnose_once`; иначе `stop` для этого рецепта. Окончательная интерпретация учитывает per-query изменения и BN/loss histories. Несколько draws делят identity и не являются независимыми датасетами.

## Данные и ограничения

Переиспользуется `audit_nive`, но не старый `prepare()`/orchestration v15. Ни один старый файл не переписан. Проверяется прежний user-confirmed fingerprint NiVe; используются только 17070 train-фото/703 identity. Внешний test и `_MK_PURE` не используются в loss. [Карточка NiVe, версия 1](https://data.mendeley.com/datasets/42wv2svztx/1) указывает CC BY 4.0; свидетельство и attribution — `SOURCE_NIVE.json`. Никаких загрузок в процессе обучения и никакого push датасета.

Помимо byte audit выполняется dHash64/Hamming<=3: NiVe train против всех organizer crop и полных кадров из размеченного `train.csv`, включая защищённые holdout. Внутренние source-похожие кадры сохраняются отдельно. Это поиск подозрений, не доказательство дубликата и не гарантия отсутствия сложных near-duplicates. При cross-domain подозрениях preflight останавливается до обучения.

После просмотра всех подозрений создать в новом run `near_duplicate_review.json` с `suspects_sha256 = digest(domain_audit['cross_domain_suspects'])` и `decisions`, где каждому external-пути соответствует `not_duplicate` или `confirmed_duplicate`. Только подтверждённый внешний дубликат исключается из нового loader; файл не удаляется, organizer не меняется. Не отмечать все пары автоматически как `not_duplicate`. Если подозрений нет, этот файл не нужен.

В `domain_audit.json` также есть распределения числа фото/групп по ID, яркости, blur при 64px, исходных размеров и фиксированные random sample ID. Яркость не считается достоверной меткой ночи. Ручной визуальный аудит не выдаётся за автоматически выполненный. Номерная устойчивость нового encoder остаётся отдельной проверкой финалиста; OCR/маски номерной зоны здесь не добавляются.

## Resume, экспорт и защита

Возобновлять тем же RUN_NAME, с неизменным кодом/config/device/runtime. Каждые 100 updates сохраняются model, optimizer, RNG, BN, история и checksum: сначала неактивный из двух resume-слотов, затем атомарно переключается `resume.json`. Сбой между этими операциями сохраняет предыдущую рабочую точку. При прерывании проигрывается лишь незавершённый блок. Выбранные checkpoint не перезаписываются. CPU synthetic тест сравнивает continuous/resumed состояния побайтно и моделирует прерывание записи; это не обещание побайтной тождественности разных устройств/версий библиотек.

После отбора создаются head-free `encoder.pt` и `encoder.onnx`, 512D, только изображения на входе; проверяются PyTorch/CPU ONNX parity и batch=1 против batch=8 с фиксированным atol 2e-5. Это исследовательский экспорт, не новый profile или полный конкурсный комплект. Порог v25 к нему не приписывается.

v25 release_decision/profiles/веса, исторические источники, organizer изображения/CSV, evaluator, прежний NiVe manifest и новый код защищены checksums. Проверки до/после прогона и после каждой ветви запрещают незаметную подмену. Готовые evaluation/export tasks переиспользуются только после проверки receipts; это не новые замеры скорости. Рабочие исходники приложения, БД и gallery-cache не изменяются.

Технический smoke делает только три disposable updates на ветвь в отдельном подкаталоге, проверяет loss/градиенты и экспорт. Настоящий N0/N1 всегда стартует с неизменённого parent, не со smoke-весов.

Стандартные проверки: `tests/test_nive_mixed.py`, `test_nive_transfer.py`, `test_overnight_training.py`, `test_retrieval_policy.py`. Для macOS переменная `ORT_DISABLE_TELEMETRY=1` задаётся **до** импортов, включая процесс pytest.
