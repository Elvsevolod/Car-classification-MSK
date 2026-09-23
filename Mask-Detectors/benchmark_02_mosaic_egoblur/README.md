# DeepMosaics и EgoBlur: второй контроль масок

Проверка без обучения на тех же 126 кропах / 140 ручных областях.
Ручной эталон, исходные изображения, ReID и MVP не изменяются.

## Протокол, зафиксированный до просмотра предсказаний

- DeepMosaics: `mosaic_position.pth`, только BiSeNet/ResNet18 для локализации мозаики.
  Восстановление скрытого изображения не используется. Подготовка — BGR, 0..1,
  короткая сторона 360, как у автора. Порог маски 64/255, как в официальном CLI.
  Два заранее определённых варианта: без расширения и с опубликованной
  постобработкой автора (box blur с размером `int(min(h,w)/20)`, второй порог 64/5).
  Сохраняем все связные области; каждую закрываем её ограничивающим прямоугольником.
  Варианты сравниваются отдельно, не выбираем порог/расширение по результатам аудита.
- EgoBlur Gen1: модели номеров и лиц, вход BGR uint8 CHW исходного кропа,
  как в официальном demo; внутреннюю подготовку выполняет TorchScript.
  Confidence > 0.25 для сопоставления с первым экспериментом, NMS IoU 0.3
  отдельно для лиц и номеров; без дополнительного расширения рамок.
  0.25 — наш заранее выбранный порог, не официальный default 0.9.
  Отчёты для номеров, лиц и их объединения сохраняются отдельно.
- Сравнение с ручной разметкой: один-к-одному IoU ≥ 0.5, покрытие площади
  ручных прямоугольников и доля областей, покрытых минимум на 90%; лишняя площадь
  измеряется отдельно. Ручные области не имеют классов, поэтому это совпадение
  с полной разметкой анонимизации, не отдельная точность определения лиц/номеров.
- Нет подбора настроек на этих 126 кропах и нет обучения на них. При недостаточном
  покрытии готовых моделей следующим шагом будет отдельная train-разметка для YOLO11.
- Для внедрения требуется не только совпадение рамок: важны редкие пропуски,
  полное закрытие нужной области, сохранение кузова и проверка на новых кадрах.
  Высокий mAP ReID на этой маленькой галерее не доказывает успешность маскирования.

Веса скачиваются отдельно в `weights/`, исключённую из Git. Источники и контрольные
суммы фиксируются в `models.json`. Никакие кадры не отправляются во внешние сервисы.

## Запуск и артефакты

Основной полный прогон: `results/benchmark.json`, визуальное сравнение —
`results/comparison.html`; выводы — [RESULTS.md](RESULTS.md).
`phase_01/results/` — предварительный прогон DeepMosaics и EgoBlur LP, выполненный,
пока скачивалась модель лиц. Итоговые выводы опираются на полный прогон в `results/`.

Из корня репозитория, с ранее установленным изолированным OpenCV:

```sh
PYTHONPATH=artifacts/mask_detector_runtime .venv/bin/python -m training.mask_specialists
PYTHONPATH=artifacts/mask_detector_runtime .venv/bin/python -m pytest -q tests/test_mask_specialists.py tests/test_mask_detection.py tests/test_clip_audit.py
```

Для повторения выберите новый пустой каталог через `--output`.
`--models deepmosaics`, `--models egoblur_lp egoblur_face` позволяют запустить часть
эксперимента. Прогресс содержит число обработанных кадров, прошедшее время и ETA.
Старые результаты намеренно не перезаписываются.

При новом checkout дополнительно нужны исходники DeepMosaics:

```sh
git clone --filter=blob:none --no-checkout https://github.com/HypoX64/DeepMosaics.git artifacts/mask_detector_sources/DeepMosaics
git -C artifacts/mask_detector_sources/DeepMosaics sparse-checkout set models
git -C artifacts/mask_detector_sources/DeepMosaics checkout b311aaac319439a0a1e8004b4a64c4222c55f38c
```

Модель использует оригинальные `BiSeNet_model.py` и `model_util.py` без изменения
архитектуры; SHA-256 исходников проверяется. Checkpoint загружается через
`torch.load(weights_only=True, map_location="cpu")` и строгий `load_state_dict`.
Подготовка входа проверена на побитное совпадение с реализацией автора.
Для перевода бинарной карты в исходный размер используем nearest-neighbor, затем
прямоугольник каждой связной области. Уверенность компонента — среднее значение
карты вероятности, не калиброванная вероятность корректного bbox.

### Веса

DeepMosaics берётся по прямой ссылке на файл из официальной папки автора, не из
случайного зеркала:

```sh
mkdir -p Mask-Detectors/benchmark_02_mosaic_egoblur/weights/deepmosaics
curl -fL 'https://drive.usercontent.google.com/download?id=11uUaHPVq5zubGP9_xAOb5O1vPu9GK0XR&export=download&confirm=t' -o Mask-Detectors/benchmark_02_mosaic_egoblur/weights/deepmosaics/mosaic_position.pth
hf download projectaria/EgoBlur ego_blur_lp.zip ego_blur_face.zip README.md script/demo_ego_blur.py --revision 9c0b319ad09d0346e8294d7bf569593d3ab865c0 --local-dir Mask-Detectors/benchmark_02_mosaic_egoblur/weights/egoblur --max-workers 1
unzip -n Mask-Detectors/benchmark_02_mosaic_egoblur/weights/egoblur/ego_blur_lp.zip ego_blur_lp.jit -d Mask-Detectors/benchmark_02_mosaic_egoblur/weights/egoblur
unzip -n Mask-Detectors/benchmark_02_mosaic_egoblur/weights/egoblur/ego_blur_face.zip ego_blur_face.jit -d Mask-Detectors/benchmark_02_mosaic_egoblur/weights/egoblur
```

В этом запуске загрузчик HF завис на face-архиве; тот же файл удалось получить через
`curl -fL` по `https://huggingface.co/projectaria/EgoBlur/resolve/9c0b319ad09d0346e8294d7bf569593d3ab865c0/ego_blur_face.zip?download=true`.
SHA-256 архива совпал с идентификатором LFS-объекта. Оба `.jit` распакованы,
проверены и сохранены; временные ZIP и незавершённая загрузка удалены для экономии места.

### Источники и лицензии

- [DeepMosaics, HypoX64](https://github.com/HypoX64/DeepMosaics/tree/b311aaac319439a0a1e8004b4a64c4222c55f38c):
  код GPL-3.0. Отдельные условия веса `mosaic_position.pth` не обнаружены в изученном
  выпуске Google Drive; это не основание автоматически приписывать весам GPL или
  включать их в поставку без проверки. Используется только детектор мозаики,
  модели восстановления не скачивались и не запускались.
- [EgoBlur, Meta Project Aria](https://huggingface.co/projectaria/EgoBlur/tree/9c0b319ad09d0346e8294d7bf569593d3ab865c0):
  Apache-2.0 по README/метаданным автора. Зафиксирована опубликованная версия Gen1;
  это не утверждение, что она оптимальна для камер нашего датасета.
  Вход BGR uint8 CHW и выход `(boxes, labels, scores, dims)` сверены с demo той же версии.

Используется CPU (2 потока), без переустановки PyTorch/torchvision и без изменения
основного окружения. Внешние обучающие изображения не добавлялись.
