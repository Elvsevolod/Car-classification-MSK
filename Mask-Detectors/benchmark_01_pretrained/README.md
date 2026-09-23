# Проверка готовых детекторов на ручной разметке

Изолированный inference-only эксперимент. Ручной эталон — 126 кропов,
140 областей из `OSNet-AIN-x1.0/audit_07_errors_masks/results/masks.json`.
Эталон, исходные изображения и MVP не изменяются.

До первого прогона зафиксированы: confidence >= 0.25; IoU сопоставления >= 0.5;
YOLO11 NMS IoU = 0.45; у YOLOv9 NMS встроен в опубликованный ONNX.
Рамки не расширяются. Детекция выполняется на исходном целевом кропе до ReID resize.
Пороги и размеры масок на этих 126 кропах не подбираются.

Детекторы ищут номерные пластины. Ручные маски не имеют классов и могут включать
лица: результаты характеризуют совпадение со ВСЕМИ ручными областями анонимизации,
а не отдельную точность детекции номеров. Все предсказания остаются непроверенными
(`reviewed: false`) и сохраняются отдельно от `masks.json`.

Модели: `ankandrew/open-image-models` YOLOv9-T 640 end-to-end ONNX и
`morsetechlab/yolov11-license-plate-detection` YOLO11n ONNX.
Версии, URL, лицензии и SHA-256 фиксируются в `models.json`.
Все вычисления локальные; фотографии никуда не загружаются.

## Результат

Прогон выполнен 20.09.2026. Выводы и ограничения — [RESULTS.md](RESULTS.md).
Численные результаты — [results/benchmark.json](results/benchmark.json).
Визуальное сравнение всех 126 кропов — [results/comparison.html](results/comparison.html):
зелёный — ручные рамки, красный — предсказания. HTML работает без сервера.

## Повторение

Все команды выполняются из корня `Car-classification-MSK`. Используется существующая
`.venv` проекта. OpenCV установлен в отдельный каталог, зависимости MVP не обновлялись:

```sh
uv pip install --target artifacts/mask_detector_runtime --no-deps opencv-python-headless==4.13.0.92
```

Веса уже скачаны локально и исключены из Git. Для нового checkout:

```sh
mkdir -p Mask-Detectors/benchmark_01_pretrained/weights/yolov9_t
curl -fL --retry 2 'https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx' -o Mask-Detectors/benchmark_01_pretrained/weights/yolov9_t/yolo-v9-t-640-license-plates-end2end.onnx
hf download morsetechlab/yolov11-license-plate-detection license-plate-finetune-v1n.onnx README.md --revision 251a30d7daedca065f56e04b0af04052c907c68f --local-dir Mask-Detectors/benchmark_01_pretrained/weights/yolo11n --max-workers 1
```

Проверки и повторный прогон:

```sh
PYTHONPATH=artifacts/mask_detector_runtime .venv/bin/python -m pytest -q tests/test_mask_detection.py tests/test_clip_audit.py
PYTHONPATH=artifacts/mask_detector_runtime .venv/bin/python -m training.mask_detection --output Mask-Detectors/benchmark_01_pretrained/results_repeat
```

Каталог вывода должен отсутствовать или быть пустым: сохранённые результаты намеренно
не перезаписываются. Нужны исходный `dataset`, ручной эталон, текущий OSNet ONNX,
`artifacts/baseline_metrics.json` и `artifacts/splits.json`. Проверяются SHA-256 весов,
фреймов, fingerprint плана и соответствие bbox. Порог отказа берётся из замороженной
калибровки; заново не подбирается. Исходные и ручные ReID-метрики пересчитываются в том же запуске.

Координаты переводятся обратно после letterbox, обрезаются по границам кропа;
левая/верхняя границы округляются вниз, правая/нижняя — вверх до целого пикселя.
Дополнительного расширения маски нет. YOLOv9 использует дробный padding из кода автора,
YOLO11 — целый left/top из примера ONNX Runtime Ultralytics.

## Источники и лицензии

- [open-image-models](https://github.com/ankandrew/open-image-models), код препроцессинга
  проверен на commit `f22000e02b30642f317cdba7755c0631638b109e`.
  MIT у обёртки **не подтверждает автоматически** лицензию весов: перед включением
  в поставку нужно отдельно выяснить условия весов и upstream YOLOv9.
  Release `assets` может изменяться, поэтому фактический файл закреплён SHA-256.
- [YOLO11 License Plate Detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/tree/251a30d7daedca065f56e04b0af04052c907c68f),
  автор morsetechlab, AGPL-3.0 по карточке и метаданным ONNX.
  Карточка предупреждает об утечке train/test в использованном Roboflow-датасете;
  опубликованные авторами метрики не используются как доказательство качества.
- [Ultralytics ONNX Runtime example](https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime/main.py)
  — источник подготовки входа и декодирования стандартного raw YOLO head.
  У скачанного YOLO11 проверен фактический выход `1 × 5 × N`, без встроенного NMS.

Внешние изображения в наш train не добавлялись. Использование готовых весов и их
перераспространение — разные вопросы; этот эксперимент не встраивает модели в MVP.
