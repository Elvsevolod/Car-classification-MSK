# Источники данных, весов и кода v40/v41

Проверено при подготовке 28.09.2026. Это исследовательский запуск, не решение о
публикации новой конкурсной модели. Данные/веса в Git не включаются.

## NiVe1303 — добавленный вручную второй датасет

**Официальная страница скачивания:**
[NiVe1303 v1, Mendeley Data](https://data.mendeley.com/datasets/42wv2svztx/1).

Ruozheng LI, DOI `10.17632/42wv2svztx.1`, CC BY 4.0. Используется только исходный
train: 17 070 фотографий, 703 identity. Test, masks и производные маскированные
изображения исключены. Файлы копируются без изменения; источник и SHA256 каждого
файла находятся в `inputs.json`. Ни camera_id, ни имя файла не являются входом модели.
Скачать отдельно нужно только при отсутствии соответствующих файлов в переносном
пакете; сам NiVe не заменяет организаторские данные и наши checkpoints.

## Наши контрольные веса

- B0_208 seed20260915 и R1_256 seeds20260915/16/17, step800: v16/review_v1/primary.
- N1: v33/low_aux_v1, step1800. Он нужен для быстрой проверки BN, а не как
  инициализация новых NiVe-абляций: они начинают с R1 seed20260915.
- Все эти checkpoints обучались на 740 primary train-ID. В пакете сохранены
  исторические manifests, selection, summaries, signatures и hashes.
- Базовая автомобильная OSNet:
  [vehicle-reid-0001, Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/vehicle-reid-0001).
  Используются существующие архитектура/загрузчик проекта, новый encoder не подменяется.

## TransReID / DeiT

- [Официальный TransReID](https://github.com/damo-cv/TransReID), Apache-2.0.
  В проекте уже есть `training/transreid_vendor` с исходниками и лицензией.
- [Официальный DeiT](https://github.com/facebookresearch/deit), Apache-2.0.
- Инициализация `deit_small_distilled_patch16_224-649709d9.pth`:
  `https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth`.
- SHA256 `649709d94f9fd790ea86c16f99d788e709b86a1f64315a19d887895f9948fb09`.
  Файл включён во входной пакет, его повторное скачивание не требуется.

## DINOv2, без ожидания доступа к DINOv3

- [Официальный код и список весов](https://github.com/facebookresearch/dinov2).
- Зафиксированный source revision: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- S14/B14 non-register, официальный pretrain LVD-142M:
  `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` и
  `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`.
- Код и указанные веса опубликованы авторами под Apache-2.0. Лицензия сохраняется
  в загруженном архиве исходников. Выполняется локальный код конкретной revision;
  его файлы сверяются с архивом перед загрузкой модели. xFormers не требуется.
- CLS, patch mean и normalized concat — три отдельных frozen представления.
  Partial fine-tune использует новую train-only проекцию/голову, не pretrained ReID-head.

## Автомобильное предобучение ResNet50-IBN

- [FastReID Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d/MODEL_ZOO.md).
- Source revision: `c9bc3ceb2f7a6438b62fb515ea3df6d1e999e95d`.
- Официальный release v0.1.1: `veri_sbs_R50-ibn.pth`,
  `vehicleid_bot_R50-ibn.pth`, `veriwild_bot_R50-ibn.pth`.
  Прямые URL закреплены в `training/research_models.py:ASSETS`.
- ResNet adapter сохраняет IBN, last_stride=1, FastReID ceil-mode maxpool,
  pretrained BNNeck и learned GeM при наличии pretrained `pool_layer.p`,
  для VeRi SBS — также non-local. Название BoT не используется как предположение об AvgPool.
  Старый classifier отбрасывается явно; остальные tensor keys загружаются строго.
- Адаптированный non-local block: Copyright 2019 JD.com Inc. JD AI, Apache-2.0.
  Полный текст: `training/research_licenses/FASTREID_LICENSE`. Изменён порядок
  умножения матриц для экономии памяти; синтетическое совпадение проверяется тестом.
- ImageNet-контроль: [официальный IBN-Net](https://github.com/XingangPan/IBN-Net),
  MIT, release v1.0 `resnet50_ibn_a-d9d0bb7b.pth`. Это ImageNet, не vehicle-ReID pretrain.
  Копия лицензии: `training/research_licenses/IBN_NET_LICENSE`.
- Условия кода не следует автоматически считать лицензией всех исходных
  автомобильных датасетов. Для внешних весов сохраняется конкретное происхождение;
  они не объявляются готовыми к распространению в релизе без отдельной проверки.
  Сами VeRi/VehicleID/VERI-Wild изображения этот runner не скачивает.

## Что означает проверка весов

Для новых источников без опубликованного полного SHA256 доверяется только
конкретному официальному HTTPS URL. Первые полученные байты фиксируются локальным
SHA256, размером и source revision; последующие запуски должны совпасть с ними.
Это **не независимое подтверждение байтов публичным авторским checksum**.
Для ImageNet IBN используется также полный SHA256 ранее проверенного локального
checkpoint, а не выдуманный полный hash на основании восьми символов имени файла.
Запись попадает в `external_sources.json` возвращаемого run. Никаких случайных весов,
автоматического strict=False или неофициальных зеркал при ошибке загрузки нет.

## Окружение и границы исследования

[Официальные команды PyTorch 2.6.0 / CUDA 12.4](https://pytorch.org/get-started/previous-versions/#v260)
использованы в setup_windows.ps1. Python 3.11, отдельное окружение и закреплённые
зависимости. Фактические версии/устройство сохраняются в manifest; результаты Mac
не выдаются за Windows/CUDA-приёмку.

Настройки v41 JSON описывают фиксированную программу этой версии, это не универсальный
интерфейс HPO. Для изменения сеток нужен согласованный новый config/код и RUN_NAME.
В этом цикле: один прежний primary fold, три прежних query/gallery draws, один seed.
Пять новых draws, дополнительный fold, многосидовое подтверждение, ConvNeXt, новый
OSNet HPO, confidence и full-train refit остаются следующими отдельными этапами.
Нового гарантированно независимого hidden-test результата пока нет.
