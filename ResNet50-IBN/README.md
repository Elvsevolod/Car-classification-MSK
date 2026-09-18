# ResNet50-IBN

Каталог содержит альтернативные backbone для честного сравнения с активной
OSNet-AIN x1.0. Эксперименты не заменяют модель MVP автоматически.

| Вариант | Статус | Назначение |
|---|---|---|
| [`variant_05_gem_bnneck`](variant_05_gem_bnneck/) | Завершён, не принят | mAP@10 0,272971 ± 0,017324; обнаружены ограничения протокола |
| [`variant_06_controlled_training`](variant_06_controlled_training/) | Готов к запуску | Frame-disjoint inner split, до 4000 шагов, фиксированный LR horizon, LR/pooling/loss абляции |

Используется официальный ImageNet checkpoint из MIT-лицензированного
[IBN-Net v1.0](https://github.com/XingangPan/IBN-Net). Сырые внешние датасеты
в проект не добавляются.
