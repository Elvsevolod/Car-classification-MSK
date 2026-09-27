# Воспроизведение обучения и происхождение

С 2026-09-27 эта историческая копия обучающих исходников хранится в **fine-tuning**.
Готовый продукт и полная папка runtime-весов `models/frozen/` — в **main**.
Пути приложения ниже относятся к продуктовой ветке/образу, а не к исследовательскому
backend этой ветки. Текущие эксперименты находятся в корневом `training/`.

Этот каталог не импортируется runtime и не попадает в Docker-образ.
Здесь сохраняется код, но не запускается обучение. Данные организаторов, NiVe,
исторические checkpoints и логи не включаются в поставку.

source/SOURCE_SHA256.json фиксирует точную версию каждого исходного файла.
Notebook сохранены с исходным кодом, но без исторического вывода.
Исходные версии находятся в research-копии; они не изменены.

- Базовая инициализация: Open Model Zoo vehicle-reid-0001, выпуск 2022.1.
  MIT: source/models/LICENSE.osnet; исходный URL и контрольные суммы в
  source/models/README.md. Точная используемая реализация архитектуры —
  source/training/osnet.py, зафиксированная SHA256 в manifest.
- MVP: рецепт variant_02_hpo_bnneck_supcon, выбранная development epoch 5.
  Исторический выбор использовал local validation; это не независимая оценка.
- R1: variant_16_review_protocol, resolution256, seed 20260915;
  variant_17_final_seed_confirmation — seeds 20260916/20260917.
  Общий замороженный шаг 800, не выбор лучшего seed по outer.
- Retrieval/threshold: variant_18_retrieval_policy; в runtime только frozen
  bundle. Повторная калибровка при инференсе запрещена.

Для воспроизведения исходных экспериментов создайте ОТДЕЛЬНУЮ исследовательскую
копию из source, установите requirements-train.txt, добавьте исходный
organizer dataset и stock ONNX с указанной контрольной суммой.
Следуйте фазам и защитным флагам соответствующих README/notebook:
pilot → confirm_inner → final; stage 17 и 18 требуют завершённых предшественников.
Сохранённые manifest/selection — спецификация протокола и замороженного рецепта,
не замена отсутствующих промежуточных результатов. Для полного повторения
предшественники надо пересчитать в новом run-каталоге. Битовая идентичность
обучения между разными GPU/версиями не обещается.

Runtime-веса находятся в models/frozen текущей рабочей копии и в
/app/models/frozen внутри поставочного Docker-образа. Source ZIP не дублирует
веса. После загрузки образа их можно извлечь для воспроизведения:

    docker create --name reid-assets-source vehicle-reid:release-integration
    docker cp reid-assets-source:/app/models/. ./models/
    docker rm reid-assets-source

Подкаталоги с названием runs сохраняют только относительные ссылки frozen
bundles и три ONNX, а не историю экспериментов.
