# Weights

Notebook создаст здесь промежуточные каталоги `stage1`, `stage2_top4`,
`stage3_top2_seeds`, а затем подпапку `<RUN_NAME>` с `last.pt`, `best_map.pt`,
`best_f1.pt`, `best_tnr.pt`, важными `epoch_XX.pt` и `osnet_best_map.onnx`.

Генерируемые `.pt` и `.onnx` исключены из Git. После сравнения лучший ONNX можно
отдельно перенести в `models/`.
