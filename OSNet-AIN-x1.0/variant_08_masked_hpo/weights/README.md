# Веса

Обучение создаст `<RUN_NAME>/stage1/`, `stage2_top4/`, `stage3_top2_seeds/`
и `selected/` с checkpoint. Итоговый ONNX находится в
`<RUN_NAME>/selected/osnet_masked_best_map.onnx` и требует маскирования перед
инференсом. Пока полноценное обучение не запускалось, новых весов нет.
Файлы весов исключены из Git; MVP автоматически не заменяется.
