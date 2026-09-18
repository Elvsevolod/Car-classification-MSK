# Веса варианта 3

Ноутбук создаёт каталоги `avg_seed_<seed>` и `gem_seed_<seed>` с `last.pt`,
`best_map.pt`, `best_f1.pt`, `best_tnr.pt` и важными `epoch_XX.pt`.

После сравнения репрезентативный checkpoint победившей группы экспортируется в
`osnet_stage3_selected.onnx`. Бинарные файлы исключены из Git.
