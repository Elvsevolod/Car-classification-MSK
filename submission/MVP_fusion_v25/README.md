# Результаты MVP_fusion_v25

ASU Team. Результаты для 1110 query и 750 объектов gallery из выданного набора.

| Файл | Формат |
| --- | --- |
| [submission.csv](submission.csv) | 1110 строк без заголовка: query ID и десять разных gallery ID |
| [candidates.csv](candidates.csv) | Заголовок `query_id,gallery_id,confidence`; 1069 принятых запросов, для 41 отказа строки нет |
| [embeddings.npy](embeddings.npy) | `float32`, `(1860, 2048)`: сначала query, затем gallery в порядке входных CSV |

Вектор состоит из нормированных блоков legacy512 и R1_1536, без общей нормировки. Ranking использует смесь legacy/R1 50/50 и k-reciprocal 20/3/0.5. Кандидат выбирается по raw R1 cosine с порогом `0.534365177154541`; confidence не является вероятностью.

[version.json](version.json) фиксирует код, веса и параметры модели. [validation.json](validation.json) содержит проверку экспорта; [export_manifest.json](export_manifest.json) сохраняет порядок ID и хеши входов. Время в [runtime_timing.json](runtime_timing.json) относится к исходному CPU-экспорту, не к официальному GPU-бенчмарку. Метрики на скрытом тесте здесь не заявляются.

Проверка файлов из этой папки на Mac/Linux:

```bash
shasum -a 256 -c SHA256SUMS
```

Датасет с изображениями в комплект не входит.
