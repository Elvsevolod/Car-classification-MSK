// Legacy fallback: production Docker serves React from frontend/dist. Kept for source-level FastAPI tests.
const $ = id => document.getElementById(id);
const canvas = $('canvas');
const context = canvas.getContext('2d');
const fields = ['x', 'y', 'w', 'h'];
let fullImage = null, imageBlob = null, lastResult = null, dragStart = null, version = 0;
let queries = [];

function status(message) { $('status').textContent = message; }
function clearResults() {
  $('results').replaceChildren(); lastResult = null; $('export').disabled = true;
}
function draw() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!fullImage) return;
  context.drawImage(fullImage, 0, 0);
  const [x, y, w, h] = fields.map(id => Number($(id).value));
  context.strokeStyle = 'red'; context.lineWidth = Math.max(2, canvas.width / 400);
  context.strokeRect(x, y, w, h);
}
function setBox(box) { fields.forEach((id, i) => { $(id).value = box ? box[i] : ''; }); draw(); }
async function loadImage(blob, box = null) {
  const current = ++version;
  $('submit').disabled = true; clearResults();
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image(); image.src = url; await image.decode();
    if (current !== version) return;
    if (image.naturalWidth * image.naturalHeight > 25000000) throw new Error('Изображение превышает 25 мегапикселей');
    fullImage = image; imageBlob = blob;
    canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
    setBox(box); $('submit').disabled = false;
    status(`Исходный размер: ${canvas.width}×${canvas.height}. ${box ? 'BBox из CSV загружен.' : 'Выделите автомобиль.'}`);
  } catch (error) { fullImage = null; imageBlob = null; draw(); status(error.message); }
  finally { URL.revokeObjectURL(url); }
}
$('file').addEventListener('change', async () => {
  const file = $('file').files[0]; if (!file) return;
  $('query').value = '';
  if (file.size > 15 * 1024 * 1024) { status('Максимальный размер — 15 МиБ'); return; }
  await loadImage(file);
});
$('query').addEventListener('change', async () => {
  const row = queries.find(row => row.image_id === $('query').value); if (!row) return;
  $('file').value = ''; $('submit').disabled = true;
  const selected = row.image_id;
  try {
    const response = await fetch(`/api/images/query/${row.image_id}`);
    if (!response.ok) throw new Error('Не удалось загрузить пример');
    const blob = await response.blob();
    if ($('query').value === selected) await loadImage(blob, fields.map(id => row[id]));
  } catch (error) { status(error.message); }
});
fields.forEach(id => $(id).addEventListener('input', () => { clearResults(); draw(); }));
function point(event) {
  const rect = canvas.getBoundingClientRect();
  const width = rect.width - 2 * canvas.clientLeft, height = rect.height - 2 * canvas.clientTop;
  return [Math.max(0, Math.min(canvas.width, Math.round((event.clientX - rect.left - canvas.clientLeft) * canvas.width / width))),
          Math.max(0, Math.min(canvas.height, Math.round((event.clientY - rect.top - canvas.clientTop) * canvas.height / height)))];
}
canvas.addEventListener('pointerdown', event => {
  if (!fullImage) return; dragStart = point(event); canvas.setPointerCapture(event.pointerId); clearResults();
});
canvas.addEventListener('pointermove', event => {
  if (!dragStart) return;
  const end = point(event);
  setBox([Math.min(dragStart[0], end[0]), Math.min(dragStart[1], end[1]), Math.abs(end[0] - dragStart[0]), Math.abs(end[1] - dragStart[1])]);
});
canvas.addEventListener('pointerup', () => { dragStart = null; });
canvas.addEventListener('pointercancel', () => { dragStart = null; });

$('search').addEventListener('submit', async event => {
  event.preventDefault(); if (!imageBlob) return;
  const box = fields.map(id => Number($(id).value));
  if (box[0] + box[2] > canvas.width || box[1] + box[3] > canvas.height) { status('BBox выходит за границы изображения'); return; }
  const data = new FormData($('search'));
  data.append('image', imageBlob, 'query.jpg');
  if (!$('threshold').value) data.delete('threshold');
  const controls = [...document.querySelectorAll('input, select, button')].map(element => [element, element.disabled]);
  controls.forEach(([element]) => { element.disabled = true; });
  canvas.style.pointerEvents = 'none';
  clearResults(); status('Поиск…');
  const current = version;
  try {
    const response = await fetch('/api/search', {method: 'POST', body: data});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
    if (current !== version) return;
    lastResult = result; $('export').disabled = false;
    status(result.refused ? `Отказ: нет кандидатов выше порога ${result.threshold.toFixed(4)}.` : `Найдено: ${result.results.length}. Время API: ${result.elapsed_ms} мс.`);
    for (const item of result.results) {
      const article = document.createElement('article');
      const image = document.createElement('img'); image.src = item.crop_url; image.alt = `Кандидат ${item.rank}`;
      const text = document.createElement('p'); text.textContent = `#${item.rank} · rerank ${item.rerank_score.toFixed(4)} · cosine ${item.similarity.toFixed(4)}\n${item.image_id}`;
      const link = document.createElement('a'); link.href = `/api/images/gallery/${item.image_id}`; link.target = '_blank'; link.rel = 'noopener'; link.textContent = 'Полный кадр';
      article.append(image, text, link); $('results').append(article);
    }
  } catch (error) { status(error.message); }
  finally {
    controls.forEach(([element, disabled]) => { element.disabled = disabled; });
    canvas.style.pointerEvents = '';
    $('submit').disabled = !imageBlob; $('export').disabled = !lastResult;
  }
});
$('export').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'search-result.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function init() {
  try {
    const response = await fetch('/api/health'); if (!response.ok) throw new Error('Сервис недоступен');
    const health = await response.json();
    $('health').textContent = `${health.model} · ${health.device} · галерея: ${health.gallery_size} объектов · порог: ${health.default_threshold?.toFixed(4) ?? 'ещё не рассчитан'}`;
    const data = await (await fetch('/api/queries?limit=1110')).json(); queries = data.items;
    queries.forEach((row, i) => { const option = document.createElement('option'); option.value = row.image_id; option.textContent = `${i + 1}. ${row.image_id}`; $('query').append(option); });
    const metricsResponse = await fetch('/api/metrics');
    $('metrics').textContent = metricsResponse.ok ? JSON.stringify(await metricsResponse.json(), null, 2) : 'Метрики пока не рассчитаны: python -m backend.evaluate';
  } catch (error) { status(error.message); }
}
init();
