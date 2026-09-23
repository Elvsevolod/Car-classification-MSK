"""FastAPI-слой: валидирует изображение/BBox и возвращает поиск или отказ."""

import io
import json
import time
import warnings
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .bootstrap import gallery_repository_from_environment
from .core import (ARTIFACTS, DATASET, MODEL_FINE_TUNED, MODEL_NAME, ROOT,
                   Encoder, Gallery, bbox, crop_image, read_rows)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000
FRONTEND = ROOT / "frontend"
FRONTEND_DIST = FRONTEND / "dist"


class QuerySearch(BaseModel):
    query_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    top_k: int = Field(default=10, ge=1, le=100)
    mode: Literal["ranking", "candidates"] = "ranking"
    threshold: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)


class Candidate(BaseModel):
    rank: int
    image_id: str
    x: int
    y: int
    w: int
    h: int
    similarity: float
    rerank_score: float
    crop_url: str


class SearchResponse(BaseModel):
    mode: str
    query_id: str | None
    gallery_size: int
    threshold: float | None
    threshold_source: str | None
    confidence: float
    refused: bool
    results: list[Candidate]
    elapsed_ms: float
    encoder_fingerprint: str


def load_metrics(encoder, artifacts):
    path = artifacts / "baseline_metrics.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get("encoder_fingerprint") != encoder.fingerprint:
        return None
    return report


def create_app(dataset=DATASET, artifacts=ARTIFACTS, gallery_repository=None):
    """Собирает API и позволяет тестам подменять датасет, артефакты и gallery-репозиторий."""
    @asynccontextmanager
    async def lifespan(app):
        app.state.encoder = Encoder()
        repository = gallery_repository or gallery_repository_from_environment()
        app.state.gallery = Gallery(app.state.encoder, dataset, repository=repository)
        app.state.queries = {r["image_id"]: r for r in read_rows(dataset / "test_query.csv")}
        yield

    app = FastAPI(title="Vehicle ReID · fine-tuned OSNet", version="0.2.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    # Production всегда отдаёт Vite dist; исходный HTML/JS остаётся только fallback для локальных Python-тестов.
    static_dir = FRONTEND_DIST if FRONTEND_DIST.exists() else FRONTEND
    # Swagger assets stay outside the Vite build and remain available offline.
    app.mount("/static/vendor", StaticFiles(directory=FRONTEND / "vendor"), name="static-vendor")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/docs", include_in_schema=False)
    def docs():
        """Локальный Swagger UI: assets включены в образ, поэтому CDN не нужен."""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} · Swagger UI",
            swagger_js_url="/static/vendor/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/vendor/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/vendor/swagger-ui/favicon-32x32.png",
        )

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    def health():
        from .rerank import ACTIVE_K1, ACTIVE_K2, ACTIVE_LAMBDA
        report = load_metrics(app.state.encoder, artifacts)
        return {"status": "ready", "model": MODEL_NAME, "fine_tuned": MODEL_FINE_TUNED,
                "embedding_dim": 512, "device": "CPU", "gallery_size": len(app.state.gallery.rows),
                "gallery_storage": type(app.state.gallery.repository).__name__,
                "encoder_fingerprint": app.state.encoder.fingerprint,
                "default_threshold": report["threshold"] if report else None,
                "reranking": {"method": "streaming k-reciprocal", "k1": ACTIVE_K1,
                              "k2": ACTIVE_K2, "lambda": ACTIVE_LAMBDA,
                              "refusal_score": "maximum raw cosine"}}

    @app.get("/api/metrics")
    def model_metrics():
        report = load_metrics(app.state.encoder, artifacts)
        if report is None:
            raise HTTPException(404, "Run python -m backend.evaluate to measure the active model")
        return report

    @app.get("/api/queries")
    def queries(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=1110)):
        rows = list(app.state.queries.values())
        return {"total": len(rows), "items": rows[offset:offset + limit]}

    @app.get("/api/images/{split}/{image_id}")
    def image(split: Literal["query", "gallery"], image_id: str, crop: bool = False):
        records = app.state.queries if split == "query" else {r["image_id"]: r for r in app.state.gallery.rows}
        row = records.get(image_id)
        if row is None:
            raise HTTPException(404, "Unknown image_id")
        path = dataset / "images" / f"{image_id}.jpg"
        if not crop:
            return FileResponse(path, media_type="image/jpeg")
        with Image.open(path) as full_image:
            output = io.BytesIO()
            crop_image(full_image, bbox(row)).save(output, format="JPEG", quality=90)
        return Response(output.getvalue(), media_type="image/jpeg")

    def search(vector, top_k, mode, threshold, started, query_id=None):
        """Единый путь поиска: порог влияет только на режим candidates, ranking всегда возвращает Top-K."""
        source = None
        if mode == "candidates":
            source = "manual" if threshold is not None else "calibration"
            if threshold is None:
                report = load_metrics(app.state.encoder, artifacts)
                if report is None:
                    raise HTTPException(409, "No calibrated threshold: run python -m backend.evaluate or supply threshold")
                threshold = report["threshold"]
        else:
            threshold = None
        results, confidence = app.state.gallery.search_with_confidence(vector, top_k, threshold)
        return {"mode": mode, "query_id": query_id, "gallery_size": len(app.state.gallery.rows),
                "threshold": threshold, "threshold_source": source, "confidence": confidence,
                "refused": mode == "candidates" and not results,
                "results": results, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "encoder_fingerprint": app.state.encoder.fingerprint}

    def uploaded_embedding(upload, box):
        """Ограничивает upload и передаёт в Encoder только проверенное JPEG/PNG изображение с BBox."""
        content = upload.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Maximum image file size is 15 MiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as uploaded:
                    if uploaded.format not in ("JPEG", "PNG"):
                        raise HTTPException(415, "Only JPEG and PNG are supported")
                    if uploaded.width * uploaded.height > MAX_PIXELS:
                        raise HTTPException(413, "Maximum image size is 25 megapixels")
                    return app.state.encoder.encode(uploaded, box)
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(413, "Image dimensions are too large")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(422, f"Invalid image or BBox: {exc}") from exc

    @app.post("/api/search", response_model=SearchResponse)
    def search_upload(
        image: Annotated[UploadFile, File()], x: Annotated[int, Form(ge=0)], y: Annotated[int, Form(ge=0)],
        w: Annotated[int, Form(gt=0)], h: Annotated[int, Form(gt=0)],
        top_k: Annotated[int, Form(ge=1, le=100)] = 10,
        mode: Annotated[Literal["ranking", "candidates"], Form()] = "ranking",
        threshold: Annotated[float | None, Form(ge=-1, le=1, allow_inf_nan=False)] = None,
    ):
        started = time.perf_counter()
        vector = uploaded_embedding(image, (x, y, w, h))
        return search(vector, top_k, mode, threshold, started)

    @app.post("/api/embedding")
    def embedding(image: Annotated[UploadFile, File()], x: Annotated[int, Form(ge=0)],
                  y: Annotated[int, Form(ge=0)], w: Annotated[int, Form(gt=0)], h: Annotated[int, Form(gt=0)]):
        vector = uploaded_embedding(image, (x, y, w, h))
        return {"embedding": vector.tolist(), "dimension": 512, "dtype": "float32", "l2_normalized": True,
                "encoder_fingerprint": app.state.encoder.fingerprint}

    @app.post("/api/search/query", response_model=SearchResponse)
    def search_query(request: QuerySearch):
        started = time.perf_counter()
        row = app.state.queries.get(request.query_id)
        if row is None:
            raise HTTPException(404, "Unknown query_id")
        with Image.open(dataset / "images" / f"{row['image_id']}.jpg") as image:
            vector = app.state.encoder.encode(image, bbox(row))
        return search(vector, request.top_k, request.mode, request.threshold, started, request.query_id)

    return app


app = create_app()
