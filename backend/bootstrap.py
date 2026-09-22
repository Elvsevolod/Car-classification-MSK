"""Композиция runtime-зависимостей: выбирает инфраструктуру, не смешивая её с ML-логикой."""

import os

from .database import DatabaseSettings
from .gallery_repository import GalleryRepository
from .postgres_gallery_repository import PostgresGalleryRepository


def gallery_repository_from_environment() -> GalleryRepository:
    """Создаёт единственное production-хранилище gallery — PostgreSQL с pgvector."""
    storage = os.environ.get("GALLERY_STORAGE", "postgres").strip().lower()
    if storage != "postgres":
        raise ValueError("GALLERY_STORAGE must be postgres")
    return PostgresGalleryRepository(DatabaseSettings.from_environment())
