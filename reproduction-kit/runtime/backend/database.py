"""Database configuration shared by PostgreSQL-backed components."""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote


@dataclass(frozen=True)
class DatabaseSettings:
    """PostgreSQL connection settings resolved from environment variables."""

    url: str

    @classmethod
    def from_environment(cls) -> "DatabaseSettings":
        explicit_url = os.environ.get("DATABASE_URL")
        if explicit_url:
            return cls(explicit_url)

        database = os.environ.get("POSTGRES_DB")
        user = os.environ.get("POSTGRES_USER")
        password = os.environ.get("POSTGRES_PASSWORD")
        if not all((database, user, password)):
            raise RuntimeError(
                "Set DATABASE_URL or POSTGRES_DB, POSTGRES_USER and POSTGRES_PASSWORD "
                "before selecting GALLERY_STORAGE=postgres"
            )
        host = os.environ.get("POSTGRES_HOST", "postgres")
        port = os.environ.get("POSTGRES_PORT", "5432")
        return cls(
            "postgresql://"
            f"{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{quote(database, safe='')}"
        )
