"""Small connectivity check for the PostgreSQL service."""
from .database import DatabaseSettings
from .postgres_gallery_repository import PostgresGalleryRepository


def main() -> None:
    result = PostgresGalleryRepository(DatabaseSettings.from_environment()).ping()
    print(f"PostgreSQL connection ready; pgvector={result['pgvector_version']}")


if __name__ == "__main__":
    main()
