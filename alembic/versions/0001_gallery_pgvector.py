"""Create the persistent pgvector gallery schema.

Revision ID: 0001_gallery_pgvector
Revises:
Create Date: 2026-09-19
"""
from alembic import op

revision = "0001_gallery_pgvector"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE gallery_items (
            image_id TEXT PRIMARY KEY,
            position INTEGER NOT NULL UNIQUE CHECK (position >= 0),
            x INTEGER NOT NULL CHECK (x >= 0),
            y INTEGER NOT NULL CHECK (y >= 0),
            w INTEGER NOT NULL CHECK (w > 0),
            h INTEGER NOT NULL CHECK (h > 0),
            embedding vector(512) NOT NULL,
            image_sha256 TEXT NOT NULL,
            encoder_fingerprint TEXT NOT NULL,
            metadata JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    op.execute(
        """
        CREATE TABLE gallery_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            fingerprint TEXT NOT NULL,
            encoder_fingerprint TEXT NOT NULL,
            preprocessing_fingerprint TEXT NOT NULL,
            csv_sha256 TEXT NOT NULL,
            gallery_size INTEGER NOT NULL CHECK (gallery_size >= 0),
            built_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            schema_version INTEGER NOT NULL DEFAULT 1
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS gallery_state")
    op.execute("DROP TABLE IF EXISTS gallery_items")
    op.execute("DROP EXTENSION IF EXISTS vector")
