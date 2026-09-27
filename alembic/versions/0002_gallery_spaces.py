"""Add immutable multi-dimension spaces without modifying the legacy MVP tables."""
from alembic import op

revision = "0002_gallery_spaces"
down_revision = "0001_gallery_pgvector"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        CREATE TABLE reid_gallery_spaces (
            fingerprint TEXT PRIMARY KEY,
            dimension INTEGER NOT NULL CHECK (dimension > 0),
            metadata JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (fingerprint, dimension)
        )
    """)
    op.execute("""
        CREATE TABLE reid_gallery_vectors (
            fingerprint TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            position INTEGER NOT NULL CHECK (position >= 0),
            image_id TEXT NOT NULL,
            embedding vector NOT NULL,
            PRIMARY KEY (fingerprint, image_id),
            UNIQUE (fingerprint, position),
            FOREIGN KEY (fingerprint, dimension)
                REFERENCES reid_gallery_spaces (fingerprint, dimension),
            CHECK (vector_dims(embedding) = dimension)
        )
    """)


def downgrade():
    op.execute("DROP TABLE reid_gallery_vectors")
    op.execute("DROP TABLE reid_gallery_spaces")
