from __future__ import annotations

import os

DEFAULT_URL = "postgresql://rag:rag@localhost:5433/rag"


def connect():
    import psycopg
    return psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_URL))
