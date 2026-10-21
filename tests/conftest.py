import uuid

import pytest


@pytest.fixture
def conn():
    """실제 Postgres(pgvector)가 필요한 테스트용. DB가 안 떠 있으면 이유를 적고 건너뛴다."""
    psycopg = pytest.importorskip("psycopg")
    from rag.db import connect
    from rag.schema import ensure_v2_schema
    try:
        c = connect(autocommit=True)
    except psycopg.OperationalError as e:
        pytest.skip(f"DB 연결 안 됨 (docker compose up -d 먼저): {e}")
    ensure_v2_schema(c)
    yield c
    c.close()


@pytest.fixture
def method(conn):
    """테스트마다 겹치지 않는 method 이름. 끝나면 그 행을 지운다 (실제 약관 데이터는 안 건드림)."""
    m = f"pytest_{uuid.uuid4().hex[:8]}"
    yield m
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE method = %s", (m,))
