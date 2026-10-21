from __future__ import annotations

import os

DEFAULT_URL = "postgresql://rag:rag@localhost:5433/rag"


def connect(autocommit: bool = False):
    """autocommit=True: conn.transaction() 블록 하나가 그대로 한 트랜잭션이 된다 (재색인처럼
    '지우고 넣기'를 한 번에 커밋해야 하는 작업에서 씀)."""
    import psycopg
    return psycopg.connect(os.environ.get("DATABASE_URL", DEFAULT_URL), autocommit=autocommit)


def connect_or_exit(autocommit: bool = True):
    """실습 스크립트용: DB가 안 떠 있으면 긴 오류 대신 할 일을 알려주고 끝낸다."""
    import psycopg
    try:
        return connect(autocommit=autocommit)
    except psycopg.OperationalError as e:
        print(f"DB 연결 안 됨: {str(e).splitlines()[0]}\n-> 'docker compose up -d'로 DB를 먼저 띄워 주세요.")
        raise SystemExit(2)
