"""chunks 표에 '문서 버전'과 '권한' 열을 붙인다. 여러 번 실행해도 안전하다 (IF NOT EXISTS).

- doc_version : 같은 문서를 몇 번째로 색인한 것인지 (1, 2, 3 ...)
- doc_sha     : 문서 전체 글자의 해시. 내용이 그대로면 다시 넣지 않는다 (중복 방지)
- allowed_groups : 이 조각을 검색 결과로 받을 수 있는 그룹 목록. 기본값은 모두에게 공개('public')

이미 데이터가 들어 있는 DB(도커 볼륨)는 init.sql이 다시 돌지 않으므로, 이 함수가 열을 추가한다.
기존 행은 기본값(doc_version=1, allowed_groups={public})을 받으므로 기존 평가 결과는 그대로다.
"""
from __future__ import annotations

V2_SQL = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_version int NOT NULL DEFAULT 1;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_sha text;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS allowed_groups text[] NOT NULL DEFAULT '{public}';
CREATE INDEX IF NOT EXISTS chunks_allowed_groups ON chunks USING gin (allowed_groups);
"""


def ensure_v2_schema(conn) -> None:
    conn.execute(V2_SQL)
