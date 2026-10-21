CREATE EXTENSION IF NOT EXISTS vector;

-- 조각 하나 = 한 행. method로 자르는 방식(baseline500, article 등)을 구분해서
-- 같은 DB 안에서 버전별 점수를 나란히 비교한다.
CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    method      text      NOT NULL,
    gen         smallint  NOT NULL,          -- 실손 세대 (2, 3, 4)
    doc         text      NOT NULL,          -- 원본 파일명
    chunk_no    int       NOT NULL,
    page_start  int       NOT NULL,
    page_end    int       NOT NULL,
    text        text      NOT NULL,
    embedding   vector(384) NOT NULL,        -- multilingual-e5-small 차원
    UNIQUE (method, doc, chunk_no)
);

CREATE INDEX IF NOT EXISTS chunks_method_gen ON chunks (method, gen);
-- 벡터 색인(HNSW/IVFFlat)은 일부러 아직 안 만든다. 조각이 수천 개 수준이라
-- 전수 비교로 충분한지 먼저 재고, 그 숫자를 근거로 결정한다.

-- 문서 버전 교체(재색인)와 검색 단계 권한 필터용 열. rag/schema.py의 V2_SQL과 같은 내용
-- (이미 만들어진 DB는 init.sql이 다시 돌지 않으므로 schema.ensure_v2_schema()가 같은 걸 실행한다).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_version int NOT NULL DEFAULT 1;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS doc_sha text;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS allowed_groups text[] NOT NULL DEFAULT '{public}';
CREATE INDEX IF NOT EXISTS chunks_allowed_groups ON chunks USING gin (allowed_groups);
