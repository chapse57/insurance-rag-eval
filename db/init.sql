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
