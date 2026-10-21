"""권한 필터를 '검색 안에서' 건다: 그 사용자가 볼 수 있는 조각만 애초에 후보로 올린다.

왜 프롬프트로 막으면 안 되나:
  "이 사용자에게 내부 문서는 말하지 마"라고 프롬프트에 써도, 내부 조각이 이미 모델 입력에 들어가 있다.
  모델이 그 지시를 어기거나, 사용자가 우회 질문을 하면 그대로 새어 나간다. 막아야 하는 곳은 검색이다.

왜 '검색 후 거르기'도 부족한가:
  상위 k개를 먼저 가져온 뒤 권한 없는 걸 빼면, 새지는 않지만 k개보다 적게 남는다.
  권한 있는 정답 조각이 k+1등에 있었다면 그냥 사라진다. WHERE로 걸면 권한 있는 것 중에서 k개를 고른다.

규칙:
  - groups는 필수 인자. 빈 목록이면 아무것도 안 나온다 (기본은 거부).
  - groups는 로그인 세션 등 서버가 확인한 값에서 가져와야 한다. 요청 본문이나 질문 글에서 받으면 안 된다.
  - 재정렬(rerank)은 이 함수 결과 안에서만 순서를 바꾸므로 권한 범위를 넓히지 않는다.

참고: 지금은 벡터 색인 없이 전수 비교라 WHERE를 걸어도 항상 k개가 나온다. HNSW 색인을 붙이면
필터에 걸러져 k개보다 적게 나올 수 있다 -> pgvector 0.8+의 반복 스캔(hnsw.iterative_scan) 설정이 필요.
"""
from __future__ import annotations

from typing import Callable, Sequence

SQL = """
SELECT gen, doc, chunk_no, page_start, page_end,
       1 - (embedding <=> %(q)s::vector) AS score, text, allowed_groups
FROM chunks
WHERE method = %(method)s
  AND (%(gen)s::smallint IS NULL OR gen = %(gen)s::smallint)
  AND allowed_groups && %(groups)s::text[]
ORDER BY embedding <=> %(q)s::vector
LIMIT %(k)s
"""

UNFILTERED_SQL = SQL.replace("  AND allowed_groups && %(groups)s::text[]\n", "")


def search_for_user(conn, question: str, groups: Sequence[str], k: int = 3, *, method: str = "baseline500",
                    gen: int | None = None, embed_query: Callable[[str], list[float]] | None = None) -> list:
    """사용자 그룹과 겹치는 조각만 상위 k개. 결과 앞 7칸은 rag.search.search()와 같아서 rerank()에 그대로 넣을 수 있다."""
    if isinstance(groups, str):
        raise TypeError("groups는 목록이어야 합니다 (예: ['public']). 문자열 하나를 넘기면 글자 단위로 쪼개짐")
    if embed_query is None:
        from rag.embed import embed_query
    from rag.embed import to_pgvector

    groups = sorted({g for g in groups if g})
    if not groups:
        return []
    return conn.execute(SQL, {"q": to_pgvector(embed_query(question)), "method": method, "gen": gen,
                              "groups": groups, "k": k}).fetchall()


def search_unfiltered(conn, question: str, k: int = 3, *, method: str = "baseline500", gen: int | None = None,
                      embed_query: Callable[[str], list[float]] | None = None) -> list:
    """비교용: 권한을 안 보고 검색 (프롬프트로만 막는 시스템이 모델에 넘기는 것과 같음)."""
    if embed_query is None:
        from rag.embed import embed_query
    from rag.embed import to_pgvector
    return conn.execute(UNFILTERED_SQL, {"q": to_pgvector(embed_query(question)), "method": method,
                                         "gen": gen, "k": k}).fetchall()


def post_filter(rows: list, groups: Sequence[str]) -> list:
    """비교용: 검색 후 거르기. 새지는 않지만 k개보다 적게 남을 수 있다."""
    g = set(groups)
    return [r for r in rows if g & set(r[7])]
