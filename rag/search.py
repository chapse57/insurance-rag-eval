"""질문 -> 가장 가까운 조각 k개.

실행:  python -m rag.search "도수치료는 1년에 몇 번까지 보장되나요" --gen 4
"""
from __future__ import annotations

import argparse
import time

from rag.db import connect
from rag.embed import _model, embed_query, to_pgvector

SQL = """
SELECT gen, doc, chunk_no, page_start, page_end,
       1 - (embedding <=> %(q)s::vector) AS score, text
FROM chunks
WHERE method = %(method)s AND (%(gen)s::smallint IS NULL OR gen = %(gen)s::smallint)
ORDER BY embedding <=> %(q)s::vector
LIMIT %(k)s
"""


def search(question: str, k: int = 3, gen: int | None = None, method: str = "baseline500"):
    q = to_pgvector(embed_query(question))
    with connect() as conn:
        return conn.execute(SQL, {"q": q, "gen": gen, "k": k, "method": method}).fetchall()


NEIGHBOR_SQL = """
SELECT chunk_no, text FROM chunks
WHERE method = %(method)s AND gen = %(gen)s AND doc = %(doc)s AND chunk_no = ANY(%(nos)s)
"""


def neighbors(row, method: str = "baseline500") -> tuple[str, str]:
    """search() 결과 한 줄 -> (바로 앞 조각 글, 바로 뒤 조각 글). 조각은 겹침 없이 이어 잘랐으므로
    앞 + 이 조각 + 뒤를 붙이면 원문이 그대로 이어진다. 없으면 빈 문자열."""
    gen, doc, no = row[0], row[1], row[2]
    with connect() as conn:
        found = dict(conn.execute(NEIGHBOR_SQL, {"method": method, "gen": gen, "doc": doc,
                                                 "nos": [no - 1, no + 1]}).fetchall())
    return found.get(no - 1, ""), found.get(no + 1, "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--gen", type=int, choices=[2, 3, 4])
    ap.add_argument("--method", default="baseline500")
    ap.add_argument("--rerank", action="store_true", help="1차 20개를 재정렬 모델로 다시 줄 세움")
    args = ap.parse_args(argv)

    t_load = time.time()
    _model()  # 모델 불러오기는 검색 시간과 따로 잰다 (첫 실행 때 몇 초 걸림)
    t0 = time.time()
    if args.rerank:
        from rag.rerank import rerank
        rows = rerank(args.question, search(args.question, 20, args.gen, args.method))[:args.k]
    else:
        rows = search(args.question, args.k, args.gen, args.method)
    print(f"질문: {args.question}  (검색 {(time.time() - t0) * 1000:.0f} ms"
          f" / 모델 로딩 {t0 - t_load:.1f}초 별도, method={args.method})\n")
    for rank, (gen, doc, no, p1, p2, score, text) in enumerate(rows, start=1):
        pages = f"{p1}쪽" if p1 == p2 else f"{p1}~{p2}쪽"
        print(f"[{rank}] {gen}세대 {pages} 조각#{no} 유사도 {score:.3f}")
        print(f"    {text[:180]}{'...' if len(text) > 180 else ''}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
