"""실습 1: 문서를 고친 뒤 다시 넣을 때 옛 버전이 남는지 세 가지 방법으로 재 본다.

실행:  python -m rag.demo_reindex               (진짜 임베딩 모델 사용)
       python -m rag.demo_reindex --fake-embed  (모델 없이 빠르게, 검색 순위는 참고용)

짧은 가상 약관(조항 5개)을 v1으로 넣고, 조항 2개를 고치고 1개를 없앤 v2(조항 4개)로 바꾼다.
  A. 덧붙이기     : v2를 새 파일 이름으로 그냥 넣음 (파일 이름을 문서 키로 쓰는 시스템에서 흔함)
  B. 번호로 덮어쓰기: v2를 조각 번호(chunk_no) 기준 upsert
  C. 통째로 교체   : rag.versioning.replace_document
그리고 같은 v2를 한 번 더 넣어 중복이 생기는지 본다.
실제 약관 데이터(baseline500)는 건드리지 않는다 (method 이름이 demo_로 시작하는 행만 쓰고 끝나면 지움).
"""
from __future__ import annotations

import argparse

from rag.chunking import Chunk
from rag.db import connect_or_exit
from rag.embed import to_pgvector
from rag.schema import ensure_v2_schema
from rag.versioning import INSERT_SQL, document_sha, replace_document

V1 = [
    (1, "제1조(목적) 이 약관은 피보험자가 질병 또는 상해로 의료기관에서 치료를 받은 경우 실제로 부담한 의료비를 보상합니다."),
    (2, "제2조(도수치료) 도수치료·체외충격파치료·증식치료는 연간 10회까지 보상하며, 1회당 자기부담금은 2만원입니다."),
    (3, "제3조(입원의료비) 입원 시 본인이 부담한 의료비의 80%를 보상하며, 연간 5천만원을 한도로 합니다."),
    (4, "제4조(해지환급금) 계약자가 계약을 해지하면 납입한 보험료의 50%를 해지환급금으로 돌려드립니다."),
    (5, "제5조(무사고 할인 특약) 직전 1년간 보험금을 청구하지 않으면 다음 해 보험료를 10% 할인합니다."),
]
V2 = [  # 2026-10 개정: 제2조·제4조 변경, 제5조(특약) 폐지
    V1[0],
    (2, "제2조(도수치료) 도수치료·체외충격파치료·증식치료는 연간 20회까지 보상하며, 1회당 자기부담금은 3만원입니다."),
    V1[2],
    (4, "제4조(해지환급금) 이 계약은 순수보장형으로, 계약을 해지해도 해지환급금이 발생하지 않습니다."),
]
OLD_MARKERS = ["연간 10회", "50%를 해지환급금", "10% 할인"]   # v2에는 없는 v1 문구
QUESTIONS = [
    ("도수치료는 1년에 몇 번까지 보장돼요?", "연간 10회"),
    ("보험 해지하면 돌려받는 돈 있어요?", "50%를 해지환급금"),
    ("보험금 청구를 안 하면 보험료 할인돼요?", "10% 할인"),
]
GEN = 4
METHODS = {"A": "demo_append", "B": "demo_upsert", "C": "demo_replace"}


def by_clause(pages) -> list[Chunk]:
    """조항 하나 = 조각 하나 (조각 경계 때문에 결과가 흐려지지 않게)."""
    return [Chunk(i, p, p, " ".join(t.split())) for i, (p, t) in enumerate(pages)]


def insert_plain(conn, method: str, doc: str, pages, embed) -> None:
    """A: 아무것도 지우지 않고 넣기."""
    chunks = by_clause(pages)
    vecs = embed([c.text for c in chunks])
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(INSERT_SQL, [(method, GEN, doc, c.chunk_no, c.page_start, c.page_end, c.text,
                                      to_pgvector(v), 1, document_sha(pages), ["public"])
                                     for c, v in zip(chunks, vecs)])


def upsert_by_chunk_no(conn, method: str, doc: str, pages, embed) -> None:
    """B: 같은 (method, doc, chunk_no)면 덮어쓰고, 없으면 넣기. 새 버전에 없는 번호는 손대지 않는다."""
    chunks = by_clause(pages)
    vecs = embed([c.text for c in chunks])
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            INSERT_SQL + " ON CONFLICT (method, doc, chunk_no) DO UPDATE SET"
            " text = EXCLUDED.text, embedding = EXCLUDED.embedding, page_start = EXCLUDED.page_start,"
            " page_end = EXCLUDED.page_end, doc_version = chunks.doc_version + 1, doc_sha = EXCLUDED.doc_sha",
            [(method, GEN, doc, c.chunk_no, c.page_start, c.page_end, c.text, to_pgvector(v), 1,
              document_sha(pages), ["public"]) for c, v in zip(chunks, vecs)])


def measure(conn, method: str, embed_query) -> dict:
    texts = [r[0] for r in conn.execute("SELECT text FROM chunks WHERE method = %s", (method,)).fetchall()]
    stale_rows = sum(any(m in t for m in OLD_MARKERS) for t in texts)
    # 질문마다 '그 질문의 옛 답' 문구만 본다 (조각이 몇 개 안 돼서 아무 옛 조각이나 세면 부풀려짐)
    q_stale_top3 = q_stale_top1 = 0
    for q, marker in QUESTIONS:
        top = [r[0] for r in conn.execute(
            "SELECT text FROM chunks WHERE method = %s ORDER BY embedding <=> %s::vector LIMIT 3",
            (method, to_pgvector(embed_query(q)))).fetchall()]
        q_stale_top3 += any(marker in t for t in top)
        q_stale_top1 += bool(top) and marker in top[0]
    return {"rows": len(texts), "stale_rows": stale_rows, "q_top3": q_stale_top3, "q_top1": q_stale_top1}


def cleanup(conn) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE method = ANY(%s)", (list(METHODS.values()),))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-embed", action="store_true", help="모델 없이 가짜 임베딩 (검색 순위는 참고용)")
    ap.add_argument("--keep", action="store_true", help="끝나고 demo_ 행을 지우지 않음 (DB에서 직접 보고 싶을 때)")
    args = ap.parse_args(argv)
    if args.fake_embed:
        from rag.fake_embed import embed_passages, embed_query
    else:
        from rag.embed import embed_passages, embed_query

    with connect_or_exit() as conn:
        ensure_v2_schema(conn)
        cleanup(conn)

        insert_plain(conn, METHODS["A"], "약관.pdf", V1, embed_passages)
        insert_plain(conn, METHODS["A"], "약관 (1).pdf", V2, embed_passages)

        insert_plain(conn, METHODS["B"], "약관", V1, embed_passages)
        upsert_by_chunk_no(conn, METHODS["B"], "약관", V2, embed_passages)

        log = []
        for pages, label in ((V1, "v1"), (V2, "v2"), (V2, "v2 한 번 더")):
            r = replace_document(conn, method=METHODS["C"], gen=GEN, doc="약관", pages=pages,
                                 chunk_fn=by_clause, embed=embed_passages)
            log.append(f"  {label:<9} -> {r.action:<9} (버전 {r.doc_version}, 지움 {r.deleted}, 넣음 {r.inserted})")

        res = {k: measure(conn, m, embed_query) for k, m in METHODS.items()}
        if not args.keep:
            cleanup(conn)

    names = {"A": "A. 덧붙이기(새 파일명)", "B": "B. 번호로 덮어쓰기", "C": "C. 통째로 교체"}
    print(f"v1 조항 {len(V1)}개 -> v2 조항 {len(V2)}개 (제2·4조 변경, 제5조 폐지)."
          f" 임베딩: {'가짜(fake)' if args.fake_embed else 'multilingual-e5-small'}\n")
    print("방법 | 조각 수 | 옛 문구 든 조각 | 질문 3개 중 옛 답이 top-3에 | 옛 답이 1등")
    print(f"(기대값) | {len(V2)} | 0 | 0/3 | 0/3")
    for k in "ABC":
        r = res[k]
        print(f"{names[k]} | {r['rows']} | {r['stale_rows']} | {r['q_top3']}/3 | {r['q_top1']}/3")
    print("\nC. 통째로 교체 기록:")
    print("\n".join(log))
    ok = res["C"] == {"rows": len(V2), "stale_rows": 0, "q_top3": 0, "q_top1": 0}
    print("\n결과:", "C는 옛 버전 0, 중복 0" if ok else "C에서 옛 버전이나 중복이 남음 -> 확인 필요")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
