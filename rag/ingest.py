"""data/processed/gen*_pages.jsonl -> 조각 -> 임베딩 -> Postgres(chunks 표).

실행:  python -m rag.ingest --method baseline500
같은 method로 다시 돌리면 그 method의 기존 행을 지우고 새로 넣는다 (다른 method는 그대로).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag.chunking import chunk_baseline
from rag.db import connect
from rag.embed import embed_passages, to_pgvector

PROCESSED = Path("data/processed")
METHODS = {
    "baseline500": lambda pages: chunk_baseline(pages, size=500),
    # 실험 1: 같은 기계적 자르기에서 크기만 바꾼다 (조각이 작을수록 한 조각에 섞이는 내용이 줄어드는지)
    "fixed200": lambda pages: chunk_baseline(pages, size=200),
    "fixed1000": lambda pages: chunk_baseline(pages, size=1000),
}


def load_pages(path: Path) -> tuple[int, str, list[tuple[int, str]], int]:
    gen, doc, pages, skipped = None, None, [], 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gen, doc = r["gen"], r["doc"]
            # 베이스라인은 일부러 거르지 않는다: 목차 페이지도 그대로 넣는다.
            # (목차가 검색을 방해하는지는 측정 후 '수리' 단계에서 확인)
            # 글자가 없는 빈 페이지와 정말 깨진 페이지만 뺀다.
            if r["status"] in ("empty", "garbled"):
                skipped += 1
                continue
            pages.append((r["page"], r["text"]))
    return gen, doc, pages, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline500", choices=sorted(METHODS))
    args = ap.parse_args(argv)

    files = sorted(PROCESSED.glob("gen*_pages.jsonl"))
    if not files:
        print("먼저 python -m rag.extract_check 를 실행해 주세요.")
        return 1

    t0 = time.time()
    with connect() as conn:
        conn.execute("DELETE FROM chunks WHERE method = %s", (args.method,))
        total = 0
        for f in files:
            gen, doc, pages, skipped = load_pages(f)
            chunks = METHODS[args.method](pages)
            vecs = embed_passages([c.text for c in chunks])
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (method, gen, doc, chunk_no, page_start, page_end, text, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)",
                    [(args.method, gen, doc, c.chunk_no, c.page_start, c.page_end, c.text, to_pgvector(v))
                     for c, v in zip(chunks, vecs)],
                )
            total += len(chunks)
            print(f"{doc}: 페이지 {len(pages)}개 사용(제외 {skipped}) -> 조각 {len(chunks)}개")
        n = conn.execute("SELECT count(*) FROM chunks WHERE method = %s", (args.method,)).fetchone()[0]
    print(f"\n{args.method}: 넣은 조각 {total}개, DB 확인 {n}개 ({'일치' if n == total else '불일치!'}), {time.time() - t0:.1f}초")
    return 0 if n == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
