"""문서가 바뀌었을 때 옛 버전 조각을 남기지 않고 '통째로 교체'하는 재색인.

흔한 실패 두 가지 (rag/demo_reindex.py가 둘 다 재현한다):
  1. 덧붙이기: 고친 문서를 새 파일처럼 그냥 넣는다 -> 옛 버전 조각이 전부 남아 검색에 같이 걸린다.
  2. 조각 번호로 덮어쓰기(upsert): 0번~N번 조각만 새 글로 바뀌고, 새 버전이 더 짧으면
     뒤쪽 옛 조각이 그대로 남는다. 조각 수가 줄어든 경우에만 생겨서 눈치채기 어렵다.

여기서 하는 방법:
  - 문서 단위로 잠그고, 한 트랜잭션 안에서 '그 문서의 조각을 전부 지우고 -> 새 조각을 넣고 ->
    넣은 수와 DB 수를 대조'한다. 하나라도 실패하면 통째로 취소(rollback)되어 옛 버전이 그대로 남는다.
    다른 사람의 검색은 커밋 전까지 옛 버전만, 커밋 후에는 새 버전만 본다 (반쯤 섞인 상태가 없다).
  - 문서 전체 글자의 해시(doc_sha)가 같으면 다시 넣지 않는다 -> 같은 파일을 두 번 올려도 중복 0.
  - 내용은 같고 권한만 바뀌면 임베딩을 다시 하지 않고 allowed_groups만 바꾼다 (권한 회수가 바로 적용됨).

문서를 구분하는 키(doc)는 '업로드한 파일 이름'이 아니라 앱에서 그 문서의 고정 ID여야 한다.
파일 이름을 키로 쓰면 '약관 (1).pdf'처럼 이름만 바뀐 재업로드가 새 문서로 들어가 실패 1이 된다.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

from rag.chunking import Chunk, chunk_baseline

Pages = Sequence[tuple[int, str]]
ChunkFn = Callable[[Pages], list[Chunk]]
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class ReindexResult:
    action: str          # inserted | replaced | unchanged | permissions_updated
    doc_version: int
    deleted: int = 0
    inserted: int = 0
    regrouped: int = 0


def document_sha(pages: Pages) -> str:
    """페이지 글자를 공백 정리 후 이어 붙인 해시. 공백만 바뀐 재업로드는 같은 문서로 본다."""
    h = hashlib.sha256()
    for page_no, text in pages:
        h.update(f"{page_no}\x1f{' '.join(text.split())}\x1e".encode("utf-8"))
    return h.hexdigest()


def normalize_groups(groups: Sequence[str]) -> list[str]:
    g = sorted({x.strip() for x in groups if x and x.strip()})
    if not g:
        raise ValueError("allowed_groups가 비어 있습니다. 아무도 못 보는 문서는 넣지 말고 지우세요.")
    return g


def _state(conn, method: str, doc: str, sha: str, groups: list[str]):
    return conn.execute(
        "SELECT count(*), coalesce(max(doc_version), 0),"
        "       coalesce(bool_and(doc_sha IS NOT DISTINCT FROM %(sha)s), false),"
        "       coalesce(bool_and(allowed_groups = %(groups)s::text[]), false)"
        " FROM chunks WHERE method = %(method)s AND doc = %(doc)s",
        {"method": method, "doc": doc, "sha": sha, "groups": groups},
    ).fetchone()


INSERT_SQL = (
    "INSERT INTO chunks (method, gen, doc, chunk_no, page_start, page_end, text, embedding,"
    "                    doc_version, doc_sha, allowed_groups)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s::text[])"
)


def replace_document(conn, *, method: str, gen: int, doc: str, pages: Pages,
                     groups: Sequence[str] = ("public",), chunk_fn: ChunkFn | None = None,
                     embed: EmbedFn | None = None) -> ReindexResult:
    """doc의 조각을 새 내용으로 통째로 교체한다. conn은 autocommit=True로 연 연결을 권장."""
    from rag.embed import to_pgvector

    groups = normalize_groups(groups)
    sha = document_sha(pages)

    n_old, version, same_sha, same_groups = _state(conn, method, doc, sha, groups)
    if n_old and same_sha:
        if same_groups:
            return ReindexResult("unchanged", version)
        with conn.transaction():
            n = conn.execute(
                "UPDATE chunks SET allowed_groups = %s::text[] WHERE method = %s AND doc = %s",
                (groups, method, doc),
            ).rowcount
        return ReindexResult("permissions_updated", version, regrouped=n)

    # 임베딩은 잠그기 전에 끝낸다 (느린 작업 동안 다른 사람의 검색·재색인을 막지 않도록).
    chunks = (chunk_fn or (lambda p: chunk_baseline(list(p), size=500)))(pages)
    if not chunks:
        raise ValueError(f"{doc}: 조각이 0개입니다 (빈 문서). 지우려면 delete_document()를 쓰세요.")
    if embed is None:
        from rag.embed import embed_passages as embed
    vecs = embed([c.text for c in chunks])
    if len(vecs) != len(chunks):
        raise RuntimeError(f"{doc}: 조각 {len(chunks)}개인데 임베딩 {len(vecs)}개")

    with conn.transaction():
        # 같은 문서를 동시에 두 번 재색인하면 번호가 엉킬 수 있어서 문서 단위로 줄을 세운다.
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{method}|{doc}",))
        version = conn.execute(
            "SELECT coalesce(max(doc_version), 0) FROM chunks WHERE method = %s AND doc = %s",
            (method, doc),
        ).fetchone()[0]
        deleted = conn.execute("DELETE FROM chunks WHERE method = %s AND doc = %s", (method, doc)).rowcount
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, [
                (method, gen, doc, c.chunk_no, c.page_start, c.page_end, c.text, to_pgvector(v),
                 version + 1, sha, groups)
                for c, v in zip(chunks, vecs)
            ])
        n = conn.execute("SELECT count(*) FROM chunks WHERE method = %s AND doc = %s",
                         (method, doc)).fetchone()[0]
        if n != len(chunks):
            raise RuntimeError(f"{doc}: 넣은 조각 {len(chunks)}개, DB {n}개 -> 취소")
    return ReindexResult("replaced" if deleted else "inserted", version + 1, deleted, len(chunks))


def delete_document(conn, *, method: str, doc: str) -> int:
    """문서를 지우면 그 문서의 조각도 전부 지운다. 지운 행 수를 돌려준다."""
    with conn.transaction():
        return conn.execute("DELETE FROM chunks WHERE method = %s AND doc = %s", (method, doc)).rowcount
