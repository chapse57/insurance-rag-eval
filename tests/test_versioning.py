import pytest

from rag import fake_embed
from rag.demo_reindex import OLD_MARKERS, V1, V2, by_clause, upsert_by_chunk_no
from rag.versioning import delete_document, document_sha, normalize_groups, replace_document


class CountingEmbed:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls += len(texts)
        return fake_embed.embed_passages(texts)


def _replace(conn, method, pages, embed=None, groups=("public",), doc="약관"):
    return replace_document(conn, method=method, gen=4, doc=doc, pages=pages, groups=groups,
                            chunk_fn=by_clause, embed=embed or fake_embed.embed_passages)


def _rows(conn, method, doc="약관"):
    return conn.execute("SELECT text, doc_version, allowed_groups FROM chunks WHERE method = %s AND doc = %s",
                        (method, doc)).fetchall()


# --- DB 없이 도는 것 ---

def test_공백만_다른_재업로드는_같은_문서로_본다():
    assert document_sha([(1, "제1조  목적\n이 약관은")]) == document_sha([(1, "제1조 목적 이 약관은")])
    assert document_sha([(1, "연간 10회")]) != document_sha([(1, "연간 20회")])


def test_빈_권한은_거부하고_권한은_정렬_중복제거():
    with pytest.raises(ValueError):
        normalize_groups([])
    with pytest.raises(ValueError):
        normalize_groups(["", "  "])
    assert normalize_groups(["b", "a", "b "]) == ["a", "b"]


# --- 실제 DB ---

def test_통째로_교체하면_옛_버전_조각이_0개(conn, method):
    assert _replace(conn, method, V1).action == "inserted"
    r = _replace(conn, method, V2)
    assert (r.action, r.doc_version, r.deleted, r.inserted) == ("replaced", 2, len(V1), len(V2))
    rows = _rows(conn, method)
    assert len(rows) == len(V2)
    assert {v for _, v, _ in rows} == {2}
    assert not any(m in t for t, _, _ in rows for m in OLD_MARKERS)


def test_같은_내용을_다시_넣으면_임베딩도_행도_그대로(conn, method):
    _replace(conn, method, V2)
    emb = CountingEmbed()
    r = _replace(conn, method, V2, embed=emb)
    assert r.action == "unchanged" and emb.calls == 0
    assert len(_rows(conn, method)) == len(V2)


def test_번호로_덮어쓰기는_새_버전이_짧으면_뒤쪽_옛_조각이_남는다(conn, method):
    """고친 방법과 비교하려고 남겨 둔 '실패하는 방법' 재현."""
    upsert_by_chunk_no(conn, method, "약관", V1, fake_embed.embed_passages)
    upsert_by_chunk_no(conn, method, "약관", V2, fake_embed.embed_passages)
    rows = _rows(conn, method)
    assert len(rows) == len(V1)
    assert sum("10% 할인" in t for t, _, _ in rows) == 1   # 폐지된 제5조가 그대로 검색 대상


def test_넣다가_실패하면_통째로_취소되어_옛_버전이_그대로(conn, method):
    _replace(conn, method, V1)
    broken = V2[:-1] + [(4, "제4조(해지환급금) 추출 중 깨진 글자\x00가 섞임")]   # PDF 추출에서 실제로 생기는 NUL 문자
    with pytest.raises(Exception):
        _replace(conn, method, broken)
    rows = _rows(conn, method)
    assert len(rows) == len(V1) and {v for _, v, _ in rows} == {1}   # 지웠던 것까지 되돌아옴


def test_권한만_바뀌면_임베딩_없이_그룹만_바뀐다(conn, method):
    _replace(conn, method, V2, groups=["underwriting"])
    emb = CountingEmbed()
    r = _replace(conn, method, V2, embed=emb, groups=["underwriting_lead"])
    assert (r.action, r.regrouped, emb.calls) == ("permissions_updated", len(V2), 0)
    assert {tuple(g) for _, _, g in _rows(conn, method)} == {("underwriting_lead",)}


def test_문서를_지우면_조각도_0개(conn, method):
    _replace(conn, method, V2)
    assert delete_document(conn, method=method, doc="약관") == len(V2)
    assert _rows(conn, method) == []


def test_다른_문서는_건드리지_않는다(conn, method):
    _replace(conn, method, V1, doc="약관A")
    _replace(conn, method, V1, doc="약관B")
    _replace(conn, method, V2, doc="약관A")
    assert len(_rows(conn, method, "약관A")) == len(V2)
    assert len(_rows(conn, method, "약관B")) == len(V1)
