import pytest

from rag import fake_embed
from rag.acl import post_filter, search_for_user, search_unfiltered
from rag.demo_acl import INTERNAL_DOC, PUBLIC_DOC, QUESTIONS
from rag.demo_reindex import by_clause
from rag.versioning import replace_document

K = 3
GUEST = ["public"]
STAFF = ["public", "underwriting"]


@pytest.fixture
def loaded(conn, method):
    common = dict(method=method, gen=4, chunk_fn=by_clause, embed=fake_embed.embed_passages)
    replace_document(conn, doc="공개약관", pages=PUBLIC_DOC, groups=["public"], **common)
    replace_document(conn, doc="내부기준", pages=INTERNAL_DOC, groups=["underwriting"], **common)
    return method


def _search(conn, method, q, groups, k=K):
    return search_for_user(conn, q, groups, k, method=method, embed_query=fake_embed.embed_query)


def _internal(rows):
    return [r for r in rows if "underwriting" in r[7]]


# --- DB 없이 도는 것 ---

def test_그룹을_문자열로_넘기면_거부():
    with pytest.raises(TypeError):
        search_for_user(None, "질문", "public")


def test_검색_후_거르기는_권한_없는_행만_뺀다():
    rows = [(4, "a", 0, 1, 1, 0.9, "공개", ["public"]), (4, "b", 0, 1, 1, 0.8, "내부", ["underwriting"])]
    assert [r[6] for r in post_filter(rows, GUEST)] == ["공개"]


# --- 실제 DB ---

def test_권한_없는_사용자에게_내부_조각이_0개(conn, loaded):
    for q, _ in QUESTIONS:
        rows = _search(conn, loaded, q, GUEST)
        assert len(rows) == K
        assert _internal(rows) == [], q


def test_필터가_없으면_내부_조각이_넘어간다(conn, loaded):
    """비교 기준: 프롬프트로만 막는 시스템은 이 조각들을 이미 모델에 넘긴 상태다."""
    leaked = sum(len(_internal(search_unfiltered(conn, q, K, method=loaded, embed_query=fake_embed.embed_query)))
                 for q, aimed in QUESTIONS if aimed)
    assert leaked > 0


def test_검색_후_거르기는_k개보다_적어지고_검색_안에서_거르면_k개(conn, loaded):
    short = 0
    for q, aimed in QUESTIONS:
        raw = search_unfiltered(conn, q, K, method=loaded, embed_query=fake_embed.embed_query)
        if len(post_filter(raw, GUEST)) < K:
            short += 1
            assert len(_search(conn, loaded, q, GUEST)) == K
    assert short > 0


def test_권한_있는_사람은_내부_조각을_받는다(conn, loaded):
    hits = sum(bool(_internal(_search(conn, loaded, q, STAFF))) for q, aimed in QUESTIONS if aimed)
    assert hits == 3


def test_빈_그룹은_아무것도_안_나온다(conn, loaded):
    assert _search(conn, loaded, "설계사 수수료는 얼마나 돼요?", []) == []


def test_권한_회수는_다시_넣지_않아도_바로_적용(conn, loaded):
    r = replace_document(conn, method=loaded, gen=4, doc="내부기준", pages=INTERNAL_DOC, groups=["underwriting_lead"],
                         chunk_fn=by_clause, embed=fake_embed.embed_passages)
    assert r.action == "permissions_updated"
    for q, aimed in QUESTIONS:
        assert _internal(_search(conn, loaded, q, STAFF)) == []
