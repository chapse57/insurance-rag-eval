import rag.rewrite as RW


def test_검색어는_다시_쓴_문장_뒤에_용어를_붙인다():
    assert RW.build_search_query("청약의 철회 가능 기간", ["청약 철회", " 보험증권 "]) == "청약의 철회 가능 기간 (청약 철회, 보험증권)"
    assert RW.build_search_query("해지환급금", []) == "해지환급금"


def test_같은_질문은_저장해_둔_검색어를_다시_쓴다(tmp_path):
    calls = []

    def fake(question):
        calls.append(question)
        return {"query": "해지환급금 지급 여부", "terms": ["해지환급금"]}

    cache = tmp_path / "rewrites.json"
    first = RW.rewrite("해지하면 돌려받는 돈 있어요?", cache_path=cache, ask=fake)
    second = RW.rewrite("해지하면 돌려받는 돈 있어요?", cache_path=cache, ask=fake)
    assert first == second == "해지환급금 지급 여부 (해지환급금)"
    assert len(calls) == 1


def test_형식이_이상하면_원래_질문으로_검색한다(tmp_path):
    assert RW.rewrite("원래 질문", cache_path=tmp_path / "c.json", ask=lambda q: {"query": ""}) == "원래 질문"
