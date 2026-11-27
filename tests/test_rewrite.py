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


def test_새_질문의_검색어는_평가용_파일이_아니라_따로_지정한_파일에_저장한다(tmp_path):
    main, live = tmp_path / "rewrites.json", tmp_path / "live.json"
    calls = []

    def fake(q):
        calls.append(q)
        return {"query": "약관 문체 질문", "terms": ["용어"]}

    RW.rewrite("평가 문항", cache_path=main, ask=fake)                       # 평가용 파일에 저장
    before = main.read_text(encoding="utf-8")
    first = RW.rewrite("화면에서 들어온 질문", cache_path=main, ask=fake, save_to=live)
    second = RW.rewrite("화면에서 들어온 질문", cache_path=main, ask=fake, save_to=live)
    assert first == second and calls == ["평가 문항", "화면에서 들어온 질문"]   # 두 번째는 저장된 것을 읽는다
    assert main.read_text(encoding="utf-8") == before                       # 평가용 파일은 그대로
    assert RW.rewrite("평가 문항", cache_path=main, ask=fake, save_to=live) and len(calls) == 2   # 평가 문항은 평가용 파일에서 읽는다
