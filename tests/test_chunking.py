from rag.chunking import chunk_baseline


def test_500자마다_자르고_페이지_범위를_기록한다():
    pages = [(1, "가" * 300), (2, "나" * 300), (3, "다" * 300)]
    chunks = chunk_baseline(pages, size=500)
    # 페이지 사이 공백 2칸 포함 902자 -> 500 + 402
    assert [len(c.text) for c in chunks] == [500, 402]
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)
    assert (chunks[1].page_start, chunks[1].page_end) == (2, 3)


def test_빈_페이지는_건너뛰고_공백은_한칸으로():
    pages = [(1, "제1조  목적\n\n이 약관은"), (2, "   \n "), (3, "보상합니다")]
    chunks = chunk_baseline(pages, size=500)
    assert len(chunks) == 1
    assert chunks[0].text == "제1조 목적 이 약관은 보상합니다"
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 3)


def test_조항_중간에서도_그냥_자른다_베이스라인의_약점():
    text = "제5조(보상하는 손해) " + "가" * 490 + " 제6조(보상하지 않는 손해) 다음 사유"
    chunks = chunk_baseline([(1, text)], size=500)
    assert "제6조" in chunks[0].text or "제6조" in chunks[1].text
    assert chunks[0].text.startswith("제5조")
    assert not chunks[1].text.startswith("제6조")  # 조항 경계가 아니라 글자 수로 잘림


def test_겹치기는_앞_조각_끝을_다음_조각_앞에_다시_넣는다():
    chunks = chunk_baseline([(1, "가" * 500 + "나" * 500)], size=500, overlap=100)
    assert chunks[1].text.startswith("가" * 100 + "나")
    assert len(chunk_baseline([(1, "가" * 1000)], size=500)) == 2        # 기본값(겹침 0)은 예전과 같다


def test_굵은_제목은_글자가_두_번씩_뽑힌_것을_한_번으로_되돌린다():
    from rag.chunking import clean_text
    page = "앞 문장입니다.\n제\n제99조\n조((보\n보험\n험금\n금))\n①계약자는 회사에 알려야 합니다."
    assert clean_text(page) == "앞 문장입니다.\n제9조(보험금)\n①계약자는 회사에 알려야 합니다."


def test_본문과_숫자만_있는_줄은_건드리지_않는다():
    from rag.chunking import clean_text
    page = "11\n22\n보험금을 지급합니다.\n1100만원"
    assert clean_text(page) == page


def test_목차_점선_줄은_뺀다():
    from rag.chunking import clean_text
    assert clean_text("제1조 보장종목..........10\n본문") == "본문"


def test_method_이름표로_임베딩_모델과_표가_정해진다():
    from rag.embed import split_method, table_for
    assert split_method("baseline500") == ("baseline500", "e5s") and table_for("baseline500") == "chunks"
    assert split_method("clean500@m3") == ("clean500", "m3") and table_for("clean500@m3") == "chunks_m3"
