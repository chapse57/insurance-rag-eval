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
