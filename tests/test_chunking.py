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


# --- struct500: 구조를 따라 자르고 이름표 붙이기 -------------------------------------------
from rag.chunking import chunk_structured, classify_line  # noqa: E402

STRUCT_PAGE = "\n".join([
    "보통약관",
    "제3관 회사가 보상하지 않는 사항",
    "제4조 (보상하지 않는 사항)",
    "(2) 질병비급여",
    "② 회사는 다음의 비급여의료비에 대해서는 보상하지 않습니다.",
] + [f"{i}. 보상하지 않는 항목 {i}번에 대한 긴 설명 문장입니다 " + "가" * 40 for i in range(1, 12)] + [
    "제5조 (특별약관에서 보상하는 사항)",
    "회사는 특별약관에서 정한 바에 따라 보상합니다. " + "나" * 160,
])


def test_제목_줄과_본문_속_인용을_구분한다():
    assert classify_line("제30조(중대사유로인한해지)") == [("art", "제30조(중대사유로인한해지)")]
    assert classify_line("2.실손의료비(비급여주사료)특별약관제1조(보장종목)") == [
        ("yak", "실손의료비(비급여주사료)특별약관"), ("art", "제1조(보장종목)")]
    assert classify_line("제1장일반사항및용어의정의제1조(보장종목)") == [
        ("part", "제1장일반사항및용어의정의"), ("art", "제1조(보장종목)")]
    assert classify_line("해당진료기록을통해객관적으로확인되는경우는제3조(보상내용)") == []   # 본문 속 인용
    assert classify_line("이 특별약관에 따른 보험금 지급 실적을 고려하여 순보험료(특별약관") == []
    assert classify_line("(3) 질병입원") == [("sub", "(3) 질병입원")]


def test_조각마다_위치_이름표를_붙이고_목록_중간이면_항의_첫머리도_붙인다():
    chunks = chunk_structured([(59, STRUCT_PAGE)], size=500)
    first, second = chunks[0], chunks[1]
    assert first.text.startswith("[보통약관 > 제3관 회사가 보상하지 않는 사항 > 제4조 (보상하지 않는 사항) > (2) 질병비급여]")
    # 목록이 500자를 넘어 두 조각으로 나뉘면, 뒤 조각에도 '보상하지 않는다'는 머리말이 따라간다
    assert "(② 회사는 다음의 비급여의료비에 대해서는 보상하지 않습니다.)" in second.text
    assert second.text.split("] ", 1)[1].split(") ", 1)[1][:2].rstrip(".").isdigit()   # 항목 번호에서 잘렸다
    # 조가 바뀌면 새 조각, 이름표의 조와 소제목도 바뀐다
    assert chunks[-1].text.startswith("[보통약관 > 제3관 회사가 보상하지 않는 사항 > 제5조 (특별약관에서 보상하는 사항)]")


def test_약관_끝의_짧은_조각은_앞_조각에_붙는다():
    page = "\n".join(["보통약관", "제3관 보험금의 지급", "제6조 (보험금의 청구)", "청구 서류는 다음과 같습니다. " + "다" * 200,
                      "제7조 (준용규정)", "이 약관에서 정하지 않은 사항은 법령을 따릅니다.", "보험료 자동납입 특별약관",
                      "제1조 (보험료납입)", "보험료는 자동이체로 납입합니다. " + "라" * 200])
    chunks = chunk_structured([(1, page)], size=500)
    assert len(chunks) == 2
    assert "제7조 (준용규정)" in chunks[0].text                       # 짧은 준용규정은 따로 조각이 되지 않는다
    assert chunks[1].text.startswith("[보험료 자동납입 특별약관 > 제1조 (보험료납입)]")


def test_mixed는_요약부를_baseline과_똑같이_자르고_본문에만_이름표를_붙인다():
    from rag.chunking import chunk_mixed
    summary = (1, "상품 요약 안내문입니다. " + "마" * 700)
    body = (2, "\n".join(["보통약관", "제1관 일반사항 및 용어의 정의", "제1조 (보장종목)", "회사는 다음을 보상합니다. " + "바" * 200]))
    chunks = chunk_mixed([summary, body], [summary, body], size=500)
    base = chunk_baseline([summary], size=500)
    assert [c.text for c in chunks[:len(base)]] == [c.text for c in base]          # 요약부는 글자까지 같다
    assert chunks[len(base)].text.startswith("[보통약관 > 제1관 일반사항 및 용어의 정의 > 제1조 (보장종목)]")
    assert [c.chunk_no for c in chunks] == list(range(len(chunks)))
