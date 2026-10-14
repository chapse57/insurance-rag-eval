from rag.extract_check import page_stat, summarize


def test_정상_한글_페이지():
    s = page_stat(1, "제3조(보상하는 사항) 회사는 피보험자가 입원하여 치료를 받은 경우")
    assert s.status == "ok"


def test_글꼴이_빠져_깨진_페이지():
    s = page_stat(1, "¿ÀÇ ¼ö ºÎ´ã ±Ý¾× " * 5)
    assert s.status == "garbled"


def test_빈_페이지():
    assert page_stat(1, "  12  ").status == "empty"


def test_요약_판정():
    pages = [page_stat(i, "회사는 보험금을 지급합니다 " * 3) for i in range(1, 10)] + [page_stat(10, "")]
    assert summarize("gen4.pdf", pages)["verdict"] == "사용 가능"


def test_목차는_깨짐이_아니라_목차로_판정():
    toc = "제6조보험금의청구........................92\n" * 6
    assert page_stat(1, toc).status == "toc"


def test_굵은글씨_겹침_흔적을_센다():
    from rag.extract_check import doubled_runs
    assert doubled_runs("((33)) 질 질병 병입 입원 원 ①회사는") == 1
    assert doubled_runs("회사는 피보험자가 질병으로 입원하여") == 0


def test_띄어쓰기_비율():
    from rag.extract_check import space_ratio
    assert space_ratio("회사는 피보험자가 질병으로 입원하여") > 0.15  # 3칸 / 한글 15자 = 0.2
    assert space_ratio("회사는피보험자가질병으로입원하여") == 0.0
