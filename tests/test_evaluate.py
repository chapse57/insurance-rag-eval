from rag.evaluate import first_hit_rank, metrics, norm


def _row(gen, text, page=1):
    return (gen, f"gen{gen}.pdf", 0, page, page, 0.5, text)


def test_띄어쓰기_무시하고_근거문구를_찾는다():
    rows = [_row(4, "엉뚱한 조각"), _row(4, "보험가입금액(350만원)한도이내에서최대50회(1년간)")]
    assert first_hit_rank(rows, 4, "최대 50회(1년간)") == 2


def test_세대가_다르면_못_맞춘_것():
    rows = [_row(3, "350만원 이내에서 50회")]
    assert first_hit_rank(rows, 4, "350만원 이내에서 50회") is None


def test_조각_경계에서_잘린_근거는_놓친_것():
    rows = [_row(4, "... 최대 50"), _row(4, "회(1년간) ...")]
    assert first_hit_rank(rows, 4, "최대 50회(1년간)") is None


def test_지표_계산():
    m = metrics([1, 2, 4, None])
    assert m["hit@3"] == 0.5             # 1등, 2등만 3위 안
    assert m["mrr@10"] == round((1 + 0.5 + 0.25) / 4, 3)


def test_norm():
    assert norm(" 제 1 조\n(목적) ") == "제1조(목적)"


def test_글자예산_안에서만_정답을_센다():
    from rag.evaluate import budget_hit
    rows = [_row(4, "가" * 1000), _row(4, "나" * 1000), _row(4, "다" * 1000), _row(4, "정답문장")]
    assert budget_hit(rows, 4, "정답문장", budget=3000) is False   # 4번째 조각은 3000자 밖
    assert budget_hit(rows, 4, "정답문장", budget=3001) is True
    small = [_row(4, "가" * 500)] * 5 + [_row(4, "정답문장")]
    assert budget_hit(small, 4, "정답문장", budget=3000) is True  # 500자 x 5 = 2500자 다음 조각
