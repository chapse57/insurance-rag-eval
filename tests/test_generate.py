from rag.generate import check, build_prompt

ROWS = [(4, "gen4.pdf", 29, 10, 10, 0.9, "비급여(3대) 도수치료 보험가입금액(350만원) 한도 이내에서 최대 50회(1년간)"),
        (4, "gen4.pdf", 30, 10, 10, 0.8, "해지환급금이 발생하지 않습니다")]


def test_인용이_조각에_있으면_통과():
    raw = '{"answer": "1년에 최대 50회입니다.", "chunk": 1, "quote": "최대 50회(1년간)"}'
    a = check("q", ROWS, raw)
    assert a.guard == "ok" and not a.refused


def test_띄어쓰기만_달라도_통과():
    raw = '{"answer": "50회", "chunk": 1, "quote": "최대50회 (1년간)"}'
    assert check("q", ROWS, raw).guard == "ok"


def test_조각에_없는_인용은_차단():
    raw = '{"answer": "1년에 100회입니다.", "chunk": 1, "quote": "최대 100회(1년간)"}'
    a = check("q", ROWS, raw)
    assert a.refused and a.guard == "quote_not_found"


def test_다른_조각을_가리키면_차단():
    raw = '{"answer": "50회", "chunk": 2, "quote": "최대 50회(1년간)"}'
    assert check("q", ROWS, raw).guard == "quote_not_found"


def test_없는_조각번호는_차단():
    assert check("q", ROWS, '{"answer": "x", "chunk": 7, "quote": "최대 50회"}').guard == "bad_chunk"


def test_모델이_모른다고_하면_거절():
    assert check("q", ROWS, '{"answer": null, "chunk": null, "quote": null}').guard == "model_refused"


def test_JSON이_아니면_거절():
    assert check("q", ROWS, "잘 모르겠습니다").guard == "bad_json"


def test_프롬프트에_조각번호와_쪽이_들어간다():
    p = build_prompt("질문", ROWS)
    assert "<1> (4세대 10쪽)" in p and "<2>" in p


def test_길이_상한에서_잘린_답은_거절():
    raw = '{"answer": "1년에 최대 50회입니다. 다만 10회 이후에는'
    assert check("q", ROWS, raw, stop_reason="max_tokens").guard == "truncated"


def test_도구로_found_false면_거절():
    assert check("q", ROWS, '{"found": false}').guard == "model_refused"


def test_도구_입력_형식도_통과():
    raw = '{"found": true, "answer": "50회", "chunk": 1, "quote": "최대 50회(1년간)"}'
    assert check("q", ROWS, raw).guard == "ok"
