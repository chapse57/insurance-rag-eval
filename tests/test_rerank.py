import rag.rerank as R


class _Fake:
    def predict(self, pairs, **kw):
        # 조각에 '정답'이 들어 있으면 높은 점수
        return [1.0 if "정답" in text else 0.0 for _, text in pairs]


def test_재정렬은_점수순으로_다시_줄세운다(monkeypatch):
    monkeypatch.setattr(R, "_model", lambda: _Fake())
    rows = [(4, "d", 0, 1, 1, 0.9, "엉뚱한 조각"), (4, "d", 1, 2, 2, 0.8, "정답 조각")]
    out = R.rerank("질문", rows)
    assert out[0][6] == "정답 조각" and len(out) == 2


def test_빈_결과는_그대로():
    assert R.rerank("질문", []) == []
