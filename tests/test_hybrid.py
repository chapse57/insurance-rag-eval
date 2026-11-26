import rag.hybrid as H
import rag.rerank as R


def row(no, text="글"):
    return (4, "gen4.pdf", no, 1, 1, 0.5, text)


def test_두_순위를_합치면_양쪽에서_모두_높은_조각이_앞에_온다():
    a = [row(1), row(2), row(3)]
    b = [row(3), row(1), row(9)]
    assert [r[2] for r in H.fuse([a, b])][:2] == [1, 3]      # 1번: 1등+2등, 3번: 3등+1등
    assert {r[2] for r in H.fuse([a, b])} == {1, 2, 3, 9}     # 한쪽에만 있는 조각도 남는다


def test_한쪽_순위의_1등은_다른_쪽에_없어도_맨_앞에_남는다():
    a = [row(7), row(2), row(3), row(4), row(5), row(6)]       # 7번: 원래 질문에서 1등, 바꿔 쓴 질문에는 없음
    b = [row(6), row(2), row(3), row(4), row(5)]               # 6번: 바꿔 쓴 질문에서 1등
    fused = H.fuse([a, b])
    assert [r[2] for r in fused].index(7) >= 3                 # 합치기만 하면 7번은 상위 3 밖으로 밀린다
    seated = H.seat_leaders(fused, [a, b])
    assert {r[2] for r in seated[:2]} == {6, 7}                # 두 순위의 1등이 맨 앞 두 자리
    assert [r[2] for r in seated[2:]] == [r[2] for r in fused if r[2] not in (6, 7)]   # 나머지 순서는 그대로


def test_거의_같은_조각은_하나만_남긴다():
    same = "보험금을 지급하지 않는 사유는 다음과 같습니다. " * 5
    rows = [row(1, same), row(61, same + "끝"), row(2, "전혀 다른 내용의 조각입니다. 해지환급금은 없습니다.")]
    assert [r[2] for r in H.drop_twins(rows)] == [1, 2]


def test_후보는_조각_번호로_중복을_없앤다():
    assert [r[2] for r in H.merge_candidates([row(5), row(2)], [row(2), row(7)])] == [2, 5, 7]


def test_재정렬_점수는_한_번_계산하면_파일에서_다시_읽는다(tmp_path, monkeypatch):
    calls = []

    class Fake:
        def predict(self, pairs, **kw):
            calls.append(len(pairs))
            return [0.25] * len(pairs)

    monkeypatch.setattr(R, "_model", lambda max_length=512: Fake())
    cache = tmp_path / "scores.jsonl"
    assert R.scores("질문", ["가", "나"], cache_path=cache) == [0.25, 0.25]
    assert R.scores("질문", ["나", "다"], cache_path=cache) == [0.25, 0.25]
    assert calls == [2, 1]                                     # '나'는 다시 계산하지 않았다
