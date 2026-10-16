"""재정렬(Reranker): 1차 검색으로 가져온 조각들을, 질문과 조각을 '같이' 읽는 모델로 순위를 다시 매긴다.

1차 검색(임베딩)은 질문과 조각을 따로 숫자로 바꿔 비교하기 때문에 말투가 다르면 놓치기 쉽다.
재정렬 모델은 둘을 한 번에 읽어서 '이 조각이 이 질문의 답이냐'를 점수로 낸다. 대신 느리다.
그래서 1차로 20개만 가져와서 그 20개 안에서만 다시 줄 세운다.
-> 정답이 1차 20개 안에 없으면 재정렬로도 못 올린다 (evaluate가 이 한계를 따로 보여준다).

모델: BAAI/bge-reranker-v2-m3 (한국어 포함 다국어, 약 2.2GB, 첫 실행 때 내려받음)
"""
from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=2)
def _model(max_length: int = 512):
    """max_length: 질문+조각을 몇 토큰까지 읽을지. 짧을수록 빠르지만 조각 뒷부분을 못 읽는다."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(MODEL_NAME, max_length=max_length)


def rerank(question: str, rows: list, max_length: int = 512) -> list:
    """rows: search() 결과. 재정렬 점수가 높은 순서로 다시 정렬해서 돌려준다."""
    if not rows:
        return rows
    scores = _model(max_length).predict([(question, r[6]) for r in rows], batch_size=8, show_progress_bar=False)
    order = sorted(range(len(rows)), key=lambda i: -float(scores[i]))
    return [rows[i] for i in order]
