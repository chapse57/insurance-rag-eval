"""글자 -> 숫자 목록(벡터). 무료 로컬 모델 multilingual-e5-small(384차원)을 쓴다.

e5 계열은 저장할 글 앞에 'passage: ', 질문 앞에 'query: '를 붙여야 성능이 제대로 나온다.
첫 실행 때 모델(약 470MB)을 내려받는다.
"""
from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vecs = _model().encode([f"passage: {t}" for t in texts], batch_size=batch_size,
                           normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    return _model().encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
