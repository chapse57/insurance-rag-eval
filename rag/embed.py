"""글자 -> 숫자 목록(벡터). 무료 로컬 모델을 쓴다.

기본은 multilingual-e5-small(384차원). e5 계열은 저장할 글 앞에 'passage: ', 질문 앞에 'query: '를
붙여야 성능이 제대로 나온다. 첫 실행 때 모델(약 470MB)을 내려받는다.

실험용으로 더 큰 모델(bge-m3, 1024차원, 약 2.3GB)도 고를 수 있다. 고르는 방법은 method 이름 뒤에
'@m3'를 붙이는 것 (예: baseline500@m3). 차원이 달라서 조각은 별도 표(chunks_m3)에 들어간다.
'@'가 없으면 지금까지와 똑같이 e5-small + chunks 표를 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384


@dataclass(frozen=True)
class EmbedSpec:
    name: str            # Hugging Face 모델 이름
    dim: int
    passage_prefix: str
    query_prefix: str
    table: str           # 이 모델의 벡터가 들어가는 표
    batch_size: int


DEFAULT_EMBEDDER = "e5s"
EMBEDDERS = {
    "e5s": EmbedSpec(MODEL_NAME, DIM, "passage: ", "query: ", "chunks", 32),
    "m3": EmbedSpec("BAAI/bge-m3", 1024, "", "", "chunks_m3", 8),
}


def split_method(method: str) -> tuple[str, str]:
    """'baseline500@m3' -> ('baseline500', 'm3'),  'baseline500' -> ('baseline500', 'e5s')."""
    base, _, key = method.partition("@")
    key = key or DEFAULT_EMBEDDER
    if key not in EMBEDDERS:
        raise ValueError(f"모르는 임베딩 모델 이름표: {key!r} (가능: {sorted(EMBEDDERS)})")
    return base, key


def table_for(method: str) -> str:
    return EMBEDDERS[split_method(method)[1]].table


@lru_cache(maxsize=2)
def _load(key: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDERS[key].name)
    if key != DEFAULT_EMBEDDER:
        model.max_seq_length = min(model.max_seq_length, 512)   # 조각이 500자 안팎이라 충분. 긴 입력으로 느려지는 것 방지
    return model


def _model(key: str = DEFAULT_EMBEDDER):
    return _load(key)


def embed_passages(texts: list[str], batch_size: int | None = None, show_progress_bar: bool = True,
                   model: str = DEFAULT_EMBEDDER) -> list[list[float]]:
    spec = EMBEDDERS[model]
    vecs = _load(model).encode([f"{spec.passage_prefix}{t}" for t in texts],
                               batch_size=batch_size or spec.batch_size,
                               normalize_embeddings=True, show_progress_bar=show_progress_bar)
    return [v.tolist() for v in vecs]


def embed_query(text: str, model: str = DEFAULT_EMBEDDER) -> list[float]:
    spec = EMBEDDERS[model]
    return _load(model).encode([f"{spec.query_prefix}{text}"], normalize_embeddings=True)[0].tolist()


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
