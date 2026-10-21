"""모델 없이 쓰는 가짜 임베딩. 테스트와 --fake-embed 실행용.

글자 1개·2개 묶음을 해시해서 384칸 벡터에 더한 뒤 길이 1로 맞춘다. 겹치는 글자가 많을수록
가까워질 뿐, 뜻은 모른다. 그래서 '권한 없는 조각이 0개인가', '옛 버전 조각이 남았나' 같은
행 단위 성질을 확인하는 데만 쓰고, 검색 품질 숫자는 진짜 모델(rag.embed)로 잰다.
"""
from __future__ import annotations

import hashlib
import math

DIM = 384


def _vec(text: str) -> list[float]:
    t = " ".join(text.split())
    v = [0.0] * DIM
    grams = list(t) + [t[i:i + 2] for i in range(len(t) - 1)]
    for g in grams:
        if g.isspace():
            continue
        h = int.from_bytes(hashlib.md5(g.encode("utf-8")).digest()[:8], "big")
        v[h % DIM] += 1.0 if (h >> 32) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    return [_vec(t) for t in texts]


def embed_query(text: str) -> list[float]:
    return _vec(text)
