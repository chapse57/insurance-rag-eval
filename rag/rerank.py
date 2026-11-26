"""재정렬(Reranker): 1차 검색으로 가져온 조각들을, 질문과 조각을 '같이' 읽는 모델로 순위를 다시 매긴다.

1차 검색(임베딩)은 질문과 조각을 따로 숫자로 바꿔 비교하기 때문에 말투가 다르면 놓치기 쉽다.
재정렬 모델은 둘을 한 번에 읽어서 '이 조각이 이 질문의 답이냐'를 점수로 낸다. 대신 느리다.
그래서 1차로 20개만 가져와서 그 20개 안에서만 다시 줄 세운다.
-> 정답이 1차 20개 안에 없으면 재정렬로도 못 올린다 (evaluate가 이 한계를 따로 보여준다).

모델: BAAI/bge-reranker-v2-m3 (한국어 포함 다국어, 약 2.2GB, 첫 실행 때 내려받음)
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

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


# ---------------------------------------------------------------------------
# 재정렬 점수 저장. 재정렬은 CPU에서 질문당 수십 초가 걸리는데, 같은 (질문, 조각)이면 점수가 항상 같다
# (반복 실행 결과가 바이트 단위로 같음을 확인함). 한 번 계산한 점수는 파일에 적어 두고 다시 쓴다.
# ---------------------------------------------------------------------------
SCORE_CACHE = Path("eval/cache/rerank_scores.jsonl")
_mem: dict[str, dict[str, float]] = {}


def score_key(question: str, text: str, max_length: int) -> str:
    return hashlib.sha1(f"{MODEL_NAME}|{max_length}|{question}|{text}".encode("utf-8")).hexdigest()


def _cache(path: Path) -> dict[str, float]:
    key = str(path)
    if key not in _mem:
        _mem[key] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        _mem[key][d["k"]] = d["s"]
    return _mem[key]


def scores(question: str, texts: list[str], max_length: int = 512, cache_path: Path = SCORE_CACHE) -> list[float]:
    """질문과 조각 글 목록 -> 재정렬 점수 목록 (같은 순서). 저장된 점수가 있으면 모델을 부르지 않는다."""
    cache = _cache(cache_path)
    keys = [score_key(question, t, max_length) for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        new = _model(max_length).predict([(question, texts[i]) for i in todo], batch_size=32, show_progress_bar=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as f:
            for i, sc in zip(todo, new):
                cache[keys[i]] = round(float(sc), 5)
                f.write(json.dumps({"k": keys[i], "s": cache[keys[i]]}) + "\n")
    return [cache[k] for k in keys]
