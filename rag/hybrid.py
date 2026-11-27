"""최종 검색 방식: 모델 두 개 + 질문 두 개 + 재정렬 + 순위 합치기.

    질문 ─┬─ 원래 질문 ───┬─ e5-small 상위 k ─┐
          │               └─ bge-m3  상위 k ─┴─ 합친 후보 -> 재정렬 -> 순위 A
          └─ 바꿔 쓴 질문 ─┬─ e5-small 상위 k ─┐
             (rag.rewrite) └─ bge-m3  상위 k ─┴─ 합친 후보 -> 재정렬 -> 순위 B
    순위 A + 순위 B (RRF) -> 각 순위의 1등을 맨 앞으로 -> 거의 같은 조각(쌍둥이) 제거 -> 상위 몇 개

왜 이렇게 하나 (60문항 측정, 2026-11-26):
- 모델 하나의 상위 20개에는 정답이 없는 문항이 있다. 두 모델 후보를 합치면 60문항 전부 후보 안에 정답이 있다.
- 원래 질문은 요약표 문항에 강하고, 약관 용어로 바꿔 쓴 질문은 조문 본문 문항에 강하다. 한쪽만 쓰면 46/60.
- 두 순위를 점수가 아니라 순위로 합친다. 재정렬 점수는 질문이 다르면 크기가 달라서 그대로 비교할 수 없다.
- 한쪽 순위의 1등은 다른 쪽 순위에 아예 없어도 상위에 남긴다 (seat_leaders). 바꿔 쓴 질문이 뜻을 틀리게 옮기면
  원래 질문의 1등이 합친 순위에서 4~5등으로 밀린다 (q03, n34). 1등 + 없음 = 1/6 + 1/104 = 0.176 인데
  6등 + 1등 = 1/11 + 1/6 = 0.258 이라서다. 이 규칙으로 52 -> 54/60, 새로 틀린 문항 0.
  주의: 틀린 문항을 들여다본 뒤에 정한 규칙이다. 이 60문항 점수는 낙관적이므로 새 문항으로 다시 확인해야 한다.
- 약관이 한 파일에 두 벌 실린 구간이 있어서 같은 내용이 상위 칸을 두 개 차지한다 -> 거의 같은 조각은 하나만 남긴다.

조각 한 줄의 모양은 rag.search.search()와 같다: (gen, doc, chunk_no, page_start, page_end, score, text)
"""
from __future__ import annotations

import re

RRF_K = 5          # 순위 합치기 상수: 1 / (RRF_K + 순위)
MISSING_RANK = 99  # 한쪽 목록에 없는 조각의 순위
TWIN_THRESHOLD = 0.6


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def merge_candidates(*row_lists: list) -> list:
    """여러 검색 결과 -> 조각 번호로 중복을 없앤 후보 목록 (조각 번호 순)."""
    seen: dict[int, tuple] = {}
    for rows in row_lists:
        for r in rows:
            seen.setdefault(r[2], r)
    return [seen[no] for no in sorted(seen)]


def rank_by_scores(rows: list, scores: list[float]) -> list:
    """점수 높은 순. 점수가 같으면 들어온 순서를 유지한다."""
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    return [rows[i] for i in order]


def fuse(ranked_lists: list[list]) -> list:
    """순위 목록 여러 개 -> RRF로 합친 순위. 한 목록에 없는 조각은 MISSING_RANK로 친다."""
    rows: dict[int, tuple] = {}
    for lst in ranked_lists:
        for r in lst:
            rows.setdefault(r[2], r)
    ranks = [{r[2]: i + 1 for i, r in enumerate(lst)} for lst in ranked_lists]

    def score(no: int) -> float:
        return sum(1 / (RRF_K + rk.get(no, MISSING_RANK)) for rk in ranks)

    return sorted(rows.values(), key=lambda r: -score(r[2]))


def seat_leaders(fused: list, ranked_lists: list[list]) -> list:
    """각 순위 목록의 1등을 합친 순위의 맨 앞으로 옮긴다. 1등끼리의 순서와 나머지 순서는 합친 순위 그대로."""
    leaders = {lst[0][2] for lst in ranked_lists if lst}
    return [r for r in fused if r[2] in leaders] + [r for r in fused if r[2] not in leaders]


def _grams(text: str, n: int = 6) -> set[str]:
    return {text[i:i + n] for i in range(0, max(1, len(text) - n + 1), 3)}


def drop_twins(rows: list, limit: int = 10, threshold: float = TWIN_THRESHOLD) -> list:
    """위에서부터 보면서, 이미 고른 조각과 글자가 거의 같은 조각은 건너뛴다. limit개가 되면 멈춘다."""
    kept: list[tuple] = []
    kept_grams: list[set[str]] = []
    for r in rows:
        g = _grams(_norm(r[6]))
        if all(len(g & k) / max(1, min(len(g), len(k))) < threshold for k in kept_grams):
            kept.append(r)
            kept_grams.append(g)
        if len(kept) >= limit:
            break
    return kept


def ranked_for_query(query: str, gen: int | None, base: str, k: int, max_length: int) -> list:
    from rag.rerank import scores
    from rag.search import search

    cands = merge_candidates(search(query, k=k, gen=gen, method=base),
                             search(query, k=k, gen=gen, method=base + "@m3"))
    return rank_by_scores(cands, scores(query, [r[6] for r in cands], max_length))


def retrieve(question: str, gen: int | None = None, top: int = 3, base: str = "baseline500", k: int = 20,
             max_length: int = 512, use_rewrite: bool = True, prompt: str = "v1", return_all: bool = False,
             llm_pick: bool = False, rewrite_save_to=None):
    """질문 -> 상위 top개 조각. return_all=True면 (상위 top개, 쌍둥이 제거 전 전체 순위)를 돌려준다.
    rewrite_save_to: 새 질문의 바꿔 쓴 검색어를 저장할 파일 (rag.rewrite.rewrite의 save_to). 평가에서는 쓰지 않는다."""
    lists = [ranked_for_query(question, gen, base, k, max_length)]
    if use_rewrite:
        from rag.rewrite import rewrite
        seen = {question}
        for version in prompt.split(","):          # "v1,v2": 프롬프트마다 바꿔 쓴 질문으로 순위를 하나씩 더 만든다
            rewritten = rewrite(question, version=version.strip(), save_to=rewrite_save_to)
            if rewritten not in seen:
                seen.add(rewritten)
                lists.append(ranked_for_query(rewritten, gen, base, k, max_length))
    fused = fuse(lists) if len(lists) > 1 else lists[0]
    best = drop_twins(seat_leaders(fused, lists), limit=max(top, 10))
    if llm_pick:                      # 상위 10개 중 답이 든 조각을 Claude가 골라 앞에 놓는다 (rag/llm_pick.py)
        from rag.llm_pick import pick
        best = pick(question, best)
    best = best[:top]
    return (best, fused) if return_all else best
