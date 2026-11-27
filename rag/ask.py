"""질문 하나에 답한다: 검색 -> 답변 생성 -> 근거 검사.

실행:  python -m rag.ask "3세대 실손 해지하면 돌려받는 돈 있어요?" --gen 3
       (기본 = 최종 검색 방식: 모델 두 개 + 질문 바꿔 쓰기 + 재정렬 + 순위 합치기. rag/hybrid.py)
       python -m rag.ask "..." --gen 3 --simple --rerank    (예전 방식: 모델 하나 + 재정렬)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag.generate import answer
from rag.search import search


LIVE_REWRITES = Path("eval/cache/rewrites_live.json")   # 평가 문항이 아닌 질문의 바꿔 쓴 검색어 (리포에 안 올라감)


def retrieve_final(question: str, gen: int, k: int = 3, method: str = "baseline500") -> list:
    """최종 검색 방식 (평가 60문항에서 Hit@3 90.0%로 잰 것과 같은 경로)."""
    from rag.hybrid import retrieve as hybrid_retrieve
    return hybrid_retrieve(question, gen, top=k, base=method, rewrite_save_to=LIVE_REWRITES)


def retrieve(question: str, gen: int, method: str, rerank: bool, k: int) -> list:
    """예전 방식: 모델 하나 (+ 재정렬). 10월 결과와 agent/ 실험이 이 경로로 측정돼서 그대로 둔다."""
    if rerank:
        from rag.rerank import rerank as rr
        return rr(question, search(question, 20, gen, method))[:k]
    return search(question, k, gen, method)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--gen", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument("--method", default="baseline500")
    ap.add_argument("--simple", action="store_true", help="예전 방식(모델 하나)으로 검색한다")
    ap.add_argument("--rerank", action="store_true", help="--simple일 때만: 1차 20개를 재정렬한다")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args(argv)

    if args.simple:
        rows = retrieve(args.question, args.gen, args.method, args.rerank, args.k)
    else:
        rows = retrieve_final(args.question, args.gen, args.k, args.method)
    a = answer(args.question, rows)
    print(f"질문: {a.question}\n답: {a.answer}")
    print(f"가드레일: {a.guard}" + (f" | 근거: 조각 {a.chunk} ({rows[a.chunk-1][3]}쪽) \"{a.quote}\"" if a.chunk and not a.refused else ""))
    print(f"토큰: {a.usage}")
    if a.guard not in ("ok", "model_refused"):
        print(f"\n[AI가 실제로 쓴 글 앞부분]\n{a.raw[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
