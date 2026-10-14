"""질문 하나에 답한다: 검색 -> (재정렬) -> 답변 생성 -> 근거 검사.

실행:  python -m rag.ask "3세대 실손 해지하면 돌려받는 돈 있어요?" --gen 3 --rerank
"""
from __future__ import annotations

import argparse

from rag.generate import answer
from rag.search import search


def retrieve(question: str, gen: int, method: str, rerank: bool, k: int) -> list:
    if rerank:
        from rag.rerank import rerank as rr
        return rr(question, search(question, 20, gen, method))[:k]
    return search(question, k, gen, method)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--gen", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument("--method", default="baseline500")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args(argv)

    rows = retrieve(args.question, args.gen, args.method, args.rerank, args.k)
    a = answer(args.question, rows)
    print(f"질문: {a.question}\n답: {a.answer}")
    print(f"가드레일: {a.guard}" + (f" | 근거: 조각 {a.chunk} ({rows[a.chunk-1][3]}쪽) \"{a.quote}\"" if a.chunk and not a.refused else ""))
    print(f"토큰: {a.usage}")
    if a.guard not in ("ok", "model_refused"):
        print(f"\n[AI가 실제로 쓴 글 앞부분]\n{a.raw[:600]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
