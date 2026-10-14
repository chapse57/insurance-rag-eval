"""평가 문제(eval/questions.csv)로 검색 점수를 잰다.

정답 판정: 같은 세대의 조각 안에 '근거 문구(evidence)'가 들어 있으면 맞춘 것.
  - 띄어쓰기는 지우고 비교한다 (2016·2018 PDF는 띄어쓰기가 사라져서 뽑히기 때문).
  - 근거 문구가 조각 경계에서 반으로 잘리면 못 맞춘 것으로 친다. 이건 '자르기' 단계의 실패라서
    일부러 그대로 둔다.
  - type이 no_answer(약관에 답이 없는 질문)인 문제는 검색 점수에서 빼고, 나중에 가드레일 시험에 쓴다.

지표:
  Hit Rate@3 = 정답 조각이 상위 3개 안에 들어온 문제 비율
  MRR@10    = 정답 조각 순위의 역수 평균 (1등 1.0, 2등 0.5, 10등 밖이면 0)
  Hit@글자예산 = 위에서부터 조각을 이어 붙여 N자(기본 3000자)가 될 때까지 안에 정답이 있는 비율.
              조각 크기가 다른 방식끼리 비교할 때 쓴다. 1000자 조각 3개(3000자)와 500자 조각 3개(1500자)를
              Hit@3으로 비교하면 큰 조각이 글자를 더 많이 보여주는 만큼 유리해서 공정하지 않다.

실행:  python -m rag.evaluate --method baseline500
결과:  화면 요약 + eval/results/{method}_{시각}.csv (문제별 순위, 1등 조각 위치)
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

QUESTIONS = Path("eval/questions.csv")
RESULTS = Path("eval/results")
K_MAX = 10          # MRR 계산 범위
K_FETCH = 20        # 글자 예산 비교를 위해 넉넉히 가져옴 (200자 조각 x 15개 = 3000자)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def first_hit_rank(rows, gen: int, evidence: str) -> int | None:
    """rows: search() 결과 [(gen, doc, chunk_no, p1, p2, score, text), ...]. 맞춘 첫 순위(1부터) 또는 None."""
    ev = norm(evidence)
    for rank, r in enumerate(rows, start=1):
        if r[0] == gen and ev in norm(r[6]):
            return rank
    return None


def budget_hit(rows, gen: int, evidence: str, budget: int) -> bool:
    """위에서부터 조각 글자 수를 누적해 budget을 넘기 전까지(넘기는 조각 포함) 정답이 있으면 True."""
    ev, used = norm(evidence), 0
    for r in rows:
        if used >= budget:
            break
        if r[0] == gen and ev in norm(r[6]):
            return True
        used += len(r[6])
    return False


def metrics(ranks: list[int | None]) -> dict:
    n = len(ranks)
    if n == 0:
        return {"n": 0, "hit@3": 0.0, "mrr@10": 0.0}
    hit3 = sum(1 for r in ranks if r is not None and r <= 3) / n
    mrr = sum(1 / r for r in ranks if r is not None and r <= K_MAX) / n
    return {"n": n, "hit@3": round(hit3, 3), "mrr@10": round(mrr, 3)}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("question") or "").strip()]


def main(argv: list[str] | None = None) -> int:
    from rag.search import search  # DB·모델이 필요한 import는 실행할 때만

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline500")
    ap.add_argument("--budget", type=int, default=3000, help="Hit@글자예산의 글자 수")
    ap.add_argument("--rerank", action="store_true", help="1차 검색 20개를 재정렬 모델로 다시 줄 세운 뒤 채점")
    args = ap.parse_args(argv)

    qs = load_questions()
    answerable = [q for q in qs if q["type"] in ("answer", "hard")]
    bad = [q["id"] for q in answerable if not norm(q["evidence"])]
    if bad:
        print(f"근거 문구가 비어 있는 문제: {bad}")
        return 1

    RESULTS.mkdir(parents=True, exist_ok=True)
    label = args.method + ("+rerank" if args.rerank else "")
    if args.rerank:
        from rag.rerank import rerank
    out = RESULTS / f"{label}_{datetime.now():%m%d_%H%M}.csv"
    ranks_by_type: dict[str, list] = {"answer": [], "hard": []}
    budget_hits = 0
    in_fetch = 0   # 1차 검색 20개 안에 정답이 있었던 문제 수 = 재정렬로 올릴 수 있는 최대치
    t0 = time.time()
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "gen", "type", "question", "rank", f"hit@{args.budget}자", "top1_page", "top1_text"])
        for q in answerable:
            gen = int(q["gen"])
            rows = search(q["question"], k=K_FETCH, gen=gen, method=args.method)
            in_fetch += first_hit_rank(rows, gen, q["evidence"]) is not None
            if args.rerank:
                rows = rerank(q["question"], rows)
            rank = first_hit_rank(rows[:K_MAX], gen, q["evidence"])
            bh = budget_hit(rows, gen, q["evidence"], args.budget)
            budget_hits += bh
            ranks_by_type[q["type"]].append(rank)
            top = rows[0] if rows else None
            top_page = f"{top[3]}-{top[4]}" if top else ""
            w.writerow([q["id"], gen, q["type"], q["question"], rank or "", int(bh), top_page,
                        (top[6][:120] if top else "")])
            mark = f"{rank}등" if rank else "놓침"
            print(f"{q['id']} [{q['type']}] {mark:>4} | {q['question']}")

    all_ranks = ranks_by_type["answer"] + ranks_by_type["hard"]
    m = metrics(all_ranks)
    print(f"\n== {label}: 문제 {m['n']}개 | Hit Rate@3 {m['hit@3']:.1%} | MRR@10 {m['mrr@10']:.3f}"
          f" | Hit@{args.budget}자 {budget_hits / max(m['n'], 1):.1%} | {time.time() - t0:.1f}초")
    for t, rs in ranks_by_type.items():
        if rs:
            mt = metrics(rs)
            print(f"   {t}: {mt['n']}개 | Hit Rate@3 {mt['hit@3']:.1%} | MRR@10 {mt['mrr@10']:.3f}")
    print(f"   1차 검색 {K_FETCH}개 안에 정답이 있던 문제: {in_fetch}/{m['n']}"
          f"{' (재정렬로 올릴 수 있는 최대치)' if args.rerank else ''}")
    skipped = len(qs) - len(answerable)
    if skipped:
        print(f"   (no_answer {skipped}개는 검색 점수에서 제외 — 가드레일 시험용)")
    print(f"문제별 결과: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
