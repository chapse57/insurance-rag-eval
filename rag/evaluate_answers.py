"""답변 단계 채점: 문제 전부 (답 없는 문제 포함).

답이 있는 문제(answer/hard):
  근거 맞춤   = 답했고, 인용한 조각 안에 정답 근거 문구가 들어 있음
  검색 탓 거절 = 거절했는데 찾아온 조각 안에 정답이 아예 없었음 (검색 단계 실패)
  생성 탓 거절 = 정답 조각을 찾아왔는데도 거절함 (AI 또는 가드레일이 놓침)
  엉뚱한 근거 = 답은 했지만 인용한 조각에 정답 근거가 없음 (사람 확인 필요)
답이 없는 문제(no_answer):
  올바른 거절 = 모름이라고 답함 / 답해버림 = 지어냈을 가능성 (사람 확인 필요)
가드레일 차단 = AI는 답했는데 인용문이 조각에 없어서 코드가 막은 수

주의: 답 문장의 내용이 맞는지는 자동으로 판정하지 않는다. 결과 CSV에서 사람이 읽어 확인한다.

실행:  python -m rag.evaluate_answers --rerank
       python -m rag.evaluate_answers --hybrid --k 10 --questions eval/questions.csv eval/questions_v2_draft.csv
       (--hybrid: 검색 채점(rag.evaluate --hybrid)과 같은 최종 검색 방식으로 조각을 가져온다. --k: AI에게 주는 조각 수)
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from rag.ask import retrieve
from rag.evaluate import QUESTIONS, RESULTS, load_questions, norm
from rag.generate import MODEL, answer

PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}  # $/100만 토큰 (입력, 출력)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="baseline500")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--k", type=int, default=3, help="AI에게 주는 조각 수")
    ap.add_argument("--hybrid", action="store_true", help="최종 검색 방식(rag.hybrid.retrieve)으로 조각을 가져온다")
    ap.add_argument("--questions", nargs="+", default=[str(QUESTIONS)], help="평가 문항 파일 (여러 개 가능)")
    args = ap.parse_args(argv)

    qs = [q for path in args.questions for q in load_questions(Path(path))]
    label = f"answers_{args.method}{'+rerank' if args.rerank else ''}_{MODEL}"
    if args.hybrid:
        from rag.hybrid import retrieve as hybrid_retrieve
        label = f"answers_hybrid_{args.method}_k{args.k}_{MODEL}"
    out = RESULTS / f"{label}_{datetime.now():%m%d_%H%M}.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    verdicts, guards, tok_in, tok_out = Counter(), Counter(), 0, 0
    ev_in_rows = 0   # 답 있는 문제 중, AI에게 준 조각 안에 정답 근거가 들어 있던 수 (= 답변 단계가 맞출 수 있는 최대치)
    t0 = time.time()
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "verdict", "guard", "answer", "quote", "cited_page", "evidence", "raw",
                    "evidence_in_chunks"])
        for q in qs:
            gen = int(q["gen"])
            if args.hybrid:
                rows = hybrid_retrieve(q["question"], gen, top=args.k, base=args.method)
            else:
                rows = retrieve(q["question"], gen, args.method, args.rerank, args.k)
            a = answer(q["question"], rows)
            guards[a.guard] += 1
            tok_in += a.usage.get("input", 0)
            tok_out += a.usage.get("output", 0)
            ev = norm(q.get("evidence", ""))
            found_ev = bool(ev) and any(ev in norm(r[6]) for r in rows)
            ev_in_rows += found_ev and q["type"] != "no_answer"
            if q["type"] == "no_answer":
                v = "올바른 거절" if a.refused else "답해버림(확인 필요)"
            elif not a.refused:
                v = "근거 맞춤" if ev in norm(rows[a.chunk - 1][6]) else "엉뚱한 근거(확인 필요)"
            else:
                v = "생성 탓 거절" if found_ev else "검색 탓 거절"
            verdicts[(q["type"] == "no_answer", v)] += 1
            page = rows[a.chunk - 1][3] if a.chunk and not a.refused else ""
            w.writerow([q["id"], q["type"], q["question"], v, a.guard, a.answer, a.quote or "", page, q.get("evidence", ""), a.raw[:400],
                        "" if q["type"] == "no_answer" else int(found_ev)])
            print(f"{q['id']} [{q['type']}] {v} | {a.answer[:60]}")

    n_ans = sum(c for (na, _), c in verdicts.items() if not na)
    n_no = sum(c for (na, _), c in verdicts.items() if na)
    print(f"\n== {label}")
    print(f"   답 있는 문제 {n_ans}개: " + ", ".join(f"{v} {c}" for (na, v), c in verdicts.items() if not na))
    print(f"   AI에게 준 조각 {args.k}개 안에 정답 근거가 있던 문제: {ev_in_rows}/{n_ans}")
    print(f"   답 없는 문제 {n_no}개: " + ", ".join(f"{v} {c}" for (na, v), c in verdicts.items() if na))
    print(f"   가드레일 판정: {dict(guards)}  (quote_not_found = AI가 답했지만 인용문이 조각에 없어 코드가 막음)")
    pin, pout = PRICE.get(MODEL, (0, 0))
    print(f"   토큰 입력 {tok_in:,} / 출력 {tok_out:,} -> 약 ${tok_in / 1e6 * pin + tok_out / 1e6 * pout:.3f} | {time.time() - t0:.0f}초")
    print(f"문제별 결과(답 문장 포함): {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
