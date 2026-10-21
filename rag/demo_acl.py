"""실습 2: 권한 없는 사용자에게 내부 문서 조각이 넘어가는지 세 가지 방법으로 재 본다.

실행:  python -m rag.demo_acl               (진짜 임베딩 모델 사용)
       python -m rag.demo_acl --fake-embed  (모델 없이 빠르게, 검색 순위는 참고용)

문서 2개를 넣는다: 고객용 공개 약관(public)과 내부 인수심사 기준(underwriting 그룹만).
사용자 2명: 고객(public), 심사팀(public + underwriting). 질문 6개 중 3개는 내부 기준을 노린다.
  1. 필터 없음      : 권한을 안 보고 상위 3개 (프롬프트로만 "말하지 마"라고 막는 시스템이 모델에 넘기는 것)
  2. 검색 후 거르기  : 상위 3개를 가져온 뒤 권한 없는 것 제거
  3. 검색 안에서 거르기: WHERE allowed_groups && 사용자 그룹 (rag.acl.search_for_user)
마지막으로 내부 문서 권한을 회수했을 때 임베딩을 다시 안 하고 바로 적용되는지 본다.
"""
from __future__ import annotations

import argparse
from functools import partial

from rag.acl import post_filter, search_for_user, search_unfiltered
from rag.db import connect_or_exit
from rag.demo_reindex import by_clause
from rag.schema import ensure_v2_schema
from rag.versioning import replace_document

METHOD = "demo_acl"
PUBLIC_DOC = [
    (1, "제1조(목적) 이 약관은 피보험자가 질병 또는 상해로 의료기관에서 치료를 받은 경우 실제로 부담한 의료비를 보상합니다."),
    (2, "제2조(도수치료) 도수치료·체외충격파치료·증식치료는 연간 20회까지 보상하며, 1회당 자기부담금은 3만원입니다."),
    (3, "제3조(입원의료비) 입원 시 본인이 부담한 의료비의 80%를 보상하며, 연간 5천만원을 한도로 합니다."),
    (4, "제4조(해지환급금) 이 계약은 순수보장형으로, 계약을 해지해도 해지환급금이 발생하지 않습니다."),
    (5, "제5조(계약 전 알릴 의무) 청약할 때 직업과 최근 5년 이내 입원·수술 이력을 사실대로 알려야 합니다."),
    (6, "제6조(보험금 청구) 보험금은 진료비 영수증과 세부내역서를 제출하면 3영업일 안에 지급합니다."),
]
INTERNAL_DOC = [
    (1, "[내부] 인수심사 기준 1. 오토바이 배달 등 이륜차 운전 직군은 보험료 30% 할증 후 인수합니다."),
    (2, "[내부] 인수심사 기준 2. 최근 3년 내 입원 이력이 2회 이상이면 인수를 거절합니다."),
    (3, "[내부] 설계사 수수료: 첫해 월 보험료의 300%, 2차년도부터 월 보험료의 5%를 지급합니다."),
    (4, "[내부] 고액 청구(500만원 이상)는 지급 전 현장 조사 대상으로 분류합니다."),
]
USERS = {"customer": ["public"], "underwriter": ["public", "underwriting"]}
QUESTIONS = [  # (질문, 내부 기준을 노린 질문인가)
    ("오토바이 배달 일을 하는데 가입할 수 있어요?", True),
    ("입원 이력이 있으면 가입이 거절되나요?", True),
    ("설계사 수수료는 얼마나 돼요?", True),
    ("도수치료는 1년에 몇 번까지 보장돼요?", False),
    ("입원하면 의료비를 얼마나 돌려받아요?", False),
    ("보험 해지하면 돌려받는 돈 있어요?", False),
]
K = 3


def is_internal(row) -> bool:
    return "underwriting" in row[7]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-embed", action="store_true", help="no model; hash-based vectors (rankings are only indicative)")
    ap.add_argument("--keep", action="store_true", help="keep the demo_acl rows in the DB after the run")
    args = ap.parse_args(argv)
    if args.fake_embed:
        from rag.fake_embed import embed_passages, embed_query
    else:
        from rag.embed import embed_passages, embed_query
    embed_passages = partial(embed_passages, show_progress_bar=False)

    with connect_or_exit() as conn:
        ensure_v2_schema(conn)
        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE method = %s", (METHOD,))
        common = dict(method=METHOD, gen=4, chunk_fn=by_clause, embed=embed_passages)
        replace_document(conn, doc="공개약관", pages=PUBLIC_DOC, groups=["public"], **common)
        replace_document(conn, doc="내부_인수심사기준", pages=INTERNAL_DOC, groups=["underwriting"], **common)

        customer = USERS["customer"]
        modes = ("1. No filter", "2. Filter after search", "3. Filter inside search")
        rows_by_mode = {m: [] for m in modes}
        for q, _ in QUESTIONS:
            raw = search_unfiltered(conn, q, K, method=METHOD, embed_query=embed_query)
            rows_by_mode[modes[0]].append(raw)
            rows_by_mode[modes[1]].append(post_filter(raw, customer))
            rows_by_mode[modes[2]].append(
                search_for_user(conn, q, customer, K, method=METHOD, embed_query=embed_query))

        staff = [search_for_user(conn, q, USERS["underwriter"], K, method=METHOD, embed_query=embed_query)
                 for q, aimed in QUESTIONS if aimed]
        staff_hits = sum(any(is_internal(r) for r in rows) for rows in staff)

        # 권한 회수: 내용은 그대로, 그룹만 underwriting -> underwriting_lead
        rv = replace_document(conn, doc="내부_인수심사기준", pages=INTERNAL_DOC, groups=["underwriting_lead"], **common)
        after = [search_for_user(conn, q, USERS["underwriter"], K, method=METHOD, embed_query=embed_query)
                 for q, aimed in QUESTIONS if aimed]
        after_hits = sum(any(is_internal(r) for r in rows) for rows in after)

        if not args.keep:
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE method = %s", (METHOD,))

    print(f"Public policy: {len(PUBLIC_DOC)} clauses + internal underwriting rules: {len(INTERNAL_DOC)} clauses."
          f" {len(QUESTIONS)} questions (3 target the internal rules), top {K}."
          f" Embedding: {'fake (hash)' if args.fake_embed else 'multilingual-e5-small'}\n")
    print("[Customer: groups = public]")
    print("Method | Questions with internal chunks | Internal chunks leaked | Questions with < 3 results")
    print("(expected for 3) | 0/6 | 0 | 0/6")
    leak_total = {}
    for mode, per_q in rows_by_mode.items():
        leaked_q = sum(any(is_internal(r) for r in rows) for rows in per_q)
        leaked_n = sum(sum(is_internal(r) for r in rows) for rows in per_q)
        short_q = sum(len(rows) < K for rows in per_q)
        leak_total[mode] = (leaked_n, short_q)
        print(f"{mode} | {leaked_q}/6 | {leaked_n} | {short_q}/6")

    print("\n[Underwriter: groups = public + underwriting, filter inside search]")
    print(f"Internal-rule questions that got internal chunks: {staff_hits}/3  (authorized users are not blocked)")
    print(f"\n[Revoke] internal rules: underwriting -> underwriting_lead: {rv.action}"
          f" (no re-embedding, {rv.regrouped} rows changed)")
    print(f"After revoke, internal-rule questions where the underwriter got internal chunks: {after_hits}/3"
          "  (expected 0)")

    ok = leak_total[modes[2]] == (0, 0) and after_hits == 0 and rv.action == "permissions_updated"
    print("\nResult:", "filter inside search: 0 leaked, 0 short, revoke applied immediately." if ok
          else "unexpected result -> check.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
