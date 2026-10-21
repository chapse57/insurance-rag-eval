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
USERS = {"고객": ["public"], "심사팀": ["public", "underwriting"]}
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
    ap.add_argument("--fake-embed", action="store_true", help="모델 없이 가짜 임베딩 (검색 순위는 참고용)")
    ap.add_argument("--keep", action="store_true", help="끝나고 demo_acl 행을 지우지 않음")
    args = ap.parse_args(argv)
    if args.fake_embed:
        from rag.fake_embed import embed_passages, embed_query
    else:
        from rag.embed import embed_passages, embed_query

    with connect_or_exit() as conn:
        ensure_v2_schema(conn)
        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE method = %s", (METHOD,))
        common = dict(method=METHOD, gen=4, chunk_fn=by_clause, embed=embed_passages)
        replace_document(conn, doc="공개약관", pages=PUBLIC_DOC, groups=["public"], **common)
        replace_document(conn, doc="내부_인수심사기준", pages=INTERNAL_DOC, groups=["underwriting"], **common)

        guest = USERS["고객"]
        rows_by_mode = {"1. 필터 없음": [], "2. 검색 후 거르기": [], "3. 검색 안에서 거르기": []}
        for q, _ in QUESTIONS:
            raw = search_unfiltered(conn, q, K, method=METHOD, embed_query=embed_query)
            rows_by_mode["1. 필터 없음"].append(raw)
            rows_by_mode["2. 검색 후 거르기"].append(post_filter(raw, guest))
            rows_by_mode["3. 검색 안에서 거르기"].append(
                search_for_user(conn, q, guest, K, method=METHOD, embed_query=embed_query))

        staff = [search_for_user(conn, q, USERS["심사팀"], K, method=METHOD, embed_query=embed_query)
                 for q, aimed in QUESTIONS if aimed]
        staff_hits = sum(any(is_internal(r) for r in rows) for rows in staff)

        # 권한 회수: 내용은 그대로, 그룹만 underwriting -> underwriting_lead
        rv = replace_document(conn, doc="내부_인수심사기준", pages=INTERNAL_DOC, groups=["underwriting_lead"], **common)
        after = [search_for_user(conn, q, USERS["심사팀"], K, method=METHOD, embed_query=embed_query)
                 for q, aimed in QUESTIONS if aimed]
        after_hits = sum(any(is_internal(r) for r in rows) for rows in after)

        if not args.keep:
            with conn.transaction():
                conn.execute("DELETE FROM chunks WHERE method = %s", (METHOD,))

    print(f"공개 약관 조항 {len(PUBLIC_DOC)}개 + 내부 기준 {len(INTERNAL_DOC)}개, 질문 {len(QUESTIONS)}개 (내부 노림 3개), 상위 {K}개."
          f" 임베딩: {'가짜(fake)' if args.fake_embed else 'multilingual-e5-small'}\n")
    print("[고객(public) 입장]")
    print("방법 | 내부 조각이 섞인 질문 | 넘어간 내부 조각 수 | 결과가 3개 미만인 질문")
    print("(기대값: 3번) | 0/6 | 0 | 0/6")
    leak_total = {}
    for mode, per_q in rows_by_mode.items():
        leaked_q = sum(any(is_internal(r) for r in rows) for rows in per_q)
        leaked_n = sum(sum(is_internal(r) for r in rows) for rows in per_q)
        short_q = sum(len(rows) < K for rows in per_q)
        leak_total[mode] = (leaked_n, short_q)
        print(f"{mode} | {leaked_q}/6 | {leaked_n} | {short_q}/6")

    print("\n[심사팀(public+underwriting) 입장, 검색 안에서 거르기]")
    print(f"내부 기준을 노린 질문 3개 중 내부 조각을 받은 질문: {staff_hits}/3  (막지 말아야 할 사람은 안 막는지)")
    print(f"\n[권한 회수] 내부 기준 그룹 underwriting -> underwriting_lead: {rv.action}"
          f" (임베딩 다시 안 함, {rv.regrouped}행 변경)")
    print(f"회수 후 심사팀이 내부 조각을 받은 질문: {after_hits}/3  (기대값 0)")

    ok = leak_total["3. 검색 안에서 거르기"] == (0, 0) and after_hits == 0 and rv.action == "permissions_updated"
    print("\n결과:", "검색 안에서 거르기: 유출 0, 결과 부족 0, 회수 즉시 적용" if ok else "기대와 다름 -> 확인 필요")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
