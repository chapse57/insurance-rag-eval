"""실손보험 약관 RAG 화면.

실행 (Upwork_2 폴더에서, .venv 켠 상태):
    streamlit run app.py
브라우저가 자동으로 열린다 (http://localhost:8501).
Docker의 PostgreSQL(5433)이 켜져 있어야 한다:  docker compose up -d
"""
from __future__ import annotations

import time

import psycopg
import streamlit as st

from rag.ask import retrieve_final
from rag.generate import MODEL, answer
from rag.search import neighbors, search
from rag.view import display_quote, guard_label, with_context

RERANK_MAX_LEN = 256   # 예전 방식에서만 쓴다 (10/16 측정: 256토큰 70.8%·질문당 8.6초, CPU)
FINAL_MAX_LEN = 512    # 최종 검색 방식이 쓰는 재정렬 길이 (rag/hybrid.py 기본값)

EXAMPLES = [
    (3, "3세대 실손 해지하면 돌려받는 돈 있어요?"),
    (4, "4세대 실손에서 도수치료는 1년에 몇 번까지 보장되나요?"),
    (3, "3세대 도수치료는 1년에 얼마까지 받을 수 있어요?"),
    (2, "2세대 실손 보험금 받으면 세금 내야 해요?"),   # 약관에 답이 없는 질문 -> 거절해야 정상
]

st.set_page_config(page_title="실손보험 약관 RAG", layout="wide")
st.title("실손보험 약관 RAG")
st.caption("약관 583쪽에서 근거를 찾아 답하고, AI가 인용한 문장이 약관에 그대로 있는지 코드로 검사합니다. "
           "근거가 없으면 답하지 않습니다.")


@st.cache_resource(show_spinner="검색 모델 불러오는 중 (처음 한 번만)...")
def warm_up(use_final: bool) -> bool:
    from rag.rerank import _model
    search("준비", 1, 3)
    if use_final:
        search("준비", 1, 3, "baseline500@m3")
        _model(FINAL_MAX_LEN)
    else:
        _model(RERANK_MAX_LEN)
    return True


st.session_state.setdefault("gen", 3)
st.session_state.setdefault("question", "")

with st.sidebar:
    st.subheader("설정")
    use_final = st.toggle("최종 검색 방식 (끄면 예전 방식: 모델 1개 + 재정렬)", value=True)
    if use_final:
        st.markdown(f"- 답변 모델: `{MODEL}`\n- 검색: 임베딩 모델 2개의 후보를 합침 → 원래 질문과 약관 용어로 바꿔 쓴 질문으로 "
                    "각각 재정렬 → 순위 합치기 → 상위 3개\n"
                    "- 평가 60문항: 상위 3개 적중률 90.0%, 답변 정확도 86.7% (답을 하나씩 읽어 판정)")
    else:
        st.markdown(f"- 답변 모델: `{MODEL}`\n- 검색: 가까운 조각 20개 → 재정렬({RERANK_MAX_LEN}토큰) → 상위 3개\n"
                    "- 10월 측정(24문항): 상위 3개 적중률 70.8%")
    st.subheader("예시 질문")
    for gen, q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state["gen"] = gen
            st.session_state["question"] = q

col1, col2 = st.columns([1, 5])
gen = col1.selectbox("세대", [2, 3, 4], key="gen")
question = col2.text_input("질문", key="question", placeholder="예: 3세대 실손 해지하면 돌려받는 돈 있어요?")
go = st.button("물어보기", type="primary")

DB_OFF = ("DB에 연결할 수 없습니다. Docker Desktop을 켠 뒤, Upwork_2 폴더에서 "
          "`docker compose up -d` 를 실행하고 다시 눌러 주세요.")

if go and question.strip():
    try:
        warm_up(use_final)
        t0 = time.perf_counter()
        if use_final:
            rows = retrieve_final(question, gen, 3)
        else:
            from rag.rerank import rerank
            rows = rerank(question, search(question, 20, gen), max_length=RERANK_MAX_LEN)[:3]
    except psycopg.OperationalError:
        st.error(DB_OFF)
        st.stop()
    t1 = time.perf_counter()
    with st.spinner("답 만드는 중..."):
        a = answer(question, rows)
    t2 = time.perf_counter()

    label, why = guard_label(a.guard)
    st.subheader("답")
    if a.refused:
        st.warning(a.answer)
    else:
        st.success(a.answer)
    page = rows[a.chunk - 1][3] if a.chunk and rows and not a.refused else None
    c1, c2, c3 = st.columns(3)
    c1.metric("근거 검사", label)
    c2.metric("근거 쪽수", f"{page}쪽" if page else "-")
    c3.metric("걸린 시간", f"검색 {t1 - t0:.1f}초 · 답 {t2 - t1:.1f}초")
    st.caption(why + (f"  |  토큰 입력 {a.usage.get('input')} · 출력 {a.usage.get('output')}" if a.usage else ""))
    if a.quote and not a.refused:
        src = rows[a.chunk - 1][6] if a.chunk and a.chunk <= len(rows) else None
        st.markdown(f"**약관 원문 인용:** “{display_quote(a.quote, src)}”")

    st.subheader("검색된 약관 조각 (AI에게 준 것 전부)")
    st.caption("흐린 글씨는 조각 경계에서 잘린 문장을 앞뒤 조각에서 이어 붙인 것입니다. AI에게는 진한 글씨 부분만 줬습니다.")
    for i, r in enumerate(rows, start=1):
        used = (a.chunk == i and not a.refused)
        title = f"조각 {i} · {r[0]}세대 {r[3]}쪽" + ("  ← 근거" if used else "")
        try:
            prev, nxt = neighbors(r)
        except psycopg.OperationalError:
            prev, nxt = "", ""
        with st.expander(title, expanded=used):
            st.markdown(with_context(r[6], a.quote if used else None, prev, nxt), unsafe_allow_html=True)
elif go:
    st.info("질문을 입력해 주세요.")
