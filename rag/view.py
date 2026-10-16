"""화면(app.py)에서 쓰는 작은 도우미. 화면 코드와 분리해서 테스트할 수 있게 둔다."""
from __future__ import annotations

import html
import re

GUARD_TEXT = {
    "ok": ("통과", "AI가 인용한 문장이 검색된 약관 조각 안에 글자 그대로 있습니다."),
    "model_refused": ("거절", "AI가 조각 안에서 답을 찾지 못했다고 제출했습니다."),
    "quote_not_found": ("차단", "AI가 인용한 문장이 조각에 없어서 답을 버렸습니다."),
    "bad_chunk": ("차단", "AI가 없는 조각 번호를 근거로 댔습니다."),
    "bad_json": ("차단", "AI 답의 형식이 깨져서 버렸습니다."),
    "truncated": ("차단", "AI 답이 길이 상한에서 잘려서 버렸습니다."),
    "no_chunks": ("거절", "검색된 조각이 없습니다."),
}


def guard_label(guard: str) -> tuple[str, str]:
    return GUARD_TEXT.get(guard, (guard, ""))


def highlight(text: str, quote: str | None) -> str:
    """조각 원문에서 인용문 부분만 <mark>로 감싼 HTML. 띄어쓰기가 달라도 찾는다."""
    safe = html.escape(text or "")
    if not quote:
        return safe
    chars = [c for c in quote if not c.isspace()]
    if len(chars) < 5:
        return safe
    pattern = r"\s*".join(re.escape(html.escape(c)) for c in chars)
    m = re.search(pattern, safe)
    if not m:
        return safe
    return safe[:m.start()] + "<mark>" + safe[m.start():m.end()] + "</mark>" + safe[m.end():]


_LEAD_FRAGMENT = re.compile(r"^\S{1,2}[,.)]\s+")


def display_quote(quote: str | None, chunk_text: str | None = None) -> str:
    """화면에 보여줄 인용문. 조각 경계에서 잘린 앞 글자(예: '써, 중도해지하는…')를 떼고 '…'를 붙인다.
    근거 검사는 원래 인용문으로 이미 끝났고, 이건 보여주기만 다듬는다."""
    q = (quote or "").strip()
    if not q:
        return q
    cut = False
    if chunk_text:
        i = chunk_text.find(q)
        if i > 0 and not chunk_text[i - 1].isspace() and " " in q:
            q, cut = q.split(" ", 1)[1].lstrip(), True   # 단어 중간에서 시작
    m = _LEAD_FRAGMENT.match(q)
    if m and len(q) > m.end():
        q, cut = q[m.end():], True                       # '써, ' 같은 1~2글자 조각
    return ("…" + q) if cut else q


_SENT_END = re.compile(r"(?:다|요)\.\s")


def lead_in(prev: str, limit: int = 300) -> str:
    """앞 조각에서 마지막 문장의 시작부터 끝까지 (이 조각 첫 문장이 잘렸을 때 앞부분을 채우는 용도)."""
    tail = (prev or "")[-limit:]
    ends = list(_SENT_END.finditer(tail))
    if ends:
        return tail[ends[-1].end():]
    return ("…" + tail) if len(prev or "") > limit else tail


def lead_out(nxt: str, limit: int = 300) -> str:
    """뒤 조각에서 첫 문장이 끝나는 곳까지 (이 조각 마지막 문장이 잘렸을 때 뒷부분을 채우는 용도)."""
    head = (nxt or "")[:limit]
    m = _SENT_END.search(head)
    if m:
        return head[:m.end()].rstrip()
    return (head + "…") if len(nxt or "") > limit else head


def with_context(text: str, quote: str | None, prev: str = "", nxt: str = "") -> str:
    """조각 본문(인용 강조 포함) 앞뒤에 이웃 조각의 잘린 문장을 흐린 글씨로 붙인 HTML."""
    grey = '<span style="opacity:0.45">{}</span>'
    before, after = lead_in(prev), lead_out(nxt)
    return ((grey.format(html.escape(before)) if before else "")
            + highlight(text, quote)
            + (grey.format(html.escape(after)) if after else ""))
