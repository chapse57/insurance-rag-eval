"""약관 글자를 조각으로 자른다.

baseline500: 일부러 허술한 첫 버전. 페이지를 이어 붙이고 500자마다 기계적으로 자른다.
조항 중간이든 표 중간이든 상관없이 자르기 때문에, 나중에 '어디서 틀렸나'를 볼 때
'자르기' 단계의 실패가 드러나도록 남겨 둔다. (고친 버전은 별도 method로 추가)
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_no: int
    page_start: int
    page_end: int
    text: str


def chunk_baseline(pages: list[tuple[int, str]], size: int = 500, overlap: int = 0) -> list[Chunk]:
    """pages: [(페이지 번호, 글자)] -> 공백을 한 칸으로 줄여 이어 붙인 뒤 size 글자마다 자른 조각들.

    overlap: 앞 조각의 끝 overlap 글자를 다음 조각 앞에 다시 넣는다 (경계에서 제목과 내용이 갈라지는 것 완화).
    기본 0 = 예전과 똑같이 자른다 (baseline500·fixed200·fixed1000 결과가 바뀌지 않도록).
    """
    if size <= 0:
        raise ValueError("size는 1 이상이어야 합니다")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap은 0 이상, size 미만이어야 합니다")
    full_parts: list[str] = []
    starts: list[int] = []      # 각 페이지가 full 문자열 어디서 시작하는지
    page_nos: list[int] = []
    pos = 0
    for page_no, text in pages:
        t = " ".join(text.split())
        if not t:
            continue
        if full_parts:
            t = " " + t
        starts.append(pos)
        page_nos.append(page_no)
        full_parts.append(t)
        pos += len(t)
    full = "".join(full_parts)

    def page_at(offset: int) -> int:
        return page_nos[bisect.bisect_right(starts, offset) - 1]

    chunks = []
    for n, i in enumerate(range(0, len(full), size - overlap)):
        piece = full[i:i + size]
        chunks.append(Chunk(n, page_at(i), page_at(i + len(piece) - 1), piece.strip()))
    return [c for c in chunks if c.text]


# ---------------------------------------------------------------------------
# 청소: 자르기 전에 페이지 글자를 고친다 (method 이름이 clean…인 설정에서만 쓴다)
#
# 1) 굵은 제목 복원. 2016·2018 PDF는 굵은 글씨를 '같은 글자 두 번 겹쳐 그리기'로 표현해서
#    "제9조(보험금을…)"이 "제 / 제99조 / 조((보 / 보험 / 험금 …"처럼 글자마다 두 번씩,
#    줄까지 쪼개져서 뽑힌다. 줄을 이어 붙였을 때 모든 글자가 정확히 두 번씩이면 한 번으로 줄인다.
# 2) 목차 줄 제거. "제1조 보장종목 ........ 10" 같은 점선 줄은 내용 없이 제목만 겹쳐서
#    진짜 조문과 검색 순위를 다툰다.
# ---------------------------------------------------------------------------
_LEADER = re.compile(r"[.·…]{6,}")
_HANGUL = re.compile(r"[가-힣]")
_RUN_LINE_MAX = 24      # 겹친 제목 줄은 짧다. 이보다 긴 줄은 본문으로 보고 건드리지 않는다


def _doubled_run(lines: list[str], i: int) -> tuple[int, str]:
    """lines[i]부터 '글자가 전부 두 번씩'인 줄 묶음을 찾는다 -> (묶음이 끝난 다음 줄 번호, 복원한 글).
    못 찾으면 (i, "")."""
    buf, best_j, best = "", i, ""
    j = i
    while j < len(lines):
        s = "".join(lines[j].split())
        if not s or len(s) > _RUN_LINE_MAX:
            break
        buf += s
        even = len(buf) // 2 * 2
        if any(buf[k] != buf[k + 1] for k in range(0, even, 2)):
            break
        j += 1
        if len(buf) % 2 == 0:
            best_j, best = j, buf[0::2]
    n_lines = best_j - i
    long_enough = (n_lines >= 2 and len(best) >= 3) or (n_lines == 1 and len(best) >= 4)
    if long_enough and _HANGUL.search(best):
        return best_j, best
    return i, ""


def clean_text(text: str) -> str:
    """한 페이지 글자 -> 굵은 제목을 복원하고 목차 점선 줄을 뺀 글자."""
    lines = [ln for ln in text.split("\n") if not _LEADER.search(ln)]
    out: list[str] = []
    i = 0
    while i < len(lines):
        j, restored = _doubled_run(lines, i)
        if j > i:
            out.append(restored)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def clean_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    return [(page_no, clean_text(text)) for page_no, text in pages]


# ---------------------------------------------------------------------------
# struct500: 약관의 구조(약관 이름 > 장·관 > 조 > 소제목)를 따라 자르고, 조각마다 그 위치를 이름표로 붙인다.
#
# 왜 (2026-11-27, 60문항 답변 채점에서 확인):
# - 500자 기계적 자르기는 목록 중간에서 잘라서, "5. 비만(E66)" 같은 한 줄이 "다음의 비급여의료비는 보상하지
#   않습니다"라는 머리말과 다른 조각에 들어간다. 검색은 그 조각을 질문과 엮지 못하고(n10), AI는 목록의 뜻을
#   뒤집어 "보상된다"고 답했다(n06).
# - 급여·비급여 조항이 거의 같은 문장으로 반복되는데, 조각에 어느 쪽인지 표시가 없어 AI가 섞었다(q02, q03).
# 그래서:
# 1) 줄 단위로 읽으면서 제목 줄을 찾는다. 조각은 줄 경계에서만 자르고, 가능하면 ①·1.·가. 같은 항목 시작에서 자른다.
# 2) 조가 바뀌면 새 조각을 시작한다. 단, 지금 조각이 너무 짧으면(min_size 미만) 다음 조와 이어 붙인다.
# 3) 조각 맨 앞에 "[보통약관 > 제3관 회사가 보상하지 않는 사항 > 제4조(보상하지 않는 사항) > (2) 질병비급여]"를 붙인다.
#    조각이 항(①) 중간에서 시작하면 그 항의 첫머리 60자도 붙인다 ("② 회사는 다음의 비급여의료비에 대해서는…").
# 4) 장·관이 처음 나오기 전(앞쪽 상품 요약부)은 제목 규칙이 세대마다 달라서 이름표 없이 줄 단위로만 자른다.
# 주의: 제목 찾기는 규칙이라 틀릴 수 있다. 본문 속 인용("제3조(보상내용)에 따라")은 줄이 제목으로 끝나고
#       그 앞이 비었거나 장·약관 이름일 때만 제목으로 쳐서 거른다.
# ---------------------------------------------------------------------------
_ART_END = re.compile(r"(제\s?\d+\s?조(?:의\s?\d+)?\s?\((?:[^()]|\([^()]*\))+\))\s*$")
_PART_IN = re.compile(r"제\s?\d+\s?[관장]")
_PART_LINE = re.compile(r"^제\s?\d+\s?[관장]\s?[^()]{2,40}$")
_SUB_LINE = re.compile(r"^\(\s?\d+\s?\)\s?[가-힣0-9][가-힣0-9\s]{1,18}$")
_BOX_LINE = re.compile(r"^□\s?[^□]{2,40}$")
_APPX_LINE = re.compile(r"^[\[【<]\s?(별표|붙임|법규)[^\]】>]{0,30}[\]】>]")
_PARA_START = re.compile(r"^[①-⑳]")
_ITEM_START = re.compile(r"^(\d{1,2}\.|[가-하]\.)")
_PARTICLE = re.compile(r"(이|에|은|는|을|를|의|로|하는|따른|따라)\s")
_LEVELS = ("yak", "box", "part", "art", "sub")


def _is_yak_name(s: str) -> bool:
    s = s.strip()
    return (0 < len(s) <= 80 and s.endswith(("보통약관", "특별약관")) and not _PARTICLE.search(s)
            and s.count("(") == s.count(")") and not s.startswith("이"))


def _short_yak(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"^무배당\s?삼성화재\s?실손의료비보험\s?\(\d{4}\.\d\)\s?", "", s)
    s = re.split(r"(?<=특별약관)\s?(?=\d+\.)", s)[-1]          # "특별약관1.실손의료비(…)특별약관" -> 뒤쪽 이름
    return re.sub(r"^\d+\.\s?", "", s).strip()


def classify_line(s: str) -> list[tuple[str, str]]:
    """한 줄 -> [(단계, 제목)]. 제목 줄이 아니면 []. 2·3세대는 '약관 이름 + 제1조(…)'가 한 줄로 붙어 있어 여러 개가 나온다."""
    s = s.strip()
    if not s or len(s) > 140:
        return []
    m = _APPX_LINE.match(s)
    if m and len(s) <= 60:
        return [("yak" if m.group(1) in ("별표", "법규") else "part", s)]
    m = _ART_END.search(s)
    if m:
        pre = s[:m.start()].strip()
        heads: list[tuple[str, str]] = []
        if pre:
            p = _PART_IN.search(pre)
            yak, part = (pre[:p.start()].strip(), pre[p.start():].strip()) if p else (pre, "")
            if yak and not _is_yak_name(yak):
                return []
            if part and not re.fullmatch(r"제\s?\d+\s?[관장].{2,40}", part):
                return []
            if yak:
                heads.append(("yak", _short_yak(yak)))
            if part:
                heads.append(("part", part))
        return heads + [("art", m.group(1))]
    if len(s) <= 45 and _PART_LINE.match(s) and (s.endswith("정의") or not re.search(r"[이에은는을를의로]$", s)):   # 끝이 조사면 줄바꿈된 본문
        return [("part", s)]
    if _is_yak_name(s) and len(s) <= 60:
        return [("yak", _short_yak(s))]
    if _SUB_LINE.match(s) or _BOX_LINE.match(s):
        return [("sub" if s.startswith("(") else "box", s)]
    return []


def _label(state: dict) -> str:
    parts = [state[k] for k in _LEVELS if state.get(k)]
    return "[" + " > ".join(parts) + "]" if parts else ""


def body_start_page(pages: list[tuple[int, str]]) -> int | None:
    """약관 본문이 시작하는 쪽 = 장·관 제목 줄이 처음 나오는 쪽. 그 앞은 상품 요약부(표·안내문)다."""
    for no, text in clean_pages(pages):
        if any(k == "part" for ln in text.split("\n") for k, _ in classify_line(ln)):
            return no
    return None


def chunk_mixed(raw_pages: list[tuple[int, str]], body_pages: list[tuple[int, str]], size: int = 500) -> list[Chunk]:
    """요약부는 baseline500과 똑같이, 본문은 chunk_structured로 자른다.

    왜: struct500을 60문항으로 재 보니(2026-11-27) 조문 문항은 32 -> 33/36, 1등 맞춤 31 -> 39로 좋아졌는데
    요약표 문항이 22 -> 18/24로 떨어졌다. 요약부에는 장·조 제목이 없어 이름표를 못 붙이면서 자르는 위치만
    바뀌었기 때문이다 (정보는 그대로인데 표가 다른 자리에서 잘림). 구조가 없는 곳은 건드리지 않는다.
    raw_pages: 목차 쪽까지 포함한 전체 (baseline500과 같은 입력), body_pages: 목차 쪽을 뺀 것.
    """
    start = body_start_page(body_pages)
    if start is None:
        return chunk_baseline(raw_pages, size=size)
    front = chunk_baseline([p for p in raw_pages if p[0] < start], size=size)
    body = chunk_structured([p for p in body_pages if p[0] >= start], size=size)
    return front + [Chunk(len(front) + i, c.page_start, c.page_end, c.text) for i, c in enumerate(body)]


def chunk_structured(pages: list[tuple[int, str]], size: int = 500, min_size: int = 150, lead_len: int = 60) -> list[Chunk]:
    """pages: [(페이지 번호, 글자)] -> 구조를 따라 자르고 이름표를 붙인 조각들. size·min_size는 이름표를 뺀 본문 글자 수."""
    lines = [(no, ln.strip()) for no, text in clean_pages(pages) for ln in text.split("\n") if ln.strip()]
    lines = [(no, ln) for no, ln in lines if not re.fullmatch(r"-\s?\d+\s?-", ln)]          # 쪽 번호 줄
    body_page = body_start_page(pages)
    start = next((i for i, (no, _) in enumerate(lines) if no == body_page), len(lines))   # 그 쪽 첫 줄부터 본문으로 본다

    chunks: list[Chunk] = []
    chunk_yak: list = []                      # 조각마다 속한 약관 이름 (짧은 꼬리 조각을 앞 조각에 붙일 때 확인)
    cur: list[tuple] = []                     # (쪽, 줄, 이 줄에서 시작하면 붙일 이름표, 약관 이름, 제목 줄 여부)
    state: dict = {}
    para = {"text": "", "open": False}

    def body_len(items) -> int:
        return sum(len(x[1]) + 1 for x in items)

    def has_body(items) -> bool:
        return any(not x[4] for x in items)

    def emit(items) -> None:
        if not items:
            return
        body = " ".join(" ".join(x[1] for x in items).split())
        if body_len(items) < min_size and chunks and chunk_yak[-1] == items[0][3]:
            # 약관 끝의 짧은 조각(준용규정 한 줄 등)은 따로 두면 이름표만 큰 조각이 되어 검색 상위로 튄다 -> 앞 조각에 붙인다
            prev = chunks[-1]
            chunks[-1] = Chunk(prev.chunk_no, prev.page_start, items[-1][0], prev.text + " " + body)
            return
        label = items[0][2]
        chunks.append(Chunk(len(chunks), items[0][0], items[-1][0], (label + " " + body).strip()))
        chunk_yak.append(items[0][3])

    for i, (no, ln) in enumerate(lines):
        heads = classify_line(ln) if i >= start else []
        if heads:
            hard = any(k in ("yak", "box", "part") for k, _ in heads)
            if cur and has_body(cur) and (hard or body_len(cur) >= min_size):
                emit(cur)
                cur = []
            for k, title in heads:
                state[k] = title
                for lower in _LEVELS[_LEVELS.index(k) + 1:]:
                    state.pop(lower, None)
            para = {"text": "", "open": False}
            if cur and not has_body(cur):          # 제목 줄만 쌓인 상태 -> 이름표를 가장 안쪽 제목 기준으로 다시 단다
                cur = [(x[0], x[1], _label(state), state.get("yak"), True) for x in cur]
            cur.append((no, ln, _label(state), state.get("yak"), True))
            continue
        if i >= start and _PARA_START.match(ln):
            para = {"text": ln, "open": not ln.rstrip().endswith("다.")}
            lead = ""
        else:
            if para["open"]:
                para["text"] += ln
                para["open"] = len(para["text"]) < lead_len and not ln.rstrip().endswith("다.")
            lead = para["text"]
        label = _label(state) if i >= start else ""
        if lead:
            short = " ".join(lead.split())
            label += f" ({short[:lead_len]}…)" if len(short) > lead_len else f" ({short})"
        if cur and body_len(cur) + len(ln) > size:
            cut = len(cur)
            for j in range(len(cur) - 1, 0, -1):
                if body_len(cur[:j]) < size * 0.4:
                    break
                if _PARA_START.match(cur[j][1]) or _ITEM_START.match(cur[j][1]):
                    cut = j
                    break
            emit(cur[:cut])
            cur = cur[cut:]
        cur.append((no, ln, label.strip(), state.get("yak"), False))
    emit(cur)
    return chunks
