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
