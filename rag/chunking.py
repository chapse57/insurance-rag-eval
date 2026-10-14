"""약관 글자를 조각으로 자른다.

baseline500: 일부러 허술한 첫 버전. 페이지를 이어 붙이고 500자마다 기계적으로 자른다.
조항 중간이든 표 중간이든 상관없이 자르기 때문에, 나중에 '어디서 틀렸나'를 볼 때
'자르기' 단계의 실패가 드러나도록 남겨 둔다. (고친 버전은 별도 method로 추가)
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_no: int
    page_start: int
    page_end: int
    text: str


def chunk_baseline(pages: list[tuple[int, str]], size: int = 500) -> list[Chunk]:
    """pages: [(페이지 번호, 글자)] -> 공백을 한 칸으로 줄여 이어 붙인 뒤 size 글자마다 자른 조각들."""
    if size <= 0:
        raise ValueError("size는 1 이상이어야 합니다")
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
    for n, i in enumerate(range(0, len(full), size)):
        piece = full[i:i + size]
        chunks.append(Chunk(n, page_at(i), page_at(i + len(piece) - 1), piece.strip()))
    return [c for c in chunks if c.text]
