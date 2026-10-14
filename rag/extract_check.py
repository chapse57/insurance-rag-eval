"""PDF에서 글자를 뽑고, 그 글자를 RAG에 써도 되는지 페이지마다 검사한다.

옛 보험 약관 PDF는 글꼴 정보가 빠져서 한글이 깨져 나오거나(예: '¿ÀÇ'),
스캔 이미지라 글자가 아예 안 나오는 경우가 있다. 이걸 모르고 자르고 검색하면
나중에 '검색이 틀렸다'로 보이지만 실제 원인은 '추출이 깨졌다'다.
그래서 자르기 전에 추출부터 숫자로 확인한다.

실행:  python -m rag.extract_check            (data/raw/gen*.pdf 전부)
결과:  화면에 문서별 요약 + data/processed/gen{N}_pages.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# 페이지 판정 기준. 약관 본문은 한글이 대부분이라 30% 미만이면 깨졌다고 본다.
MIN_HANGUL_RATIO = 0.30
MIN_CHARS = 20  # 이보다 적으면 빈 페이지(또는 이미지만 있는 페이지)
# 목차는 "제6조보험금의청구.......92"처럼 점선이 많아 한글 비율이 낮다. 깨짐과 구분한다.
TOC_LEADERS = 5
_LEADER = re.compile(r"[.·…]{5,}")


@dataclass
class PageStat:
    page: int
    chars: int          # 공백 뺀 글자 수
    hangul: int         # 완성형 한글 글자 수
    weird: int          # 깨진 글자 흔적 (사용자 정의 영역, U+FFFD, 제어문자)
    status: str         # ok / empty / toc / garbled

    @property
    def hangul_ratio(self) -> float:
        return self.hangul / self.chars if self.chars else 0.0


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def _is_weird(ch: str) -> bool:
    o = ord(ch)
    return (0xE000 <= o <= 0xF8FF) or ch == "�" or (o < 32 and ch not in "\n\r\t")


def page_stat(page_no: int, text: str) -> PageStat:
    body = re.sub(r"\s+", "", text)
    chars = len(body)
    hangul = sum(_is_hangul(c) for c in body)
    weird = sum(_is_weird(c) for c in body)
    if chars < MIN_CHARS:
        status = "empty"
    elif len(_LEADER.findall(text)) >= TOC_LEADERS:
        status = "toc"
    elif hangul / chars < MIN_HANGUL_RATIO:
        status = "garbled"
    else:
        status = "ok"
    return PageStat(page_no, chars, hangul, weird, status)


def doubled_runs(text: str) -> int:
    """굵은 제목을 같은 글자를 두 번 찍어 만든 PDF는 '질 질병 병입 입원 원'처럼 뽑힌다.
    앞 토막의 끝 글자 = 다음 토막의 첫 글자인 짧은 토막이 3개 이상 이어지면 1군데로 센다."""
    toks = text.split()
    runs, cur = 0, 1
    for a, b in zip(toks, toks[1:]):
        if len(a) <= 2 and len(b) <= 2 and a[-1] == b[0]:
            cur += 1
            continue
        runs += cur >= 3
        cur = 1
    return runs + (cur >= 3)


def space_ratio(text: str) -> float:
    """한글 1글자당 띄어쓰기 수. 보통 한국어 문장은 0.3 안팎, 띄어쓰기가 빠지면 크게 낮아진다."""
    hangul = sum(_is_hangul(c) for c in text)
    return text.count(" ") / hangul if hangul else 0.0


def gen_from_name(path: Path) -> int:
    m = re.search(r"gen(\d)", path.stem)
    if not m:
        raise ValueError(f"파일명에 세대 번호가 없습니다: {path.name} (gen2.pdf 처럼 이름을 바꿔 주세요)")
    return int(m.group(1))


def extract(path: Path) -> list[tuple[PageStat, str]]:
    import fitz  # PyMuPDF

    out = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            out.append((page_stat(i, text), text))
    return out


def summarize(name: str, pages: list[PageStat]) -> dict:
    n = len(pages)
    ok = sum(p.status == "ok" for p in pages)
    empty = sum(p.status == "empty" for p in pages)
    toc = sum(p.status == "toc" for p in pages)
    garbled = sum(p.status == "garbled" for p in pages)
    chars = sum(p.chars for p in pages)
    hangul = sum(p.hangul for p in pages)
    content = n - empty - toc  # 본문이 있어야 할 페이지
    if content and ok / content >= 0.9:
        verdict = "사용 가능"
    elif content and ok / content >= 0.5:
        verdict = "부분 깨짐 - 깨진 페이지 확인 필요"
    else:
        verdict = "사용 불가 - OCR 또는 다른 PDF 필요"
    return {
        "doc": name, "pages": n, "ok": ok, "empty": empty, "toc": toc, "garbled": garbled,
        "hangul_ratio": round(hangul / chars, 3) if chars else 0.0, "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdfs", nargs="*", type=Path, help="기본값: data/raw/gen*.pdf")
    ap.add_argument("--sample-page", type=int, default=0, help="이 페이지 글자 앞부분을 보여줌 (0이면 가운데 페이지)")
    args = ap.parse_args(argv)

    pdfs = args.pdfs or sorted(RAW_DIR.glob("gen*.pdf"))
    if not pdfs:
        print("data/raw/ 에 gen2.pdf, gen3.pdf, gen4.pdf 를 넣어 주세요.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        gen = gen_from_name(pdf)
        rows = extract(pdf)
        stats = [s for s, _ in rows]
        summ = summarize(pdf.name, stats)
        print(f"\n== {pdf.name} (세대 {gen}) ==")
        print(f"  페이지 {summ['pages']} | 정상 {summ['ok']} | 목차 {summ['toc']} | 빈 페이지 {summ['empty']}"
              f" | 깨짐 {summ['garbled']} -> {summ['verdict']}")
        ok_text = "".join(t for st, t in rows if st.status == "ok")
        d_runs = sum(doubled_runs(t) for st, t in rows if st.status == "ok")
        d_pages = sum(doubled_runs(t) > 0 for st, t in rows if st.status == "ok")
        print(f"  띄어쓰기 비율 {space_ratio(ok_text):.3f} (보통 0.3 안팎)"
              f" | 글자 중복 흔적 {d_runs}군데 ({d_pages}쪽)")
        bad = [s.page for s in stats if s.status != "ok"]
        if bad:
            print(f"  정상 아닌 페이지: {bad[:30]}{' ...' if len(bad) > 30 else ''}")
        idx = (args.sample_page - 1) if args.sample_page else len(rows) // 2
        idx = max(0, min(idx, len(rows) - 1))
        sample = re.sub(r"\s+", " ", rows[idx][1]).strip()[:200]
        print(f"  {idx + 1}쪽 앞부분: {sample}")

        out = OUT_DIR / f"gen{gen}_pages.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for s, text in rows:
                rec = {"gen": gen, "doc": pdf.name, **asdict(s),
                       "hangul_ratio": round(s.hangul_ratio, 3), "text": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
