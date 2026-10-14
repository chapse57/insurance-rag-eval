"""답변 생성 + 근거 검사(가드레일).

흐름: 검색(+재정렬)으로 찾은 조각 k개 -> Claude에게 '이 조각만 근거로 답하고, 원문을 그대로 인용하라' 요청
      -> 코드가 인용문이 실제로 그 조각 안에 글자 그대로 있는지 확인
      -> 없으면 AI 답을 버리고 '모름'으로 바꾼다 (지어낸 답 차단)

AI를 믿지 않고 코드로 확인하는 부분이 이 프로젝트의 핵심이다.
API 키는 Upwork_2/.env 의 ANTHROPIC_API_KEY 에서 읽는다 (리포에 올리지 않음).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

MODEL = os.environ.get("RAG_MODEL", "claude-haiku-5-5")
REFUSAL = "약관에서 근거를 찾지 못해 답할 수 없습니다."

PROMPT = """당신은 실손의료보험 약관 안내원입니다. 아래 [조각]만 근거로 [질문]에 답하세요.

규칙:
1. 조각에 답이 없으면 found=false 로 제출합니다. 추측하지 마세요.
2. 답이 있으면 근거가 된 조각 번호(chunk)와, 그 조각에서 답의 근거가 되는 문장을 **한 글자도 바꾸지 말고** 그대로 복사해 quote에 넣습니다 (10~80자).
3. answer는 고객이 이해하기 쉬운 한두 문장으로 씁니다.
4. 반드시 submit_answer 도구로 제출합니다.

[질문]
{question}

[조각]
{chunks}
"""


# 형식을 '부탁'하지 않고 도구 스키마로 '강제'한다.
# tool_choice로 이 도구를 지정하면 모델이 앞에 설명 글을 쓰지 않고 바로 이 형식으로만 답한다.
# (처음엔 프롬프트로 JSON만 내라고 부탁했는데, 앞에 긴 글을 써서 30문제 중 7개가 길이 상한에서 잘렸다)
TOOL = {
    "name": "submit_answer",
    "description": "약관 조각을 근거로 한 답을 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {"type": "boolean", "description": "조각 안에 답이 있으면 true"},
            "answer": {"type": "string", "description": "고객용 한두 문장 답"},
            "chunk": {"type": "integer", "description": "근거 조각 번호 (1부터)"},
            "quote": {"type": "string", "description": "근거 조각에서 글자 그대로 복사한 문장"},
        },
        "required": ["found"],
    },
}


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    chunk: int | None = None
    quote: str | None = None
    guard: str = ""            # 가드레일 판정: ok / model_refused / truncated / bad_json / bad_chunk / quote_not_found
    raw: str = ""
    usage: dict = field(default_factory=dict)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def load_env(path: Path = Path(".env")) -> dict:
    """.env를 읽어 환경변수에 넣는다. PC에 이미 같은 이름의 값이 있어도 .env가 이긴다 (이 프로젝트 키를 확실히 쓰려고)."""
    found = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():   # 메모장 BOM 대비
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                found[k.strip()] = v.strip().strip('"').strip("'")
        os.environ.update(found)
    return found


def check_key() -> int:
    """키를 화면에 드러내지 않고 상태만 보여준다: python -m rag.generate"""
    before = os.environ.get("ANTHROPIC_API_KEY")
    found = load_env()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    print(f".env 파일: {'있음' if Path('.env').is_file() else '없음 (폴더이거나 이름이 다름)'}")
    print(f".env 안에 ANTHROPIC_API_KEY: {'있음' if 'ANTHROPIC_API_KEY' in found else '없음'}")
    if before and before != key:
        print("PC 환경변수에 다른 키가 있었음 -> .env 키로 덮어씀")
    print(f"키 길이 {len(key)}자 | 'sk-ant-'로 시작: {key.startswith('sk-ant-')} | 끝 4자리: ...{key[-4:] if key else ''}")
    bad_chars = any(c in key for c in " \t\"'")
    hangul = any("\uac00" <= c <= "\ud7a3" for c in key)
    print(f"공백·따옴표 섞임: {bad_chars} | 한글 섞임: {hangul}")
    return 0


def build_prompt(question: str, rows: list) -> str:
    parts = [f"<{i}> ({r[0]}세대 {r[3]}쪽)\n{r[6]}" for i, r in enumerate(rows, start=1)]
    return PROMPT.format(question=question, chunks="\n\n".join(parts))


def check(question: str, rows: list, raw: str, stop_reason: str = "end_turn") -> Answer:
    """AI가 낸 원문(raw)을 가드레일로 검사한다. AI 호출과 분리해서 테스트할 수 있게 둔다."""
    if stop_reason == "max_tokens":   # 답이 길이 상한에서 잘림 -> 형식을 믿을 수 없음
        return Answer(question, REFUSAL, True, guard="truncated", raw=raw)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return Answer(question, REFUSAL, True, guard="bad_json", raw=raw)
    if data.get("found") is False or not data.get("answer"):
        return Answer(question, REFUSAL, True, guard="model_refused", raw=raw)
    try:
        idx = int(data.get("chunk"))
    except (TypeError, ValueError):
        idx = -1
    if not 1 <= idx <= len(rows):
        return Answer(question, REFUSAL, True, guard="bad_chunk", raw=raw)
    quote = data.get("quote") or ""
    if len(norm(quote)) < 5 or norm(quote) not in norm(rows[idx - 1][6]):
        return Answer(question, REFUSAL, True, chunk=idx, quote=quote, guard="quote_not_found", raw=raw)
    return Answer(question, data["answer"], False, chunk=idx, quote=quote, guard="ok", raw=raw)


def ask_model(prompt: str) -> tuple[str, dict, str]:
    import anthropic

    load_env()
    client = anthropic.Anthropic()
    msg = client.messages.create(model=MODEL, max_tokens=1024, tools=[TOOL],
                                 tool_choice={"type": "tool", "name": TOOL["name"]},
                                 messages=[{"role": "user", "content": prompt}])
    tool = next((b for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
    text = json.dumps(tool.input, ensure_ascii=False) if tool else \
        "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text, {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}, msg.stop_reason


def answer(question: str, rows: list) -> Answer:
    if not rows:
        return Answer(question, REFUSAL, True, guard="no_chunks")
    raw, usage, stop = ask_model(build_prompt(question, rows))
    a = check(question, rows, raw, stop)
    a.usage = usage
    return a


if __name__ == "__main__":
    raise SystemExit(check_key())
