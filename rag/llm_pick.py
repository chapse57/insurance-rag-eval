"""마지막 고르기: 재정렬까지 거친 상위 10개 조각 중에서 Claude가 '답이 들어 있는 조각'을 최대 3개 고른다.

왜: 재정렬 모델은 말이 다르면 못 알아본다 ("살 빼려고" <-> "비만(E66)", "취소" <-> "청약의 철회").
    질문을 약관 용어로 바꿔 써서 재정렬하면 이번엔 바꿔 쓰기가 뜻을 틀리게 옮긴 문항에서 오답이 올라온다.
    60문항에서 정답은 거의 항상 상위 10개 안에는 있었다 -> 그 10개를 읽고 고르는 일만 Claude에게 맡긴다.
- 고른 조각을 앞에 놓고, 나머지는 원래 순서대로 뒤에 붙인다. 형식이 이상하면 원래 순서를 그대로 쓴다.
- 같은 (질문, 조각들)이면 저장해 둔 결과를 다시 쓴다 (eval/cache/llm_picks.json).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag.generate import MODEL, load_env

CACHE = Path("eval/cache/llm_picks.json")

PROMPT = """고객이 실손의료보험에 대해 [질문]했습니다. 아래 [조각]들은 약관에서 검색된 글입니다.
질문에 대한 답이나 답의 직접적인 근거가 들어 있는 조각을 고르세요.

규칙:
1. 고객의 일상 표현과 약관 용어가 달라도 같은 뜻이면 같은 것으로 봅니다.
2. 주제만 비슷하고 답이 없는 조각은 고르지 않습니다.
3. 관련이 높은 순서로 최대 3개의 조각 번호를 submit_picks 도구로 제출합니다. 하나도 없으면 빈 목록을 제출합니다.
4. 조각은 <조각 번호="N"> 과 </조각> 사이의 글입니다. 번호는 그 태그에 적힌 N을 그대로 씁니다.

[질문]
{question}

[조각]
{chunks}
"""

TOOL = {
    "name": "submit_picks",
    "description": "답이 들어 있는 조각 번호를 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {"chunks": {"type": "array", "items": {"type": "integer"},
                                  "description": "조각 번호, 관련이 높은 순서, 최대 3개"}},
        "required": ["chunks"],
    },
}


def ask_model(prompt: str) -> list:
    import anthropic

    load_env()
    msg = anthropic.Anthropic().messages.create(
        model=MODEL, max_tokens=200, tools=[TOOL], tool_choice={"type": "tool", "name": TOOL["name"]},
        messages=[{"role": "user", "content": prompt}])
    tool = next((b for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
    return list(tool.input.get("chunks", [])) if tool else []


def reorder(rows: list, picks: list) -> list:
    """picks(1부터 세는 조각 번호)를 앞에 놓고 나머지는 원래 순서대로. 범위 밖·중복 번호는 버린다."""
    chosen = []
    for p in picks:
        if isinstance(p, int) and not isinstance(p, bool) and 1 <= p <= len(rows) and p - 1 not in chosen:
            chosen.append(p - 1)
    chosen = chosen[:3]
    return [rows[i] for i in chosen] + [r for i, r in enumerate(rows) if i not in chosen]


def pick(question: str, rows: list, cache_path: Path = CACHE, ask=ask_model) -> list:
    if len(rows) <= 1:
        return rows
    prompt = PROMPT.format(question=question, chunks="\n\n".join(f'<조각 번호="{i}">\n{r[6]}\n</조각>' for i, r in enumerate(rows, 1)))
    key = hashlib.sha256(f"{MODEL}|{prompt}|{json.dumps(TOOL, sort_keys=True)}".encode("utf-8")).hexdigest()[:24]
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    if key not in cache:
        cache[key] = ask(prompt)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return reorder(rows, cache[key])
