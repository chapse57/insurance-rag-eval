"""질문 바꿔 쓰기: 고객의 일상 말투 질문 -> 약관 용어로 된 검색어.

왜: 고객은 "취소", "돌려받는 돈", "살 빼려고"라고 묻는데 약관에는 "청약의 철회", "해지환급금", "비만"이라고
적혀 있다. 말이 다르면 임베딩 검색도 재정렬도 정답 조각을 못 알아본다 (평가 60문항에서 놓친 것의 대부분).
검색하기 전에 Claude가 질문을 약관 문체로 바꾸고, 그 조항에 나올 법한 용어를 덧붙인다.

- 답을 만들지 않는다. 검색어만 만든다 (숫자·기간을 지어내지 못하게 프롬프트에서 막는다).
- 결과는 eval/rewrites.json에 저장해 두고 다시 쓴다. 같은 질문·같은 프롬프트·같은 모델이면 API를 다시 부르지 않는다
  -> 평가를 다시 돌려도 검색어가 바뀌지 않는다 (비용도 안 든다).

확인:  python -m rag.rewrite "4세대 실손 중간에 해지하면 돌려받는 돈 있어요?"
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from rag.generate import MODEL, load_env

CACHE = Path("eval/rewrites.json")

PROMPT = """당신은 실손의료보험 약관 검색을 돕습니다. 고객이 일상 말투로 한 [질문]을, 보험 약관에서 해당 내용을 찾기 좋은 검색어로 바꿉니다.

규칙:
1. 질문에 답하지 마세요. 숫자·기간·금액을 지어내지 마세요.
2. query: 질문의 뜻은 그대로 두고 약관 문체의 한 문장으로 다시 씁니다. 일상 표현은 약관에서 쓰는 용어로 바꿉니다.
3. terms: 그 내용을 다루는 약관 조항에 실제로 나올 법한 용어 3~6개. 질문과 무관한 용어는 넣지 않습니다.
4. '2세대', '3세대', '4세대' 같은 세대 표현은 뺍니다 (세대는 따로 걸러서 검색합니다).
5. 반드시 submit_query 도구로 제출합니다.

[질문]
{question}
"""

TOOL = {
    "name": "submit_query",
    "description": "약관 검색어를 제출한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "약관 문체로 다시 쓴 한 문장"},
            "terms": {"type": "array", "items": {"type": "string"}, "description": "약관에 나올 법한 용어 3~6개"},
        },
        "required": ["query", "terms"],
    },
}

PROMPT_V2 = """당신은 실손의료보험 약관 검색을 돕습니다. 고객이 일상 말투로 한 [질문]을, 보험 약관에서 해당 내용을 찾기 좋은 검색어로 바꿉니다.

규칙:
1. 질문에 답하지 마세요. 숫자·기간·금액을 지어내지 마세요.
2. query: 질문이 묻는 것만 약관 용어로 짧게 다시 씁니다 (40자 이내). 질문에 없는 상황·조건을 덧붙이지 마세요. 질문에 나온 사람(계약자, 피보험자, 가족 등)과 행위는 바꾸지 않습니다.
3. terms: 그 내용을 다루는 약관 조항의 제목이나 조문에 실제로 나올 법한 용어 3~6개. 일상 표현에 대응하는 약관 용어를 우선하고, 확신이 없는 용어는 넣지 않습니다.
4. '2세대', '3세대', '4세대' 같은 세대 표현은 뺍니다 (세대는 따로 걸러서 검색합니다).
5. 반드시 submit_query 도구로 제출합니다.

[질문]
{question}
"""

PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2}   # v1: 처음 쓴 것(60문항에서 측정함). v2: 짧게 + 질문에 없는 내용 금지
DEFAULT_PROMPT = "v1"


def prompt_sha(version: str) -> str:
    text = PROMPTS[version] + json.dumps(TOOL, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


PROMPT_SHA = prompt_sha("v1")


def build_search_query(query: str, terms: list[str]) -> str:
    terms = [t.strip() for t in terms if t and t.strip()]
    return f"{query.strip()} ({', '.join(terms)})" if terms else query.strip()


def ask_model(question: str, version: str = DEFAULT_PROMPT) -> dict:
    import anthropic

    load_env()
    client = anthropic.Anthropic()
    msg = client.messages.create(model=MODEL, max_tokens=300, tools=[TOOL],
                                 tool_choice={"type": "tool", "name": TOOL["name"]},
                                 messages=[{"role": "user", "content": PROMPTS[version].format(question=question)}])
    tool = next((b for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
    return dict(tool.input) if tool else {}


def _load_cache(path: Path) -> dict:
    """{프롬프트 해시: {질문: {query, terms, model}}}. 프롬프트를 바꿔도 예전 검색어가 지워지지 않는다."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data and all(isinstance(v, dict) and "query" in v for v in data.values()):   # 예전 형식(질문 -> 결과)
        new: dict = {}
        for q, v in data.items():
            new.setdefault(v.get("prompt_sha", PROMPT_SHA), {})[q] = {k: v[k] for k in ("query", "terms", "model")}
        return new
    return data


def rewrite(question: str, cache_path: Path = CACHE, ask=None, version: str = DEFAULT_PROMPT,
            save_to: Path | None = None) -> str:
    """질문 -> 검색어. 형식이 이상하게 오면 원래 질문을 그대로 돌려준다 (검색이 망가지지 않게).

    save_to: 새로 만든 검색어를 cache_path 대신 여기에 저장한다. 화면·명령줄에서 들어온 아무 질문이나
    평가용 파일(eval/rewrites.json, 리포에 올라감)에 섞여 들어가지 않게 하려는 것. 읽을 때는 둘 다 본다.
    """
    sha = prompt_sha(version)
    hit = _load_cache(cache_path).get(sha, {}).get(question)
    if not (hit and hit.get("model") == MODEL) and save_to is not None:
        hit = _load_cache(save_to).get(sha, {}).get(question)
    if not (hit and hit.get("model") == MODEL):
        data = ask(question) if ask else ask_model(question, version)
        query, terms = data.get("query"), data.get("terms")
        if not isinstance(query, str) or not query.strip() or not isinstance(terms, list):
            return question
        hit = {"query": query, "terms": [str(t) for t in terms], "model": MODEL}
        target = save_to if save_to is not None else cache_path
        cache = _load_cache(target)
        cache.setdefault(sha, {})[question] = hit
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return build_search_query(hit["query"], hit["terms"])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    for q in sys.argv[1:]:
        print(f"{q}\n -> {rewrite(q)}\n")
