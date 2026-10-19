from dataclasses import replace
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from rag.ask import retrieve
from rag.generate import answer
from agent.number_guard import missing_numbers

MAX_TRIES = 2
K_STEPS = [3, 5, 8, 12, 20]
RETRY_GUARDS = {"quote_not_found", "model_refused", "number_not_found"}

class State(TypedDict):
    question: str
    gen: int
    tries: int
    rows: list
    result: object
    history: list

def search(state: State):
    k = K_STEPS[state["tries"]]
    rows = retrieve(state["question"], state["gen"], "baseline500", True, k)
    return {"rows": rows, "tries": state["tries"] + 1}

def generate(state: State):
    a = answer(state["question"], state["rows"])
    miss = []
    if a.guard == "ok":
        miss = missing_numbers(a.answer, state["rows"], state["question"])
        if miss:
            a = replace(a, guard="number_not_found")
    k = K_STEPS[state["tries"] - 1]
    print(f"[{state['tries']}번째] 조각 {k}개 -> 판정: {a.guard}" + (f" (근거 없는 숫자: {miss})" if miss else ""))
    step = {"k": k, "guard": a.guard, "usage": a.usage, "rows": state["rows"],
            "answer": a.answer, "quote": a.quote, "miss": miss}
    return {"result": a, "history": state["history"] + [step]}

def route(state: State):
    g = state["result"].guard
    if g == "ok":
        return END
    if state["tries"] >= MAX_TRIES:
        return END
    if g in RETRY_GUARDS:
        return "search"
    return END

graph = StateGraph(State)
graph.add_node("search", search)
graph.add_node("generate", generate)
graph.add_edge(START, "search")
graph.add_edge("search", "generate")
graph.add_conditional_edges("generate", route, ["search", END])
app = graph.compile()

