import time, json
from rag.evaluate import load_questions, first_hit_rank
from agent import graph as agent_v2

agent_v2.MAX_TRIES = 5
records = []
for q in load_questions():
    t0 = time.time()
    out = agent_v2.app.invoke({"question": q["question"], "gen": int(q["gen"]),
                               "tries": 0, "rows": [], "history": []})
    steps = []
    for h in out["history"]:
        hit = None
        if q["type"] != "no_answer":
            hit = first_hit_rank(h["rows"], int(q["gen"]), q["evidence"]) is not None
        steps.append({"k": h["k"], "guard": h["guard"], "usage": h["usage"], "hit": hit,
                      "answer": h["answer"], "quote": h["quote"], "miss": h["miss"]})
    records.append({"id": q["id"], "type": q["type"], "question": q["question"],
                    "evidence": q["evidence"], "sec": round(time.time() - t0, 1), "steps": steps})
    print(q["id"], q["type"], [s["guard"] for s in steps])

with open("agent/results/eval_v2_raw.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=1)

def outcome(steps, cap):
    used = steps[:cap]
    for i, s in enumerate(used):
        if s["guard"] == "ok":
            return "ok", used[:i + 1]
    return used[-1]["guard"], used

def tok(u, w):
    return sum(v for k, v in (u or {}).items() if w in k and isinstance(v, (int, float)))

ans = [r for r in records if r["type"] != "no_answer"]
no = [r for r in records if r["type"] == "no_answer"]
print(f"\n상한 | 답있음 ok/{len(ans)} | 근거검색됨/{len(ans)} | 숫자가드 막음 | 답없음 거절/{len(no)} | 호출 | 토큰 in/out")
for cap in range(1, 6):
    ok = hit = blocked = ref = tin = tout = calls = 0
    for r in records:
        g, used = outcome(r["steps"], cap)
        calls += len(used)
        tin += sum(tok(s["usage"], "input") for s in used)
        tout += sum(tok(s["usage"], "output") for s in used)
        if r["type"] != "no_answer":
            ok += g == "ok"
            hit += bool(used[-1]["hit"])
            blocked += g == "number_not_found"
        else:
            ref += g != "ok"
    print(f"{cap} | {ok} | {hit} | {blocked} | {ref} | {calls} | {tin}/{tout}")

