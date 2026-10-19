import json
from rag.ask import retrieve
from rag.evaluate import load_questions, first_hit_rank
qs = {q["id"]: q for q in load_questions()}
recs = json.load(open("agent/results/eval_v2_raw.json", encoding="utf-8"))
for r in recs:
    q = qs[r["id"]]
    if q["type"] == "no_answer":
        continue
    for s in r["steps"]:
        rows = retrieve(q["question"], int(q["gen"]), "baseline500", True, s["k"])
        s["hit"] = first_hit_rank(rows, int(q["gen"]), q["evidence"]) is not None
    print(r["id"], [s["hit"] for s in r["steps"]])
with open("agent/results/eval_v2_raw.json", "w", encoding="utf-8") as f:
    json.dump(recs, f, ensure_ascii=False, indent=1)

def outcome(steps, cap):
    used = steps[:cap]
    for i, s in enumerate(used):
        if s["guard"] == "ok":
            return "ok", used[:i + 1]
    return used[-1]["guard"], used

ans = [r for r in recs if r["type"] != "no_answer"]
print(f"\n상한 | 답있음 ok/{len(ans)} | 근거검색됨/{len(ans)}")
for cap in range(1, 6):
    ok = sum(outcome(r["steps"], cap)[0] == "ok" for r in ans)
    hit = sum(bool(outcome(r["steps"], cap)[1][-1]["hit"]) for r in ans)
    print(f"{cap} | {ok} | {hit}")
print("\n[ok인데 근거검색 안 됨: 상한 2]")
for r in ans:
    g, used = outcome(r["steps"], 2)
    if g == "ok" and not used[-1]["hit"]:
        print(r["id"])

