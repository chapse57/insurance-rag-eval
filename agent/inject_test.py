from rag.ask import retrieve
from agent.number_guard import missing_numbers
q = "4세대 실손인데 건강보험 안 되는 진료를 입원 없이 받으면 제 돈은 얼마나 나가요?"
rows = retrieve(q, 4, "baseline500", True, 3)
real = "공제금액은 3만원과 보장대상 의료비의 30% 중 큰 금액입니다."
fake = "공제금액은 7만원과 보장대상 의료비의 45% 중 큰 금액입니다."
print("진짜 답 ->", missing_numbers(real, rows, q))
print("가짜 답 ->", missing_numbers(fake, rows, q))

