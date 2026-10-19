from agent.number_guard import numbers_in, missing_numbers
print(numbers_in("공제금액은 3만원과 보장대상 의료비의 30% 중 큰 금액이며, 연간 통원 100회 한도. 3세대는 1,000원"))
print("기대: ['3만원', '30%', '100회', '1000원']")
rows = [("통원 1회당 비급여 의료비에서 공제금액 3만원을 뺀 금액",)]
print(missing_numbers("공제금액은 3만원과 30% 중 큰 금액", rows))
print("기대: ['30%']")
print(missing_numbers("40세 남자는 3만원", rows, "40세 남자 보험료"))
print("기대: []")

