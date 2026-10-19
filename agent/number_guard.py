import re

UNIT = r"(?:만원|천원|원|%|회|일|개월|년|세(?!대))"
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?\s*" + UNIT)

def norm(s):
    return re.sub(r"\s+", "", s or "")

def numbers_in(text):
    return [norm(m.group()).replace(",", "") for m in NUM.finditer(text or "")]

def row_text(r):
    if isinstance(r, dict):
        vals = r.values()
    elif isinstance(r, (list, tuple)):
        vals = r
    else:
        vals = [getattr(r, "text", r)]
    return " ".join(str(v) for v in vals)

def missing_numbers(answer_text, rows, question=""):
    pool = norm(" ".join(row_text(r) for r in rows)).replace(",", "")
    asked = set(numbers_in(question))
    return [n for n in numbers_in(answer_text) if n not in asked and n not in pool]

