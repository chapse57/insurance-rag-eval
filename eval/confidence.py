"""검색 Hit@3의 신뢰구간과 버전 간 차이의 유의성.

표본이 24문항뿐이라 점수 하나만으로는 개선인지 우연인지 구분되지 않는다.
- 각 버전: Wilson 95% 신뢰구간 + 부트스트랩 95% 구간
- 버전 쌍: 같은 24문항이므로 대응표본 -> McNemar 정확검정(이항)
"""
import csv, math, random, sys
from pathlib import Path
from itertools import combinations

RESULTS = Path(__file__).parent / "results"
RUNS = {   # 설정 -> 그 설정으로 돌린 실행 파일들 (11/26 고정 평가셋 기준)
    "baseline500":      ["baseline500_1126_1419.csv"],
    "fixed200":         ["fixed200_1126_1419.csv"],
    "fixed1000":        ["fixed1000_1126_1420.csv"],
    "+rerank20_len512": ["baseline500+rerank20_len512_1126_1420.csv"],
    "+rerank20_len256": ["baseline500+rerank20_len256_1126_1414.csv"],
    "+rerank10_len512": ["baseline500+rerank10_len512_1126_1431.csv"],
    "+rerank40_len256": ["baseline500+rerank40_len256_1126_1508.csv"],
}

K = 3  # Hit@3

def load(fn):
    """{질문id: 1/0} — 정답 조항이 상위 K안에 들어왔는가"""
    out = {}
    for r in csv.DictReader(open(RESULTS / fn, encoding="utf-8-sig")):
        rank = r["rank"].strip()
        out[r["id"]] = 1 if (rank.isdigit() and 1 <= int(rank) <= K) else 0
    return out

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

def bootstrap(v, B=20000, seed=42):
    rng = random.Random(seed)
    n = len(v)
    s = sorted(sum(rng.choice(v) for _ in range(n)) / n for _ in range(B))
    return (s[int(.025*B)], s[int(.975*B)])

def mcnemar_exact(a, b):
    """대응표본 이항검정. a,b는 {id:0/1}. 반환 (개선된수, 나빠진수, 양측 p)"""
    ids = sorted(set(a) & set(b))
    bb = sum(1 for i in ids if a[i] == 0 and b[i] == 1)   # 새로 맞음
    cc = sum(1 for i in ids if a[i] == 1 and b[i] == 0)   # 새로 틀림
    n = bb + cc
    if n == 0: return bb, cc, 1.0
    m = min(bb, cc)
    tail = sum(math.comb(n, i) for i in range(m+1)) * (0.5**n)
    return bb, cc, min(1.0, 2*tail)

def main():
    data, repeat = {}, {}
    for name, fns in RUNS.items():
        runs = [load(f) for f in fns if (RESULTS / f).exists()]
        if not runs: continue
        data[name] = runs[0]
        repeat[name] = (len(runs), all(r == runs[0] for r in runs))

    print(f"Hit@{K} — 답이 있는 질문 기준, 표본 24문항\n")
    print(f"{'설정':<20}{'점수':>12}   {'Wilson 95%':<22}{'부트스트랩 95%':<22}반복실행")
    for name, d in data.items():
        v = list(d.values()); k, n = sum(v), len(v)
        lo, hi = wilson(k, n); blo, bhi = bootstrap(v)
        cnt, same = repeat[name]
        tag = f"{cnt}회 " + ("전부 동일" if same else "불일치")
        print(f"{name:<20}{k:>2}/{n} ({k/n*100:4.1f}%)   "
              f"[{lo*100:4.1f}% ~ {hi*100:5.1f}%]      [{blo*100:4.1f}% ~ {bhi*100:5.1f}%]     {tag}")

    print("\n설정 간 차이 (같은 24문항, McNemar 정확검정)\n")
    for x, y in combinations(list(data), 2):
        bb, cc, p = mcnemar_exact(data[x], data[y])
        gap = (sum(data[y].values()) - sum(data[x].values())) / 24 * 100
        print(f"{x:>18} -> {y:<18} {gap:+5.1f}%p | "
              f"새로맞음 {bb}, 새로틀림 {cc} | p={p:.4f}  "
              f"{'유의함' if p < 0.05 else '구분 안 됨'}")

if __name__ == "__main__":
    main()
