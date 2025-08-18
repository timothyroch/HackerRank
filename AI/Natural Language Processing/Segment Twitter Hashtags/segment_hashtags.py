import sys, math, re
from typing import Dict, List, Tuple

# ---------- Parameters ----------
UNK_COST_PER_CHAR = 8.0     # penalty per unknown character
MAX_WORD_LEN = 24
COUNTRY_ABBRS = {"us","usa","uk","uae","eu","prc"}

# ---------- Input ----------
def read_input():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return 0, []
    n = int(data[0].strip())
    tags = [line.strip() for line in data[1:1+n]]
    return n, tags

# ---------- Dictionary ----------
def load_wordlist(path="wordlist.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [w.strip().lower() for w in f if w.strip().isalpha()]
    except:
        return []

def build_cost(words: List[str]) -> Tuple[Dict[str, float], int]:
    unique = list(dict.fromkeys(words))  # dedupe
    N = len(unique)
    if N == 0:
        return {}, 0
    # Use word rank as cost: earlier words cheaper
    costs = {w: math.log((i+1) * math.log(N+1)) for i, w in enumerate(unique)}
    maxlen = max(len(w) for w in unique)
    return costs, maxlen

# ---------- Cost wrapper ----------
class WordCost:
    def __init__(self, wc: Dict[str,float], maxlen: int):
        self.wc, self.maxlen = wc, maxlen
    def __call__(self, w: str) -> float:
        lw = w.lower()
        if lw in self.wc:
            return self.wc[lw]
        if lw in COUNTRY_ABBRS:
            return 1.0
        if lw.isdigit():
            return 0.8
        return UNK_COST_PER_CHAR * len(lw)

# ---------- Segmentation ----------
def segment_chunk(chunk: str, wc: WordCost) -> List[str]:
    L = len(chunk)
    best = [0.0] + [float("inf")] * L
    back = [-1] * (L+1)
    limit = min(wc.maxlen, MAX_WORD_LEN)

    for i in range(1, L+1):
        for j in range(max(0, i-limit), i):
            cost = best[j] + wc(chunk[j:i])
            if cost < best[i]:
                best[i], back[i] = cost, j

    out = []
    i = L
    while i > 0 and back[i] >= 0:
        j = back[i]
        out.append(chunk[j:i])
        i = j
    out.reverse()
    return out if out else [chunk]

def segment_hashtag(tag: str, wc: WordCost) -> List[str]:
    s = tag.lstrip("#").lower()
    if not s:
        return []
    parts = re.findall(r"[a-z]+|\d+", s)
    result = []
    for p in parts:
        if p.isdigit():
            result.append(p)
        else:
            result.extend(segment_chunk(p, wc))
    return result

# ---------- Main ----------
def main():
    words = load_wordlist("wordlist.txt")
    if not words:
        print("Error: wordlist.txt missing or empty, generate it first.", file=sys.stderr)
        sys.exit(1)

    wc_map, maxlen = build_cost(words)
    wc = WordCost(wc_map, maxlen)

    _, tags = read_input()
    for t in tags:
        print(" ".join(segment_hashtag(t, wc)))

if __name__ == "__main__":
    main()
