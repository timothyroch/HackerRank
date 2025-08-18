import sys, re
from collections import Counter

def main():
    txt = sys.stdin.read().lower()
    words = re.findall(r"[a-z']+", txt)
    cnt = Counter(words)
    # Filter out very short words 
    cnt = {w: c for w, c in cnt.items() if len(w) >= 2}
    # Sort by descending frequency
    sorted_words = sorted(cnt.items(), key=lambda x: -x[1])
    # Output top ~50,000 words
    for w, c in sorted_words[:50000]:
        print(w)

if __name__ == "__main__":
    main()
