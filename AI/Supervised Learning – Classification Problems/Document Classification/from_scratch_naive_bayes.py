"""
My Solution – Multinomial Naive Bayes Text Classifier (From Scratch)
--------------------------------------------------------------------

I approached this document classification problem by building my own
Multinomial Naive Bayes model entirely from scratch — no external ML
libraries. The goal is to classify each test document into one of the
eight categories (1–8) using the labeled training data provided in
"trainingdata.txt".

How I understood the problem:
    - The training file starts with the number of examples (N),
      followed by N lines, each with a category label (1–8) and the
      processed document text.
    - The program will then read test documents from standard input,
      where the first line is the number of test cases (T), followed by
      T lines of document text.
    - For each test document, I need to output the predicted category
      on its own line.

How my solution works:
    1. Tokenization:
        I lowercase all text, keep only letters and digits, and split
        on anything else (punctuation becomes spaces).
    2. Feature representation:
        I turn the tokens into a bag-of-words (word counts per document).
    3. Training:
        I build a vocabulary from the training data, count how often
        each word appears in each class, and store total counts.
        I also compute class priors (frequency of each class).
    4. Smoothing:
        I use Laplace smoothing (alpha=1.0) so unseen words don’t
        zero out probabilities.
    5. Prediction:
        For each test document, I compute the log-probability of the
        document under each class (log prior + sum over words), and
        pick the class with the highest score.
        If a document has no valid tokens, I fall back to the class
        with the highest prior.

Why I like this approach:
    - It shows I understand the math behind Naive Bayes instead of
      relying on a library.
    - It’s simple, explainable, and matches the multinomial model.
    - It’s flexible enough to handle empty inputs and unseen words.


"""


import sys
from collections import defaultdict
import math

# ---------- Tokenization ----------
def tokenize(text):
    # Lowercase + keep letters/numbers only; replace others by space
    out = []
    w = []
    for ch in text.lower():
        if ch.isalnum():
            w.append(ch)
        else:
            if w:
                out.append(''.join(w))
                w = []
    if w:
        out.append(''.join(w))
    return out

def bag_of_words(tokens):
    bow = defaultdict(int)
    for t in tokens:
        bow[t] += 1
    return bow

# ---------- Read training data from trainingdata.txt ----------
def read_training(path="trainingdata.txt"):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return lines
        try:
            n = int((first or "0").strip())
        except ValueError:
            n = 0
        for _ in range(n):
            raw = f.readline()
            if not raw:
                break
            raw = raw.rstrip("\n")
            if not raw:
                continue
            parts = raw.split(" ", 1)
            try:
                label = int(parts[0])
            except ValueError:
                continue
            if not (1 <= label <= 8):
                continue
            text = parts[1] if len(parts) > 1 else ""
            lines.append((label, text))
    return lines

# ---------- Train Multinomial Naive Bayes ----------
class NBTextClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.vocab = {}
        self.class_counts = [0]*9       # classes 1..8
        self.word_counts = [defaultdict(int) for _ in range(9)]  # per-class word totals
        self.total_words_in_class = [0]*9
        self.priors = [0.0]*9
        self.V = 0

    def fit(self, labeled_texts):
        # Build vocab and aggregate counts
        for cls, text in labeled_texts:
            self.class_counts[cls] += 1
            toks = tokenize(text)
            for w in toks:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
                self.word_counts[cls][w] += 1
                self.total_words_in_class[cls] += 1

        self.V = len(self.vocab)
        N = sum(self.class_counts)
        if N == 0:
            return
        for c in range(1, 9):
            self.priors[c] = self.class_counts[c] / N

    def score_class(self, c, text_bow):
        # If no training docs for this class, give -inf
        if self.class_counts[c] == 0 or self.priors[c] == 0.0:
            return float("-inf")
        s = math.log(self.priors[c])
        den = self.total_words_in_class[c] + self.alpha * self.V
        # Accumulate log-likelihoods for words present in the test doc
        for w, cnt in text_bow.items():
            if w in self.vocab:
                num = self.word_counts[c][w] + self.alpha
                s += cnt * (math.log(num) - math.log(den))
        return s

    def predict_one(self, text):
        toks = tokenize(text)
        bow = bag_of_words(toks)
        # If nothing remains after tokenization, fall back to most frequent class
        if not bow:
            # argmax prior
            best_c, best_prior = 1, -1.0
            for c in range(1, 9):
                if self.priors[c] > best_prior:
                    best_c, best_prior = c, self.priors[c]
            return best_c
        best_c, best_s = 1, float("-inf")
        for c in range(1, 9):
            sc = self.score_class(c, bow)
            if sc > best_s:
                best_c, best_s = c, sc
        return best_c

# ---------- Main I/O ----------
def main():
    # Train
    train = read_training("trainingdata.txt")
    clf = NBTextClassifier(alpha=1.0)
    clf.fit(train)

    # Predict on test input from STDIN
    data = sys.stdin.read().splitlines()
    if not data:
        return
    try:
        T = int((data[0] or "0").strip())
    except ValueError:
        T = 0
    idx = 1
    for _ in range(T):
        text = data[idx] if idx < len(data) else ""
        idx += 1
        pred = clf.predict_one(text)
        print(pred)

if __name__ == "__main__":
    main()
