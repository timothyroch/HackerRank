"""
Quora Answer Quality Classifier — Problem & Approach Notes
---------------------------------------------------------

Problem (HackerRank):
- Task: Predict whether a Quora answer is "good" (+1) or "bad" (-1).
- Input format:
    Line 1: "N M"  → N = #training rows, M = #features (indices are 1..M)
    Next N lines:  "<id> <label> k:v k:v ..."  where label is {+1, -1}
    Next line:     "q"  → #test rows
    Next q lines:  "<id> k:v k:v ..."  (no label; we must predict it)
- Output format:
    q lines: "<id> +1" or "<id> -1"


Approaches I tried (and why the final one worked):
1) Linear SVM (LinearSVC) with L1 penalty
   - Pros: strong baseline for high-dimensional sparse problems; fast.
   - Result: ~83.1/100 for me here; close but below the cutoff.
   - Notes: needed careful loss/penalty compatibility (L1 requires squared_hinge & dual=False).

2) SGDClassifier / LogisticRegression (elastic-net, averaging)
   - Pros: flexible, can approximate linear SVM/logistic; quick to iterate.
   - Result: under the LC baseline with my quick settings; sensitive to α/C and l1_ratio.

3) Naive Bayes (GaussianNB)
   - Pros: simple, fast; useful if features look Gaussian-ish per class.
   - Result: not competitive on this dataset compared to linear/ensemble.

4) Final choice — RandomForest + DictVectorizer (this file)
   - Why it worked: tree ensembles naturally handle non-linear interactions,
     mixed feature scales, and irrelevant features without heavy preprocessing.
   - DictVectorizer lets me map the sparse "k:v" lists directly into a matrix
     using feature indices as keys; RandomForest trains on the dense array.
   - Simple, robust, and passed the judge where others were either slower
     (or scored just below the cutoff).

Implementation outline:
- Parse header -> read N train lines -> build list-of-dict features + labels.
- DictVectorizer converts {feature_index: value} dicts to a consistent matrix.
- Fit RandomForest on the training matrix.
- Parse q test lines -> transform with the SAME vectorizer -> predict.
"""

import sys
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

# -------------------------
# skips empty/whitespace-only lines
# -------------------------
def read_nonempty() -> str:
    while True:
        line = sys.stdin.readline()
        if not line:  
            return ""  
        line = line.strip()
        if line:
            return line

# -------------------------
# Parsing helpers
# -------------------------
def read_header() -> Tuple[int, int]:
    s = read_nonempty()
    if not s:
        return 0, 0
    n_str, m_str = s.split()
    return int(n_str), int(m_str)

def parse_train_lines(n: int) -> Tuple[List[Dict[int, float]], List[int]]:
    X_dict, y = [], []
    for _ in range(n):
        parts = read_nonempty().split()
        # parts: [id, label, "k:v", ...]
        label = int(parts[1])
        feats = {int(k): float(v) for k, v in (p.split(":", 1) for p in parts[2:])}
        y.append(label)
        X_dict.append(feats)
    return X_dict, y

def read_q() -> int:
    s = read_nonempty()
    return int(s) if s else 0

def parse_test_lines(q: int) -> Tuple[List[str], List[Dict[int, float]]]:
    ids, X_dict = [], []
    for _ in range(q):
        parts = read_nonempty().split()
        ids.append(parts[0])
        feats = {int(k): float(v) for k, v in (p.split(":", 1) for p in parts[1:])}
        X_dict.append(feats)
    return ids, X_dict

# -------------------------
# Model train/predict 
# -------------------------
def train_model(X_dict: List[Dict[int, float]], y: List[int]) -> Tuple[DictVectorizer, RandomForestClassifier]:
    vec = DictVectorizer()                 
    X = vec.fit_transform(X_dict).toarray()  
    clf = RandomForestClassifier()         
    clf.fit(X, y)
    return vec, clf

def predict(vec: DictVectorizer, clf: RandomForestClassifier,
            X_dict: List[Dict[int, float]]) -> List[int]:
    X = vec.transform(X_dict).toarray()    # densify to match training
    return clf.predict(X).tolist()

# -------------------------
# Main
# -------------------------
def main() -> None:
    n, m = read_header()
    if n <= 0:
        return

    X_train_dict, y_train = parse_train_lines(n)
    vec, clf = train_model(X_train_dict, y_train)

    q = read_q()
    ids, X_test_dict = parse_test_lines(q)
    preds = predict(vec, clf, X_test_dict)

    for ans_id, pred in zip(ids, preds):
        print(f"{ans_id} {pred:+d}")

if __name__ == "__main__":
    main()
