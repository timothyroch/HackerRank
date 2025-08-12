"""
Craigslist Category Classifier — Problem & Approach Notes
---------------------------------------------------------

Problem (HackerRank):
- Task: Predict the Craigslist category for each post, given its city, section, and heading.

- Background:
  Craigslist has 9 major sections (jobs, resumes, gigs, personals, housing, community, services,
  for-sale, discussion forums). In this challenge we focus on 4 sections:
    * for-sale
    * housing
    * community
    * services

  We classify posts into 16 categories (each category maps to exactly one section):
    activities, appliances, artists, automotive, cell-phones, childcare, general,
    household-services, housing, photography, real-estate, shared, temporary,
    therapeutic, video-games, wanted-housing

- Input format (test stream via STDIN):
    Line 1: integer N (1 <= N <= 22,000)
    Next N lines: each line is a JSON object with fields:
      - "city"    (ASCII string)
      - "section" (ASCII string; one of the four sections above)
      - "heading" (UTF-8 string; post title)
    The test JSON does NOT include "category" — that’s what we must predict.

- Output format:
    Exactly N lines, each containing the predicted category for the corresponding test JSON.

- Training file:
    A file named "training.json" is present in the working directory.
    Line 1: integer T (number of training rows, ~20k).
    Next T lines: each line is a JSON object with the same fields as test, plus:
      - "category" (the ground-truth label)

My approach (why this worked for me):
1) Field-aware TF-IDF:
   - I vectorize heading, section, and city separately with TF-IDF.
   - Headings carry most signal, so I use word uni/bi-grams and a token pattern that preserves
     hyphens/dots/apostrophes (e.g., "video-games", "paris.en"). Sections also get uni/bi-grams,
     cities get unigrams.
   - I then hstack the three blocks and apply simple scalar weights to emphasize headings.

2) Linear SVM (LinearSVC):
   - Strong baseline for high-dimensional sparse text; trains and predicts quickly.
   - Robust on noisy, short headings.

Implementation outline:
- Read and parse training.json (count on first line, then T JSON lines).
- Read test JSONs from STDIN (count on first line, then N JSON lines).
- Build three TF-IDF vectorizers (heading/section/city), fit on training fields only.
- Transform both train and test with the SAME vectorizers; hstack with simple weights.
- Train LinearSVC on the training matrix; predict categories for the test matrix.
- Print predictions, one per line, in order.
"""

import sys, json
from scipy.sparse import hstack
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# Constants & tokenization
# -------------------------
TRAIN = "training.json"

# Preserve tokens like "video-games", "paris.en", "kid's"
TOKEN_PATTERN = r"(?u)[A-Za-z0-9][A-Za-z0-9\-\.\']+"

# -------------------------
# I/O helpers
# -------------------------
def read_training(path):
    """Read training data: first line is count T, followed by T JSON lines."""
    X_sec, X_head, X_city, y = [], [], [], []
    with open(path, "rb") as f:
        first = f.readline()
        if not first:
            return X_sec, X_head, X_city, y
        n = int(first.decode().strip())
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            obj = json.loads(line.decode())
            # Defensive: handle missing/null values
            X_sec.append((obj.get("section") or "").strip())
            X_head.append((obj.get("heading") or "").strip())
            X_city.append((obj.get("city") or "").strip())
            y.append(obj["category"])
    return X_sec, X_head, X_city, y

def read_test_from_stdin():
    """Read test data from STDIN: first line is N, followed by N JSON lines."""
    X_sec, X_head, X_city = [], [], []
    first = sys.stdin.buffer.readline()
    if not first:
        return X_sec, X_head, X_city
    n = int(first.decode().strip())
    for _ in range(n):
        line = sys.stdin.buffer.readline()
        if not line:
            break
        obj = json.loads(line.decode())
        X_sec.append((obj.get("section") or "").strip())
        X_head.append((obj.get("heading") or "").strip())
        X_city.append((obj.get("city") or "").strip())
    return X_sec, X_head, X_city

# -------------------------
# Main training + inference
# -------------------------
def main():
    # Load data
    tr_sec, tr_head, tr_city, y = read_training(TRAIN)
    if not y:
        return
    te_sec, te_head, te_city = read_test_from_stdin()
    if not te_sec:
        return

    # -------------------------
    # Feature extractors 
    # -------------------------
    v_head = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=1.0,
    )
    v_sec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=1.0,
    )
    v_city = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
        sublinear_tf=True,
        min_df=1,
        max_df=1.0,
    )

    # Fit on training fields only
    Xh_tr = v_head.fit_transform(tr_head)
    Xs_tr = v_sec.fit_transform(tr_sec)
    Xc_tr = v_city.fit_transform(tr_city)

    # Weighted concatenation (heading strongest, city weakest)
    X_tr = hstack([Xs_tr * 1.0, Xh_tr * 1.3, Xc_tr * 0.7], format="csr")

    # -------------------------
    # Classifier
    # -------------------------
    clf = LinearSVC(C=1.5, tol=1e-4, dual=True)
    clf.fit(X_tr, y)

    # -------------------------
    # Predict on test split
    # -------------------------
    Xh_te = v_head.transform(te_head)
    Xs_te = v_sec.transform(te_sec)
    Xc_te = v_city.transform(te_city)
    X_te = hstack([Xs_te * 1.0, Xh_te * 1.3, Xc_te * 0.7], format="csr")

    preds = clf.predict(X_te)

    # -------------------------
    # Output 
    # -------------------------
    out = sys.stdout
    for p in preds:
        print(p, file=out)


if __name__ == "__main__":
    main()
