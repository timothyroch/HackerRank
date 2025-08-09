"""
Stack Exchange Question Classifier
----------------------------------

Problem Statement:
------------------
This program solves the "Stack Exchange Question Classifier" challenge.

Stack Exchange is a network of Q&A sites with many different topics. In this
challenge, we are given a training dataset containing questions from 10 chosen
topics (e.g., Electronics, Mathematics, Photography). Each training example
includes:
    - question: The text from the title of the question (UTF-8 string)
    - excerpt: A short excerpt from the body of the question (UTF-8 string)
    - topic: The topic label (ASCII string)

Our task is to build a model that, given only the question and excerpt, predicts
the correct topic from these 10 classes.

Input Format:
-------------
Training:
    - A file named "training.json" in the current directory.
    - First line: integer N (number of training examples).
    - Next N lines: valid JSON objects with fields:
        "question" (string), "excerpt" (string), "topic" (string)

Testing (stdin):
    - First line: integer M (number of test examples).
    - Next M lines: valid JSON objects with fields:
        "question" (string), "excerpt" (string)
      Note: topic is omitted in test data; we must predict it.


My Approach:
------------
I chose a straightforward supervised text classification pipeline using scikit-learn:

1. Data Loading
   - Read the training file ("training.json") line by line.
   - Parse each JSON object to extract the question, excerpt, and topic.
   - Maintain a mapping from topic strings to integer labels.
   - Each training example contributes two separate samples: the question text
     and the excerpt text, both assigned the same topic label. This "up-weights"
     the training data and helps the model learn from both short summaries
     and fuller descriptions.

2. Feature Extraction
   - Use TfidfVectorizer to convert text into TF-IDF-weighted n-grams.
   - Configure it for:
       * lowercasing
       * removing English stopwords
       * using unigrams and bigrams
       * sublinear term frequency scaling
       * filtering out very common terms (max_df=0.1)

3. Model Training
   - Use SGDClassifier with a linear SVM (hinge loss).
   - L2 regularization (`penalty="l2"`), small alpha for less regularization.
   - Train for a limited number of iterations to balance speed and accuracy.

4. Prediction
   - Read test data from stdin.
   - For each test example, concatenate question and excerpt into one string.
   - Transform using the fitted TfidfVectorizer.
   - Predict topic labels with the trained classifier.
   - Map predicted label indices back to topic strings and print them.

This approach is lightweight, trains quickly even on the full dataset, and
leverages TF-IDF + linear models which are well-suited for high-dimensional
sparse text classification tasks.
"""

import sys, json
from typing import List, Tuple
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def read_training(path: str) -> Tuple[List[str], List[int], List[str]]:
    X, y = [], []                                    # X = training text samples, y = topic indices
    topic_to_idx, topics = {}, []                    # mapping from topic name to index and reverse list
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        try:
            n = int((first or "0").strip())          # number of training samples
        except ValueError:
            n = 0
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            topic = j.get("topic")
            q = j.get("question", "")
            ex = j.get("excerpt", "")
            if not topic:
                continue
            # Assign numeric index to topic if new
            if topic not in topic_to_idx:
                topic_to_idx[topic] = len(topics)
                topics.append(topic)
            idx = topic_to_idx[topic]
            X.extend([q, ex])
            y.extend([idx, idx])
    return X, y, topics

def read_test_from_stdin() -> List[str]:
    data = sys.stdin.read().splitlines()
    if not data:
        return []
    try:
        n = int((data[0] or "0").strip())             # number of test samples
    except ValueError:
        n =  0
    out = []
    for i in range(1, n+1):
        if i >= len(data): break
        s = data[i].strip()
        if not s:
            out.append("")
            continue
        try:
            j  = json.loads(s)
        except json.JSONDecodeError:
            out.append("")
            continue
        out.append(j.get("question", "") +  " " + j.get("excerpt", ""))
    return out

def main():
    # Load and prepare training data
    X_train, y_train, topics = read_training("training.json")
    if not X_train:
        return
      
    # Convert text to TF-IDF features
    vec = TfidfVectorizer(
      lowercase=True,
      strip_accents="unicode",
      stop_words="english",
      ngram_range=(1, 2),
      sublinear_tf=True,
      max_df=0.1,
    )
    Xtr = vec.fit_transform(X_train)

    # Train linear classifier (SVM via SGD)
    clf = SGDClassifier(
      loss="hinge",
      alpha=2e-5,
      penalty="l2",
      max_iter=30,
      random_state=0,
      class_weight=None
    ).fit(Xtr, y_train)

    X_test = read_test_from_stdin()
    if not X_test:
        return
    Xte  = vec.transform(X_test)
    preds = clf.predict(Xte)

    for i in preds:
        print(topics[i])

if __name__ == "__main__":
    main()
