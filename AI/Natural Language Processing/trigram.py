"""
HackerRank — Most Frequent Trigram
----------------------------------

Problem:
    Given a large chunk of text, identify the most frequently occurring trigram.
    A trigram is a sequence of three consecutive words within the same sentence.
    Words are case-insensitive, separated by single spaces, and sentences end
    with a period ('.'). Trigrams cannot cross sentence boundaries.

    If multiple trigrams have the same maximum frequency, return the one that
    appears first in the text. The output should be in lowercase.

Constraints:
    - Input size < 10 KB
    - Input contains only letters, spaces, and periods
    - Output: most popular trigram in lowercase, words separated by single spaces

Examples:
    Input:
        I came from the moon. He went to the other room. She went to the drawing room.
    Output:
        went to the

    Input:
        I love to dance. I like to dance I. like to play chess.
    Output:
        i love to
"""

import sys

def most_frequent_trigram(text: str) -> str:
    text = text.lower()
    counts = {}
    best_count = 0
    best_trigram = ""

    # Split sentences on '.' and extract trigrams within each
    for sent in text.split('.'):
        words = sent.split()
        for i in range(len(words) - 2):
            tri = ' '.join(words[i:i+3])
            counts[tri] = counts.get(tri, 0) + 1
            # Update best trigram if higher count, or same count but seen earlier
            if counts[tri] > best_count:
                best_count = counts[tri]
                best_trigram = tri

    return best_trigram

if __name__ == "__main__":
    text = sys.stdin.read()
    print(most_frequent_trigram(text))
