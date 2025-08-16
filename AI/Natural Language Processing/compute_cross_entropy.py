"""
HackerRank — Cross-Entropy from Perplexity
------------------------------------------

Task:
    Given a perplexity value of a bigram model, compute the
    cross-entropy (in bits per word) to two decimal places.

Notes:
    Perplexity and cross-entropy are related as:
        cross_entropy = log2(perplexity)

Example:
    Input: 170
    Output: 7.41
"""

import math

def perplexity_to_cross_entropy(perplexity: float) -> float:
    """Return cross-entropy (bits/word) from perplexity."""
    if perplexity <= 0:
        raise ValueError("Perplexity must be positive.")
    return math.log2(perplexity)

if __name__ == "__main__":
    perplexity = 170
    ce = perplexity_to_cross_entropy(perplexity)
    print(f"{ce:.2f}")
