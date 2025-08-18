# Hashtag Segmentation (HackerRank Solution)

## Problem Statement

This project is my solution to a HackerRank problem:

> **Task:**  
> Given a set of Twitter hashtags, split each hashtag into its constituent words.  
>  
> Example:  
> - `wearethepeople` -> `we are the people`  
> - `mentionyourfaves` -> `mention your faves`  
> - `nowplaying` -> `now playing`  
> - `thewalkingdead` -> `the walking dead`  
> - `followme` -> `follow me`  

### Input Format
- The first line contains an integer `N`, the number of hashtags.  
- Each of the next `N` lines contains one hashtag.  

### Output Format
- For each hashtag, print the segmented version as space-separated words.

### Constraints
- Hashtags will not contain named entities other than country names and abbreviations (e.g., US, UK, UAE).  
- Hashtags may occasionally contain slang (e.g., *faves* for *favorites*).  
- Scoring depends on the percentage of hashtags segmented correctly.  

---

## My Solution

I implemented a **dynamic programming word segmentation algorithm**:

- I treat segmentation as a **shortest path problem**, where each candidate word in a split has a cost.  
- Common words have low cost, rare/unknown words have high cost.  
- I use a **dictionary built from real text** to assign probabilities.  
- For each hashtag, my code tries all possible splits and picks the lowest-cost segmentation.  

### Key Features
- Handles numbers (`iphone14` -> `iphone 14`).  
- Recognizes country abbreviations (`UK`, `US`).  
- Penalizes unknown words but still allows slang when necessary.  
- Runs fast enough for thousands of hashtags.  

---

## Data Source

This algorithm needs a **dictionary of words** to work. Instead of manually embedding a list, I generated one from **Project Gutenberg**, which is free and public domain.  

1. I downloaded a large book (e.g., *Pride and Prejudice* by Jane Austen):  
   https://www.gutenberg.org/files/1342/1342-0.txt  

2. I built a helper script `build_wordlist.py` that:  
   - Extracts all alphabetic words from the text.  
   - Counts frequencies.  
   - Outputs a list of the most common ~50,000 words.  

3. The segmentation script (`segment_hashtags.py`) loads this list (`wordlist.txt`) and uses word rank as a probability model.  

This way, my code adapts to real English usage without requiring any special training data.

---

## Usage

1. **Generate the wordlist** (once):
   ```bash
   curl -s https://www.gutenberg.org/files/1342/1342-0.txt -o pride_prejudice.txt
   python3 build_wordlist.py < pride_prejudice.txt > wordlist.txt
   ```

2. **Run segmentation**:

   ```bash
   python3 segment_hashtags.py < input.txt > output.txt
   ```

### Example

**input.txt**

```
5
wearethepeople
mentionyourfaves
nowplaying
thewalkingdead
followme
```

> **leads to** 


**output.txt**

```
we are the people
mention your faves
now playing
the walking dead
follow me
```

---

## Files

* `segment_hashtags.py` -> main solution (reads input, segments hashtags).
* `build_wordlist.py` -> generate `wordlist.txt` from any Gutenberg text.

---

## Notes

* This approach balances **accuracy** with **simplicity**: it doesn’t require heavy machine learning models, just text frequency analysis.
* You can swap in different Gutenberg texts or even combine multiple books to expand the vocabulary.
* The better the wordlist, the better the segmentation performance.

---
