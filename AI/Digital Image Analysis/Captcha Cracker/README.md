# Captcha Cracker – HackerRank Challenge

This project is my solution to the [**HackerRank “Captcha Cracker” problem**](https://www.hackerrank.com/challenges/the-captcha-cracker).

The challenge:

* Each captcha image is always **5 characters long** (A–Z, 0–9).
* Font, spacing, and style are fixed.
* Background and foreground colors are consistent.
* The input is provided as a **plain-text grid** of BGR pixel values (`R=30`, `C=60`).
* I must output the exact 5-character captcha text.

---

## My Approach

I took a simple **template-matching approach**:

1. **Generate templates**
   I used the 25 labeled captcha samples provided by HackerRank.

   * Each captcha is segmented into 5 bands.
   * Each character is cropped, resized to `24x16`, and stored.
   * All patches are grouped under their label, giving me a library of exemplars.
   * This step is handled by `make_templates.py`, which outputs a `TEMPLATES` dict.

2. **Recognition**
   For unseen captchas, I run `captcha_tr.py`:

   * Convert the BGR input to grayscale and then binary (ink vs. background).
   * Segment into 5 character bands using vertical projections.
   * Crop and resize each band to `24x16`.
   * Compare against all templates using **mean squared error (MSE)**.
   * The best match determines the predicted character.

This works reliably because the captcha generator is deterministic in style (no skew, same spacing, consistent colors).

---

## How to Run

### 1. Build templates from the training set

The provided inputs are in `input/` and labels in `output/`:

```bash
python make_templates.py > templates_snippet.py
```

This creates a `templates_snippet.py` file containing the `TEMPLATES` dictionary.

### 2. Recognize a captcha

Run the solver on any captcha text file:

```bash
python captcha_tr.py < input/input00.txt
```

Expected output for `input00.txt` would be the 5-character label, e.g.:

```
EGYK4
```

---

## Files

* `make_templates.py` – Builds the character template dictionary (`TEMPLATES`).
* `captcha_tr.py` – Uses the templates to solve unseen captchas.
* `templates_snippet.py` – Auto-generated Python snippet containing `TEMPLATES`.

