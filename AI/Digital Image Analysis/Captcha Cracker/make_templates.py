import sys
import os
import numpy as np
from typing import Dict, List, Tuple


# ---------- parsing & binarization ----------
def read_bgr_txt_path(path: str) -> np.ndarray:
    """
    Read an image stored in the challenge's plain-text BGR format.

    File format:
      - First line: "R C" (rows, cols)
      - Next R lines: C tokens "B,G,R" separated by spaces

    Args:
        path: Path to the text file.

    Returns:
        Image as a NumPy array of shape (R, C, 3) with dtype=uint8 (B,G,R order).
    """
    with open(path, "r") as fp:
        R, C = map(int, fp.readline().split())
        img = np.zeros((R, C, 3), dtype=np.uint8)
        for r in range(R):
            row = fp.readline().split()
            for c, pix in enumerate(row):
                b, g, r_ = map(int, pix.split(','))
                img[r, c] = [b, g, r_]
    return img


def to_binary(img: np.ndarray) -> np.ndarray:
    """
    Convert color image to a binary mask where 1 indicates "ink" (dark).

    Args:
        img: (R, C, 3) uint8 array in B,G,R order.

    Returns:
        Binary (R, C) uint8 array with values {0,1}.
    """
    # Standard luminance weights for B,G,R
    gray = (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(np.uint8)
    # Use mean-based threshold but not lower than 128
    thr = max(128, int(gray.mean()))
    return (gray < thr).astype(np.uint8)  # 1 = ink


# ---------- vertical banding ----------
def smooth_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Smooth a 1D signal with a length-k uniform kernel.

    Args:
        x: 1D array to smooth.
        k: Kernel size; if <=1 returns x unchanged.

    Returns:
        Smoothed 1D array (same length as x).
    """
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode='same')


def find_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find inclusive index runs of consecutive 1's in a binary 1D mask.

    Args:
        mask: 1D array of {0,1}.

    Returns:
        List of (start, end) index pairs (inclusive).
    """
    runs: List[Tuple[int, int]] = []
    on = False
    s = 0
    for i, v in enumerate(mask):
        if v and not on:
            on, s = True, i
        if not v and on:
            on = False
            runs.append((s, i - 1))
    if on:
        runs.append((s, len(mask) - 1))
    return runs


def merge_close_runs(runs: List[Tuple[int, int]], max_gap: int = 2) -> List[Tuple[int, int]]:
    """
    Merge runs if the gap between them is <= max_gap.

    Args:
        runs: List of (start, end) runs (inclusive).
        max_gap: Maximum allowed zero-gap between runs to merge.

    Returns:
        Merged runs as a list of (start, end).
    """
    if not runs:
        return []
    out: List[Tuple[int, int]] = [runs[0]]
    for a, b in runs[1:]:
        A, B = out[-1]
        if a - B - 1 <= max_gap:
            out[-1] = (A, b)
        else:
            out.append((a, b))
    return out


def guess_char_bands(bw: np.ndarray) -> List[Tuple[int, int]]:
    """
    Guess the 5 vertical character bands in a binary captcha image.

    Approach:
      - Sum ink per column, smooth, and threshold to detect character areas.
      - Merge close runs and try to get exactly 5 bands.
      - If not 5, fall back to equal-width slicing.
      - Tighten each band by trimming empty columns on both sides.

    Args:
        bw: Binary image (R, C) with values {0,1}.

    Returns:
        List of 5 (left, right) inclusive column indices.
    """
    col_sum = bw.sum(axis=0)
    s = smooth_1d(col_sum, 5)
    thr = max(1, int(0.20 * s.max()))
    mask = (s >= thr).astype(np.uint8)
    runs = merge_close_runs(find_runs(mask), 2)

    if len(runs) != 5:
        # Fallback: equal slices with small overlaps
        C = bw.shape[1]
        step = C // 5
        runs = [
            (max(0, i * step - 1), C - 1 if i == 4 else min(C - 1, (i + 1) * step + 1))
            for i in range(5)
        ]

    # Tighten band boundaries by trimming leading/trailing zero columns
    tight: List[Tuple[int, int]] = []
    for L, R in runs:
        seg = col_sum[L:R + 1]
        if seg.sum() == 0:
            tight.append((L, R))
            continue
        l = 0
        while l < len(seg) and seg[l] == 0:
            l += 1
        r = 0
        while r < len(seg) and seg[-1 - r] == 0:
            r += 1
        tight.append((L + l, R - r))
    return tight


# ---------- top/bottom crop & resizing ----------
def tight_crop(bw_band: np.ndarray) -> np.ndarray:
    """
    Remove empty rows at the top and bottom of a band.

    Args:
        bw_band: Binary image (h, w).

    Returns:
        Cropped binary image (h', w). Returns input if completely empty.
    """
    rows = bw_band.sum(axis=1)
    if rows.sum() == 0:
        return bw_band
    top = int(np.argmax(rows > 0))
    bottom = len(rows) - 1 - int(np.argmax(rows[::-1] > 0))
    return bw_band[top:bottom + 1, :]


def resize_nn(a: np.ndarray, H: int = 24, W: int = 16) -> np.ndarray:
    """
    Resize a 2D binary image to (H, W) using nearest-neighbor sampling.

    Args:
        a: Input 2D array.
        H: Target height.
        W: Target width.

    Returns:
        Resized array of shape (H, W), dtype=uint8.
    """
    h, w = a.shape
    r_idx = (np.linspace(0, h - 1, H)).astype(int)
    c_idx = (np.linspace(0, w - 1, W)).astype(int)
    return a[r_idx][:, c_idx].astype(np.uint8)


# ---------- build templates from inputXX.txt / outputXX.txt ----------
def read_label(path: str) -> str:
    """
    Read a single-line captcha label string from a file.

    Args:
        path: Path to outputNN.txt.

    Returns:
        The label string (expected length 5).
    """
    with open(path, "r") as f:
        return f.read().strip()


def main(in_dir: str = "input") -> None:
    """
    Build character templates from paired training files and print a Python snippet.

    For each inputNN.txt (image) with corresponding outputNN.txt (label),
    this script:
      - Parses and binarizes the image.
      - Segments into 5 character bands.
      - Crops and resizes each band to 24x16.
      - Buckets each patch under its character label.

    Finally, it prints a ready-to-paste snippet that defines:
      - `import numpy as np`
      - `TEMPLATES = { 'A': [np.array(...), ...], '0': [...], ... }`

    Args:
        in_dir: Directory containing inputNN.txt files (default: "input").
                Corresponding labels are read from "output/outputNN.txt".
    """
    # Gather pairs inputNN.txt + outputNN.txt
    names = sorted([f for f in os.listdir(in_dir) if f.startswith("input") and f.endswith(".txt")])
    T: Dict[str, List[np.ndarray]] = {}  # char -> list of (24,16) patches

    for inf in names:
        # Map inputNN.txt -> outputNN.txt
        base = inf.replace("input", "output")
        out_path = os.path.join("output", base)
        if not os.path.exists(out_path):
            # Skip if label is missing
            continue

        label = read_label(out_path)
        img = read_bgr_txt_path(os.path.join(in_dir, inf))
        bw = to_binary(img)
        bands = guess_char_bands(bw)

        if len(bands) != 5:
            # Fallback: equal slices
            C = bw.shape[1]
            step = C // 5
            bands = [
                (max(0, i * step - 1), C - 1 if i == 4 else min(C - 1, (i + 1) * step + 1))
                for i in range(5)
            ]

        # Build (24,16) template patches and bucket by char
        for i, (L, R) in enumerate(bands):
            ch = label[i]
            band = bw[:, L:R + 1]
            band = tight_crop(band)
            patch = resize_nn(band, 24, 16)
            T.setdefault(ch, []).append(patch)

    # Print a paste-ready Python snippet:
    print("import numpy as np")
    print("TEMPLATES = {")
    for ch, lst in sorted(T.items()):
        print(f"    {ch!r}: [")
        for arr in lst:
            # Emit patch as a literal list of 0/1 rows for portability
            print("        np.array([")
            for row in arr:
                print("            " + repr(row.tolist()) + ",")
            print("        ], dtype=np.uint8),")
        print("    ],")
    print("}")

    # Also a tiny note on stderr (not part of the snippet)
    sys.stderr.write(f"Built templates for {len(T)} characters.\n")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "input"
    main(d)
