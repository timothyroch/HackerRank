"""

 Count the Faces — HackerRank

 ------------------------------------------------------------

 Problem summary (from the challenge):
 You are given a single image of a meeting or group photo and must
 output one integer: the number of human faces in the image.
 HackerRank supplies the image as a text grid via STDIN:
 - First line: R C  (rows, cols)
 - Next R lines: each line has C tokens, each token is "B,G,R"
 with 0..255 integers for Blue, Green, Red
 Output: a single integer (face count). Test cases are up to \~15MB.



 My approach (what I ended up shipping):

 1) A fast, reliable baseline for clean photos:
 - Convert BGR -> grayscale
 - Histogram equalization (equalizeHist) for lighting robustness
 - OpenCV Haar frontal face cascade (frontalface\_default)
 - minSize is set relative to the image size so tiny noise is ignored

 2) A careful fallback for hard photos (small, blurry, low light, side faces):
 - If the image is small, upsample with bicubic interpolation
 - Build stronger gray variants (CLAHE + adaptive gamma; light Gaussian)
 - Use an additional frontal cascade (frontalface\_alt2)
 - Run a profile cascade on the original and a horizontally flipped image
 - Merge all detections with IoU-based NMS and center-based de-duplication

 3) Gray-aware behavior:
 - If the photo is low-chroma (near grayscale), skip skin-color checks,
 upsample a bit more, and relax minSize/minNeighbors a notch.

 4) Hybrid policy:
 - On crisp, high-resolution color images I trust the baseline and only add
 non-overlapping boxes from the robust pass
 - On small/blur/gray images I trust the robust pass and only add
 non-overlapping boxes from the baseline
 - Final boxes are re-merged (NMS + center de-dup) before counting



 Why I chose this:

 - The “innocent” single-cascade version performed best on large, sharp photos
but missed faces on low-res or profile views.
 - A fully aggressive pipeline found more faces on hard images but tended to
 overcount on the easy ones (duplicate hits from multiple cascades).
 - The hybrid policy keeps the strengths of both: it preserves the accuracy of
 the simple path on good images and only pulls in extra evidence when the
 scene is challenging.



 Notes on parameters:

 - scaleFactor and minNeighbors are kept conservative in the first pass and
 relaxed only if needed to recover missed faces.
 - minSize scales with the shorter image side; for tiny images I upsample first.
 - NMS IoU around 0.6–0.65 plus center de-dup (about 0.45–0.50 of box size)
 is a good balance to remove near-duplicates without deleting true neighbors.
 - For profile faces I also run on the flipped image so left/right profiles are
 handled with the same cascade.



 How I tested iteratively:

 - First validated the baseline detector on high-res examples (worked best).
 - Observed systematic misses on low-res/blur/profile images.
 - Added the robust pass and saw overcounting from duplicate rectangles.
 - Introduced NMS + center de-dup, then the “merge only non-overlapping extras”
 rule to stabilize counts across both easy and hard photos.



 Usage:

 - Locally: run with a normal image path or pipe the HackerRank grid to STDIN.


"""

import sys, os, math
from typing import List, Tuple, Optional
import numpy as np
import cv2

Box = Tuple[int,int,int,int]

# ---------------- IO ----------------
def read_bgr_from_stdin() -> Optional[np.ndarray]:
    # Parse HackerRank's "rows cols + B,G,R grid" text input into a NumPy BGR image
    first = sys.stdin.readline().strip()
    if not first:
        return None
    rows, cols = map(int, first.split())
    try:
        data = np.empty((rows, cols, 3), dtype=np.uint8)
        for r in range(rows):
            parts = sys.stdin.readline().strip().split()
            row_arr = np.fromiter((int(c) for tok in parts for c in tok.split(",")),
                                  dtype=np.int32, count=cols*3).reshape(cols,3)
            data[r] = row_arr.astype(np.uint8)
        return data
    except Exception:
        buf = sys.stdin.read()
        vals = buf.replace(",", " ").split()
        need = rows*cols*3
        if len(vals) < need:
            vals += ["0"]*(need-len(vals))
        flat = np.fromiter((int(x) for x in vals[:need]), dtype=np.uint16, count=need)
        return flat.reshape(rows, cols, 3).astype(np.uint8)

# ---------------- utils ----------------
def _nms(boxes: List[Box], iou: float=0.65) -> List[Box]:
    # Apply Non-Maximum Suppression (NMS) to remove overlapping face boxes
    if not boxes: return []
    b = np.array(boxes, dtype=np.float32)
    x1,y1 = b[:,0], b[:,1]
    x2,y2 = x1+b[:,2], y1+b[:,3]
    a = (x2-x1+1)*(y2-y1+1)
    order = a.argsort()[::-1]
    keep=[]
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iouv = inter/(a[i] + a[order[1:]] - inter + 1e-9)
        order = order[np.where(iouv<=iou)[0]+1]
    return b[keep].astype(np.int32).tolist()

def _dedup_centers(boxes: List[Box], frac: float=0.50) -> List[Box]:
    # merges overlapping detections with close centers:
    if len(boxes)<=1: return boxes
    centers = [(x+0.5*w, y+0.5*h, max(w,h)) for x,y,w,h in boxes]
    used = [False]*len(boxes); keep=[]
    for i in range(len(boxes)):
        if used[i]: continue
        used[i]=True
        cx,cy,s = centers[i]
        for j in range(i+1,len(boxes)):
            if used[j]: continue
            cx2,cy2,s2 = centers[j]
            thr = frac*max(s,s2)
            if abs(cx-cx2)<=thr and abs(cy-cy2)<=thr:
                used[j]=True
        keep.append(i)
    return [boxes[i] for i in keep]

def _lap_var(gray: np.ndarray) -> float:
    # blur detection via Laplacian variance
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _is_low_chroma(img_bgr: np.ndarray) -> bool:
    # check if image is nearly grayscale (low color saturation)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[...,1].astype(np.float32)/255.0
    return float(np.median(s)) < 0.08

def _clahe(gray: np.ndarray) -> np.ndarray:
    # local contrast enhancement (CLAHE)
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8)).apply(gray)

def _gamma(gray: np.ndarray) -> np.ndarray:
    # gamma correction based on brightness
    m = float(np.mean(gray))/255.0
    g = 1.0 if m<=1e-6 else max(0.5, min(2.2, math.log(0.5)/math.log(m)))
    lut = np.array([min(255, int((i/255.0)**g*255.0 + 0.5)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, lut)

def _unsharp(gray: np.ndarray, sigma: float=1.0, amount: float=1.2) -> np.ndarray:
    # sharpening via unsharp masking
    blur = cv2.GaussianBlur(gray, (0,0), sigma)
    out = cv2.addWeighted(gray, 1.0+amount, blur, -amount, 0)
    return out

def _skin_fraction(img_bgr: np.ndarray, box: Box) -> float:
    # estimate fraction of skin pixels inside a candidate box
    x,y,w,h = box
    H,W = img_bgr.shape[:2]
    x0 = max(0, x + int(0.08*w)); y0 = max(0, y + int(0.08*h))
    x1 = min(W, x + w - int(0.08*w)); y1 = min(H, y + h - int(0.08*h))
    if x1 <= x0 or y1 <= y0: return 0.0
    roi = img_bgr[y0:y1, x0:x1]
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb = YCrCb[...,0], YCrCb[...,1], YCrCb[...,2]
    m = (Cr>=133)&(Cr<=173)&(Cb>=77)&(Cb<=127)&(Y>20)&(Y<245)
    return float(np.count_nonzero(m))/float(m.size)

def _filter_faces(img_bgr: np.ndarray, boxes: List[Box], use_skin: bool) -> List[Box]:
    # discard boxes that are too small, oddly shaped, or lack skin
    if not boxes: return []
    H,W = img_bgr.shape[:2]
    min_side = max(1, min(H,W))
    out=[]
    for (x,y,w,h) in boxes:
        if w < 0.02*min_side or h < 0.02*min_side: continue
        ar = h/float(w+1e-6)
        if ar < 0.65 or ar > 1.8: continue
        if use_skin and _skin_fraction(img_bgr, (x,y,w,h)) < 0.15: continue
        out.append((x,y,w,h))
    return out

# ---------------- cascades ----------------
def _load(name: str) -> Optional[cv2.CascadeClassifier]:
    # Load a Haar cascade by name, ensuring the file exists and the classifier is valid.  
    base = getattr(cv2.data, "haarcascades", "")
    path = os.path.join(base, name)
    if not (base and os.path.exists(path)): return None
    c = cv2.CascadeClassifier(path)
    return c if not c.empty() else None

def _c_frontal_default(): return _load("haarcascade_frontalface_default.xml")
def _c_frontal_alt2():   return _load("haarcascade_frontalface_alt2.xml")
def _c_profile():        return _load("haarcascade_profileface.xml")

# ---------------- detectors ----------------
def detect_baseline(img_bgr: np.ndarray, gray_mode: bool) -> List[Box]:
    # Run a baseline Haar cascade face detection on the image (with optional gray-mode tweaks for blurry/dark inputs).  
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if gray_mode:
        gray = _unsharp(gray, 1.0, 0.8)
    H,W = gray.shape
    min_side = min(H,W)
    min_size = max(20, min_side//24 if not gray_mode else min_side//32)
    cas = _c_frontal_default()
    if cas is None: return []
    det = cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=(5 if gray_mode else 4),
                               minSize=(min_size, min_size))
    return det.tolist() if det is not None and len(det)>0 else []

def detect_robust(img_bgr: np.ndarray, gray_mode: bool) -> List[Box]:
    # More robust face detector: resizes small images, enhances contrast if needed, and runs multiple cascades (frontal/profile) with strict & loose settings.  
    H0,W0 = img_bgr.shape[:2]
    min_side = min(H0,W0)
    scale = 1.0
    if min_side < 300: scale = 1.8
    if min_side < 180: scale = 2.4
    if scale != 1.0:
        img = cv2.resize(img_bgr, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_CUBIC)
    else:
        img = img_bgr

    g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(g0))
    if gray_mode or mean < 70 or mean > 185:
        g = _unsharp(_gamma(_clahe(g0)), 1.2, 1.0)
    else:
        g = cv2.GaussianBlur(g0, (3,3), 0)

    H,W = g.shape
    ms = max(12, int(min(H,W)*(0.025 if gray_mode else 0.03)))
    strict = dict(scaleFactor=1.05, minNeighbors=(5 if gray_mode else 6), minSize=(ms,ms))
    loose  = dict(scaleFactor=1.03, minNeighbors=(2 if gray_mode else 3),
                  minSize=(max(10,int(ms*0.85)),)*2)

    c_f1, c_f2, c_p = _c_frontal_alt2(), _c_frontal_default(), _c_profile()
    def run(params):
        # Helper routine for `detect_robust`: runs frontal & profile cascades (with mirrored profiles too), falls back to looser params if needed, rescales boxes back, then applies NMS + deduplication to clean up results.  
        cur=[]
        for c in (c_f1, c_f2):
            if c is None: continue
            d = c.detectMultiScale(g, **params)
            if d is not None and len(d)>0: cur += d.tolist()
        if c_p is not None:
            d = c_p.detectMultiScale(g, **params)
            if d is not None and len(d)>0: cur += d.tolist()
            gf = cv2.flip(g,1)
            df = c_p.detectMultiScale(gf, **params)
            if df is not None and len(df)>0:
                for x,y,w0,h0 in df.tolist():
                    cur.append((W-x-w0, y, w0, h0))
        return cur

    boxes = run(strict)
    if not boxes: boxes = run(loose)
    if not boxes: return []

    if scale != 1.0:
        boxes = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for x,y,w,h in boxes]
    boxes = _nms(boxes, iou=0.65)
    boxes = _dedup_centers(boxes, frac=0.50)
    return boxes

# ---------------- hybrid policy ----------------
def count_faces(img_bgr: np.ndarray) -> int:
    if img_bgr is None or img_bgr.size==0: return 0
    gray_mode = _is_low_chroma(img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = _lap_var(gray)
    min_side = min(img_bgr.shape[:2])
    high_quality = (not gray_mode) and (blur >= 100.0) and (min_side >= 450)

    base = detect_baseline(img_bgr, gray_mode)
    robust = detect_robust(img_bgr, gray_mode)

    # merge policy:
    # - on high-quality color: trust baseline; add only boxes not overlapping it
    # - on gray/low-quality: trust robust; add baseline only if non-overlapping
    primary = base if high_quality else robust
    secondary = robust if high_quality else base

    primary = _dedup_centers(_nms(primary, iou=0.65), frac=0.50)
    extras = []
    for b in secondary:
        if all(_iou(b, a) < 0.30 for a in primary):
            extras.append(b)

    use_skin = (not gray_mode)
    merged = _filter_faces(img_bgr, primary + extras, use_skin)
    merged = _dedup_centers(_nms(merged, iou=0.65), frac=0.50)
    return len(merged)

# small helper for IoU in merge step
def _iou(a: Box, b: Box) -> float:
    ax1, ay1, aw, ah = a; ax2, ay2 = ax1+aw, ay1+ah
    bx1, by1, bw, bh = b; bx2, by2 = bx1+bw, by1+bh
    xx1, yy1 = max(ax1,bx1), max(ay1,by1)
    xx2, yy2 = min(ax2,bx2), min(ay2,by2)
    w, h = max(0, xx2-xx1), max(0, yy2-yy1)
    inter = w*h
    if inter<=0: return 0.0
    return inter/float(aw*ah + bw*bh - inter + 1e-9)

# ---------------- main ----------------
if __name__ == "__main__":
    img = None
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    if img is None:
        img = read_bgr_from_stdin()
    print(count_faces(img))
