# precompute_longest_segment.py
import os, glob, re, argparse
import numpy as np
import cv2

def natural_key(s):
    m = re.search(r"(\d+)", os.path.basename(s))
    return int(m.group(1)) if m else s

def load_landmarks(lms_dir):
    lms_files = sorted(glob.glob(os.path.join(lms_dir, "*.lms")), key=natural_key)
    if not lms_files:
        raise FileNotFoundError(f"No .lms files found in {lms_dir}")
    L = []
    for f in lms_files:
        pts = np.loadtxt(f, dtype=np.float32)  # shape [K,2]
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Bad landmark file: {f}")
        L.append(pts)
    L = np.stack(L, 0)  # [T,K,2]
    return L, lms_files

def detect_cuts_via_landmarks(L, pos_thresh=0.08, scale_thresh=0.12):
    """
    Cut when there is a large jump in landmark center (translation) or face scale.
    Thresholds are relative to the median face size (robust to resolution).
    """
    T, K, _ = L.shape
    centers = L.mean(1)                             # [T,2]
    diffs   = L - centers[:, None, :]               # [T,K,2]
    scale   = np.sqrt((diffs**2).sum((1,2))/K)      # [T]
    s_med   = np.median(scale) + 1e-6

    pos_jump = np.linalg.norm(np.diff(centers, axis=0), axis=1) / s_med
    scl_jump = np.abs(np.diff(scale) / (scale[:-1] + 1e-6))

    cut_idxs = np.where((pos_jump > pos_thresh) | (scl_jump > scale_thresh))[0] + 1
    bounds = np.concatenate([[0], cut_idxs, [T]])
    segments = [(int(bounds[i]), int(bounds[i+1])) for i in range(len(bounds)-1)]
    return segments, centers, scale

def pick_longest_segment(segments):
    lengths = [b-a for a,b in segments]
    if not lengths:
        raise RuntimeError("No segments produced.")
    idx = int(np.argmax(lengths))
    return segments[idx], lengths[idx]

def save_segment(dataset_dir, seg):
    a, b = seg
    seg_idx = np.arange(a, b, dtype=np.int32)   # [a, a+1, ..., b-1]
    out_npy = os.path.join(dataset_dir, "longest_segment.npy")
    np.save(out_npy, seg_idx)
    print(f"[OK] Saved longest segment indices ({len(seg_idx)} frames): {out_npy}")
    print(f"      span: [{a}, {b-1}]")

def maybe_check_images(dataset_dir, img_dirname="full_body_img", seg=None):
    # Optional sanity check: ensure images exist for the span (fast)
    if seg is None: return
    img_dir = os.path.join(dataset_dir, img_dirname)
    if not os.path.isdir(img_dir): return
    a, b = seg
    missing = []
    for i in (a, a+(b-a)//2, b-1):
        fp = os.path.join(img_dir, f"{i}.jpg")
        if not os.path.exists(fp): missing.append(fp)
    if missing:
        print("[WARN] Some expected images not found:", *missing, sep="\n       ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset dir used by inference.py")
    ap.add_argument("--lms_dirname", default="landmarks", help="Dir name of landmark files inside dataset")
    ap.add_argument("--pos_thresh", type=float, default=0.08, help="Relative center jump to call a cut")
    ap.add_argument("--scale_thresh", type=float, default=0.12, help="Relative scale jump to call a cut")
    args = ap.parse_args()

    lms_dir = os.path.join(args.dataset, args.lms_dirname)
    L, _ = load_landmarks(lms_dir)
    segments, _, _ = detect_cuts_via_landmarks(L, args.pos_thresh, args.scale_thresh)
    longest, Llen = pick_longest_segment(segments)
    print(f"[INFO] Found {len(segments)} segments. Longest length: {Llen}")
    save_segment(args.dataset, longest)
    maybe_check_images(args.dataset, seg=longest)

if __name__ == "__main__":
    main()
