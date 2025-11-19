# precompute_bg_path.py
import os, glob, numpy as np, cv2

def load_landmarks(lms_dir):
    files = sorted(glob.glob(os.path.join(lms_dir, "*.lms")),
                   key=lambda p: int(os.path.basename(p).split('.')[0]))
    all_pts = []
    for p in files:
        pts = np.loadtxt(p, dtype=np.float32)  # (K,2)
        all_pts.append(pts)
    L = np.stack(all_pts, 0)  # [T, K, 2]
    return L, files

def descriptors_from_landmarks(L, img_h, img_w, pca_dims=8):
    T, K, _ = L.shape
    centers = L.mean(1)                            # [T,2]
    diffs = L - centers[:, None, :]               # [T,K,2]
    scale = np.sqrt((diffs**2).sum((1,2))/K)      # RMS size per frame [T]

    # normalize shape for pose/shape descriptor
    norm = diffs / (scale[:, None, None] + 1e-6)  # [T,K,2]

    # in-plane angle via per-frame PCA of landmarks
    angles = np.empty(T, np.float32)
    for t in range(T):
        X = norm[t].reshape(-1, 2)
        C = np.cov(X.T)
        w, v = np.linalg.eig(C)
        v1 = v[:, np.argmax(w)]
        angles[t] = np.arctan2(v1[1], v1[0])      # [-pi, pi]

    # low-dim shape embedding via PCA (SVD) on flattened normalized shapes
    X = norm.reshape(T, -1)
    mu = X.mean(0, keepdims=True)
    X0 = X - mu
    U, S, Vt = np.linalg.svd(X0, full_matrices=False)
    Vp = Vt[:pca_dims]                             # [pca_dims, D]
    Z = X0 @ Vp.T                                  # [T, pca_dims]

    # translation normalized by image size
    cxy = centers / np.array([img_w, img_h], dtype=np.float32)[None, :]

    # final descriptor = [cosθ, sinθ, Z..., cx, cy]
    desc = np.concatenate(
        [np.cos(angles)[:, None], np.sin(angles)[:, None], Z, cxy], axis=1
    ).astype(np.float32)
    return desc, centers, scale

def segment_by_cuts(centers, scale, pos_thresh=0.04, scale_thresh=0.06):
    # large jumps in center/scale → shot cuts (thresholds are relative to median scale)
    T = centers.shape[0]
    s_med = np.median(scale) + 1e-6
    pos_jump = np.linalg.norm(np.diff(centers, axis=0), axis=1) / s_med
    scl_jump = np.abs(np.diff(scale)/ (scale[:-1] + 1e-6))
    cut_idxs = np.where((pos_jump > pos_thresh) | (scl_jump > scale_thresh))[0] + 1
    bounds = [0] + cut_idxs.tolist() + [T]
    segments = [np.arange(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]
    return segments

def link_segments_min_cost(segments, desc, w_pose=1.0, w_trans=1.0):
    # pose dims = 2 + pca_dims ; trans dims = last 2 dims
    D = desc.shape[1]
    pose_slice = slice(0, D-2)
    trans_slice = slice(D-2, D)
    w = np.ones(D, dtype=np.float32)
    w[pose_slice] *= w_pose
    w[trans_slice] *= w_trans

    path = []
    last = None
    for seg in segments:
        if len(seg) == 0:
            continue
        if last is None:
            path.extend(seg.tolist())
            last = seg[-1]
            continue
        # choose rotation of this segment that best matches previous tail
        d_last = desc[last]
        costs = []
        for j in range(len(seg)):
            d0 = desc[seg[j]]
            c = np.sqrt(((w * (d0 - d_last))**2).sum())
            costs.append(c)
        jbest = int(np.argmin(costs))
        srot = np.concatenate([seg[jbest:], seg[:jbest]])
        path.extend(srot.tolist())
        last = path[-1]

    # optional: freeze a few frames at the global seam to hide residual jump
    seam_cost = np.sqrt(((w * (desc[path[0]] - desc[path[-1]]))**2).sum())
    if seam_cost > 0.5:  # tuneable; duplicates add no runtime cost
        path = path + [path[-1]] * 3
    return np.array(path, dtype=np.int32)

def main(dataset_dir):
    img_dir = os.path.join(dataset_dir, "full_body_img")
    lms_dir = os.path.join(dataset_dir, "landmarks")
    exm = cv2.imread(os.path.join(img_dir, "0.jpg"))
    h, w = exm.shape[:2]

    L, _ = load_landmarks(lms_dir)
    desc, centers, scale = descriptors_from_landmarks(L, h, w)
    segments = segment_by_cuts(centers, scale)
    bg_path = link_segments_min_cost(segments, desc)

    np.save(os.path.join(dataset_dir, "bg_path.npy"), bg_path)
    print(f"[OK] Saved bg_path with {len(bg_path)} indices to {os.path.join(dataset_dir,'bg_path.npy')}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="same folder you pass to --dataset in inference")
    args = ap.parse_args()
    main(args.dataset)
