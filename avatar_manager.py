"""
load_labels_map

save_labels_map

mint_avatar_id

ensure_default_labels

list_avatar_labels

label_to_id

_find_first_frame

get_avatar_preview
"""
import cv2 
import glob, pickle
from PIL import Image
import secrets
import json
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
from pathlib import Path
import gradio as gr

from configs import AVATAR_DIR, LABELS_PATH
from ultralight.unet import Model

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu")

# ----------------- Simple in-memory cache -----------------
_AVATAR_CACHE: Dict[str, tuple] = {}  # avatar_id -> (model, full_frames, face_frames, bboxes)

# App helpers
# ---------------------- Label map (public → private) ----------------------
def load_labels_map() -> dict:
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_labels_map(d: dict) -> None:
    AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def mint_avatar_id() -> str:
    """Opaque internal folder name; never shown to users."""
    return f"ultralight_{secrets.token_hex(4)}"  # e.g., avt_7f3a9c2b

def ensure_default_labels() -> None:
    """Optionally seed labels for any existing known avatars (if present on disk).
    This does not expose them to users beyond their label.
    """
    m = load_labels_map()
    defaults = {
        "林奇宏校長": "ultralight_principal_tsmc",
        "林御專老師": "ultralight_prof_1_6-1",
        "蘇信寧老師": "ultralight_prof_2_2-1",
        "蘇育陞老師": "ultralight_prof_3_4-2",
        "陳佳宏老師": "ultralight_prof_4",
        "曾院介老師": "ultralight_prof_5",
    }
    changed = False
    for lbl, internal_id in defaults.items():
        if (AVATAR_DIR / internal_id).exists() and lbl not in m:
            m[lbl] = internal_id
            changed = True
    if changed:
        save_labels_map(m)


def list_avatar_labels() -> List[str]:
    """Only show labels that are mapped and actually exist on disk."""
    m = load_labels_map()
    labels = []
    for lbl, internal_id in m.items():
        if (AVATAR_DIR / internal_id).exists():
            labels.append(lbl)
    # labels.sort()
    return labels


def label_to_id(label: str) -> str:
    m = load_labels_map()
    if label not in m:
        raise gr.Error("Unknown avatar label.")
    return m[label]

# Seed mapping for any already-installed avatars
ensure_default_labels()

# ---------------------- Avatar preview helpers ----------------------

def _find_first_frame(internal_id: str) -> Optional[Path]:
    """Return path to the first frame image for the avatar, if any.
    Prefer data/avatars/<id>/full_imgs/00000000.png, otherwise pick the first .png/.jpg sorted.
    """
    base = AVATAR_DIR / internal_id / "full_imgs"
    if not base.exists():
        return None
    # Preferred canonical name
    candidate = base / "00000000.png"
    if candidate.exists():
        return candidate
    # Fallback: first png/jpg by name
    pics = sorted([*base.glob("*.png"), *base.glob("*.jpg"), *base.glob("*.jpeg")])
    return pics[0] if pics else None


def get_avatar_preview(label: str) -> Optional[Image.Image]:
    """Load preview as a PIL Image to avoid leaking internal file paths via URLs."""
    if not label:
        return None
    try:
        internal_id = label_to_id(label)
    except gr.Error:
        return None
    path = _find_first_frame(internal_id)
    if not path or not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


# Processing helpers
def _read_imgs_sorted(folder: Path) -> List[np.ndarray]:
    # Accept common image extensions; sorted numerically by stem if possible.
    exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.webp")
    paths = []
    for patt in exts:
        paths.extend(glob.glob(str(folder / patt)))
    def stem_num(p):
        try:
            return int(Path(p).stem)
        except Exception:
            return 10**9  # non-numeric last
    paths = sorted(paths, key=stem_num)
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: 
            continue
        frames.append(img)
    if not frames:
        raise FileNotFoundError(f"No images found in {folder}")
    return frames

# ----------------- Avatar loading -----------------
def load_avatar_ultralight(avatar_id: str) -> Tuple[torch.nn.Module, List[np.ndarray], List[np.ndarray], List[List[int]]]:
    print(f"Loading avatar: {avatar_id}")
    root = Path("data")/"avatars"/avatar_id
    full_dir = root/"full_imgs"
    face_dir = root/"face_imgs"
    coords_pkl = root/"coords.pkl"
    weight_pth = root/"ultralight.pth"

    if not weight_pth.exists():
        raise FileNotFoundError(f"Missing weights: {weight_pth}")
    if not coords_pkl.exists():
        raise FileNotFoundError(f"Missing coords: {coords_pkl}")
    if not full_dir.exists():
        raise FileNotFoundError(f"Missing dir: {full_dir}")
    if not face_dir.exists():
        raise FileNotFoundError(f"Missing dir: {face_dir}")

    with open(coords_pkl, "rb") as f:
        coord_list_cycle = pickle.load(f)  # list of [x1,y1,x2,y2]

    full_frames = _read_imgs_sorted(full_dir)
    face_frames = _read_imgs_sorted(face_dir)

    model = Model(6, 'hubert').to(DEVICE)
    state = torch.load(str(weight_pth), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, full_frames, face_frames, coord_list_cycle

def prepare_avatar(avatar_id: str):
    """
    Preload and cache the avatar (model, frames, bboxes).
    Idempotent; safe to call multiple times.
    """
    if avatar_id in _AVATAR_CACHE:
        return
    entry = load_avatar_ultralight(avatar_id)
    _AVATAR_CACHE[avatar_id] = entry

def get_cached_avatar(avatar_id: str):
    """
    Return cached avatar tuple, loading it if needed.
    """
    if avatar_id not in _AVATAR_CACHE:
        prepare_avatar(avatar_id)
    return _AVATAR_CACHE[avatar_id]
