"""
run_ultralight_preprocess

run_syncnet_training

run_avatar_training

run_gen_avatar

full_auto_train_pipeline
"""
import shutil
import subprocess
from pathlib import Path
import gradio as gr
from datetime import datetime

from configs import ULTRALIGHT_DIR, AVATAR_DIR, JOBS_DIR
from configs import PROCESS_PY, SYNCNET_PY, TRAIN_PY, GENAVATAR_PY
from configs import MIN_DURATION

from utils import _ensure_25fps, _probe_duration

from avatar_manager import prepare_avatar, load_labels_map, mint_avatar_id, \
    list_avatar_labels, save_labels_map, get_avatar_preview


def run_ultralight_preprocess(video_in_dataset: Path, asr: str, progress=None) -> Path:
    """Run preprocessing on a video placed inside its dataset dir.
    Returns the dataset dir (i.e., video_in_dataset.parent).
    """
    assert video_in_dataset.exists(), f"Missing input: {video_in_dataset}"
    dataset_dir = video_in_dataset.parent
    # --- NEW: auto-fix FPS to 25 for hubert alignment ---
    if asr.lower() == "hubert":
        fixed = _ensure_25fps(video_in_dataset, dataset_dir / "input_25fps.mp4")
        if fixed != video_in_dataset:
            # replace original with fixed to keep downstream paths simple
            video_in_dataset.unlink(missing_ok=True)
            fixed.rename(video_in_dataset)
    if progress:
        progress(0.08, desc="[1/4] Preprocessing: extract audio/images/landmarks/ASR features…")
    cmd = [
        "python", str(PROCESS_PY), str(video_in_dataset), "--asr", asr
    ]
    subprocess.check_call(cmd, cwd=str(ULTRALIGHT_DIR))
    return dataset_dir


def run_syncnet_training(dataset_dir: Path, asr: str, out_dir: Path, epochs: int = 50, progress=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.30, desc="[2/4] Training SyncNet…")
    cmd = [
        "python", str(SYNCNET_PY),
        "--save_dir", str(out_dir),
        "--dataset_dir", str(dataset_dir),
        "--asr", asr,
        "--epochs", str(epochs)
    ]
    subprocess.check_call(cmd, cwd=str(ULTRALIGHT_DIR))
    ckpt = out_dir / "best.pth"
    if not ckpt.exists():
        # Fallback: last epoch
        cands = sorted(out_dir.glob("*.pth"))
        if not cands:
            raise RuntimeError("SyncNet produced no checkpoints.")
        ckpt = cands[-1]
    return ckpt


def run_avatar_training(dataset_dir: Path, asr: str, syncnet_ckpt: Path, out_dir: Path, epochs: int = 100, progress=None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.55, desc="[3/4] Training avatar model…")
    cmd = [
        "python", str(TRAIN_PY),
        "--use_syncnet",
        "--syncnet_checkpoint", str(syncnet_ckpt),
        "--dataset_dir", str(dataset_dir),
        "--save_dir", str(out_dir),
        "--asr", asr,
        "--epochs", str(epochs)
    ]
    subprocess.check_call(cmd, cwd=str(ULTRALIGHT_DIR))
    ckpt = out_dir / "best.pth"
    if not ckpt.exists():
        cands = sorted(out_dir.glob("*.pth"))
        if not cands:
            raise RuntimeError("Avatar training produced no checkpoints.")
        ckpt = cands[-1]
    return ckpt


def run_gen_avatar(dataset_dir: Path, avatar_ckpt: Path, internal_id: str, tmp_results_dir: Path, progress=None) -> Path:
    """Run genavatar.py which writes to ./results/avatars/<id>. We'll move it to data/avatars/<id>."""
    if progress:
        progress(0.82, desc="[4/4] Generating avatar assets…")
    # genavatar.py writes hardcoded under ./results/avatars/<avatar_id>
    cmd = [
        "python", str(GENAVATAR_PY),
        "--dataset", str(dataset_dir),
        "--checkpoint", str(avatar_ckpt),
        "--avatar_id", internal_id,
    ]
    subprocess.check_call(cmd, cwd=str(ULTRALIGHT_DIR))

    src_avatar_root = ULTRALIGHT_DIR / "results" / "avatars" / internal_id
    if not src_avatar_root.exists():
        raise RuntimeError(f"Expected gen outputs at {src_avatar_root} not found.")

    # Move/copy into our app-managed location
    dst_avatar_root = AVATAR_DIR / internal_id
    dst_avatar_root.mkdir(parents=True, exist_ok=True)

    # Copy folders/files
    for name in ["full_imgs", "face_imgs", "coords.pkl", "ultralight.pth"]:
        sp = src_avatar_root / name
        dp = dst_avatar_root / name
        if sp.is_dir():
            if dp.exists():
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
        elif sp.exists():
            shutil.copy2(sp, dp)

    # Also copy loudness.npy created by process.py
    lp = dataset_dir / "loudness.npy"
    if lp.exists():
        shutil.copy2(lp, dst_avatar_root / "loudness.npy")

    # Warm cache
    prepare_avatar(internal_id)
    return dst_avatar_root


def full_auto_train_pipeline(video_file, display_label: str, asr: str, n_syncnet_epochs: int = 50, n_ultralight_epochs: int = 200, progress=gr.Progress()):
    display_label = (display_label or "").strip()
    if not video_file:
        raise gr.Error("Please upload a talking-head video.")
    if not display_label:
        raise gr.Error("Please provide a new avatar label (e.g., Alice).")

    # Map & collision check
    mapping = load_labels_map()
    if display_label in mapping:
        raise gr.Error("This label already exists. Choose a different label.")

    # Check duration
    video_path = Path(video_file.name)
    duration_sec = _probe_duration(video_path)

    if duration_sec == 0.0:
        # This means ffprobe failed, could be a corrupt or invalid file
        raise gr.Error("Could not determine video duration. The file may be corrupt or in an unsupported format.")
    if duration_sec < MIN_DURATION:
        raise gr.Error(f"Video is too short ({duration_sec:.1f}s). Required minimum duration is 3 minutes.")

    # Stage workspace
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = JOBS_DIR / f"train_{tag}"
    dataset_dir = work_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_video = dataset_dir / "input.mp4"  # name doesn't matter; process.py uses dirname()
    shutil.copy(Path(video_file.name), local_video)

    try:
        # 1) Preprocess
        ds_dir = run_ultralight_preprocess(local_video, asr=asr, progress=progress)
        # 2) SyncNet
        sync_ckpt = run_syncnet_training(ds_dir, asr=asr, out_dir=work_dir / "syncnet_ckpt", epochs=n_syncnet_epochs, progress=progress)
        # 3) Avatar training
        avt_ckpt = run_avatar_training(ds_dir, asr=asr, syncnet_ckpt=sync_ckpt, out_dir=work_dir / "avatar_ckpt", epochs=n_ultralight_epochs, progress=progress)
        # 4) Generate assets + publish
        internal_id = mint_avatar_id()
        dst_root = run_gen_avatar(ds_dir, avt_ckpt, internal_id, tmp_results_dir=work_dir / "gen", progress=progress)

        # Persist mapping and refresh UI
        mapping[display_label] = internal_id
        save_labels_map(mapping)

        preview_img = get_avatar_preview(display_label)
        progress(1.0, desc="Training complete.")
        return (
            f"Avatar published: {display_label}",
            gr.update(choices=list_avatar_labels(), value=display_label),
            gr.update(choices=list_avatar_labels(), value=display_label),
            preview_img,
            preview_img,
        )
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"Training step failed. Check logs. CMD: {' '.join(e.cmd)}")