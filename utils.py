"""
_ensure_dir

_safe_stem

extract_audio_if_needed

zip_files

_probe_fps

_probe_duration

_ensure_25fps
"""
import subprocess
from pathlib import Path
from datetime import datetime
import zipfile
from typing import List
import gradio as gr

# --- import configs --- #
from configs import ULTRALIGHT_DIR, SUPPORTED_AUDIO_EXTS

# App.py utilities
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _safe_stem(p: Path) -> str:
    try:
        return p.stem
    except Exception:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_audio_if_needed(src_path: Path, dst_wav: Path) -> Path:
    """Ensure an audio WAV for processing.
    - If container/codec (mp4/mov/mkv/m4a/aac), extract to 16k mono wav via ffmpeg.
    - Otherwise return original (processing.py can handle resampling as needed).
    """
    ext = src_path.suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        raise gr.Error(f"Unsupported file type: {ext}")

    if ext in {".mp4", ".mov", ".mkv", ".m4a", ".aac"}:
        _ensure_dir(dst_wav.parent)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src_path),
            "-ac", "1", "-ar", "16000", str(dst_wav)
        ]
        subprocess.check_call(cmd)
        return dst_wav

    return src_path

def zip_files(files: List[Path], zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    return zip_path

def _probe_fps(video_path: Path) -> float:
    """Return stream FPS as float using ffprobe (fallback to 0 on failure)."""
    try:
        out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=nw=1:nk=1",
        str(video_path)
        ], cwd=str(ULTRALIGHT_DIR), text=True).strip()
        if "/" in out:
            num, den = out.split("/")
            return float(num) / float(den)
        return float(out)
    except Exception:
        return 0.0
    
def _probe_duration(video_path: Path) -> float:
    """Return stream duration in seconds as float using ffprobe (fallback to 0 on failure)."""
    try:
        out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
        ], cwd=str(ULTRALIGHT_DIR), text=True).strip()
        return float(out)
    except Exception:
        return 0.0


def _ensure_25fps(video_path: Path, dst_path: Path) -> Path:
    """Transcode to 25 FPS (CFR) if needed; otherwise return original path."""
    fps = _probe_fps(video_path)
    if abs(fps - 25.0) < 1e-3:
        return video_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode to CFR 25fps, keep audio
    cmd = [
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    "-i", str(video_path),
    "-vf", "fps=25",
    "-r", "25",
    "-pix_fmt", "yuv420p",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-c:a", "aac", "-b:a", "192k",
    str(dst_path)
    ]
    subprocess.check_call(cmd, cwd=str(ULTRALIGHT_DIR))
    return dst_path


# processing.py utilities
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _check_call(args: list[str]):
    subprocess.check_call(args, stderr=subprocess.STDOUT)

def _ffprobe_duration_seconds(path: str) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.STDOUT
        ).decode("utf-8","ignore").strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0