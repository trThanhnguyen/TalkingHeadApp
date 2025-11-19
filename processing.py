# processing.py — OFFLINE Ultralight pipeline (audio → lip-synced video → mux with audio)
# Requirements: ffmpeg/ffprobe in PATH, PyTorch, OpenCV, numpy
# Avatar assets expected at: data/avatars/<avatar_id>/{full_imgs,face_imgs,coords.pkl,ultralight.pth}
# V8

from __future__ import annotations
import os, subprocess
from pathlib import Path
from typing import List
import pyloudnorm as pln

import numpy as np
import cv2
import torch
import soundfile as sf
import librosa
from tqdm import tqdm
import time

# Ultralight components (from your repo)
from ultralight.audio2feature import Audio2Feature
from avatar_manager import get_cached_avatar

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu")


# ----------------- Utilities -----------------
from utils import _ensure_dir, _check_call, _ffprobe_duration_seconds

def _make_even_first_dim(tensor):
    size = list(tensor.size())
    if size[0] % 2 == 1:
        size[0] -= 1
        return tensor[:size[0]]
    return tensor

def _decode_to_wav16k_mono(in_audio: str, out_wav: str):
    """
    Decode ANY common audio container/codec (m4a/mp4/mp3/ogg/flac/wav, even video files) to 16k mono WAV.
    Requires ffmpeg in PATH.
    """
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_audio,
        "-vn",          # ignore video if present (e.g., .mp4 containers)
        "-ac", "1",     # mono
        "-ar", "16000", # 16kHz
        "-f", "wav",
        out_wav
    ]
    _check_call(args)

def _read_and_norm_mono_16k(decoded_wav_path: str, loudness_path: str | None, out_normed_wav: str):
    """
    Read a 16k mono WAV, normalize to target LUFS (avatar profile or -14 LUFS fallback),
    peak-limit to -1 dBFS, and save to out_normed_wav (WAV).
    Returns: (float32 waveform @16k, path_to_normed_wav)
    """
    speech, sr = sf.read(decoded_wav_path, dtype="float32")
    if sr != 16000:
        # Shouldn't happen if we decoded with ffmpeg, but keep as safety.
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000

    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)

    # loudness target
    target_lufs = -14.0
    if loudness_path:
        try:
            target_lufs = float(np.load(loudness_path))
        except Exception:
            print(f"Warning: failed to read loudness profile {loudness_path}, fallback to -14 LUFS.")

    # LUFS normalization
    meter = pln.Meter(sr)  # ITU-R BS.1770
    try:
        loudness = meter.integrated_loudness(speech)
        speech = pln.normalize.loudness(speech, loudness, target_lufs)
    except Exception as e:
        print(f"LUFS normalization failed ({e}); continuing without LUFS normalization.")

    # simple peak limiter to -1 dBFS
    peak = float(np.max(np.abs(speech)) + 1e-12)
    peak_db = 20*np.log10(peak)
    peak_dbfs = -1.0
    if peak_db > peak_dbfs:
        gain = 10**((peak_dbfs - peak_db)/20)
        speech = speech * gain

    # final safety
    speech = np.clip(speech, -1.0, 1.0).astype(np.float32, copy=False)

    # write normalized file as WAV
    sf.write(out_normed_wav, speech, 16000, subtype="PCM_16")
    return speech, out_normed_wav

def _mirror_index(size: int, index: int) -> int:
    # Ping-pong looping through avatar frames.
    turn = index // size
    res = index % size
    return res if (turn % 2 == 0) else (size - res - 1)

# ----------------- Audio features -----------------

def extract_features(audio_wav_16k_waveform: np.ndarray) -> np.ndarray:
    """
    Use Audio2Feature to get per-frame features for Ultralight.
    Returns: (T, 2, 1024) after reshape in this implementation.
    """
    a2f = Audio2Feature()
    feats = None
    try:
        feats = a2f.get_hubert_from_16k_speech(audio_wav_16k_waveform)
        feats = _make_even_first_dim(feats).reshape(-1, 2, 1024)
        feats = np.asarray(feats)
    except Exception as e:
        print(f"An error occurred during audio feature extraction: {e}")
        raise RuntimeError("Cannot extract audio feature by Audio2Feature")
    return feats.astype(np.float32, copy=False)

# ----------------- Frame synthesis -----------------

def _make_ultralight_input(face_img: np.ndarray) -> torch.Tensor:
    crop = face_img[4:164, 4:164].copy()
    masked = crop.copy()
    masked = cv2.rectangle(masked, (5, 5, 150, 145), (0, 0, 0), -1)  # match LightReal
    img_real_ex = crop.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_masked  = masked.transpose(2, 0, 1).astype(np.float32) / 255.0
    six = np.concatenate([img_real_ex, img_masked], axis=0)  # (6,160,160)
    return torch.from_numpy(six)

def _paste_back(pred_face_rgb: np.ndarray, full_img: np.ndarray, face_img: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = y2 - y1, x2 - x1
    face_crop = face_img.copy()
    face_crop[4:164, 4:164] = np.clip(pred_face_rgb, 0, 255).astype(np.uint8)
    face_resized = cv2.resize(face_crop, (w, h), interpolation=cv2.INTER_AREA)
    out = full_img.copy()
    out[y1:y2, x1:x2] = face_resized
    return out

def get_audio_features_3d(features: np.ndarray, index: int) -> np.ndarray:
    left = max(0, index - 8)
    right = min(features.shape[0], index + 8)
    seg = features[left:right]  # (<=16, 2, 1024)
    pad_left = max(0, 8 - (index - left))
    pad_right = max(0, 16 - pad_left - seg.shape[0])
    if pad_left:
        seg = np.concatenate([np.zeros((pad_left, *seg.shape[1:]), seg.dtype), seg], axis=0)
    if pad_right:
        seg = np.concatenate([seg, np.zeros((pad_right, *seg.shape[1:]), seg.dtype)], axis=0)
    return seg.reshape(16, 2048)

def get_audio_features(features: np.ndarray, index: int) -> np.ndarray:
    left = max(0, index - 8)
    right = min(features.shape[0], index + 8)
    seg = features[left:right]
    pad_left = max(0, 8 - (index - left))
    pad_right = max(0, 16 - pad_left - seg.shape[0])
    if pad_left:
        seg = np.concatenate([np.zeros((pad_left, seg.shape[1]), seg.dtype), seg], axis=0)
    if pad_right:
        seg = np.concatenate([seg, np.zeros((pad_right, seg.shape[1]), seg.dtype)], axis=0)
    return seg

def synthesize_frames_ultralight(model, full_frames, face_frames, bboxes, feats, mode="hubert", batch_size=32):
    T = feats.shape[0]
    n_avatar = len(face_frames)
    outs = []

    for i0 in tqdm(range(0, T, batch_size), desc="Synthesizing"):
        i1 = min(T, i0 + batch_size)
        this = i1 - i0

        img_batch, mel_batch = [], []

        for k in range(this):
            idx  = i0 + k
            aidx = _mirror_index(n_avatar, idx)
            face = face_frames[aidx]
            six  = _make_ultralight_input(face)
            img_batch.append(six)

            feat16 = get_audio_features_3d(feats, idx)
            if mode == "hubert":
                feat = feat16.reshape(32, 32, 32)
            elif mode == "wenet":
                feat = feat16.reshape(128, 16, 32)
            else:
                raise ValueError(f"Unsupported mode {mode}")
            mel_batch.append(torch.from_numpy(feat))

        img_batch_t = torch.stack(img_batch, 0).to(DEVICE)
        mel_batch_t = torch.stack(mel_batch, 0).to(DEVICE)

        with torch.no_grad():
            pred = model(img_batch_t, mel_batch_t)
        pred = (pred.detach().float().cpu().numpy().transpose(0,2,3,1) * 255.0)

        for k in range(this):
            idx  = i0 + k
            aidx = _mirror_index(n_avatar, idx)
            full = full_frames[aidx]
            face = face_frames[aidx]
            bbox = bboxes[aidx]
            pred_face = pred[k]
            outs.append(_paste_back(pred_face, full, face, bbox))

    return outs

# ----------------- Video writing -----------------
def write_video_and_mux_audio(frames: List[np.ndarray], fps: int, job_dir: Path, audio_wav_16k: str) -> str:
    """
    Pipe frames directly to ffmpeg stdin, bypassing disk I/O, 
    and mux audio in ONE single command.
    """
    if not frames:
        raise ValueError("Received no frames to write.")
    
    # Get frame dimensions from the first frame
    first_frame = frames[0]
    height, width, _ = first_frame.shape
    
    result_path = str((job_dir / "result.mp4").resolve())

    # This single command reads raw video from stdin, audio from a file,
    # and encodes/muxes them into the final MP4.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        
        # Input 1: Raw video frames from stdin (pipe)
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",  # Frame size WxH
        "-pix_fmt", "bgr24",       # Pixel format from OpenCV (BGR)
        "-r", str(fps),            # Framerate
        "-i", "-",                 # Stdin
        
        # Input 2: Audio file
        "-i", audio_wav_16k,
        
        # Output settings
        "-c:v", "libx264",         # Standard video codec
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",     # Ensures compatibility
        "-c:a", "aac",             # Standard audio codec
        "-b:a", "128k",
        "-shortest",               # Finish when audio finishes
        result_path
    ]

    # Start the ffmpeg process
    try:
        proc = subprocess.Popen(cmd, 
                              stdin=subprocess.PIPE, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)

        # Write frames to ffmpeg's stdin
        for frame in tqdm(frames, desc="🚀 Encoding video (fast)"):
            # .tobytes() is the magic. No more cv2.imwrite.
            proc.stdin.write(frame.tobytes())
        
        # Wait for ffmpeg to finish
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            # Print error if ffmpeg fails
            print("--- FFMPEG STDOUT ---")
            print(stdout.decode())
            print("--- FFMPEG STDERR ---")
            print(stderr.decode())
            raise RuntimeError(f"ffmpeg command failed with code {proc.returncode}")

    except (IOError, BrokenPipeError) as e:
        print(f"Error writing to ffmpeg stdin: {e}")
        # Try to get error message
        try:
            stdout, stderr = proc.communicate(timeout=1)
            print("--- FFMPEG STDERR ---")
            print(stderr.decode())
        except Exception:
            pass # Failsafe
        raise RuntimeError("ffmpeg process failed during frame writing.")
    
    # The old temp dir cleanup is gone because we never made the temp dir
    return result_path

# ----------------- Public entry -----------------

def run_render(audio_path: str, avatar_id: str, model: str = "ultralight", fps: int = 25, resolution: str = "1920x1080") -> str:
    """
    Offline Ultralight:
      1) decode ANY input to 16k mono WAV with ffmpeg
      2) normalize loudness to avatar profile (or -14 LUFS fallback)
      3) extract per-frame features at target fps
      4) synthesize frames using Ultralight and avatar assets (cached if prepared)
      5) mux frames with user's audio -> MP4
    """
    assert model.lower() in ("ultralight","ultralight-digital-human","ul"), "This entry implements Ultralight. (Adapt here for MuseTalk/Wav2Lip)"
    root = Path("data")/"avatars"/avatar_id
    loudness_path = None
    lp = root/"loudness.npy"
    if lp.is_file():
        loudness_path = str(lp)

    job_dir = Path(audio_path).parent
    _ensure_dir(job_dir)

    # 1) Decode arbitrary input to a clean 16k mono WAV
    decoded_wav = str((job_dir / "audio_decoded_16k_mono.wav").resolve())
    _decode_to_wav16k_mono(audio_path, decoded_wav)

    # Duration sanity check after decode
    if _ffprobe_duration_seconds(decoded_wav) <= 0.0:
        raise RuntimeError("Decoded audio has zero duration. Is the input empty or unsupported?")

    # 2) Loudness normalization -> produce a canonical audio_normed.wav
    normed_wav = str((job_dir / "audio_normed.wav").resolve())
    speech16, normed_audio = _read_and_norm_mono_16k(decoded_wav, loudness_path, normed_wav)

    # 3) Features
    start_ext_aud = time.perf_counter()
    feats = extract_features(speech16)
    end_ext_aud = time.perf_counter()
    ext_aud_time = end_ext_aud - start_ext_aud
    print(f"extracting audio time: {ext_aud_time:.03f}")
    if feats.shape[0] == 0:
        raise RuntimeError("No audio features extracted; is the audio empty or too short?")

    # 4) Avatar & model (cached if prepared)
    model_net, full_frames, face_frames, bboxes = get_cached_avatar(avatar_id)

    # 5) Synthesize frames
    start_synz = time.perf_counter()
    frames = synthesize_frames_ultralight(model_net, full_frames, face_frames, bboxes, feats, batch_size=32)
    end_synz= time.perf_counter()
    synz_time = end_synz - start_synz
    print(f"Synthesizing frames time: {synz_time:.03f}")

    # 6) Write video and mux with normalized audio
    mp4 = write_video_and_mux_audio(frames, fps=fps, job_dir=job_dir, audio_wav_16k=normed_audio)
    return mp4
