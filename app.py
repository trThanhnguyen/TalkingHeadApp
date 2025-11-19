"""
app.py — version 10
Features
- Hides internal avatar folder names (users see labels only)
- Batch rendering (multi audio/video → multi MP4)
- Auto-train avatar: upload video → preprocess (Ultralight) → publish → refresh UI
- FPS fixed to 25
- new in v7: Avatar preview image (first frame from each avatar's data folder) without exposing folder names
- new in v8: training video duration check >= 3 minutes, access control
- new in v10: update single rendering file naming and batch rendering progress bar
This file expects your core functions in processing.py:
  - prepare_avatar(avatar_id: str) -> None
  - run_render(audio_path: str, avatar_id: str, fps: int) -> str  # returns path to mp4
"""
from __future__ import annotations
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import gradio as gr
from PIL import Image

# --------------- Import configs ----------------- #
from configs import JOBS_DIR, FIXED_FPS, ASR_BACKEND, N_SYNCNET_EPOCHS, N_ULTRALIGHT_EPOCHS

# ---------------------- Core imports ----------------------
from processing import run_render  # noqa: E402
from avatar_manager import prepare_avatar

# ---------------------- Utilities ----------------------
from utils import _ensure_dir, _safe_stem, extract_audio_if_needed, zip_files

# --------------------- Avatar management ------------------ # 
from avatar_manager import list_avatar_labels, label_to_id, get_avatar_preview

# ---------------------- Samples library ---------------------- #
from sample_manager import resolve_sample, sample_choices

# ---------------------- Training pipeline ---------------------$
from training_pipeline import full_auto_train_pipeline


def cb_use_sample(name: str):
    p = resolve_sample(name)
    if not p:
        raise gr.Error("Sample not found.")
    return gr.update(value=str(p))


def cb_use_samples(names: Sequence[str]):
    paths = []
    for n in names or []:
        p = resolve_sample(n)
        if p:
            paths.append(str(p))
    if not paths:
        raise gr.Error("Select one or more samples.")
    return gr.update(value=paths)


def cb_prepare_avatar(avatar_label: str) -> str:
    if not avatar_label:
        raise gr.Error("Select an avatar.")
    internal_id = label_to_id(avatar_label)
    prepare_avatar(internal_id)
    return f"Avatar prepared: {avatar_label}"


def cb_load_ui():
    """
    Called by demo.load() to populate UI components dynamically.
    Ensures dropdowns have the most recent avatar list on page load.
    """
    labels = list_avatar_labels()
    default_label = labels[0] if labels else None
    preview = get_avatar_preview(default_label)

    return (
        gr.update(choices=labels, value=default_label),  # Update dd_avatar
        gr.update(choices=labels, value=default_label),  # Update dd_avatar_b
        preview,                                        # Update img_preview
        preview                                         # Update img_preview_b
    )


def cb_render_single(audio_file, avatar_label: str, progress=gr.Progress(track_tqdm=True)):
    if audio_file is None:
        raise gr.Error("Please upload an audio/video file.")
    if not avatar_label:
        raise gr.Error("Select an avatar.")

    internal_id = label_to_id(avatar_label)

    job_dir = JOBS_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_single")
    _ensure_dir(job_dir)
    src_path = Path(audio_file.name)
    dst_in = job_dir / f"input{src_path.suffix}"
    shutil.copy(src_path, dst_in)

    progress(0.05, desc="Checking/Extracting audio…")
    safe_audio = extract_audio_if_needed(dst_in, job_dir / "extracted_16k.wav")

    progress(0.10, desc="Preparing avatar…")
    prepare_avatar(internal_id)

    progress(0.15, desc="Rendering (offline)…")
    start_render_time = time.perf_counter()
    out_mp4 = run_render(str(safe_audio), avatar_id=internal_id, fps=FIXED_FPS)
    end_render_time = time.perf_counter()
    render_time = end_render_time - start_render_time
    print(f"rendering takes {render_time:.3f} seconds.")

    final_mp4_path = job_dir / f"{_safe_stem(src_path)}.mp4"
    if Path(out_mp4) != final_mp4_path:
        shutil.move(out_mp4, final_mp4_path)

    progress(1.0, desc="Done")
    return str(final_mp4_path)


# def cb_render_batch(files, avatar_label: str, progress=gr.Progress()):
def cb_render_batch(files, avatar_label: str, progress=gr.Progress(track_tqdm=False)):
    if not files:
        raise gr.Error("Please upload one or more audio/video files.")
    if not avatar_label:
        raise gr.Error("Select an avatar.")

    internal_id = label_to_id(avatar_label)
    
    # --- FIX: New progress logic ---
    progress(0.0, desc="Preparing avatar…")
    prepare_avatar(internal_id)

    results: List[Path] = []
    job_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = JOBS_DIR / f"batch_{job_tag}"
    _ensure_dir(job_dir)

    N = len(files)
    if N == 0:
        progress(1.0, desc="No files to process.")
        return [], None

    # Define progress sections: 80% for rendering, 10% for zipping
    render_share = 0.90
    zip_share = 0.10
    
    for i, f in enumerate(files, start=1):
        src = Path(f.name)
        # Calculate progress *before* starting the step
        step_progress = ((i - 1) / N) * render_share
        progress(step_progress, desc=f"Rendering {i}/{N}: {_safe_stem(src)}…")
        
        dst_in = job_dir / f"input_{i:03d}{src.suffix}"
        shutil.copy(src, dst_in)
        safe_audio = extract_audio_if_needed(dst_in, job_dir / f"extracted_{i:03d}_16k.wav")
        
        # This is the long-running part
        out_mp4 = run_render(str(safe_audio), avatar_id=internal_id, fps=FIXED_FPS)
        
        final_mp4 = job_dir / f"{_safe_stem(src)}.mp4"
        shutil.move(out_mp4, final_mp4)
        results.append(final_mp4)

    # All rendering is done, now we zip
    progress(render_share, desc=f"Zipping {N} results…")
    zip_path = job_dir / f"batch_results_{job_tag}.zip"
    zip_files(results, zip_path)

    progress(1.0, desc="Batch complete.")
    # --- End Fix ---
    
    return [str(p) for p in results], str(zip_path)


def cb_auto_train(video_file, display_label: str, progress=gr.Progress()):
    """Thin wrapper that runs the full 4-step training pipeline with a fixed ASR backend."""

    return full_auto_train_pipeline(video_file, display_label, ASR_BACKEND, n_syncnet_epochs=N_SYNCNET_EPOCHS, n_ultralight_epochs=N_ULTRALIGHT_EPOCHS, progress=progress)


def cb_preview(avatar_label: str):
    return get_avatar_preview(avatar_label)


# ---------------------- UI ----------------------
with gr.Blocks(title="Talking Avatars - Gradio UI") as demo:
    gr.Markdown("# Talking Avatars — Ultralight\n")
    gr.Markdown("""Ultralight Talking Avatars presented by GradIO UI. Supports batch rendering and automatic training.\n
                Please prepare your audio files with talking content in advance, you can use our [voice-cloning toolkit](https://app.aix-nycu.tw/).  
                Upload the audio here to animate the avatars.  
                To train your own avatar, please go to "Auto-train avatar" tab.""")

    with gr.Tabs():
        # -------- Render Tab --------
        with gr.Tab("Render"):
            with gr.Row():
                with gr.Column(scale=1):
                    dd_avatar = gr.Dropdown(
                        label="Avatar"
                        # `choices` and `value` will be set by demo.load()
                    )
                    btn_prepare = gr.Button("Confirm Avatar Selection")
                    status_single = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    img_preview = gr.Image(label="Avatar preview", interactive=False)
            with gr.Row():
                with gr.Column(scale=2):
                    audio_in = gr.File(label="Upload your audio/video file here:")
                with gr.Column(scale=1):
                    dd_sample = gr.Dropdown(choices=sample_choices(), label="Or use one of our sample audio files")
                    btn_use_sample = gr.Button("Load sample")
                btn_render = gr.Button("Render", variant="primary")

            out_video = gr.Video(label="Result MP4", interactive=False)

            btn_use_sample.click(cb_use_sample, inputs=[dd_sample], outputs=[audio_in])
            btn_prepare.click(cb_prepare_avatar, inputs=[dd_avatar], outputs=[status_single])
            btn_render.click(cb_render_single, inputs=[audio_in, dd_avatar], outputs=[out_video])
            dd_avatar.change(cb_preview, inputs=[dd_avatar], outputs=[img_preview])

        # -------- Batch Tab --------
        with gr.Tab("Batch render"):
            with gr.Row():
                with gr.Column(scale=1):
                    dd_avatar_b = gr.Dropdown(
                        label="Avatar"
                        # `choices` and `value` will be set by demo.load()
                    )
                with gr.Column(scale=1):
                    img_preview_b = gr.Image(label="Avatar preview", interactive=False)
            with gr.Row():
                with gr.Column(scale=2):
                    files_in = gr.Files(label="Upload many audio/video files")
                with gr.Column(scale=1):
                    chk_samples = gr.CheckboxGroup(choices=sample_choices(), label="Add sample audios")
                    btn_add_samples = gr.Button("Add samples")
            btn_batch = gr.Button("Run batch")
            out_files = gr.Files(label="Rendered MP4 files")
            out_zip = gr.File(label="ZIP of all results")


            btn_add_samples.click(cb_use_samples, inputs=[chk_samples], outputs=[files_in])
            btn_batch.click(cb_render_batch, inputs=[files_in, dd_avatar_b], outputs=[out_files, out_zip])
            dd_avatar_b.change(cb_preview, inputs=[dd_avatar_b], outputs=[img_preview_b])

        # -------- Auto-train Tab --------
        with gr.Tab("Auto-train avatar"):
            gr.Markdown("""Upload a clear video of a talking person. The system will train and publish it. Then it becomes available in the dropdowns of "Render" and "Batch Render" tabs.<br>
                        Training Video Requirements:<br>
                        - Duration: ~5 minutes of continuous speak, shorter videos may yield suboptimal results
                        - Visual: Only one speaker with complete face visible in every frame
                        - Audio: Clear speech without noise or echo
                        - Frame Rate: 25FPS
                        - Resolution: Flexible, but face must be clearly visible
                        - Environment: Good lighting, minimal background noise.
                        """)
            gr.Markdown("Please expect 6-8 hours for the avatar to be ready.")
            video_in = gr.File(label="Talking-head video")
            display_label = gr.Textbox(label="New avatar label (name your avatar)", placeholder="e.g., Alice")
            btn_train = gr.Button("Start training")
            train_msg = gr.Textbox(label="Status", interactive=False)

            # Refresh both dropdowns and both previews after training
            btn_train.click(
                cb_auto_train,
                inputs=[video_in, display_label],
                outputs=[train_msg, dd_avatar, dd_avatar_b, img_preview, img_preview_b]
            )

    # Dynamically load choices and previews for all components
    demo.load(
        cb_load_ui,
        outputs=[dd_avatar, dd_avatar_b, img_preview, img_preview_b]
    )

    # Long-running jobs support
    demo.queue(api_open=False, max_size=32)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        auth=("nycu.ai", "nycu@ai")
    )
