"""
_sec_to_samples

_create_beep

ensure_sample_library

list_sample_audios

sample_choices

resolve_sample
"""
from pathlib import Path
import numpy as np
from typing import List, Optional
import soundfile as sf
from configs import SAMPLES_DIR

def _sec_to_samples(seconds: float, sr: int) -> int:
    return int(round(seconds * sr))


def _create_beep(path: Path, sr: int = 16000, freq: float = 440.0, dur: float = 1.0) -> Path:
    t = np.linspace(0, dur, _sec_to_samples(dur, sr), endpoint=False)
    x = 0.15 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), x, sr)
    return path


def ensure_sample_library() -> None:
    """Ensure at least two sample audios exist in data/samples.
    If none provided by you, we synthesize short beeps as placeholders."""
    existing = list(SAMPLES_DIR.glob("*.wav")) + list(SAMPLES_DIR.glob("*.mp3"))
    if existing:
        return
    _create_beep(SAMPLES_DIR / "sample_beep_A.wav", freq=440, dur=1.2)
    _create_beep(SAMPLES_DIR / "sample_beep_B.wav", freq=660, dur=1.0)


def list_sample_audios() -> List[Path]:
    ensure_sample_library()
    files = sorted(SAMPLES_DIR.glob("*.wav")) + sorted(SAMPLES_DIR.glob("*.mp3"))
    return files


def sample_choices() -> List[str]:
    return [p.name for p in list_sample_audios()]


def resolve_sample(name: str) -> Optional[Path]:
    for p in list_sample_audios():
        if p.name == name:
            return p
    return None