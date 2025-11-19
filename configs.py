from pathlib import Path

# ---------------------- Paths & Config ----------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = (ROOT_DIR / "data").resolve()
AVATAR_DIR = DATA_DIR / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR = (ROOT_DIR / "jobs").resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)
LABELS_PATH = AVATAR_DIR / "labels.json"  # {label: internal_id}
SAMPLES_DIR = DATA_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mov", ".mkv"}
FIXED_FPS = 25

#  Auto-train 
ULTRALIGHT_DIR = ROOT_DIR / "ultralight" # scripts live next to this app by default
PROCESS_PY = ULTRALIGHT_DIR / "data_utils" / "process.py"
SYNCNET_PY  = ULTRALIGHT_DIR / "syncnet.py"
TRAIN_PY    = ULTRALIGHT_DIR / "train.py"
GENAVATAR_PY= ULTRALIGHT_DIR / "genavatar.py"

MIN_DURATION = 180  # 3 minutes
ASR_BACKEND = "hubert"
N_SYNCNET_EPOCHS = 50
N_ULTRALIGHT_EPOCHS = 200