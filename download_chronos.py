# download_chronos.py
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_DIR = Path("models/chronos-2")
REPO_ID = "amazon/chronos-2"  # change if your repo name differs

if MODEL_DIR.exists():
    print(f"Model already exists at: {MODEL_DIR}")
else:
    print("Downloading model...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print("Download complete.")
