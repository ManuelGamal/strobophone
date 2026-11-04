"""Upload the web app to a Hugging Face Space (Gradio SDK, which is free; Docker
Spaces are paid). The Space runs space/space_main.py, which serves the same
FastAPI app on port 7860; Gradio itself is not used.

    hf auth login                      # once, with a write token
    .venv/Scripts/python scripts/deploy_space.py USER/strobophone

Then set the secret MODEL_API_KEY in the Space settings. Only space/*, app/ and
strobophone/ are uploaded (never .env or data/).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    repo_id = sys.argv[1]
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for f in (ROOT / "space").iterdir():
            shutil.copy(f, Path(tmp) / f.name)
        for sub in ("app", "strobophone"):
            shutil.copytree(ROOT / sub, Path(tmp) / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        api.upload_folder(folder_path=tmp, repo_id=repo_id, repo_type="space", commit_message="Deploy StroboPhone")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
