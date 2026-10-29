"""Upload the web app to a Hugging Face Docker Space.

    hf auth login                      # once, with a write token
    .venv/Scripts/python scripts/deploy_space.py USER/strobophone

Then set the secret MODEL_API_KEY in the Space settings. Only the app, the
package, requirements and Dockerfile are uploaded (never .env or data/).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_README = """---
title: StroboPhone
emoji: ⏱️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: A phone video in, a physics measurement with error bars out.
---

Live demo of [StroboPhone](https://github.com/ManuelGamal/strobophone), an entry
for the Meta Global AI Developer Hackathon 2026. Source and evaluation live in
the GitHub repository.
"""


def main() -> None:
    repo_id = sys.argv[1]
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(SPACE_README, encoding="utf-8")
        api.upload_file(path_or_fileobj=readme, path_in_repo="README.md", repo_id=repo_id, repo_type="space")
    api.upload_folder(
        folder_path=ROOT,
        repo_id=repo_id,
        repo_type="space",
        allow_patterns=["Dockerfile", "requirements.txt", "app/**", "strobophone/**"],
        ignore_patterns=["**/__pycache__/**", "**/*.pyc"],
    )
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
