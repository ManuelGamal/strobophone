"""Kaggle job: download IRIS (rasulkhanbayov/IRIS, CC BY-NC 4.0) take by take,
shrink each video so it can be pulled back cheaply, record its exact timing,
and delete the 4K original. No API keys are used here; tracking and fitting run
locally on the small clips.

Output (/kaggle/working/iris_small): <Class>/<setting>/<take>.mp4, meta.json,
parameters.json, README.md (licence notice).
"""

import json
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "rasulkhanbayov/IRIS"
OUT = Path("/kaggle/working/iris_small")
TMP = Path("/kaggle/tmp/iris")
# class -> (output height, seconds to keep or None for all)
PLAN = {
    "Falling_ball": (1080, None),
    "Dropping_ball": (1080, None),
    "Sliding_cone": (1080, None),
    "Pendulum": (720, 20),
}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_read_frames,width,height,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)["streams"][0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for f in ("parameters.json", "README.md"):
        shutil.copy(hf_hub_download(REPO, f, repo_type="dataset", local_dir=TMP), OUT / f)
    files = [s.rfilename for s in HfApi().dataset_info(REPO).siblings]
    meta = {}
    for cls, (height, keep_s) in PLAN.items():
        for name in sorted(f for f in files if f.startswith(cls + "/") and f.endswith(".mp4")):
            src = Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=TMP))
            dst = OUT / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
            if keep_s:
                cmd += ["-t", str(keep_s)]
            cmd += ["-vf", f"scale=-2:{height}:flags=lanczos", "-an", "-c:v", "libx264", "-crf", "18",
                    "-preset", "medium", "-pix_fmt", "yuv420p", str(dst)]
            subprocess.run(cmd, check=True)
            meta[name] = {"original": probe(src), "small": probe(dst), "kept_seconds": keep_s}
            print(name, meta[name]["original"]["width"], "->", meta[name]["small"]["height"],
                  round(dst.stat().st_size / 1e6, 1), "MB", flush=True)
            src.unlink()
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    (OUT / "LICENSE_NOTICE.txt").write_text(
        "Derived from IRIS (https://huggingface.co/datasets/rasulkhanbayov/IRIS), CC BY-NC 4.0. "
        "Downscaled copies for non-commercial evaluation only; do not redistribute.\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*.mp4")) / 1e6
    print(f"done: {len(meta)} clips, {total:.0f} MB")


if __name__ == "__main__":
    main()
