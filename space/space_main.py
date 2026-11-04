import spaces
import uvicorn

from app.server import app


# The free Space tier is ZeroGPU, which stops a Space that registers no GPU function.
# StroboPhone needs no GPU (SAM 3.1 and Muse Spark run on the Meta Model API).
@spaces.GPU
def _no_gpu_needed() -> None:
    return None


# ZeroGPU normally reports this from Gradio's launch(); this app is plain FastAPI.
try:
    from spaces.zero.client import startup_report

    startup_report()
except Exception as e:  # not on ZeroGPU, or the report failed: serve anyway
    print("ZeroGPU startup report:", e)

uvicorn.run(app, host="0.0.0.0", port=7860)
