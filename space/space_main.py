import spaces
import uvicorn

from app.server import app


# The free Space tier is ZeroGPU, which refuses to start without one GPU function.
# StroboPhone needs no GPU (SAM 3.1 and Muse Spark run on the Meta Model API).
@spaces.GPU
def _no_gpu_needed() -> None:
    return None


uvicorn.run(app, host="0.0.0.0", port=7860)
