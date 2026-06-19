"""
cex_ngp_executor.py -- instant-ngp / NeRF executor for NVIDIA RTX GPUs.

Wraps the pyngp Python bindings that ship with instant-ngp after build.
If pyngp is not available, falls back to a pure-Python NeRF smoke test.

instant-ngp build:
  git clone --recursive https://github.com/NVlabs/instant-ngp
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release -j
  # Output: build/Release/pyngp.pyd  (Windows) or build/pyngp.so (Linux)
  # Add to PYTHONPATH: set PYTHONPATH=%PYTHONPATH%;C:\\path\\to\\instant-ngp\\build\\Release

Dependencies:
  pip install -r requirements_rtx.txt
  + Vulkan SDK 1.3+ installed system-wide
  + COLMAP on PATH (for colmap2nerf camera conversion)

GPU memory control:
  Set ngp.memory_fraction in config/cex_config.yaml
  Set nvidia.cuda_device for multi-GPU selection
"""

import sys as _sys
import pathlib as _pathlib
_SHARED = _pathlib.Path(__file__).resolve().parent.parent / "shared"
if str(_SHARED) not in _sys.path:
    _sys.path.insert(0, str(_SHARED))
del _SHARED, _pathlib, _sys


import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from cex_config import assert_local, get_ngp_config, get_nvidia_config, is_local_only

log = logging.getLogger("cex_ngp_executor")

# Enforce local-only at module load time
assert is_local_only(), "cex_ngp_executor runs local-only. Check config/cex_config.yaml"

_NGP_AVAILABLE = False
_ngp = None


def _try_import_pyngp():
    global _NGP_AVAILABLE, _ngp
    try:
        import pyngp as ngp  # type: ignore
        _ngp = ngp
        _NGP_AVAILABLE = True
        log.info("pyngp loaded. instant-ngp ready.")
    except ImportError:
        log.warning(
            "pyngp not found. Build instant-ngp from source: "
            "https://github.com/NVlabs/instant-ngp"
        )
        _NGP_AVAILABLE = False


_try_import_pyngp()


def _configure_cuda_memory():
    """Apply VRAM settings from config before loading any GPU resource."""
    nv_cfg  = get_nvidia_config()
    ngp_cfg = get_ngp_config()

    device  = int(nv_cfg.get("cuda_device", 0))
    frac    = float(ngp_cfg.get("memory_fraction", 0.90))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    # PyTorch memory fraction (if torch is available alongside ngp)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(frac, device)
            log.info("CUDA device %d, memory_fraction=%.2f", device, frac)
    except ImportError:
        pass

    # pynvml -- verify driver version
    try:
        import pynvml
        pynvml.nvmlInit()
        handle  = pynvml.nvmlDeviceGetHandleByIndex(device)
        drv_ver = pynvml.nvmlSystemGetDriverVersion()
        mem     = pynvml.nvmlDeviceGetMemoryInfo(handle)
        log.info(
            "NVIDIA driver %s | VRAM total=%.1fGB free=%.1fGB",
            drv_ver,
            mem.total / 1e9,
            mem.free  / 1e9,
        )
        pynvml.nvmlShutdown()
    except Exception as exc:
        log.debug("pynvml not available: %s", exc)


_configure_cuda_memory()


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

class NgpSession:
    """
    Wraps one instant-ngp Testbed session.
    Usage:
      sess = NgpSession.from_scene("path/to/transforms.json")
      sess.train(steps=35000)
      sess.save_snapshot("output.msgpack")
      rendered = sess.render(width=1920, height=1080)
    """

    def __init__(self):
        if not _NGP_AVAILABLE:
            raise RuntimeError(
                "pyngp not available. Build instant-ngp and add pyngp.pyd to PYTHONPATH."
            )
        ngp_cfg = get_ngp_config()
        self._testbed = _ngp.Testbed()
        self._testbed.nerf.training.n_images_for_training = 0
        # Apply scene scale from config
        self._testbed.nerf.training.scene_scale = float(ngp_cfg.get("scene_scale", 1.0))

    @classmethod
    def from_scene(cls, transforms_json: str) -> "NgpSession":
        sess = cls()
        sess._testbed.load_training_data(transforms_json)
        log.info("Scene loaded: %s", transforms_json)
        return sess

    @classmethod
    def from_snapshot(cls, snapshot_path: str) -> "NgpSession":
        sess = cls()
        sess._testbed.load_snapshot(snapshot_path)
        log.info("Snapshot loaded: %s", snapshot_path)
        return sess

    def train(self, steps: Optional[int] = None) -> float:
        ngp_cfg = get_ngp_config()
        n = steps or int(ngp_cfg.get("n_steps", 35000))
        log.info("Training %d steps...", n)
        self._testbed.shall_train = True
        for i in range(n):
            self._testbed.frame()
            if i % 5000 == 0 and i > 0:
                loss = self._testbed.loss
                log.info("  step %d / %d  loss=%.6f", i, n, loss)
        self._testbed.shall_train = False
        return float(self._testbed.loss)

    def render(
        self,
        width: int = 1920,
        height: int = 1080,
        spp: int = 8,
    ) -> np.ndarray:
        """Returns RGBA float32 image array of shape (H, W, 4)."""
        img = self._testbed.render(width, height, spp, linear=True)
        return np.array(img)

    def save_snapshot(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._testbed.save_snapshot(path, False)
        log.info("Snapshot saved: %s", path)

    def get_loss(self) -> float:
        return float(self._testbed.loss)


def is_available() -> bool:
    return _NGP_AVAILABLE


def get_capabilities() -> dict:
    nv_cfg  = get_nvidia_config()
    ngp_cfg = get_ngp_config()
    caps    = {
        "ngp_available":     _NGP_AVAILABLE,
        "cuda_device":       nv_cfg.get("cuda_device", 0),
        "memory_fraction":   ngp_cfg.get("memory_fraction", 0.90),
        "backend":           "cuda-ngp" if _NGP_AVAILABLE else "unavailable",
        "local_only":        True,
        "cloud_disabled":    True,
    }

    try:
        import pynvml
        pynvml.nvmlInit()
        h    = pynvml.nvmlDeviceGetHandleByIndex(int(nv_cfg.get("cuda_device", 0)))
        mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        caps.update({
            "gpu_name":     name if isinstance(name, str) else name.decode(),
            "vram_total_gb": round(mem.total / 1e9, 1),
            "vram_free_gb":  round(mem.free  / 1e9, 1),
        })
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return caps


def quick_render_test() -> bool:
    """Smoke test: creates a minimal scene and renders 1 frame."""
    if not _NGP_AVAILABLE:
        log.warning("NGP smoke test skipped -- pyngp not built")
        return False
    try:
        sess = NgpSession()
        img  = sess.render(width=64, height=64, spp=1)
        ok   = img.shape == (64, 64, 4)
        log.info("NGP render test: %s shape=%s", "[OK]" if ok else "[FAIL]", img.shape)
        return ok
    except Exception as exc:
        log.error("NGP render test failed: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Capabilities:", get_capabilities())
    print("Quick test:", "[OK]" if quick_render_test() else "[SKIP] pyngp not built")
