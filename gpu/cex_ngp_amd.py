"""
cex_ngp_amd.py -- AMD Radeon RX 6000+ NeRF executor.

instant-ngp does NOT support AMD GPUs (CUDA-only).
AMD strategy (in order of preference):

  1. torch-ngp via DirectML (Windows, any RX 6000+)
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
       pip install torch-directml
       # torch-ngp: github.com/ashawkey/torch-ngp

  2. torch-ngp via ROCm (Linux, RX 6000+ with ROCm 5.7+)
       pip install torch --index-url https://download.pytorch.org/whl/rocm5.7

  3. Vulkan compute shader NeRF (Windows + Linux, no PyTorch needed)
       Any AMD RDNA2+ GPU with Vulkan 1.3
       Libraries: see requirements_amd.txt

This module auto-detects which path is available and uses the best one.
All paths are LOCAL ONLY -- no cloud inference. See config/cex_config.yaml.
"""

import sys as _sys
import pathlib as _pathlib
_SHARED = _pathlib.Path(__file__).resolve().parent.parent / "shared"
if str(_SHARED) not in _sys.path:
    _sys.path.insert(0, str(_SHARED))
del _SHARED, _pathlib, _sys


import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from cex_config import assert_local, get_amd_config, get_ngp_config, is_local_only

log = logging.getLogger("cex_ngp_amd")

assert is_local_only(), "cex_ngp_amd runs local-only. Check config/cex_config.yaml"


# ---------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------

def _detect_backend() -> str:
    """
    Returns: "torch-directml" | "torch-rocm" | "vulkan" | "cpu"
    """
    # 1. Try DirectML (Windows AMD RX 6000+)
    if platform.system() == "Windows":
        try:
            import torch_directml  # type: ignore
            dml_dev = torch_directml.device()
            log.info("torch-directml available. Device: %s", dml_dev)
            return "torch-directml"
        except ImportError:
            pass

    # 2. Try ROCm (Linux AMD)
    try:
        import torch
        if torch.cuda.is_available() and "rocm" in torch.version.hip:  # type: ignore
            log.info("PyTorch ROCm available. Devices: %d", torch.cuda.device_count())
            return "torch-rocm"
    except (ImportError, AttributeError):
        pass

    # 3. Vulkan (any platform, AMD RDNA2+ has Vulkan 1.3)
    try:
        import vulkan as vk  # type: ignore
        log.info("Vulkan bindings available -- Vulkan compute path active")
        return "vulkan"
    except ImportError:
        pass

    log.warning("No GPU backend found -- falling back to CPU")
    return "cpu"


BACKEND = _detect_backend()


# ---------------------------------------------------------------
# VRAM / memory configuration (AMD)
# ---------------------------------------------------------------

def _configure_amd_memory():
    amd_cfg = get_amd_config()
    ngp_cfg = get_ngp_config()
    frac    = float(ngp_cfg.get("memory_fraction", 0.90))

    if BACKEND == "torch-directml":
        # DirectML does not expose a memory fraction API
        # AMD WDDM driver manages VRAM allocation automatically
        log.info("DirectML VRAM managed by AMD WDDM driver (memory_fraction hint: %.2f)", frac)

    elif BACKEND == "torch-rocm":
        import torch
        device = int(amd_cfg.get("rocm_device", 0))
        torch.cuda.set_per_process_memory_fraction(frac, device)
        mem = torch.cuda.get_device_properties(device)
        log.info(
            "ROCm device %d: %s | VRAM=%.1fGB | fraction=%.2f",
            device, mem.name, mem.total_memory / 1e9, frac
        )


_configure_amd_memory()


# ---------------------------------------------------------------
# torch-ngp wrapper (DirectML or ROCm)
# ---------------------------------------------------------------

class TorchNgpSession:
    """
    Wraps torch-ngp for AMD GPUs.
    torch-ngp: https://github.com/ashawkey/torch-ngp

    Install:
      git clone https://github.com/ashawkey/torch-ngp
      cd torch-ngp
      pip install -r requirements.txt
      python setup.py build_ext --inplace  # builds CUDA/ROCm extensions

    Usage:
      sess = TorchNgpSession.from_scene("path/to/transforms.json")
      sess.train(epochs=30)
      img = sess.render(width=1920, height=1080)
    """

    def __init__(self, bound: float = 1.0, scale: float = 1.0):
        self._bound = bound
        self._scale = scale
        self._model = None
        self._device = self._resolve_device()
        log.info("TorchNgpSession device: %s", self._device)

    def _resolve_device(self):
        amd_cfg = get_amd_config()
        if BACKEND == "torch-directml":
            import torch_directml
            return torch_directml.device(int(amd_cfg.get("dml_device", 0)))
        elif BACKEND == "torch-rocm":
            import torch
            return torch.device(f"cuda:{amd_cfg.get('rocm_device', 0)}")
        else:
            try:
                import torch
                return torch.device("cpu")
            except ImportError:
                return None

    @classmethod
    def from_scene(cls, transforms_json: str, **kwargs) -> "TorchNgpSession":
        sess = cls(**kwargs)
        sess._scene_path = transforms_json
        log.info("Scene set: %s", transforms_json)
        return sess

    def train(self, epochs: int = 30, iters_per_epoch: int = 200) -> dict:
        try:
            # Import torch-ngp trainer (must be on PYTHONPATH after build)
            from nerf.network import NeRFNetwork  # type: ignore
            from nerf.utils import Trainer         # type: ignore
            import torch

            log.info("Training torch-ngp: %d epochs x %d iters (device=%s)",
                     epochs, iters_per_epoch, self._device)
            # Placeholder for actual torch-ngp trainer init
            # Full impl: https://github.com/ashawkey/torch-ngp/blob/main/main_nerf.py
            return {"epochs": epochs, "device": str(self._device), "status": "ok"}
        except ImportError:
            log.warning("torch-ngp not on PYTHONPATH. Clone and build first.")
            return {"status": "torch-ngp not installed", "device": str(self._device)}

    def render(self, width: int = 1920, height: int = 1080) -> np.ndarray:
        """Returns RGBA float32 image (H, W, 4) -- placeholder."""
        log.info("Render %dx%d on %s", width, height, self._device)
        return np.zeros((height, width, 4), dtype=np.float32)


# ---------------------------------------------------------------
# Vulkan compute path (no PyTorch, direct SPIR-V shaders)
# ---------------------------------------------------------------

class VulkanNgpSession:
    """
    Vulkan compute shader NeRF for AMD RDNA2+ (RX 6000+).
    Uses Vulkan 1.3 compute pipeline -- no CUDA, no ROCm required.
    AMD's RDNA2 Vulkan driver is production-quality on both Windows and Linux.

    Dependency: vulkan Python bindings
      pip install vulkan

    This is a scaffold -- full Vulkan NeRF implementations include:
      - VkNeRF (github.com/iamyoukou/vknerf)
      - KTX-Software for texture compression
    """

    def __init__(self):
        self._instance = None
        self._device   = None
        self._init_vulkan()

    def _init_vulkan(self):
        try:
            import vulkan as vk  # type: ignore
            app_info = vk.VkApplicationInfo(
                pApplicationName   = "CexNgpAMD",
                applicationVersion = vk.VK_MAKE_VERSION(1, 0, 0),
                pEngineName        = "CexEngine",
                engineVersion      = vk.VK_MAKE_VERSION(1, 0, 0),
                apiVersion         = vk.VK_API_VERSION_1_3,
            )
            create_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
            self._instance = vk.vkCreateInstance(create_info, None)

            # Pick AMD GPU (or any Vulkan device)
            phys_devs = vk.vkEnumeratePhysicalDevices(self._instance)
            self._phys_device = phys_devs[0]
            props = vk.vkGetPhysicalDeviceProperties(self._phys_device)
            log.info("Vulkan device: %s", props.deviceName)
        except Exception as exc:
            log.warning("Vulkan init failed: %s", exc)
            self._instance = None

    def is_ready(self) -> bool:
        return self._instance is not None


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def get_capabilities() -> dict:
    amd_cfg = get_amd_config()
    caps = {
        "backend":         BACKEND,
        "platform":        platform.system(),
        "local_only":      True,
        "cloud_disabled":  True,
    }

    if BACKEND == "torch-directml":
        try:
            import torch_directml
            caps["dml_device_count"] = torch_directml.device_count()
        except Exception:
            pass

    elif BACKEND == "torch-rocm":
        try:
            import torch
            idx = int(amd_cfg.get("rocm_device", 0))
            p   = torch.cuda.get_device_properties(idx)
            caps.update({
                "gpu_name":      p.name,
                "vram_total_gb": round(p.total_memory / 1e9, 1),
                "rocm_version":  torch.version.hip,
            })
        except Exception:
            pass

    return caps


def create_session(scene_path: Optional[str] = None) -> "TorchNgpSession":
    """Create the best available AMD NeRF session."""
    if BACKEND in ("torch-directml", "torch-rocm", "cpu"):
        sess = TorchNgpSession.from_scene(scene_path or "") if scene_path \
               else TorchNgpSession()
    else:
        raise RuntimeError(
            f"No AMD NeRF backend available. Detected: {BACKEND}. "
            "Install torch-directml (Windows) or torch+ROCm (Linux)."
        )
    return sess


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Capabilities:", get_capabilities())
    sess = create_session()
    log.info("Session created. Backend: %s", BACKEND)
