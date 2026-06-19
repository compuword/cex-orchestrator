"""
cex_gpu_executor.py -- DirectML/CUDA ONNX inference executor.

Runs on the GPU server machine (Windows with RTX or AMD RX 6000+).
Provider priority:
  1. DmlExecutionProvider   (Windows: any DirectX 12 GPU -- RTX + AMD + Intel Arc)
  2. CUDAExecutionProvider  (Linux NVIDIA, or Windows with CUDA toolkit installed)
  3. CPUExecutionProvider   (fallback -- always available)
"""

import sys as _sys
import pathlib as _pathlib
_SHARED = _pathlib.Path(__file__).resolve().parent.parent / "shared"
if str(_SHARED) not in _sys.path:
    _sys.path.insert(0, str(_SHARED))
del _SHARED, _pathlib, _sys


import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

log = logging.getLogger("cex_gpu_executor")

MODEL_CACHE_SIZE = int(os.environ.get("CEX_MODEL_CACHE_SIZE", "8"))
DML_DEVICE_ID   = int(os.environ.get("CEX_DML_DEVICE_ID", "0"))


def _detect_providers() -> List[str]:
    """Return the ordered provider list for this machine."""
    available = ort.get_available_providers()
    chain: List[str] = []

    if "DmlExecutionProvider" in available:
        chain.append("DmlExecutionProvider")
        log.info("DirectML available (device %d)", DML_DEVICE_ID)
    if "CUDAExecutionProvider" in available:
        chain.append("CUDAExecutionProvider")
        log.info("CUDA available")
    if "ROCMExecutionProvider" in available:
        chain.append("ROCMExecutionProvider")
        log.info("ROCm available")

    chain.append("CPUExecutionProvider")
    return chain


_PROVIDERS = _detect_providers()


class GpuModelCache:
    """LRU cache for ONNX sessions -- avoids re-loading on every call."""

    def __init__(self, max_size: int = MODEL_CACHE_SIZE):
        self._max   = max_size
        self._cache: "OrderedDict[str, ort.InferenceSession]" = OrderedDict()

    def _key(self, model_bytes: bytes) -> str:
        return hashlib.sha256(model_bytes).hexdigest()[:16]

    def get_or_load(self, model_bytes: bytes) -> ort.InferenceSession:
        key = self._key(model_bytes)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        sess = self._load(model_bytes)
        self._cache[key] = sess
        if len(self._cache) > self._max:
            evicted = next(iter(self._cache))
            del self._cache[evicted]
            log.debug("Evicted model %s from cache", evicted)
        return sess

    def _load(self, model_bytes: bytes) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # suppress ORT info logs

        provider_opts = {}
        if "DmlExecutionProvider" in _PROVIDERS:
            provider_opts["DmlExecutionProvider"] = {"device_id": DML_DEVICE_ID}
        if "CUDAExecutionProvider" in _PROVIDERS:
            provider_opts["CUDAExecutionProvider"] = {"device_id": DML_DEVICE_ID}

        providers_with_opts = [
            (p, provider_opts.get(p, {})) for p in _PROVIDERS
        ]

        sess = ort.InferenceSession(model_bytes, opts, providers=providers_with_opts)
        active = sess.get_providers()[0]
        log.info("Session loaded. Active provider: %s", active)
        return sess


_CACHE = GpuModelCache()


def run_inference(
    model_bytes: bytes,
    input_tensors: Dict[str, np.ndarray],
    output_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Run inference on the GPU. Returns (output_tensors, latency_ms).
    Thread-safe: each call gets a session from the shared cache.
    """
    sess  = _CACHE.get_or_load(model_bytes)
    t0    = time.perf_counter()

    if output_names is None:
        output_names = [o.name for o in sess.get_outputs()]

    results = sess.run(output_names, input_tensors)
    latency = (time.perf_counter() - t0) * 1000.0

    return dict(zip(output_names, results)), latency


def get_capabilities() -> dict:
    """Return server capability info (included in HEARTBEAT_ACK)."""
    return {
        "ort_version": ort.__version__,
        "providers":   _PROVIDERS,
        "gpu_available": any(
            p in _PROVIDERS
            for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider")
        ),
        "active_provider": _PROVIDERS[0],
        "dml_device_id": DML_DEVICE_ID,
        "model_cache_size": MODEL_CACHE_SIZE,
        "platform": "gpu",
    }


def smoke_test() -> bool:
    """Run a tiny inference to verify the GPU path is functional."""
    try:
        import onnx
        from onnx import helper, TensorProto

        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "smoke", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        model_bytes = model.SerializeToString()
        inp  = {"X": np.array([[-1, 0, 1, 2]], dtype=np.float32)}
        out, lat = run_inference(model_bytes, inp, ["Y"])

        expected = np.array([[0, 0, 1, 2]], dtype=np.float32)
        ok = np.allclose(out["Y"], expected)
        log.info("Smoke test [%s] latency=%.2fms provider=%s",
                 "[OK]" if ok else "[FAIL]", lat, _PROVIDERS[0])
        return ok
    except Exception as exc:
        log.error("Smoke test failed: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Capabilities:", get_capabilities())
    print("Smoke test:", "[OK]" if smoke_test() else "[FAIL]")
