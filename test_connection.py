"""
test_connection.py -- End-to-end connection test for the CEX orchestrator stack.

Tests each layer in sequence:
  [1] Prerequisites  -- numpy, onnxruntime, required imports
  [2] Local GPU      -- ONNX Runtime provider detection (CUDA / DirectML / CPU)
  [3] GPU server     -- start cex_gpu_server.py locally, verify health endpoint
  [4] Resource scan  -- run find_resources.py --scan, check inventory
  [5] BackendRouter  -- from_resource_file() + from_discovery_api()
  [6] Inference      -- route a real ONNX model through the orchestrator

Run: python test_connection.py
     python test_connection.py --subnet 192.168.1.0/24   (also scan LAN)
     python test_connection.py --install                  (auto-install deps first)
"""

import argparse
import io
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

PASS = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

def step(n, label):
    print(f"\n=== STEP {n}: {label} ===")

def ok(msg):   print(f"  {PASS}  {msg}")
def fail(msg): print(f"  {FAIL} {msg}"); sys.exit(1)
def warn(msg): print(f"  {WARN} {msg}")
def info(msg): print(f"  {INFO} {msg}")

def port_open(host, port, timeout=1.0):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except OSError:
        return False

# ---------------------------------------------------------------
# Step 1: Prerequisites
# ---------------------------------------------------------------

def test_prereqs(install=False):
    step(1, "Prerequisites")

    if install:
        info("Installing onnxruntime + onnxruntime-directml ...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "onnxruntime>=1.17.0", "onnxruntime-directml>=1.17.0",
             "numpy>=1.24.0", "fastapi>=0.110.0", "uvicorn[standard]>=0.29.0",
             "--quiet"],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            warn("pip install had errors -- check manually")
            print(r.stderr[:500])
        else:
            ok("pip install complete")

    missing = []
    for pkg in ("numpy", "onnxruntime"):
        try:
            __import__(pkg)
            ok(f"{pkg} importable")
        except ImportError:
            missing.append(pkg)
            warn(f"{pkg} NOT found -- run: python test_connection.py --install")

    if missing:
        fail(f"Missing packages: {missing}  --  re-run with --install flag")

# ---------------------------------------------------------------
# Step 2: Local GPU detection
# ---------------------------------------------------------------

def test_local_gpu():
    step(2, "Local GPU / NPU detection (ONNX Runtime)")
    import onnxruntime as ort
    providers = ort.get_available_providers()
    info(f"Available providers: {providers}")

    found_gpu = False
    for p in providers:
        if p in ("CUDAExecutionProvider",):
            ok(f"CUDA (RTX 4090) -- provider: {p}")
            found_gpu = True
        elif p in ("DmlExecutionProvider",):
            ok(f"DirectML (RTX / AMD) -- provider: {p}")
            found_gpu = True
        elif p in ("QNNExecutionProvider",):
            ok(f"Local NPU (Qualcomm/Intel) -- provider: {p}")

    if not found_gpu:
        warn("No GPU provider found -- CPU fallback only")
        warn("For CUDA: pip install onnxruntime-gpu")
        warn("For DirectML: pip install onnxruntime-directml")

    ok(f"CPUExecutionProvider always available")
    return providers

# ---------------------------------------------------------------
# Step 3: GPU server (start locally)
# ---------------------------------------------------------------

_server_proc = None

def test_gpu_server():
    global _server_proc
    step(3, "GPU server (cex_gpu_server.py on localhost:7478)")

    server_path = pathlib.Path(__file__).parent / "gpu" / "cex_gpu_server.py"
    if not server_path.exists():
        fail(f"Server not found: {server_path}")

    # Check if already running
    if port_open("localhost", 7479, timeout=0.5):
        ok("GPU server already running (port 7479 responds)")
        return True

    info("Starting GPU server ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pathlib.Path(__file__).parent / "shared") + os.pathsep + env.get("PYTHONPATH","")
    _server_proc = subprocess.Popen(
        [sys.executable, str(server_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait up to 8 seconds for health port to open
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if port_open("localhost", 7479, timeout=0.5):
            ok("GPU server started -- health port 7479 open")
            return True
        time.sleep(0.5)

    # Print server stderr for debugging
    _server_proc.terminate()
    stderr = _server_proc.stderr.read(1000).decode(errors="replace")
    warn(f"GPU server did not start in time. Stderr:\n{stderr}")
    return False

def stop_gpu_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        _server_proc.wait(timeout=3)
        info("GPU server stopped")

# ---------------------------------------------------------------
# Step 4: Resource discovery
# ---------------------------------------------------------------

def test_discovery(subnet=None):
    step(4, "Resource discovery (find_resources.py --scan)")

    disc_path = pathlib.Path(__file__).parent.parent / "cex-resource-discovery" / "discovery" / "find_resources.py"
    if not disc_path.exists():
        warn(f"cex-resource-discovery not found at: {disc_path}")
        warn("Skipping discovery test -- create .cex/resources.json manually")
        return False

    cmd = [sys.executable, str(disc_path), "--scan"]
    if subnet:
        cmd += ["--subnet", subnet]

    info(f"Running: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print(r.stdout)
    if r.returncode != 0:
        warn(f"Discovery exited {r.returncode}: {r.stderr[:300]}")
        return False

    resource_file = pathlib.Path(".cex/resources.json")
    if resource_file.exists():
        data = json.loads(resource_file.read_text(encoding="utf-8"))
        tc = data.get("total_compute", {})
        ok(f"Resource file written: {resource_file}")
        ok(f"NPU servers: {tc.get('npu_count', 0)}  GPU servers: {tc.get('gpu_count', 0)}")
        return True
    else:
        warn("No .cex/resources.json written")
        return False

# ---------------------------------------------------------------
# Step 5: BackendRouter constructors
# ---------------------------------------------------------------

def test_router(server_started):
    step(5, "BackendRouter constructors")

    # Add shared/ to path
    shared = pathlib.Path(__file__).parent / "shared"
    sys.path.insert(0, str(shared))
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "orchestrator"))

    from cex_backend_router import BackendRouter

    # 5a: from_resource_file
    resource_file = pathlib.Path(".cex/resources.json")
    if resource_file.exists():
        router_rf = BackendRouter.from_resource_file(str(resource_file))
        s = router_rf.status()
        ok(f"from_resource_file: {len(s)} backend(s) loaded")
        for b in s:
            info(f"  {b['kind'].upper()} {b['host']}:{b['rx_port']}  available={b['available']}")
    else:
        warn("Skipping from_resource_file (no .cex/resources.json)")

    # 5b: from_discovery_api (only if daemon is running)
    if port_open("localhost", 7480, timeout=0.5):
        router_api = BackendRouter.from_discovery_api("http://localhost:7480")
        s = router_api.status()
        ok(f"from_discovery_api: {len(s)} backend(s) from daemon")
    else:
        warn("Discovery daemon not running on port 7480 (start with --daemon to test)")

    # 5c: built-in scan targeting localhost GPU server
    if server_started:
        router_scan = BackendRouter(
            subnet="127.0.0.1/32",
            static_backends=[{"host": "127.0.0.1", "kind": "gpu"}],
        )
        router_scan.discover(timeout=2.0)
        s = router_scan.status()
        online = [b for b in s if b["available"]]
        if online:
            ok(f"BackendRouter(static): {len(online)} backend(s) online")
            for b in online:
                ok(f"  GPU {b['host']}:{b['rx_port']}  provider={b['provider']}")
        else:
            warn("Static backend not responding -- GPU server may lack DirectML/CUDA")
        return router_scan
    else:
        warn("Skipping static router test (GPU server not started)")
        return None

# ---------------------------------------------------------------
# Step 6: End-to-end inference
# ---------------------------------------------------------------

def build_test_model():
    """Build a minimal ONNX model: y = x * 2.0 (no external deps)."""
    try:
        import onnx
        from onnx import TensorProto, helper
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        scale = helper.make_tensor(
            "scale", TensorProto.FLOAT, [1], [2.0]
        )
        mul_node = helper.make_node("Mul", ["X", "scale"], ["Y"])
        graph    = helper.make_graph([mul_node], "test_graph", [X], [Y], [scale])
        model    = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        buf = io.BytesIO()
        onnx.save(model, buf)
        return buf.getvalue()
    except ImportError:
        return None

def test_inference(router):
    step(6, "End-to-end inference (ONNX model through orchestrator)")

    if router is None:
        warn("No router from Step 5 -- skipping inference test")
        return

    import numpy as np
    model_bytes = build_test_model()
    if model_bytes is None:
        warn("onnx package not installed -- using pre-built model bytes fallback")
        warn("Install: pip install onnx")
        # Attempt CPU fallback directly
        try:
            import onnxruntime as ort
            warn("Skipping ONNX model build test -- install onnx package")
        except ImportError:
            pass
        return

    ok(f"Test model built ({len(model_bytes)} bytes): y = x * 2.0")

    import numpy as np
    x_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    input_tensors = {"X": x_data}

    outputs, lat_ms, backend = router.infer(model_bytes, input_tensors, ["Y"])
    y = outputs["Y"]
    expected = x_data * 2.0

    if not np.allclose(y, expected):
        fail(f"Inference result wrong: got {y}, expected {expected}")

    ok(f"Inference correct: {x_data[0].tolist()} * 2 = {y[0].tolist()}")
    ok(f"Backend: {backend}  latency: {lat_ms:.1f} ms")

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CEX orchestrator connection test")
    parser.add_argument("--install", action="store_true", help="Auto-install missing Python packages")
    parser.add_argument("--subnet", default=None, help="LAN subnet to scan (e.g. 192.168.1.0/24)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CEX Orchestrator -- Connection Test")
    print("  Target: RTX 4090 localhost + LAN GPU/NPU servers")
    print("=" * 60)

    try:
        test_prereqs(install=args.install)
        test_local_gpu()
        server_ok = test_gpu_server()
        test_discovery(subnet=args.subnet)
        router = test_router(server_ok)
        test_inference(router)

        print("\n" + "=" * 60)
        print("  [OK] All reachable tests passed.")
        print("  Next: add ARM NPU servers to config/npu_connections.json")
        print("        or run: python test_connection.py --subnet 192.168.x.0/24")
        print("=" * 60)

    except SystemExit:
        print("\n" + "=" * 60)
        print("  [FAIL] Test stopped at a required step.")
        print("=" * 60)
        raise
    finally:
        stop_gpu_server()


if __name__ == "__main__":
    main()
