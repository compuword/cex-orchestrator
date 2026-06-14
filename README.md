# CEX Orchestrator -- Distributed GPU/NPU Inference

Routes ONNX Runtime inference from any Windows 10/11 machine to the best
available GPU or NPU server on the LAN. No cloud required.

## Architecture

```
[Client: any Windows 10/11 machine]
  ONNX Runtime app
    providers=['CexGpuExecutionProvider']  OR  ['CexNpuExecutionProvider']
        |
        v (auto-routed by BackendRouter)
        |
        +-- GPU server found? --> [Windows/Linux machine with RTX or AMD RX 6000+]
        |                          DirectML (Windows) / CUDA (Linux)
        |                          Port: 7478 (CXNP)
        |
        +-- NPU server found? --> [Qualcomm ARM Linux server]
        |                          QNN HTP (HW NPU)
        |                          Port: 7474 (CXNP)
        |
        +-- No servers?       --> Local CPUExecutionProvider (fallback)
```

## GPU Support Matrix

| Hardware | OS | Provider | Min Driver |
|----------|----|----------|-----------|
| NVIDIA RTX (any) | Windows 10/11 | DirectML | Game Ready 451+ |
| AMD Radeon RX 6000+ | Windows 10/11 | DirectML | Adrenalin 21.4+ |
| AMD Radeon RX 6000+ | Linux | ROCm | ROCm 5.4+ |
| Intel Arc A-series | Windows 10/11 | DirectML | 31.0.101+ |
| NVIDIA RTX (any) | Linux | CUDA | CUDA 11.8+ |

**DirectML** is the recommended single EP for Windows: covers RTX + AMD RX 6000+
with one codebase, no CUDA install required.

## Quick Start

### Run GPU Server (on the machine with the RTX or AMD GPU)

```powershell
pip install -r requirements.txt
python gpu/cex_gpu_server.py --rx-port 7478 --health-port 7479
```

Verify:
```
curl http://localhost:7479/health
```

### Discover + Route from a Client

```python
from orchestrator.cex_backend_router import BackendRouter

router = BackendRouter(subnet="192.168.1.0/24")
router.discover()

import onnx
model_bytes = open("my_model.onnx", "rb").read()
inputs = {"input": np.random.rand(1, 3, 224, 224).astype("float32")}

outputs, latency_ms, backend = router.infer(model_bytes, inputs)
print(f"Backend: {backend}, latency: {latency_ms:.1f}ms")
```

## Network Ports

| Port | Role | Protocol | Auth |
|------|------|----------|------|
| 7474 | NPU inference RX | TCP/CXNP | X-CEX-Key |
| 7476 | NPU health | HTTP/JSON | none |
| 7478 | GPU inference RX | TCP/CXNP | X-CEX-Key |
| 7479 | GPU health | HTTP/JSON | none |

All ports > 1024. Set `CEX_NET_KEY` env var for auth.

## Components

| Path | Description |
|------|-------------|
| `shared/cex_gpu_protocol.py` | CXNP wire protocol (same as NPU, different ports) |
| `gpu/cex_gpu_executor.py` | ONNX executor with DirectML/CUDA/ROCm auto-detection |
| `gpu/cex_gpu_server.py` | TCP inference server (port 7478) |
| `orchestrator/cex_backend_router.py` | LAN discovery + GPU/NPU routing + CPU fallback |

## Windows GPU Setup (DirectML)

DirectML is built into Windows 10/11. No extra install needed.

```powershell
pip install onnxruntime-directml
python gpu/cex_gpu_server.py
```

GPU shows in Task Manager > Performance > GPU automatically.
No kernel driver needed (unlike the NPU tab which needs cex-npu-windows).

## Linux GPU Setup (CUDA)

```bash
pip install onnxruntime-gpu   # NVIDIA CUDA
# or
pip install onnxruntime-rocm  # AMD ROCm
python gpu/cex_gpu_server.py
```

## Requirements

```
numpy>=1.24.0
onnxruntime-directml>=1.17.0  # Windows: RTX + AMD
# OR
onnxruntime-gpu>=1.17.0       # Linux NVIDIA
# OR
onnxruntime>=1.17.0           # CPU fallback only
```

## Related Projects

- [cex-npu-linux](https://github.com/compuword/cex-npu-linux) -- Qualcomm ARM NPU server
- [cex-npu-windows](https://github.com/compuword/cex-npu-windows) -- Windows NPU client + WDDM driver
