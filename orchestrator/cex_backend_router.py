"""
cex_backend_router.py -- discovers and routes inference to best available backend.

Backend priority (auto-detected via LAN scan + health check):
  1. GPU server  (:7478) -- RTX / AMD RX 6000+ via DirectML
  2. NPU server  (:7474) -- Qualcomm ARM via QNN
  3. Local CPU   -- fallback via onnxruntime CPUExecutionProvider

Usage:
  # Option 1: built-in LAN scan
  router = BackendRouter(subnet="192.168.1.0/24")
  router.discover()

  # Option 2: read resource file (from cex-resource-discovery)
  router = BackendRouter.from_resource_file(".cex/resources.json")

  # Option 3: query discovery daemon API (cex-resource-discovery, port 7480)
  router = BackendRouter.from_discovery_api("http://localhost:7480")

  outputs, lat, backend = router.infer(model_bytes, input_tensors)
"""

import logging
import pathlib
import socket
import struct
import threading
import time
import urllib.request
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from cex_gpu_protocol import (
    CXNP_HEADER_SIZE,
    MsgType,
    build_heartbeat,
    build_infer_request,
    decode_header,
    parse_infer_response,
)

log = logging.getLogger("cex_router")


# ---------------------------------------------------------------
# Port assignments (mirrors network_config.yaml)
# ---------------------------------------------------------------
PORT_NPU_RX     = 7474
PORT_NPU_HEALTH = 7476
PORT_GPU_RX     = 7478
PORT_GPU_HEALTH = 7479


class BackendInfo:
    def __init__(self, host: str, rx_port: int, health_port: int, kind: str):
        self.host        = host
        self.rx_port     = rx_port
        self.health_port = health_port
        self.kind        = kind       # "gpu" | "npu"
        self.available   = False
        self.latency_ms  = 9999.0
        self.caps: dict  = {}

    def probe(self, timeout: float = 2.0) -> bool:
        try:
            url = f"http://{self.host}:{self.health_port}/health"
            with urllib.request.urlopen(url, timeout=timeout) as r:
                self.caps = json.loads(r.read())
            self.available = True
            log.info("[%s] %s:%d online. Provider: %s",
                     self.kind, self.host, self.rx_port,
                     self.caps.get("active_provider", "?"))
        except Exception as exc:
            self.available = False
            log.debug("[%s] %s:%d offline: %s", self.kind, self.host, self.rx_port, exc)
        return self.available

    def __repr__(self) -> str:
        status = "online" if self.available else "offline"
        return f"<{self.kind.upper()} {self.host}:{self.rx_port} {status}>"


class BackendRouter:
    def __init__(
        self,
        subnet: str = "192.168.1.0/24",
        static_backends: Optional[List[dict]] = None,
        probe_interval: float = 30.0,
    ):
        self._subnet   = subnet
        self._backends: List[BackendInfo] = []
        self._lock     = threading.Lock()
        self._probe_interval = probe_interval

        # Static backends: [{"host": "x.x.x.x", "kind": "gpu"}, ...]
        static_infos = []
        for b in (static_backends or []):
            kind = b.get("kind", "gpu")
            rx   = PORT_GPU_RX     if kind == "gpu" else PORT_NPU_RX
            hlt  = PORT_GPU_HEALTH if kind == "gpu" else PORT_NPU_HEALTH
            bi   = BackendInfo(b["host"], rx, hlt, kind)
            self._backends.append(bi)
            static_infos.append(bi)

        # Probe static backends immediately (don't wait for first probe_loop tick)
        if static_infos:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(static_infos)) as pool:
                pool.map(lambda bi: bi.probe(timeout=3.0), static_infos)

        # Background probe thread
        t = threading.Thread(target=self._probe_loop, daemon=True)
        t.start()

    # ---------------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------------
    def discover(self, timeout: float = 2.0) -> List[BackendInfo]:
        """Scan LAN subnet for GPU (:7478) and NPU (:7474) servers."""
        import ipaddress, concurrent.futures

        net = ipaddress.ip_network(self._subnet, strict=False)
        hosts = list(net.hosts())
        found: List[BackendInfo] = []

        def probe_host(ip):
            candidates = [
                BackendInfo(str(ip), PORT_GPU_RX, PORT_GPU_HEALTH, "gpu"),
                BackendInfo(str(ip), PORT_NPU_RX, PORT_NPU_HEALTH, "npu"),
            ]
            live = []
            for b in candidates:
                try:
                    s = socket.create_connection((b.host, b.health_port), timeout=0.5)
                    s.close()
                    live.append(b)
                except OSError:
                    pass
            return live

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            for results in pool.map(probe_host, hosts):
                found.extend(results)

        with self._lock:
            existing = {(b.host, b.kind): b for b in self._backends}
            to_probe = []
            for b in found:
                key = (b.host, b.kind)
                if key in existing:
                    to_probe.append(existing[key])  # probe existing object
                else:
                    self._backends.append(b)
                    to_probe.append(b)

        for b in to_probe:
            b.probe(timeout=timeout)

        log.info("Discovery complete. Found: %s", [str(b) for b in found])
        return found

    def _probe_loop(self):
        while True:
            time.sleep(self._probe_interval)
            with self._lock:
                bs = list(self._backends)
            for b in bs:
                b.probe(timeout=2.0)

    # ---------------------------------------------------------------
    # Routing: pick best available backend
    # ---------------------------------------------------------------
    def _pick(self) -> Optional[BackendInfo]:
        with self._lock:
            # Priority: GPU first, then NPU
            for kind in ("gpu", "npu"):
                available = [b for b in self._backends if b.kind == kind and b.available]
                if available:
                    return min(available, key=lambda b: b.latency_ms)
        return None

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    def infer(
        self,
        model_bytes: bytes,
        input_tensors: Dict[str, np.ndarray],
        output_names: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Tuple[Dict[str, np.ndarray], float, str]:
        """
        Route inference to best backend.
        Returns (output_tensors, latency_ms, backend_label).
        Falls back to local CPU if no servers available.
        """
        backend = self._pick()
        if backend is None:
            return self._local_cpu(model_bytes, input_tensors, output_names)

        rid = request_id or _new_rid()
        try:
            outputs, lat = self._remote_infer(
                backend, rid, model_bytes, input_tensors, output_names, timeout
            )
            backend.latency_ms = lat
            return outputs, lat, f"{backend.kind}:{backend.host}"
        except Exception as exc:
            log.warning("Backend %s failed: %s -- falling back to CPU", backend, exc)
            backend.available = False
            return self._local_cpu(model_bytes, input_tensors, output_names)

    def _remote_infer(
        self,
        backend: BackendInfo,
        request_id: str,
        model_bytes: bytes,
        input_tensors: Dict[str, np.ndarray],
        output_names: Optional[List[str]],
        timeout: float,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        frame = build_infer_request(request_id, model_bytes, input_tensors, output_names)
        t0    = time.perf_counter()

        with socket.create_connection((backend.host, backend.rx_port), timeout=timeout) as s:
            s.sendall(frame)
            hdr = _recv_exact(s, CXNP_HEADER_SIZE)
            msg_type, payload_len = decode_header(hdr)
            payload = _recv_exact(s, payload_len) if payload_len else b""

        if msg_type == MsgType.ERROR:
            err_len = struct.unpack_from("<I", payload)[0]
            raise RuntimeError(payload[4:4 + err_len].decode())

        if msg_type != MsgType.INFER_RESPONSE:
            raise RuntimeError(f"Unexpected response type 0x{msg_type:02X}")

        resp = parse_infer_response(payload)
        lat  = (time.perf_counter() - t0) * 1000.0
        return resp["output_tensors"], lat

    def _local_cpu(
        self,
        model_bytes: bytes,
        input_tensors: Dict[str, np.ndarray],
        output_names: Optional[List[str]],
    ) -> Tuple[Dict[str, np.ndarray], float, str]:
        log.warning("No backends available -- running on local CPU")
        import onnxruntime as ort
        t0   = time.perf_counter()
        sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        outs = sess.run(output_names, input_tensors)
        lat  = (time.perf_counter() - t0) * 1000.0
        names = output_names or [o.name for o in sess.get_outputs()]
        return dict(zip(names, outs)), lat, "local:cpu"

    # ---------------------------------------------------------------
    # Alternative constructors (from cex-resource-discovery)
    # ---------------------------------------------------------------

    @classmethod
    def from_resource_file(
        cls,
        path: str = ".cex/resources.json",
        probe_interval: float = 30.0,
    ) -> "BackendRouter":
        """
        Build a router from a .cex/resources.json file written by
        cex-resource-discovery (python discovery/find_resources.py --scan).

        The router skips the LAN scan and starts with the pre-built inventory.
        Backends that were online at scan time are marked available immediately;
        the background probe loop keeps them up to date.
        """
        p = pathlib.Path(path)
        if not p.exists():
            log.warning("Resource file not found: %s -- starting with empty router", path)
            return cls(probe_interval=probe_interval)

        data      = json.loads(p.read_text(encoding="utf-8"))
        resources = data.get("resources", [])

        router = cls.__new__(cls)
        router._subnet         = ""
        router._backends       = []
        router._lock           = threading.Lock()
        router._probe_interval = probe_interval

        for r in resources:
            kind = r.get("kind", "gpu")
            if kind in ("local_gpu", "local_npu"):
                continue  # local resources don't have TCP backends
            b = BackendInfo(
                host        = r["host"],
                rx_port     = r.get("rx_port", PORT_GPU_RX if kind == "gpu" else PORT_NPU_RX),
                health_port = r.get("health_port", PORT_GPU_HEALTH if kind == "gpu" else PORT_NPU_HEALTH),
                kind        = kind,
            )
            b.available  = r.get("available", False)
            b.latency_ms = r.get("latency_ms", 9999.0)
            b.caps       = r.get("capabilities", {})
            router._backends.append(b)

        log.info("from_resource_file: loaded %d backends from %s", len(router._backends), path)
        t = threading.Thread(target=router._probe_loop, daemon=True)
        t.start()
        return router

    @classmethod
    def from_discovery_api(
        cls,
        url: str = "http://localhost:7480",
        probe_interval: float = 30.0,
        timeout: float = 5.0,
    ) -> "BackendRouter":
        """
        Build a router by querying the cex-resource-discovery daemon API.

        The daemon must be running:
          python discovery/find_resources.py --daemon --subnet 192.168.1.0/24

        Falls back to an empty router if the API is unreachable.
        """
        try:
            api_url = url.rstrip("/") + "/resources"
            with urllib.request.urlopen(api_url, timeout=timeout) as resp:
                data = json.loads(resp.read())
            log.info("from_discovery_api: connected to %s", url)
        except Exception as exc:
            log.warning("from_discovery_api: unreachable (%s) -- starting empty", exc)
            return cls(probe_interval=probe_interval)

        router = cls.__new__(cls)
        router._subnet         = ""
        router._backends       = []
        router._lock           = threading.Lock()
        router._probe_interval = probe_interval

        for r in data.get("resources", []):
            kind = r.get("kind", "gpu")
            if kind in ("local_gpu", "local_npu"):
                continue
            b = BackendInfo(
                host        = r["host"],
                rx_port     = r.get("rx_port", PORT_GPU_RX if kind == "gpu" else PORT_NPU_RX),
                health_port = r.get("health_port", PORT_GPU_HEALTH if kind == "gpu" else PORT_NPU_HEALTH),
                kind        = kind,
            )
            b.available  = r.get("available", False)
            b.latency_ms = r.get("latency_ms", 9999.0)
            b.caps       = r.get("capabilities", {})
            router._backends.append(b)

        log.info("from_discovery_api: loaded %d backends from %s", len(router._backends), url)
        t = threading.Thread(target=router._probe_loop, daemon=True)
        t.start()
        return router

    def status(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "kind":      b.kind,
                    "host":      b.host,
                    "rx_port":   b.rx_port,
                    "available": b.available,
                    "latency_ms": round(b.latency_ms, 1),
                    "provider":  b.caps.get("active_provider", "unknown"),
                }
                for b in self._backends
            ]


def _recv_exact(s: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Server closed connection")
        buf.extend(chunk)
    return bytes(buf)


def _new_rid() -> str:
    import uuid
    return str(uuid.uuid4())[:8]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = BackendRouter(subnet="192.168.1.0/24")
    discovered = router.discover()
    print("Backends found:", len(discovered))
    for b in discovered:
        print(" ", b)
    print("\nRouter status:")
    for s in router.status():
        print(" ", s)
