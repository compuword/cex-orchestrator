"""
cex_llama_rpc.py -- llama.cpp RPC integration for CEX orchestrator.

Two components:
  RpcComputeNode     -- remote llama-rpc-server (ARM NPU, spare GPU, CPU)
                        port 50052, binary RPC over TCP, no HTTP health
  LlamaServerManager -- manages local llama-server process with --rpc flags
                        port 8080, OpenAI-compatible HTTP API

Architecture:
  [Windows RTX 4090]                    [ARM Linux NPU]
  llama-server                          llama-rpc-server :50052
    --model llama-3.1-70b-Q4_K_M.gguf
    --rpc   192.168.1.100:50052  ------>  computes assigned layers
    --n-gpu-layers 99            (RTX handles remaining layers locally)
    --port 8080

  Result: 70B model split across 24GB VRAM + ARM HTP compute.
  Neither device could run it alone.

Model format: GGUF (quantized). NOT ONNX -- this is a separate protocol
path from CXNP. Both coexist in BackendRouter.

Ports:
  50052  llama-rpc-server (compute node, one per device)
  8080   llama-server HTTP API (orchestrator, OpenAI-compatible)

Install:
  Windows (CUDA):  .\\install\\install_llama_rpc_windows.ps1
  Linux ARM (QNN): ./server/install_llama_rpc.sh
"""

import json
import logging
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

log = logging.getLogger("cex_llama_rpc")

RPC_DEFAULT_PORT    = 50052
SERVER_DEFAULT_PORT = 8080

# Common llama-server binary locations
_BIN_CANDIDATES = [
    "llama-server",
    "llama-server.exe",
    r"C:\llama.cpp\build\bin\Release\llama-server.exe",
    r"C:\llama.cpp\build\Release\llama-server.exe",
    "/usr/local/bin/llama-server",
    "/usr/bin/llama-server",
    str(pathlib.Path.home() / "llama.cpp/build/bin/llama-server"),
    str(pathlib.Path.home() / "llama.cpp/build/llama-server"),
]

_RPC_BIN_CANDIDATES = [
    "llama-rpc-server",
    "llama-rpc-server.exe",
    r"C:\llama.cpp\build\bin\Release\llama-rpc-server.exe",
    r"C:\llama.cpp\build\Release\llama-rpc-server.exe",
    "/usr/local/bin/llama-rpc-server",
    "/usr/bin/llama-rpc-server",
    str(pathlib.Path.home() / "llama.cpp/build/bin/llama-rpc-server"),
    str(pathlib.Path.home() / "llama.cpp/build/llama-rpc-server"),
]


def _find_binary(candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if shutil.which(c):
            return shutil.which(c)
        if pathlib.Path(c).exists():
            return c
    return None


# ---------------------------------------------------------------
# RpcComputeNode
# ---------------------------------------------------------------

class RpcComputeNode:
    """
    Represents a llama-rpc-server running on a remote machine.

    Health check: TCP connect only (no HTTP endpoint in llama-rpc-server).
    Availability: port 50052 open = server alive.
    """

    def __init__(self, host: str, port: int = RPC_DEFAULT_PORT):
        self.host       = host
        self.port       = port
        self.kind       = "llama_rpc"
        self.available  = False
        self.latency_ms = 9999.0
        self.last_seen: Optional[str] = None

    def probe(self, timeout: float = 2.0) -> bool:
        t0 = time.perf_counter()
        try:
            s = socket.create_connection((self.host, self.port), timeout=timeout)
            s.close()
            self.latency_ms = (time.perf_counter() - t0) * 1000
            self.available  = True
            self.last_seen  = _now()
            log.info("[RPC] %s:%d online (%.1f ms)", self.host, self.port, self.latency_ms)
            return True
        except OSError as exc:
            self.available = False
            log.debug("[RPC] %s:%d offline: %s", self.host, self.port, exc)
            return False

    def to_rpc_arg(self) -> str:
        """Returns 'host:port' string for llama-server --rpc flag."""
        return f"{self.host}:{self.port}"

    def to_dict(self) -> dict:
        return {
            "id":         f"llama_rpc-{self.host}",
            "kind":       self.kind,
            "host":       self.host,
            "port":       self.port,
            "rx_port":    self.port,
            "health_port": self.port,
            "available":  self.available,
            "latency_ms": round(self.latency_ms, 1),
            "last_seen":  self.last_seen,
            "capabilities": {"protocol": "llama.cpp RPC", "port": self.port},
        }

    def __repr__(self) -> str:
        status = "online" if self.available else "offline"
        return f"<RPC {self.host}:{self.port} {status}>"

    @classmethod
    def scan_lan(
        cls,
        subnet: str,
        port: int = RPC_DEFAULT_PORT,
        timeout: float = 0.5,
        max_workers: int = 64,
    ) -> List["RpcComputeNode"]:
        """Scan subnet for llama-rpc-server nodes. Returns online nodes only."""
        import ipaddress
        net   = ipaddress.ip_network(subnet, strict=False)
        hosts = [str(h) for h in net.hosts()]
        found: List[RpcComputeNode] = []

        def probe_host(ip: str) -> Optional[RpcComputeNode]:
            n = cls(ip, port)
            return n if n.probe(timeout=timeout) else None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for result in pool.map(probe_host, hosts):
                if result is not None:
                    found.append(result)

        log.info("RPC scan (%s): %d node(s) found", subnet, len(found))
        return found


# ---------------------------------------------------------------
# LlamaServerManager
# ---------------------------------------------------------------

class LlamaServerManager:
    """
    Manages a llama-server process with optional RPC compute backends.

    llama-server loads the GGUF model and distributes layer computation
    across local GPU (-ngl) and remote RPC nodes (--rpc host:port,...).

    The HTTP API (port 8080) is OpenAI-compatible:
      GET  /health
      POST /v1/chat/completions
      POST /v1/completions
      GET  /v1/models
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        api_port: int = SERVER_DEFAULT_PORT,
        n_gpu_layers: int = 99,
        ctx_size: int = 4096,
        parallel: int = 4,
        llama_server_bin: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ):
        self.model_path   = pathlib.Path(model_path) if model_path else None
        self.api_port     = api_port
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size     = ctx_size
        self.parallel     = parallel
        self.extra_args   = extra_args or []
        self._bin         = llama_server_bin or _find_binary(_BIN_CANDIDATES)
        self._proc: Optional[subprocess.Popen] = None
        self._lock        = threading.Lock()

    # ---------------------------------------------------------------
    # Binary / capability discovery
    # ---------------------------------------------------------------

    @staticmethod
    def find_binary() -> Optional[str]:
        return _find_binary(_BIN_CANDIDATES)

    @staticmethod
    def find_rpc_binary() -> Optional[str]:
        return _find_binary(_RPC_BIN_CANDIDATES)

    @staticmethod
    def get_capabilities() -> dict:
        """
        Detect llama.cpp installation and available compute backends.
        Does NOT require a model -- just probes the environment.
        """
        server_bin = _find_binary(_BIN_CANDIDATES)
        rpc_bin    = _find_binary(_RPC_BIN_CANDIDATES)
        caps = {
            "llama_server_bin": server_bin,
            "llama_rpc_bin":    rpc_bin,
            "server_installed": server_bin is not None,
            "rpc_installed":    rpc_bin is not None,
            "backends": [],
        }

        if server_bin:
            try:
                r = subprocess.run(
                    [server_bin, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                caps["version"] = (r.stdout + r.stderr).strip().splitlines()[0]
            except Exception:
                caps["version"] = "unknown"

            # Detect available compute via --list-backends (llama.cpp >= b3800)
            try:
                r = subprocess.run(
                    [server_bin, "--list-backends"],
                    capture_output=True, text=True, timeout=5
                )
                raw = r.stdout + r.stderr
                for line in raw.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        caps["backends"].append(line)
            except Exception:
                pass

        return caps

    # ---------------------------------------------------------------
    # Process lifecycle
    # ---------------------------------------------------------------

    def build_command(self, rpc_nodes: Optional[List[RpcComputeNode]] = None) -> List[str]:
        if not self._bin:
            raise RuntimeError(
                "llama-server binary not found. "
                "Run: install\\install_llama_rpc_windows.ps1"
            )
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                "Download a GGUF model, e.g.: "
                "huggingface-cli download bartowski/Llama-3.1-8B-Instruct-GGUF "
                "--include 'Llama-3.1-8B-Instruct-Q4_K_M.gguf' --local-dir models/"
            )

        cmd = [
            self._bin,
            "--model",         str(self.model_path),
            "--n-gpu-layers",  str(self.n_gpu_layers),
            "--host",          "0.0.0.0",
            "--port",          str(self.api_port),
            "--ctx-size",      str(self.ctx_size),
            "--parallel",      str(self.parallel),
        ]

        if rpc_nodes:
            online = [n for n in rpc_nodes if n.available]
            if online:
                rpc_str = ",".join(n.to_rpc_arg() for n in online)
                cmd += ["--rpc", rpc_str]
                log.info("RPC backends: %s", rpc_str)

        cmd += self.extra_args
        return cmd

    def start(
        self,
        rpc_nodes: Optional[List[RpcComputeNode]] = None,
        wait_timeout: float = 60.0,
    ) -> bool:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                log.info("llama-server already running (pid %d)", self._proc.pid)
                return True

            cmd = self.build_command(rpc_nodes)
            log.info("Starting: %s", " ".join(cmd))

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

        log.info("llama-server pid=%d, waiting for /health ...", self._proc.pid)
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                out = self._proc.stdout.read(2000).decode(errors="replace")
                log.error("llama-server exited early:\n%s", out)
                return False
            try:
                url = f"http://localhost:{self.api_port}/health"
                with urllib.request.urlopen(url, timeout=1) as r:
                    if r.status == 200:
                        log.info("[OK] llama-server ready on port %d", self.api_port)
                        return True
            except Exception:
                pass
            time.sleep(1.0)

        log.error("llama-server did not become ready in %.0fs", wait_timeout)
        return False

    def stop(self):
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                log.info("llama-server stopped")
            self._proc = None

    def is_running(self) -> bool:
        if self._proc is None or self._proc.poll() is not None:
            return False
        return self.health().get("status") == "ok"

    # ---------------------------------------------------------------
    # Inference API (OpenAI-compatible)
    # ---------------------------------------------------------------

    def health(self) -> dict:
        try:
            url = f"http://localhost:{self.api_port}/health"
            with urllib.request.urlopen(url, timeout=2) as r:
                return json.loads(r.read())
        except Exception:
            return {"status": "offline"}

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> str:
        """
        OpenAI-compatible chat completion.

        messages: [{"role": "user", "content": "Hello"}, ...]
        Returns: assistant reply string
        """
        body = json.dumps({
            "model":       "local",
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{self.api_port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        return resp["choices"][0]["message"]["content"]

    def completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> str:
        """OpenAI-compatible text completion."""
        body = json.dumps({
            "prompt":      prompt,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "stream":      False,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{self.api_port}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        return resp["choices"][0]["text"]

    def list_models(self) -> List[dict]:
        """GET /v1/models -- returns loaded model info."""
        try:
            url = f"http://localhost:{self.api_port}/v1/models"
            with urllib.request.urlopen(url, timeout=2) as r:
                return json.loads(r.read()).get("data", [])
        except Exception:
            return []

    def __repr__(self) -> str:
        status = "running" if self.is_running() else "stopped"
        return f"<LlamaServer :{self.api_port} {status} model={self.model_path}>"


# ---------------------------------------------------------------
# Standalone RPC server (for running on compute node)
# ---------------------------------------------------------------

class LlamaRpcServer:
    """
    Wraps llama-rpc-server for running on a compute node.

    Used when THIS machine should act as an RPC backend
    (e.g. a second GPU machine or ARM server not running the main model).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = RPC_DEFAULT_PORT,
        bin_path: Optional[str] = None,
    ):
        self.host     = host
        self.port     = port
        self._bin     = bin_path or _find_binary(_RPC_BIN_CANDIDATES)
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        if not self._bin:
            raise RuntimeError(
                "llama-rpc-server binary not found. "
                "Run: install\\install_llama_rpc_windows.ps1"
            )
        cmd = [self._bin, "--host", self.host, "--port", str(self.port)]
        log.info("Starting RPC server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd)
        time.sleep(1.0)
        return self._proc.poll() is None

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            log.info("llama-rpc-server stopped")

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ---------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------

def get_capabilities() -> dict:
    return LlamaServerManager.get_capabilities()


def scan_rpc_nodes(subnet: str, port: int = RPC_DEFAULT_PORT) -> List[RpcComputeNode]:
    return RpcComputeNode.scan_lan(subnet, port=port)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    caps = get_capabilities()
    print("\n=== llama.cpp RPC capabilities ===")
    print(f"  llama-server:     {caps['llama_server_bin'] or 'NOT FOUND'}")
    print(f"  llama-rpc-server: {caps['llama_rpc_bin']    or 'NOT FOUND'}")
    if caps.get("version"):
        print(f"  Version:          {caps['version']}")
    if caps.get("backends"):
        print(f"  Backends:         {', '.join(caps['backends'])}")
    if not caps["server_installed"]:
        print("\n  [!!] Install: .\\install\\install_llama_rpc_windows.ps1")
