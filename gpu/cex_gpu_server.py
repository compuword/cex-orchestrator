"""
cex_gpu_server.py -- GPU inference TCP server (CXNP protocol).

Same wire protocol as cex_npu_server.py.
Runs on: Windows 10/11 with RTX or AMD RX 6000+ (DirectML).
         Linux with NVIDIA (CUDA) or AMD (ROCm).

Ports:
  7478 -- CXNP inference (above 1024, expansion range from network_config.yaml)
  7479 -- HTTP health endpoint (no auth)

Usage:
  python cex_gpu_server.py [--rx-port 7478] [--health-port 7479]
"""

import argparse
import http.server
import json
import logging
import os
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from cex_gpu_executor import get_capabilities, run_inference, smoke_test
from cex_gpu_protocol import (
    CXNP_HEADER_SIZE,
    CXNP_MAGIC,
    MsgType,
    build_error,
    build_heartbeat_ack,
    build_infer_response,
    decode_header,
    decode_tensors,
    encode_tensors,
    parse_infer_request,
)

log = logging.getLogger("cex_gpu_server")

DEFAULT_RX_PORT     = int(os.environ.get("CEX_GPU_RX_PORT",     "7478"))
DEFAULT_HEALTH_PORT = int(os.environ.get("CEX_GPU_HEALTH_PORT", "7479"))
MAX_WORKERS         = int(os.environ.get("CEX_GPU_WORKERS",     "4"))
AUTH_KEY            = os.environ.get("CEX_NET_KEY", "")

# Shared state for health endpoint
_stats_lock  = threading.Lock()
_stats       = {"requests_total": 0, "requests_ok": 0, "requests_err": 0}
_gpu_util    = 0.0


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def _handle_client(conn: socket.socket, addr) -> None:
    global _gpu_util
    log.info("Client connected: %s", addr)
    try:
        while True:
            raw_hdr = _recv_exact(conn, CXNP_HEADER_SIZE)
            msg_type, payload_len = decode_header(raw_hdr)

            payload = b""
            if payload_len:
                payload = _recv_exact(conn, payload_len)

            if msg_type == MsgType.HEARTBEAT:
                ack = build_heartbeat_ack(npu_util_pct=_gpu_util * 100.0, queue_depth=0)
                conn.sendall(ack)

            elif msg_type == MsgType.INFER_REQUEST:
                with _stats_lock:
                    _stats["requests_total"] += 1
                try:
                    req = parse_infer_request(payload)
                    outputs, latency = run_inference(
                        req["model_bytes"],
                        req["input_tensors"],
                        req["output_names"] or None,
                    )
                    _gpu_util = min(1.0, _gpu_util + 0.3)
                    resp = build_infer_response(req["request_id"], outputs, latency)
                    conn.sendall(resp)
                    with _stats_lock:
                        _stats["requests_ok"] += 1
                except Exception as exc:
                    log.error("Inference error: %s", exc)
                    conn.sendall(build_error(str(exc)))
                    with _stats_lock:
                        _stats["requests_err"] += 1
                finally:
                    _gpu_util = max(0.0, _gpu_util - 0.3)

            else:
                conn.sendall(build_error(f"Unknown msg_type 0x{msg_type:02X}"))
    except (ConnectionError, OSError) as e:
        log.debug("Client %s disconnected: %s", addr, e)
    finally:
        conn.close()


def _health_server(health_port: int) -> None:
    """Minimal HTTP server on health_port returning JSON stats."""
    caps = get_capabilities()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path not in ("/health", "/"):
                self.send_response(404); self.end_headers(); return
            with _stats_lock:
                body = json.dumps({
                    **caps,
                    **_stats,
                    "gpu_util_pct": round(_gpu_util * 100, 1),
                    "uptime_s": round(time.monotonic(), 0),
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.HTTPServer(("0.0.0.0", health_port), Handler)
    log.info("Health endpoint: http://0.0.0.0:%d/health", health_port)
    srv.serve_forever()


def run_server(rx_port: int = DEFAULT_RX_PORT, health_port: int = DEFAULT_HEALTH_PORT) -> None:
    log.info("=== CEX GPU Server ===")
    caps = get_capabilities()
    log.info("Providers: %s", caps["providers"])
    log.info("Active: %s", caps["active_provider"])

    if not smoke_test():
        log.warning("Smoke test failed -- GPU may not be available, falling back to CPU")

    # Health thread
    threading.Thread(target=_health_server, args=(health_port,), daemon=True).start()

    # Inference TCP server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", rx_port))
        srv.listen(32)
        log.info("Listening on 0.0.0.0:%d (CXNP)", rx_port)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            while True:
                conn, addr = srv.accept()
                pool.submit(_handle_client, conn, addr)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="CEX GPU inference server")
    parser.add_argument("--rx-port",     type=int, default=DEFAULT_RX_PORT)
    parser.add_argument("--health-port", type=int, default=DEFAULT_HEALTH_PORT)
    args = parser.parse_args()
    run_server(rx_port=args.rx_port, health_port=args.health_port)
